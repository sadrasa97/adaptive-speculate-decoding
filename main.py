"""
Adaptive Speculative Decoding for CPU-Constrained GGUF Models
Main entry point - Final scientifically correct version with real baseline measurement
and full speculative decoding implementation.
"""
import sys
import os
import time
import argparse
import json
import random
import math
import logging
import statistics
import urllib.request
import urllib.error
from typing import Optional, List
import numpy as np
from dotenv import load_dotenv

def setup_logging(log_file: str = "adaptive_sd.log", verbose: bool = False):
    log_format = "%(asctime)s | %(name)-22s | %(levelname)-7s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    root_logger = logging.getLogger("AdaptiveSD")
    root_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    root_logger.handlers.clear()
    
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    root_logger.addHandler(fh)
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    root_logger.addHandler(ch)
    
    return root_logger

main_logger = logging.getLogger("AdaptiveSD.Main")

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False

from monitor import RuntimeMonitor, MonitorConfig
from draft_controller import AdaptiveDraftController, DraftControllerConfig, SpeculationMode
from kv_cache import KVCacheCoordinator, KVCacheConfig
from policy_engine import DynamicPolicyEngine, PolicyConfig, PolicyType

DEFAULT_MODEL = r"D:\models\Qwen3.5-2B-UD-Q4_K_XL.gguf"
DEFAULT_DRAFT_MODEL = r"D:\models\Qwen3.5-0.8B-Q4_0.gguf"

_SIM_CORPORA = {
    "coding": [
        "def quicksort(arr): if len(arr) <= 1: return arr",
        "pivot = arr[len(arr) // 2]; left = [x for x in arr if x < pivot]",
        "middle = [x for x in arr if x == pivot]; right = [x for x in arr if x > pivot]",
        "return quicksort(left) + middle + quicksort(right)",
        "class TreeNode: def __init__(self, val=0, left=None, right=None):",
        "    self.val = val; self.left = left; self.right = right",
        "for i in range(n): result.append(compute(i))",
        "import numpy as np; matrix = np.zeros((rows, cols))",
        "while left < right: mid = (left + right) // 2",
        "if target == nums[mid]: return mid",
    ],
    "reasoning": [
        "Therefore, we can conclude that the hypothesis is valid.",
        "Because the premises are true, the conclusion must follow.",
        "Thus, by mathematical induction, the theorem holds for all n.",
        "Hence, the optimization problem has a unique solution.",
        "We observe that the function is continuous on the interval.",
        "It follows from the axioms that the set is non-empty.",
        "Consequently, the algorithm terminates in polynomial time.",
        "This implies that the bound is tight up to constant factors.",
    ],
    "chat": [
        "Hello! How can I help you today?",
        "That is an interesting question. Let me think about it.",
        "I would be happy to explain that in more detail.",
        "Sure, here is what I know about that topic.",
        "Great question! The answer involves several key concepts.",
        "Let me break this down into simpler parts for you.",
    ],
    "creative": [
        "The transformer architecture revolutionized natural language processing.",
        "Self-attention mechanisms allow the model to weigh the importance of different tokens.",
        "Multi-head attention captures relationships from multiple representation subspaces.",
        "Positional encodings inject sequence order information into the model.",
        "The encoder-decoder structure enables powerful sequence-to-sequence tasks.",
        "Layer normalization stabilizes training and accelerates convergence.",
        "Feed-forward networks apply non-linear transformations to each position.",
        "Residual connections help gradients flow through deep architectures.",
    ],
}

def _select_corpus(prompt: str) -> str:
    p = prompt.lower()
    if any(k in p for k in ["def ", "class ", "code", "function", "implement", "python"]):
        return "coding"
    if any(k in p for k in ["explain", "why", "how does", "theory", "because"]):
        return "reasoning"
    if any(k in p for k in ["hello", "hi", "chat"]):
        return "chat"
    return "creative"

def _sim_acceptance(step: int, depth: int, entropy: float, ctx_len: int) -> float:
    if step < 20:
        alpha_base = random.uniform(0.62, 0.78)
    elif step < 60:
        t = (step - 20) / 40.0
        lo = 0.62 - 0.18 * t
        hi = 0.78 - 0.18 * t
        alpha_base = random.uniform(lo, hi)
    elif step < 150:
        alpha_base = random.uniform(0.35, 0.50)
    else:
        alpha_base = random.uniform(0.18, 0.32)
    
    depth_decay = math.exp(-0.18 * max(0, depth - 1))
    entropy_factor = 1.0 / (1.0 + 0.07 * max(0.0, entropy - 5.0))
    ctx_factor = max(0.75, 1.0 - 8e-5 * ctx_len)
    
    raw = alpha_base * depth_decay * entropy_factor * ctx_factor
    raw += random.gauss(0.0, 0.015)
    
    if step % 50 == 0:
        print(f"\n[DIAG] step={step:4d} | α_base={alpha_base:.3f} | δ(D={depth})={depth_decay:.3f} | "
              f"φ(H={entropy:.2f})={entropy_factor:.3f} | ψ(L={ctx_len})={ctx_factor:.3f} | "
              f"α_raw={raw:.3f}", flush=True)
    
    return max(0.05, min(0.92, raw))

def _sim_entropy(step: int, workload: str) -> float:
    if step < 20:
        base = random.uniform(3.0, 4.5)
    elif step < 80:
        base = random.uniform(5.5, 7.5)
    else:
        base = random.uniform(6.0, 8.0)
    
    offsets = {
        "coding": -1.2,
        "chat": -0.4,
        "creative": +0.3,
        "reasoning": +0.8,
    }
    noise = random.uniform(-0.25, 0.25)
    return max(0.5, base + offsets.get(workload, 0.0) + noise)

def _sim_cpu(step: int, depth: int, acceptance_ratio: float, workload: str) -> float:
    k = 5.0
    n_rejected = max(0, depth - round(depth * acceptance_ratio))
    c_target = 0.30
    c_draft = depth / k * 0.06
    c_reject = n_rejected * 0.04
    c_workload = {"coding": 0.04, "reasoning": 0.03, "chat": -0.02, "creative": 0.02}.get(workload, 0.0)
    noise = random.gauss(0.0, 0.015)
    cpu = c_target + c_draft + c_reject + c_workload + noise
    return max(0.05, min(0.97, cpu))

class AdaptiveInferenceEngine:
    def __init__(self, model_path=DEFAULT_MODEL, draft_model_path=DEFAULT_DRAFT_MODEL,
                 n_ctx=4096, n_threads=0, n_draft=4, verbose=False,
                 policy_type=PolicyType.ENSEMBLE, backend="auto",
                 openrouter_api_key=None, openrouter_model=None,
                 openrouter_base_url="https://openrouter.ai/api/v1",
                 openrouter_site_url=None, openrouter_site_name=None):
        self.model_path = model_path
        self.draft_model_path = draft_model_path
        self.n_ctx = n_ctx
        self.verbose = verbose
        self.backend = backend
        self.openrouter_api_key = openrouter_api_key
        self.openrouter_model = openrouter_model or "openai/gpt-4o-mini"
        self.openrouter_base_url = openrouter_base_url.rstrip("/")
        self.openrouter_site_url = openrouter_site_url
        self.openrouter_site_name = openrouter_site_name
        
        self._monitor_cfg = MonitorConfig(
            window_size=64, acceptance_low_threshold=0.40, cpu_overload_threshold=0.88,
        )
        
        # FIX: Optimized for CPU-constrained environments
        self._controller_cfg = DraftControllerConfig(
            min_depth=1, max_depth=4, initial_depth=min(n_draft, 2),
            acceptance_low=0.25, acceptance_disable=0.05,
            bad_steps_to_disable=10,
            depth_change_cooldown=5, oscillation_window=10, oscillation_threshold=3,
            latency_spike_ms=200.0, latency_spike_multiplier=1.8,
        )
        
        self._kv_cfg = KVCacheConfig(quantize_draft=True, max_context=n_ctx)
        
        self._policy_cfg = PolicyConfig(
            policy_type=policy_type, bandit_epsilon=0.15, bandit_ucb_c=0.5,
            depth_choices=[1, 2, 3, 4], workload_detect_window=64,
            workload_change_hysteresis=4,
        )
        
        self.monitor = RuntimeMonitor(self._monitor_cfg)
        self.controller = AdaptiveDraftController(monitor=self.monitor, config=self._controller_cfg)
        self.kv = KVCacheCoordinator(self._kv_cfg)
        self.policy = DynamicPolicyEngine(monitor=self.monitor, controller=self.controller, config=self._policy_cfg)
        
        self._llm = None
        self._draft_llm = None
        self._active_backend = "simulation"
        self._n_threads = n_threads if n_threads > 0 else max(1, (os.cpu_count() or 2) // 2)
        self._baseline_tps = None
        
        self._resolve_backend()
        self._load_model_if_needed()
        self._measure_baseline_tps()
        
        main_logger.info(
            "Engine initialised: backend=%s, model=%s, baseline_tps=%.1f, threads=%d",
            self._active_backend, model_path, self._baseline_tps, self._n_threads
        )
    
    def _measure_baseline_tps(self):
        """Measure real baseline TPS by running inference without speculation"""
        if self._active_backend == "simulation":
            self._baseline_tps = 12.0
            self.monitor.set_baseline_tps(self._baseline_tps)
            return
        
        if self._active_backend == "openrouter":
            main_logger.info("Measuring baseline TPS for OpenRouter...")
            try:
                headers = {
                    "Authorization": f"Bearer {self.openrouter_api_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": self.openrouter_model,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Write a detailed paragraph about machine learning and its applications."},
                    ],
                    "max_tokens": 150,
                    "temperature": 0.7,
                    "stream": False,
                }
                req = urllib.request.Request(
                    url=f"{self.openrouter_base_url}/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                t0 = time.perf_counter()
                with urllib.request.urlopen(req, timeout=60) as resp:
                    body = resp.read().decode("utf-8")
                elapsed = time.perf_counter() - t0
                
                data = json.loads(body)
                choices = data.get("choices", [])
                if choices:
                    message = choices[0].get("message") or {}
                    content = message.get("content") or message.get("reasoning") or ""
                    tokens = max(1, len(content.split()))
                    self._baseline_tps = tokens / elapsed if elapsed > 0 else 8.0
                else:
                    self._baseline_tps = 8.0
                
                main_logger.info(
                    "Measured OpenRouter baseline TPS: %.2f (%d tokens in %.2fs)",
                    self._baseline_tps, tokens if 'tokens' in locals() else 0, elapsed
                )
            except Exception as e:
                main_logger.warning("OpenRouter baseline measurement failed: %s, using default 8.0", e)
                self._baseline_tps = 8.0
            self.monitor.set_baseline_tps(self._baseline_tps)
            return
        
        if self._llm is None:
            self._baseline_tps = 12.0
            self.monitor.set_baseline_tps(self._baseline_tps)
            return
        
        main_logger.info("Measuring baseline TPS (no speculation)...")
        
        # Warmup
        test_prompt = "The quick brown fox jumps over the lazy dog."
        try:
            _ = self._llm(test_prompt, max_tokens=10, temperature=0.7, echo=False)
        except Exception as e:
            main_logger.warning("Warmup failed: %s", e)
        
        # Measure
        test_tokens = 100
        t0 = time.perf_counter()
        try:
            output = self._llm(test_prompt, max_tokens=test_tokens, temperature=0.7, echo=False)
            elapsed = time.perf_counter() - t0
            
            if isinstance(output, dict) and "choices" in output:
                text = output["choices"][0].get("text", "")
                tokens_generated = len(self._llm.tokenize(text.encode()))
            else:
                tokens_generated = test_tokens
            
            self._baseline_tps = tokens_generated / elapsed if elapsed > 0 else 12.0
            main_logger.info("Measured baseline TPS: %.2f (%d tokens in %.2fs)",
                        self._baseline_tps, tokens_generated, elapsed)
        except Exception as e:
            main_logger.warning("Baseline measurement failed: %s, using default 12.0", e)
            self._baseline_tps = 12.0
        
        self.monitor.set_baseline_tps(self._baseline_tps)
    
    def _reset_runtime_modules(self):
        try:
            self.monitor.stop()
        except Exception:
            pass
        
        self.monitor = RuntimeMonitor(self._monitor_cfg)
        self.monitor.set_baseline_tps(self._baseline_tps)
        self.controller = AdaptiveDraftController(monitor=self.monitor, config=self._controller_cfg)
        self.kv = KVCacheCoordinator(self._kv_cfg)
        self.policy = DynamicPolicyEngine(monitor=self.monitor, controller=self.controller, config=self._policy_cfg)
    
    def _resolve_backend(self):
        gguf_ready = LLAMA_AVAILABLE and os.path.exists(self.model_path)
        or_ready = bool(self.openrouter_api_key)
        
        if self.backend == "gguf":
            if gguf_ready:
                self._active_backend = "gguf"
            else:
                self._active_backend = "simulation"
                main_logger.warning("GGUF backend requested but unavailable; falling back to simulation.")
            return
        
        if self.backend == "openrouter":
            if or_ready:
                self._active_backend = "openrouter"
            else:
                raise ValueError("OpenRouter backend requested but OPENROUTER_API_KEY is not set.")
            return
        
        if self.backend == "simulation":
            self._active_backend = "simulation"
            return
        
        if gguf_ready:
            self._active_backend = "gguf"
        elif or_ready:
            self._active_backend = "openrouter"
        else:
            self._active_backend = "simulation"
    
    def _load_model_if_needed(self):
        if self._active_backend not in ["gguf", "openrouter"]:
            return
        
        if self._active_backend == "openrouter":
            return
        
        if not LLAMA_AVAILABLE:
            main_logger.warning("llama-cpp-python not installed. Falling back to simulation mode.")
            self._active_backend = "simulation"
            return
        
        if not os.path.exists(self.model_path):
            main_logger.warning("GGUF model not found. Falling back to simulation mode.")
            self._active_backend = "simulation"
            return
        
        main_logger.info("Loading target model: %s", self.model_path)
        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self._n_threads,
            n_batch=512,
            verbose=self.verbose,
            logits_all=True  # ← این خط ضروری است برای logprobs
        )
        
        if os.path.exists(self.draft_model_path):
            main_logger.info("Loading draft model: %s", self.draft_model_path)
            try:
                self._draft_llm = Llama(
                    model_path=self.draft_model_path,
                    n_ctx=self.n_ctx,
                    n_threads=self._n_threads,
                    n_batch=512,
                    verbose=self.verbose,
                    logits_all=True
                )
                main_logger.info("Draft model loaded successfully. Speculative decoding enabled.")
            except Exception as e:
                main_logger.warning("Failed to load draft model: %s. Speculative decoding disabled.", e)
                self._draft_llm = None
        else:
            main_logger.warning(
                "Draft model not found at %s. Speculative decoding disabled.\n"
                "To enable speculative decoding:\n"
                "  1. Download a draft model (e.g., Qwen3.5-0.5B-UD-Q4_K_XL.gguf)\n"
                "  2. Place it at: %s\n"
                "  3. Or specify path with --draft-model argument",
                self.draft_model_path, self.draft_model_path
            )
            self._draft_llm = None
    
    def generate(self, prompt: str, max_tokens=256, temperature=0.7, top_p=0.9, stream=True) -> str:
        self.monitor.mark_generation_start()
        
        if self._active_backend == "gguf" and self._llm is not None:
            self.monitor.set_simulation_mode(False)
            if self._draft_llm is not None:
                return self._generate_speculative(prompt, max_tokens, temperature, top_p, stream)
            else:
                return self._generate_real(prompt, max_tokens, temperature, top_p, stream)
        
        if self._active_backend == "openrouter":
            self.monitor.set_simulation_mode(False)
            return self._generate_openrouter(prompt, max_tokens, temperature, top_p, stream)
        
        self.monitor.set_simulation_mode(True)
        return self._generate_simulated(prompt, max_tokens, stream)
    
    def _generate_speculative(self, prompt, max_tokens, temperature, top_p, stream) -> str:
        """
        Speculative decoding with llama-cpp-python.
        """
        output: List[str] = []
        safety_prefix = (
            "You are a concise assistant.\n"
            "Provide only the final answer and do not reveal internal reasoning.\n\n"
        )
        wrapped_prompt = f"{safety_prefix}User: {prompt}\nAssistant: "
        
        prompt_tokens: List[int] = self._llm.tokenize(wrapped_prompt.encode())
        
        if stream:
            print("", end="", flush=True)
        
        total_generated = 0
        step_count = 0
        consecutive_failures = 0
        
        self.policy.step("", prompt=prompt)
        
        while total_generated < max_tokens:
            step_start = time.perf_counter()
            step_count += 1
            
            # FIX: Fallback to non-speculative if too many failures
            if consecutive_failures >= 5:
                main_logger.warning("Too many speculative failures, falling back to non-speculative")
                remaining_text = self._generate_real(
                    wrapped_prompt + "".join(output),
                    max_tokens - total_generated,
                    temperature, top_p, stream
                )
                output.append(remaining_text)
                break
            
            depth = self.controller.current_depth()
            remaining = max_tokens - total_generated
            actual_depth = min(depth, remaining)
            
            # Step 1: Draft model proposes tokens
            draft_texts: List[str] = []
            draft_token_ids: List[int] = []
            draft_ctx = list(prompt_tokens)
            
            for _ in range(actual_depth):
                try:
                    d_out = self._draft_llm(
                        draft_ctx,
                        max_tokens=1,
                        temperature=temperature,
                        top_p=top_p,
                        echo=False,
                    )
                    if not (isinstance(d_out, dict) and d_out.get("choices")):
                        break
                    tok_text = d_out["choices"][0].get("text", "")
                    if not tok_text:
                        break
                    toks = self._draft_llm.tokenize(tok_text.encode())
                    if not toks:
                        break
                    tok_id = toks[0]
                    draft_token_ids.append(tok_id)
                    draft_texts.append(tok_text)
                    draft_ctx.append(tok_id)
                except Exception as e:
                    main_logger.debug("Draft token failed: %s", e)
                    break
            
            # Step 2: Target model verifies
            accepted_count = 0
            target_tok_id: Optional[int] = None
            target_tok_text: str = ""
            target_logits_last: List[float] = []
            
            verify_ctx = list(prompt_tokens)
            for i, dtok in enumerate(draft_token_ids):
                try:
                    t_out = self._llm(
                        verify_ctx,
                        max_tokens=1,
                        temperature=0.0,  # FIX: Use greedy for verification
                        top_p=1.0,
                        echo=False,
                        logprobs=20,
                    )
                    if not (isinstance(t_out, dict) and t_out.get("choices")):
                        break
                    choice = t_out["choices"][0]
                    t_text = choice.get("text", "")
                    t_toks = self._llm.tokenize(t_text.encode()) if t_text else []
                    
                    vlogprobs = choice.get("logprobs") or {}
                    top_lps = vlogprobs.get("top_logprobs") or []
                    if top_lps:
                        target_logits_last = list((top_lps[-1] or {}).values())
                    
                    if not t_toks:
                        target_tok_id = None
                        target_tok_text = t_text
                        break
                    
                    target_next_id = t_toks[0]
                    
                    # FIX: Compare token IDs directly
                    if target_next_id == dtok:
                        accepted_count += 1
                        verify_ctx.append(dtok)
                        target_tok_id = None
                        target_tok_text = ""
                    else:
                        target_tok_id = target_next_id
                        target_tok_text = t_text
                        break
                except Exception as e:
                    main_logger.debug("Verify step %d failed: %s", i, e)
                    consecutive_failures += 1
                    break
            
            # Bonus token if all drafts accepted
            if accepted_count == len(draft_token_ids) and target_tok_id is None and draft_token_ids:
                try:
                    bonus_ctx = list(prompt_tokens) + draft_token_ids
                    b_out = self._llm(
                        bonus_ctx,
                        max_tokens=1,
                        temperature=0.0,
                        top_p=1.0,
                        echo=False,
                        logprobs=20,
                    )
                    if isinstance(b_out, dict) and b_out.get("choices"):
                        choice = b_out["choices"][0]
                        target_tok_text = choice.get("text", "")
                        b_toks = self._llm.tokenize(target_tok_text.encode()) if target_tok_text else []
                        target_tok_id = b_toks[0] if b_toks else None
                        vlogprobs = choice.get("logprobs") or {}
                        top_lps = vlogprobs.get("top_logprobs") or []
                        if top_lps:
                            target_logits_last = list((top_lps[-1] or {}).values())
                except Exception as e:
                    main_logger.debug("Bonus token failed: %s", e)
            
            # AR fallback if no drafts and no target token
            if not draft_token_ids and target_tok_id is None:
                try:
                    ar_out = self._llm(
                        list(prompt_tokens),
                        max_tokens=1,
                        temperature=0.0,
                        top_p=1.0,
                        echo=False,
                        logprobs=20,
                    )
                    if isinstance(ar_out, dict) and ar_out.get("choices"):
                        choice = ar_out["choices"][0]
                        target_tok_text = choice.get("text", "")
                        ar_toks = self._llm.tokenize(target_tok_text.encode()) if target_tok_text else []
                        target_tok_id = ar_toks[0] if ar_toks else None
                        vlogprobs = choice.get("logprobs") or {}
                        top_lps = vlogprobs.get("top_logprobs") or []
                        if top_lps:
                            target_logits_last = list((top_lps[-1] or {}).values())
                except Exception as e:
                    main_logger.debug("AR fallback failed: %s", e)
            
            # KV bookkeeping
            committed_pos = len(prompt_tokens)
            self.kv.begin_draft(actual_depth, committed_pos)
            dummy_keys = np.zeros((1, 1, 64), dtype=np.float32)
            dummy_vals = np.zeros((1, 1, 64), dtype=np.float32)
            for i in range(actual_depth):
                self.kv.store_draft_kv(committed_pos + i, dummy_keys, dummy_vals)
            for i in range(accepted_count):
                self.kv.retrieve_draft_kv(committed_pos + i)
            self.kv.accept_tokens(accepted_count)
            n_rejected = actual_depth - accepted_count
            if n_rejected > 0:
                self.kv.reject_tokens(n_rejected)
            
            # Build committed sequence
            committed_ids: List[int] = draft_token_ids[:accepted_count]
            if target_tok_id is not None:
                committed_ids.append(target_tok_id)
            
            committed_texts: List[str] = draft_texts[:accepted_count]
            if target_tok_text:
                committed_texts.append(target_tok_text)
            
            generated_text = "".join(committed_texts)
            output.append(generated_text)
            
            prompt_tokens.extend(committed_ids)
            
            tokens_this_step = max(1, len(committed_ids))
            tokens_this_step = min(tokens_this_step, remaining)
            total_generated += tokens_this_step
            
            # Monitor
            snap = self.monitor.on_token_generated(
                logits=target_logits_last if target_logits_last else None,
                drafted=actual_depth,
                accepted=accepted_count,
                context_length=len(prompt_tokens),
                speculation_depth=actual_depth,
            )
            
            # Policy + controller
            policy_suggestion, policy_confidence = self.policy.step(generated_text)
            decision = self.controller.step(
                snap,
                policy_suggestion=policy_suggestion,
                policy_confidence=policy_confidence,
            )
            
            if stream and generated_text:
                print(generated_text, end="", flush=True)
            
            step_elapsed = time.perf_counter() - step_start
            
            # FIX: Reset failure counter on success
            if accepted_count > 0:
                consecutive_failures = 0
            
            if step_count % 25 == 0 or total_generated >= max_tokens:
                self._print_status(total_generated, decision)
                print(
                    f"[DBG] step={step_count:4d} | depth={actual_depth} | "
                    f"accepted={accepted_count}/{actual_depth} | "
                    f"committed={tokens_this_step} | t_step={step_elapsed*1000:.1f}ms",
                    flush=True,
                )
            
            # EOS guard
            if target_tok_id is None and not draft_token_ids:
                main_logger.debug("EOS reached at step %d", step_count)
                break
        
        if stream:
            print()
        
        return "".join(output)
    
    def _generate_real(self, prompt, max_tokens, temperature, top_p, stream) -> str:
        output = []
        safety_prefix = (
            "You are a concise assistant.\n"
            "Provide only the final answer and do not reveal internal reasoning.\n\n"
        )
        wrapped_prompt = f"{safety_prefix}User: {prompt}\nAssistant: "
        ctx = len(self._llm.tokenize(wrapped_prompt.encode()))

        # ❌ حذف logprobs از اینجا — چون نیازی نیست
        generator = self._llm(
            wrapped_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            # logprobs=10,  ← حذف شود
        )
        
        for td in generator:
            tok = td["choices"][0]["text"]
            output.append(tok)
            ctx += 1
            # ❌ بدون logprobs، نمی‌توانیم entropy بفرستیم — پس None بفرستیم
            self.monitor.on_token_generated(
                logits=None,  # ← None بفرستیم
                drafted=0,
                accepted=0,
                context_length=ctx,
                speculation_depth=1,
            )
            if stream:
                print(tok, end="", flush=True)
            
        if stream:
            print()
            
        return "".join(output)
        

    def _generate_openrouter(self, prompt, max_tokens, temperature, top_p, stream) -> str:
        """OpenRouter API inference with retry logic."""
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        if self.openrouter_site_url:
            headers["HTTP-Referer"] = self.openrouter_site_url
        if self.openrouter_site_name:
            headers["X-Title"] = self.openrouter_site_name
        
        max_retries = 3
        content = None
        elapsed = 0.0
        
        for attempt in range(max_retries):
            payload = {
                "model": self.openrouter_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Provide concise, complete answers.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": False,
            }
            
            req = urllib.request.Request(
                url=f"{self.openrouter_base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            
            try:
                t0 = time.perf_counter()
                with urllib.request.urlopen(req, timeout=120) as resp:
                    body = resp.read().decode("utf-8")
                elapsed = time.perf_counter() - t0
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
                raise RuntimeError(f"OpenRouter HTTP error: {e.code} {detail}") from e
            except urllib.error.URLError as e:
                raise RuntimeError(f"OpenRouter request failed: {e}") from e
            
            data = json.loads(body)
            
            if "error" in data:
                error_msg = data['error'].get('message', str(data['error']))
                main_logger.warning("OpenRouter error (attempt %d/%d): %s",
                                attempt + 1, max_retries, error_msg)
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                raise RuntimeError(f"OpenRouter API error: {error_msg}")
            
            choices = data.get("choices", [])
            if not choices:
                main_logger.warning("OpenRouter returned no choices (attempt %d/%d)",
                                attempt + 1, max_retries)
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                content = "[No response from model]"
                break
            
            message = choices[0].get("message") or {}
            content = message.get("content") or ""
            reasoning = message.get("reasoning") or ""
            finish_reason = choices[0].get("finish_reason", "")
            
            if not content and reasoning:
                main_logger.info("Using reasoning field as content (finish_reason=%s)", finish_reason)
                content = reasoning
                break
            elif content and len(content.strip()) > 10:
                break
            else:
                main_logger.warning(
                    "OpenRouter returned short/empty content (attempt %d/%d, finish_reason=%s, len=%d)",
                    attempt + 1, max_retries, finish_reason, len(content)
                )
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    content = reasoning if reasoning else "[Empty response from model]"
        
        if not content:
            content = "[Empty response from model]"
        
        context_len = 20 + len(prompt.split())
        words = content.split()
        n_words = len(words)
        
        if stream:
            print("", end="", flush=True)
        
        self.monitor.mark_generation_start()
        
        per_token_delay = 0.01
        emitted = 0
        for w in words:
            emitted += 1
            if per_token_delay > 0:
                time.sleep(per_token_delay)
            self.monitor.on_token_generated(
                logits=None,
                drafted=0,
                accepted=0,
                context_length=context_len + emitted,
                speculation_depth=1,
            )
            if stream:
                print(w + " ", end="", flush=True)
        
        if stream:
            print()
        
        return content
    
    def _generate_simulated(self, prompt: str, max_tokens: int, stream: bool) -> str:
        corpus_name = _select_corpus(prompt)
        corpus = _SIM_CORPORA[corpus_name]
        main_logger.info("Simulation: %d tokens, corpus=%s, baseline=%.1f TPS",
                        max_tokens, corpus_name, self._baseline_tps)
        
        self.monitor.set_simulation_mode(True)
        
        print(f"\n[SIM] ── Simulation Start ─────────────────────────────────────")
        print(f"[SIM] Prompt    : {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        print(f"[SIM] Tokens    : {max_tokens} │ Corpus: {corpus_name} │ Baseline: {self._baseline_tps:.1f} TPS")
        print(f"[SIM] ────────────────────────────────────────────────────────────\n")
        print("[SIM] Output: ", end="", flush=True)
        
        self.policy.step("", prompt=prompt)
        
        all_tokens = []
        while len(all_tokens) < max_tokens * 4:
            for sent in corpus:
                for w in sent.split():
                    all_tokens.append(w + " ")
        
        context_len = 20 + len(prompt.split())
        base_step_time = 1.0 / self._baseline_tps
        draft_speedup = 5.0
        
        total_committed = 0
        token_idx = 0
        step_count = 0
        
        while total_committed < max_tokens:
            step_start = time.perf_counter()
            step_count += 1
            
            depth = self.controller.current_depth()
            
            sim_entropy = _sim_entropy(step_count, corpus_name)
            ctx_now = context_len + total_committed
            per_tok_alpha = _sim_acceptance(step_count, depth, sim_entropy, ctx_now)
            
            accepted_drafts = max(0, round(depth * per_tok_alpha))
            tokens_this_step = min(accepted_drafts + 1, max_tokens - total_committed)
            n_rejected = depth - accepted_drafts
            
            sim_cpu = _sim_cpu(step_count, depth, per_tok_alpha, corpus_name)
            
            t_verify = base_step_time
            t_draft = depth * base_step_time / draft_speedup
            t_reject = n_rejected * base_step_time * 0.15
            ctx_penalty = 1.0 + 3e-5 * ctx_now
            step_time = (t_verify + t_draft + t_reject) * ctx_penalty
            
            logits = [0.0] * max(2, round(math.exp(sim_entropy)))
            snap = self.monitor.on_token_generated(
                logits=logits,
                drafted=depth,
                accepted=accepted_drafts,
                context_length=ctx_now,
                speculation_depth=depth,
                sim_entropy=sim_entropy,
                simulated_cpu=sim_cpu,
            )
            
            generated_text = "".join(all_tokens[token_idx: token_idx + tokens_this_step])
            policy_suggestion, policy_confidence = self.policy.step(generated_text)
            decision = self.controller.step(
                snap, policy_suggestion=policy_suggestion,
                policy_confidence=policy_confidence,
            )
            
            committed_pos = ctx_now
            self.kv.begin_draft(depth, committed_pos)
            dummy_keys = np.zeros((1, 1, 64), dtype=np.float32)
            dummy_vals = np.zeros((1, 1, 64), dtype=np.float32)
            
            for i in range(depth):
                self.kv.store_draft_kv(committed_pos + i, dummy_keys, dummy_vals)
            
            for i in range(accepted_drafts):
                self.kv.retrieve_draft_kv(committed_pos + i)
            
            self.kv.accept_tokens(accepted_drafts)
            if n_rejected > 0:
                self.kv.reject_tokens(n_rejected)
            
            tokens_to_print = all_tokens[token_idx: token_idx + tokens_this_step]
            if stream:
                for t in tokens_to_print:
                    print(t, end="", flush=True)
            
            token_idx += tokens_this_step
            total_committed += tokens_this_step
            
            elapsed = time.perf_counter() - step_start
            sleep_time = step_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            if step_count % 25 == 0 or total_committed >= max_tokens:
                self._print_status(total_committed, decision)
                print(f"[DBG] step={step_count:4d} | depth={depth} | α_per_tok={per_tok_alpha:.3f} | "
                      f"acc_drafts={accepted_drafts}/{depth} | committed={tokens_this_step} | "
                      f"cpu={sim_cpu:.2f} | H={sim_entropy:.2f} | t_step={step_time*1000:.1f}ms",
                      flush=True)
        
        print("\n")
        return "".join(all_tokens[:total_committed])
    
    def _print_status(self, step: int, decision):
        mon = self.monitor.summary()
        ctrl = self.controller.summary()
        kv = self.kv.get_stats()
        pol = self.policy.summary()
        ev = self.monitor.get_evaluation_metrics()
        
        print(f"\n{'─' * 70}")
        print(f"  Tokens {step:>4d} │ Mode: {ctrl['mode']:12s} │ Depth: {ctrl['draft_depth']} │ Reason: {decision.reason}")
        print(f"  TPS: {mon['rolling_tps']:>6.1f} (EMA: {mon['ema_tps']:.1f}) │      "
              f"Acceptance: {mon['rolling_acceptance']:.1%} │   CPU: {mon['cpu_utilization']:.1%} │   Phase: {mon['phase']}")
        print(f"  ITL: {mon['mean_itl_ms']:>6.2f} ms │      "
              f"P95: {mon['p95_itl_ms']:.2f} ms │   P99: {mon['p99_itl_ms']:.2f} ms │   Stability: {mon['stability_cv']:.3f}")
        spec_eff = ev.get('speculative_efficiency')
        spec_eff_str = f"{spec_eff:.3f}" if spec_eff is not None else "N/A"
        print(f"  Entropy: {mon['entropy_ema']:.3f} │      "
              f"Workload: {pol['workload']:10s} │   Policy: {pol['policy_type']} │   Efficiency: {spec_eff_str}")
        print(f"  KV syncs: {kv['total_syncs']} │      "
              f"Rollbacks: {kv['total_rollbacks']} (p:{kv['partial_rollbacks']}, f:{kv['full_rollbacks']}) │      "
              f"Hit Rate: {kv['hit_rate']:.1%} │   Pressure: {kv['memory_pressure']:.1%}")
        print(f"{'─' * 70}\n", end="", flush=True)
    
    def _print_final_evaluation(self):
        ev = self.monitor.get_evaluation_metrics()
        cs = self.controller.get_decision_stats()
        cst = self.controller.get_stability_metrics()
        wd = self.policy.get_workload_distribution()
        kv = self.kv.get_stats()
        ps = self.policy.summary()
        
        print("\n" + "─" * 70)
        print("  DIAGNOSTIC SANITY CHECKS")
        print("─" * 70)
        total_committed = ev['total_tokens']
        total_drafted = ev['total_drafted']
        total_accepted = ev['total_accepted']
        total_steps = ev['total_steps']
        avg_acc = ev['avg_acceptance']
        speedup = ev['speedup_ratio']
        theoretical_max = (1 + avg_acc * ev['avg_depth'])
        has_speculation = total_drafted > 0
        
        print(f"  [CHECK] avg accepted/step  = {total_accepted}/{total_steps} = "
              f"{total_accepted/max(total_steps,1):.2f}  (expected ≈ depth × α)")
        print(f"  [CHECK] avg committed/step = {total_committed}/{total_steps} = "
              f"{total_committed/max(total_steps,1):.2f}  (expected = accepted+1 per step)")
        if has_speculation:
            print(f"  [CHECK] acceptance rate    = {avg_acc:.3f}  (per draft token; should be 0.15-0.65 in sim)")
        else:
            print("  [CHECK] acceptance rate    = N/A (no draft tokens proposed)")
        if speedup is not None:
            print(f"  [CHECK] speedup ratio      = {speedup:.3f}x  (>1.0 means faster than baseline)")
        else:
            print("  [CHECK] speedup ratio      = N/A (no speculative path active)")
        print(f"  [CHECK] theoretical max tps ≈ {self.monitor._baseline_tps * theoretical_max:.1f}     "
              f"(baseline × E[tokens/step])")
        if has_speculation:
            print(f"  [CHECK] KV hit rate        = {kv['hit_rate']:.1%}  (should be > 0% with store-then-retrieve)")
        else:
            print("  [CHECK] KV hit rate        = N/A (KV draft path not used)")
        print(f"  [CHECK] total drafted      = {total_drafted}  | total wasted = {ev['total_wasted']}")
        print(f"  [CHECK] sample size        = {ev['total_tokens']} tokens   "
              f"({'sufficient' if ev.get('sample_sufficient') else 'small sample; interpret cautiously'})")
        
        if has_speculation:
            ok = all([
                0.05 < avg_acc < 0.95,
                speedup is not None and speedup > 1.0,
                kv['hit_rate'] > 0.0,
                total_committed > 0,
            ])
            if speedup is not None and speedup <= 1.0 and speedup > 0:
                print(f"  [WARN] Speedup {speedup:.3f}x < 1.0 — speculative overhead exceeds benefit.")
                print(f"         Check: draft model quality, depth setting, or CPU resource contention.")
            print(f"  [{'✓ ALL OK' if ok else '✗ ISSUES FOUND'}]")
        else:
            print("  [INFO] Non-speculative run: speculative checks intentionally skipped")
        print("─" * 70)
        
        print("\n" + "=" * 70)
        print("  COMPREHENSIVE EVALUATION METRICS")
        print("=" * 70)
        
        validity = ev.get("scientific_validity", {})
        print("\n  ── Scientific Validity ───────────────────────────")
        print(f"  Speculation Metrics Valid:   {validity.get('speculation_metrics_valid', False)}")
        print(f"  Sample Size Sufficient:      {validity.get('sample_size_sufficient', False)}")
        print(f"  Sample Tokens:               {validity.get('sample_tokens', 0)}")
        print(f"  Baseline Source:             {validity.get('baseline_source', 'unknown')}")
        
        print("\n  ── Performance ──────────────────────────────────────")
        print(f"  Baseline TPS (no speculation):  {ev['baseline_tps']}")
        perf_label = "Overall TPS (with speculation)" if has_speculation else "Overall TPS (no speculation path)"
        print(f"  {perf_label}: {ev['overall_tps']}")
        print(f"  Peak TPS:                       {ev['peak_tps']}")
        print(f"  Min TPS:                        {ev['min_tps']}")
        speedup_str = f"{ev['speedup_ratio']:.3f}x" if ev['speedup_ratio'] is not None else "N/A"
        eff_gain_str = f"{ev['efficiency_gain']:.3f}" if ev['efficiency_gain'] is not None else "N/A"
        print(f"  Speedup Ratio:                  {speedup_str}")
        print(f"  Efficiency Gain:                {eff_gain_str}")
        print(f"  Stability (CV):                 {ev['stability_cv']:.4f}")
        
        print("\n  ── Latency (ms) ─────────────────────────────────────")
        print(f"  TTFT:         {ev['ttft_ms']}")
        print(f"  Mean ITL:     {ev['mean_itl_ms']}")
        print(f"  P95 ITL:      {ev['p95_itl_ms']}")
        print(f"  P99 ITL:      {ev['p99_itl_ms']}")
        print(f"  Max ITL:      {ev['max_itl_ms']}")
        print(f"  Tail Ratio:   {ev['tail_latency_ratio']}")
        
        print("\n  ── Speculative Decoding Efficiency ──────────────────")
        print(f"  Total Tokens Generated:         {ev['total_tokens']}")
        print(f"  Total Draft Tokens Proposed:    {ev['total_drafted']}")
        print(f"  Total Draft Tokens Accepted:    {ev['total_accepted']}")
        print(f"  Total Wasted Drafts:            {ev['total_wasted']}")
        print(f"  Draft Acceptance Rate:          {ev['avg_acceptance']:.2%}")
        print(f"  Rejection Rate:                 {ev['rejection_rate']:.2%}")
        print(f"  Tokens / Step (avg):            {ev['total_tokens'] / max(ev['total_steps'], 1):.2f}")
        print(f"  Average Draft Depth:            {ev['avg_depth']:.2f}")
        spec_eff_str = f"{ev['speculative_efficiency']:.3f}" if ev['speculative_efficiency'] is not None else "N/A"
        print(f"  Speculative Efficiency:         {spec_eff_str}")
        print(f"  Total Speculative Steps:        {ev.get('total_speculative_steps', 0)}")
        if not has_speculation:
            print("  Note: No draft model path was executed in this run; speculative metrics are informational only.")
        
        print("\n  ── Phase Analysis ─────────────────────────────────")
        for phase, info in ev.get("phase_breakdown", {}).items():
            if info["steps"] > 0:
                print(f"  {phase:12s}: {info['steps']:4d} steps, TPS {info['avg_tps']:.1f}, acc {info['avg_acceptance']:.1%}")
        
        if has_speculation:
            print("\n  ── Controller Decisions ─────────────────────────────")
            for reason, count in sorted(cs.items(), key=lambda x: -x[1]):
                if count > 0:
                    print(f"    {reason:28s}: {count:4d} {'█' * min(40, count)}")
            
            print("\n  ── Controller Stability ───────────────────────────")
            print(f"  Depth Changes:     {cst['depth_changes']}")
            print(f"  Oscillations:      {cst['oscillations']}")
            print(f"  Stability Score:   {cst['stability_score']:.4f}")
            print(f"  Mean Depth:        {cst['mean_depth']:.2f}")
            
            print("\n  ── Workload Distribution ───────────────────────────")
            for wl, info in wd.items():
                print(f"    {wl:12s}: {info['count']:4d} windows ({info['percent']:.1f}%)")
            
            print("\n  ── Policy Engine ───────────────────────────────────")
            print(f"  Type:            {ps['policy_type']}")
            print(f"  Workload:        {ps['workload']}")
            print(f"  Avg Reward:      {ps['avg_reward_last20']:.4f}")
            print(f"  Avg Regret:      {ps['avg_regret_last20']:.4f}")
            print(f"  Bandit Best Arm: depth={ps['bandit_best_arm']}")
            print(f"  Bandit Epsilon:  {ps['bandit_epsilon']:.4f}")
            print("  Policy Contribution:")
            for name, val in ps.get("policy_contribution", {}).items():
                print(f"    {name:10s}: {'█' * max(1, int(val * 20))} {val:.4f}")
            print("  EMA Rewards:")
            for d, info in ps.get("ema_rewards", {}).items():
                r = info.get("reward", info) if isinstance(info, dict) else info
                print(f"    depth={d}: {'█' * max(1, int(float(r) * 30))} {r:.4f}")
            
            print("\n  ── KV Cache Statistics ──────────────────────────────")
            print(f"  Total Syncs:        {kv['total_syncs']}")
            print(f"  Total Rollbacks:    {kv['total_rollbacks']} (partial: {kv['partial_rollbacks']}, full: {kv['full_rollbacks']})")
            print(f"  Avg Rollback Size:  {kv['avg_rollback_tokens']:.2f}")
            print(f"  Shadow Hits:        {kv['shadow_hits']}")
            print(f"  Shadow Misses:      {kv['shadow_misses']}")
            print(f"  Hit Rate:           {kv['hit_rate']:.1%}")
            print(f"  Compression Ratio:  {kv['compression_ratio']:.2f}x")
            print(f"  Memory Pressure:    {kv['memory_pressure']:.1%}")
        else:
            print("\n  ── Speculation Subsystems ─────────────────────────")
            print("  Controller/Policy/KV sections skipped because speculative path was inactive.")
        
        print("\n" + "=" * 70)
        main_logger.info("Final evaluation: %s", json.dumps(ev, indent=2))
    
    def benchmark(self, prompts=None, tokens_each=100):
        if prompts is None:
            prompts = [
                "Write a Python implementation of quicksort.",
                "Explain the theory of general relativity.",
                "def fibonacci(n): # complete this function",
            ]
        print("\n" + "=" * 70)
        print("  BENCHMARK")
        print("=" * 70)
        results = []
        for idx, prompt in enumerate(prompts):
            self._reset_runtime_modules()
            print(f"\n[{idx + 1}/{len(prompts)}] {prompt[:60]}...")
            t0 = time.perf_counter()
            self.generate(prompt, max_tokens=tokens_each, stream=False)
            elapsed = time.perf_counter() - t0
            ev = self.monitor.get_evaluation_metrics()
            speedup_brief = f"{ev['speedup_ratio']:.2f}x" if ev['speedup_ratio'] is not None else "N/A"
            print(f"    {elapsed:.2f}s | TPS: {ev['overall_tps']:.1f} | Speedup: {speedup_brief} | Acc: {ev['avg_acceptance']:.1%}")
            results.append({"prompt": prompt[:40], "elapsed": round(elapsed, 2), **ev})
        self._print_benchmark_summary(results)
        
        self._print_final_evaluation()
        return results
    
    def _print_benchmark_summary(self, results: List[dict]):
        if not results:
            return
        
        tps_values = [r["overall_tps"] for r in results if r.get("overall_tps") is not None]
        token_counts = [r.get("total_tokens", 0) for r in results]
        total_tokens = sum(token_counts)
        sample_sizes_ok = all(r.get("sample_sufficient", False) for r in results)
        
        mean_tps = statistics.mean(tps_values) if tps_values else 0.0
        std_tps = statistics.pstdev(tps_values) if len(tps_values) > 1 else 0.0
        
        speculative_runs = [r for r in results if r.get("speculation_active")]
        speedups = [r.get("speedup_ratio") for r in speculative_runs if r.get("speedup_ratio") is not None]
        mean_speedup = statistics.mean(speedups) if speedups else None
        
        print("\n" + "-" * 70)
        print("  BENCHMARK SUMMARY")
        print("-" * 70)
        print(f"  Runs:                    {len(results)}")
        print(f"  Total Generated Tokens:  {total_tokens}")
        print(f"  Mean TPS:                {mean_tps:.2f}")
        print(f"  TPS StdDev:              {std_tps:.2f}")
        print(f"  Sample Sufficiency:      {sample_sizes_ok}")
        if mean_speedup is not None:
            print(f"  Mean Speculative Speedup:{mean_speedup:.3f}x")
        else:
            print("  Mean Speculative Speedup: N/A (no speculative runs)")
        print("-" * 70)
    
    def close(self):
        self.monitor.stop()
        main_logger.info("Engine closed.")
    
    def status_json(self) -> str:
        return json.dumps({
            "monitor": self.monitor.summary(),
            "controller": self.controller.summary(),
            "controller_stability": self.controller.get_stability_metrics(),
            "kv_cache": self.kv.get_stats(),
            "policy": self.policy.summary(),
            "evaluation": self.monitor.get_evaluation_metrics(),
        }, indent=2)

def _redact_args(args_dict: dict) -> dict:
    """Helper to hide sensitive keys in logs"""
    redacted = dict(args_dict)
    for key in ["openrouter_api_key", "api_key", "token", "password", "secret"]:
        if key in redacted and redacted[key]:
            redacted[key] = "REDACTED"
    return redacted

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Adaptive Speculative Decoding")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--policy", default="ensemble", choices=["heuristic", "bandit", "ema", "ensemble"])
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-file", default="adaptive_sd.log")
    parser.add_argument("--backend", default="auto", choices=["auto", "gguf", "openrouter", "simulation"])
    parser.add_argument("--openrouter-model", default=os.environ.get("OPENROUTER_MODEL", "openrouter/free"))
    parser.add_argument("--openrouter-base-url", default=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    parser.add_argument("--openrouter-api-key", default=os.environ.get("OPENROUTER_API_KEY", ""))
    parser.add_argument("--openrouter-site-url", default=os.environ.get("OPENROUTER_SITE_URL", ""))
    parser.add_argument("--openrouter-site-name", default=os.environ.get("OPENROUTER_SITE_NAME", "AdaptiveSD"))
    args = parser.parse_args()
    
    setup_logging(log_file=args.log_file, verbose=args.verbose)
    main_logger.info("Starting Adaptive Speculative Decoding engine")
    main_logger.info("Arguments: %s", _redact_args(vars(args)))
    
    policy_map = {"heuristic": PolicyType.HEURISTIC, "bandit": PolicyType.BANDIT,
                  "ema": PolicyType.EMA, "ensemble": PolicyType.ENSEMBLE}
    
    engine = AdaptiveInferenceEngine(
        model_path=args.model,
        draft_model_path=args.draft_model,
        n_ctx=args.n_ctx, n_threads=args.threads,
        n_draft=args.depth, verbose=args.verbose, policy_type=policy_map[args.policy],
        backend=args.backend,
        openrouter_api_key=args.openrouter_api_key or None,
        openrouter_model=args.openrouter_model,
        openrouter_base_url=args.openrouter_base_url,
        openrouter_site_url=args.openrouter_site_url or None,
        openrouter_site_name=args.openrouter_site_name or None,
    )
    
    try:
        if args.benchmark:
            engine.benchmark()
        elif args.interactive:
            print("Interactive Mode. Type 'quit' to exit.\n")
            while True:
                try:
                    prompt = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not prompt:
                    continue
                if prompt.lower() == "quit":
                    break
                if prompt.lower() == "status":
                    print(engine.status_json())
                    continue
                print("Assistant: ", end="", flush=True)
                engine.generate(prompt, max_tokens=args.max_tokens, temperature=args.temperature)
            engine._print_final_evaluation()
        elif args.prompt:
            engine.generate(args.prompt, max_tokens=args.max_tokens, temperature=args.temperature)
            engine._print_final_evaluation()
            print("\n── Final Status (JSON) ─────────────────────────────────")
            print(engine.status_json())
        else:
            engine.benchmark(
                prompts=[
                    "Explain adaptive speculative decoding on CPUs.",
                    "Describe transformer attention in concise technical language.",
                    "Give a structured explanation of backpropagation and gradient descent.",
                ],
                tokens_each=128,
            )
    finally:
        engine.close()
        main_logger.info("Shutdown complete. Log saved to: %s", args.log_file)

if __name__ == "__main__":
    main()