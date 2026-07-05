"""
Module 4: Dynamic Policy Engine
Stable workload detection, proper bandit, confidence-based ensemble.
"""
import time
import math
import random
import logging
import collections
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Deque
from enum import Enum
from monitor import RuntimeMonitor, InferenceSnapshot
from draft_controller import AdaptiveDraftController, SpeculationMode

logger = logging.getLogger("AdaptiveSD.Policy")

class WorkloadType(Enum):
    CODING = "coding"
    REASONING = "reasoning"
    CHAT = "chat"
    CREATIVE = "creative"
    UNKNOWN = "unknown"

class PolicyType(Enum):
    HEURISTIC = "heuristic"
    BANDIT = "bandit"
    EMA = "ema"
    ENSEMBLE = "ensemble"
    # ── Baselines added for comparative evaluation (reviewer-requested) ──
    # These are NOT reproductions of the original papers' trained/learned
    # components (no distillation data, no released checkpoints were
    # available); they are rule-based re-implementations of each method's
    # *core control idea*, plugged into the exact same controller / KV-cache
    # / monitor / backend as AdaptiveSD so the comparison is apples-to-apples
    # on identical hardware and identical prompts. This limitation is
    # reported alongside every comparison table generated from these runs
    # (see main.py: run_baseline_comparison).
    FIXED_DEPTH = "fixed_depth"            # static depth, no adaptation at all
    PARALLEL_SD = "parallel_sd"            # always-max-depth, verify-everything baseline
    DYNAMIC_LOOKAHEAD = "dynamic_lookahead"  # depth follows a moving-average acceptance rate only
    SPECDEC_PLUS_APPROX = "specdec_plus_approx"  # entropy/confidence-threshold stopping rule

@dataclass
class PolicyConfig:
    policy_type: PolicyType = PolicyType.ENSEMBLE
    bandit_epsilon: float = 0.15
    bandit_decay: float = 0.995
    bandit_min_epsilon: float = 0.02
    bandit_ucb_c: float = 0.5
    ema_alpha: float = 0.1
    workload_detect_window: int = 64
    ensemble_heuristic_weight: float = 0.5
    ensemble_bandit_weight: float = 0.3
    ensemble_ema_weight: float = 0.2
    reward_tps_weight: float = 0.4
    reward_acceptance_weight: float = 0.2
    reward_latency_weight: float = 0.3
    reward_cpu_weight: float = 0.1
    depth_choices: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    confidence_threshold: float = 0.6
    workload_change_hysteresis: int = 4
    # Baseline-specific knobs
    fixed_depth_value: int = 2               # depth used by FIXED_DEPTH baseline
    lookahead_window: int = 20               # moving-average window for DYNAMIC_LOOKAHEAD
    lookahead_kp: float = 6.0                # proportional gain: depth change per unit acceptance error
    specdec_entropy_threshold: float = 5.5   # stop drafting once EMA token entropy exceeds this

@dataclass
class PolicyStep:
    chosen_depth: int
    policy_type: PolicyType
    workload: WorkloadType
    reward: float
    regret: float
    tps: float
    acceptance: float
    latency_penalty: float
    timestamp: float = field(default_factory=time.perf_counter)

class _BanditArm:
    def __init__(self, depth: int):
        self.depth = depth
        self.pulls = 0
        self.total_reward = 0.0
        self.avg_reward = 0.0
        self.regret_history: Deque[float] = collections.deque(maxlen=50)

    def update(self, reward: float):
        self.pulls += 1
        self.total_reward += reward
        self.avg_reward = self.total_reward / self.pulls

    def ucb_score(self, total_pulls: int, c: float = 0.5) -> float:
        if self.pulls == 0:
            return float("inf")
        return self.avg_reward + c * math.sqrt(math.log(total_pulls + 1) / self.pulls)

    def get_regret(self, optimal_reward: float) -> float:
        regret = optimal_reward - self.avg_reward
        self.regret_history.append(regret)
        return regret

    def avg_regret(self) -> float:
        if not self.regret_history:
            return 0.0
        return sum(self.regret_history) / len(self.regret_history)

class HeuristicPolicy:
    def __init__(self, config: PolicyConfig):
        self.cfg = config
        self._confidence = 0.5

    def choose_depth(self, workload: WorkloadType, snap: Optional[InferenceSnapshot],
                     controller: AdaptiveDraftController) -> Tuple[int, float]:
        base = controller.current_depth()
        if workload == WorkloadType.CODING:
            depth = min(self.cfg.depth_choices[-1], base + 1)
        elif workload == WorkloadType.CREATIVE:
            depth = max(self.cfg.depth_choices[0], base)
        elif workload == WorkloadType.REASONING:
            depth = min(base, 4)
        else:
            depth = base
        self._confidence = 0.7 if workload != WorkloadType.UNKNOWN else 0.3
        return depth, self._confidence

    def confidence(self) -> float:
        return self._confidence

class BanditPolicy:
    def __init__(self, config: PolicyConfig):
        self.cfg = config
        self._arms: Dict[int, _BanditArm] = {d: _BanditArm(d) for d in config.depth_choices}
        self._epsilon = config.bandit_epsilon
        self._total_pulls = 0
        self._last_depth: Optional[int] = None
        self._confidence = 0.5

    def choose_depth(self) -> Tuple[int, float]:
        if random.random() < self._epsilon:
            chosen = random.choice(self.cfg.depth_choices)
            self._confidence = 0.4
        else:
            chosen = max(self._arms.values(),
                         key=lambda a: a.ucb_score(self._total_pulls, self.cfg.bandit_ucb_c)).depth
            self._confidence = 0.8
        self._last_depth = chosen
        self._total_pulls += 1
        self._epsilon = max(self.cfg.bandit_min_epsilon, self._epsilon * self.cfg.bandit_decay)
        return chosen, self._confidence

    def update(self, reward: float):
        if self._last_depth is not None:
            self._arms[self._last_depth].update(reward)

    def get_regret(self) -> float:
        pulled = [a for a in self._arms.values() if a.pulls > 0]
        if not pulled or self._last_depth is None:
            return 0.0
        optimal = max(a.avg_reward for a in pulled)
        return self._arms[self._last_depth].get_regret(optimal)

    def best_arm(self, min_pulls: int = 5) -> int:
        # FIX: an arm with a single lucky pull could otherwise dominate avg_reward.
        # Only trust arms sampled enough times; fall back to "most-pulled" arm otherwise.
        reliable = [a for a in self._arms.values() if a.pulls >= min_pulls]
        if reliable:
            return max(reliable, key=lambda a: a.avg_reward).depth
        pulled = [a for a in self._arms.values() if a.pulls > 0]
        return max(pulled, key=lambda a: a.pulls).depth if pulled else self.cfg.depth_choices[0]

    def arm_summary(self, min_pulls: int = 5) -> dict:
        return {d: {"pulls": a.pulls, "avg_reward": round(a.avg_reward, 4),
                    "avg_regret": round(a.avg_regret(), 4),
                    "reliable": a.pulls >= min_pulls} for d, a in self._arms.items()}

    def confidence(self) -> float:
        return self._confidence

class EMAPolicy:
    # FIX: minimum number of updates an arm needs before its EMA reward is
    # treated as trustworthy. Without this, an arm updated once or twice
    # (e.g. depth 3/4, which the controller rarely visits) could report a
    # spuriously high reward from noise alone and get reported/selected as
    # "better" with no statistical basis (see reviewer point 2c).
    MIN_SAMPLES_FOR_CONFIDENCE = 8

    def __init__(self, config: PolicyConfig):
        self.cfg = config
        self._ema_rewards: Dict[int, float] = {d: 0.5 for d in config.depth_choices}
        self._ema_variance: Dict[int, float] = {d: 0.1 for d in config.depth_choices}
        self._update_count: Dict[int, int] = {d: 0 for d in config.depth_choices}
        self._last_depth: Optional[int] = None

    def choose_depth(self) -> Tuple[int, float]:
        reliable = {d: v for d, v in self._ema_rewards.items()
                    if self._update_count[d] >= self.MIN_SAMPLES_FOR_CONFIDENCE}
        if reliable:
            best = max(reliable, key=reliable.get)
            confidence = 0.6
        else:
            # Not enough evidence anywhere yet: fall back to the depth with the
            # most samples rather than trusting a noisy EMA.
            best = max(self._update_count, key=self._update_count.get)
            confidence = 0.3
        self._last_depth = best
        return best, confidence

    def update(self, depth: int, reward: float):
        if depth in self._ema_rewards:
            alpha = self.cfg.ema_alpha
            old = self._ema_rewards[depth]
            self._ema_rewards[depth] = alpha * reward + (1.0 - alpha) * old
            self._ema_variance[depth] = alpha * ((reward - old) ** 2) + (1.0 - alpha) * self._ema_variance[depth]
            self._update_count[depth] += 1

    def summary(self) -> dict:
        # FIX: expose sample count + reliability flag alongside each reward so
        # downstream reporting (plots, tables) can't present a low-n estimate
        # as if it were as trustworthy as a high-n one.
        return {d: {"reward": round(v, 4), "variance": round(self._ema_variance[d], 6),
                    "n_samples": self._update_count[d],
                    "reliable": self._update_count[d] >= self.MIN_SAMPLES_FOR_CONFIDENCE}
                for d, v in self._ema_rewards.items()}

    def confidence(self) -> float:
        n_reliable = sum(1 for d, c in self._update_count.items()
                          if c >= self.MIN_SAMPLES_FOR_CONFIDENCE)
        if n_reliable == 0:
            return 0.3
        return 0.6

class FixedDepthPolicy:
    """Weak/no-adaptation baseline: always suggests the same constant depth.
    This is the honest, explicit version of the "depth-0/constant" baseline
    the reviewer noted the paper was implicitly (and only partially) compared
    against in section 4.7 — now runnable directly via --policy fixed_depth."""
    def __init__(self, config: PolicyConfig):
        self.cfg = config

    def choose_depth(self) -> Tuple[int, float]:
        return self.cfg.fixed_depth_value, 1.0  # depth, confidence(always max: nothing to be uncertain about)


class ParallelSDPolicy:
    """Always-maximum-depth baseline, approximating the "verify everything in
    parallel, don't adapt" philosophy of parallel speculative-decoding methods
    (batch-verify a wide draft tree every step rather than tuning depth).
    No load-shedding: this intentionally never backs off, so its CPU/latency
    tail behaviour under contention is a fair thing to show *against*."""
    def __init__(self, config: PolicyConfig):
        self.cfg = config

    def choose_depth(self) -> Tuple[int, float]:
        return max(self.cfg.depth_choices), 1.0


class DynamicLookaheadPolicy:
    """Single-signal adaptive baseline: depth is driven ONLY by a moving
    average of recent acceptance ratio via a simple proportional controller.
    Unlike AdaptiveSD's ensemble (bandit + EMA-reward + heuristic + CPU/entropy/
    latency-variance rules), this baseline ignores CPU load, token entropy,
    latency variance, and workload type entirely — representative of simpler
    "dynamic lookahead length" schemes that resize the speculation window from
    acceptance statistics alone."""
    def __init__(self, config: PolicyConfig):
        self.cfg = config
        self._window: Deque[float] = collections.deque(maxlen=config.lookahead_window)
        self._depth = min(config.depth_choices, key=lambda d: abs(d - 2))

    def observe(self, acceptance_ratio: float):
        self._window.append(acceptance_ratio)

    def choose_depth(self) -> Tuple[int, float]:
        if len(self._window) < max(3, self.cfg.lookahead_window // 4):
            return self._depth, 0.3  # low confidence: not enough samples yet
        avg_acc = sum(self._window) / len(self._window)
        # Proportional control around a 50% acceptance set-point: above 50%
        # acceptance -> push depth up, below -> push depth down.
        error = avg_acc - 0.5
        target = self._depth + self.cfg.lookahead_kp * error
        self._depth = max(min(self.cfg.depth_choices), min(max(self.cfg.depth_choices), round(target)))
        confidence = min(1.0, len(self._window) / self.cfg.lookahead_window)
        return self._depth, confidence


class SpecDecPlusApproxPolicy:
    """
    CRITICAL FIX: Overhead-aware depth selection
    """
    def __init__(self, config: PolicyConfig):
        self.cfg = config
        self._ema_entropy = 0.0
        self._entropy_initialized = False
        self._recent_acceptance = []
        self._recent_speedups = []
        
    def choose_depth(self, ema_entropy: float, acceptance_ratio: float = 0.5, 
                     current_speedup: Optional[float] = None) -> Tuple[int, float]:
        """
        CRITICAL FIX: Consider overhead when choosing depth
        """
        max_d = max(self.cfg.depth_choices)
        min_d = min(self.cfg.depth_choices)
        
        # Update EMA entropy
        if not self._entropy_initialized:
            self._ema_entropy = ema_entropy
            self._entropy_initialized = True
        else:
            alpha = 0.15
            self._ema_entropy = alpha * ema_entropy + (1.0 - alpha) * self._ema_entropy
        
        # Track recent acceptance
        self._recent_acceptance.append(acceptance_ratio)
        if len(self._recent_acceptance) > 10:
            self._recent_acceptance.pop(0)
        
        avg_acceptance = sum(self._recent_acceptance) / len(self._recent_acceptance)
        
        # ═══════════════════════════════════════════════════════════════
        # CRITICAL FIX: If speedup is low, disable speculation
        # ═══════════════════════════════════════════════════════════════
        if current_speedup is not None:
            self._recent_speedups.append(current_speedup)
            if len(self._recent_speedups) > 5:
                self._recent_speedups.pop(0)
            
            avg_speedup = sum(self._recent_speedups) / len(self._recent_speedups)
            
            # If speculation is not helping, return depth=1 (no speculation)
            if avg_speedup < 0.8:
                return 1, 1.0  # depth=1 means no speculation
        
        # Original logic
        if self._ema_entropy <= 0:
            entropy_depth = max_d
        else:
            ratio = self._ema_entropy / self.cfg.specdec_entropy_threshold
            entropy_depth = max_d - ratio * (max_d - min_d)
            entropy_depth = max(min_d, min(max_d, round(entropy_depth)))
        
        if avg_acceptance > 0.7:
            acc_depth = max_d
        elif avg_acceptance > 0.5:
            acc_depth = max(min_d + 1, max_d - 1)
        elif avg_acceptance > 0.3:
            acc_depth = max(min_d, max_d - 2)
        else:
            acc_depth = min_d
        
        final_depth = min(entropy_depth, acc_depth)
        
        agreement = 1.0 - abs(entropy_depth - acc_depth) / (max_d - min_d)
        confidence = max(0.2, min(1.0, 0.5 + 0.5 * agreement))
        
        return final_depth, confidence    
class DynamicPolicyEngine:
    def __init__(self, monitor: RuntimeMonitor, controller: AdaptiveDraftController,
                 config: Optional[PolicyConfig] = None):
        self.monitor = monitor
        self.controller = controller
        self.cfg = config or PolicyConfig()
        self._heuristic = HeuristicPolicy(self.cfg)
        self._bandit = BanditPolicy(self.cfg)
        self._ema = EMAPolicy(self.cfg)
        self._fixed = FixedDepthPolicy(self.cfg)
        self._parallel = ParallelSDPolicy(self.cfg)
        self._lookahead = DynamicLookaheadPolicy(self.cfg)
        self._specdec = SpecDecPlusApproxPolicy(self.cfg)
        
        self._workload: WorkloadType = WorkloadType.UNKNOWN
        self._token_buffer: Deque[str] = collections.deque(maxlen=self.cfg.workload_detect_window)
        self._step_history: Deque[PolicyStep] = collections.deque(maxlen=500)
        self._last_snap: Optional[InferenceSnapshot] = None
        self._step_count = 0
        self._workload_history: Dict[str, int] = collections.defaultdict(int)

        self._candidate_workload: Optional[WorkloadType] = None
        self._candidate_count = 0
        self._prompt_analyzed = False

        logger.info("DynamicPolicyEngine initialised (type=%s, arms=%s)",
                    self.cfg.policy_type.value, self.cfg.depth_choices)

    def step(self, token_text: str = "", prompt: str = "") -> Tuple[int, float]:
        self._step_count += 1
        snap = self.monitor.get_current_snapshot()

        if not self._prompt_analyzed and prompt:
            self._workload = self._analyze_prompt(prompt)
            self._workload_history[self._workload.value] += 1
            self._prompt_analyzed = True

        if token_text and token_text.strip():
            self._token_buffer.append(token_text.strip())

        if len(self._token_buffer) >= self.cfg.workload_detect_window and \
           self._step_count % self.cfg.workload_detect_window == 0:
            detected = self._detect_workload()
            if detected != self._workload and detected != WorkloadType.UNKNOWN:
                if detected == self._candidate_workload:
                    self._candidate_count += 1
                else:
                    self._candidate_workload = detected
                    self._candidate_count = 1
                if self._candidate_count >= self.cfg.workload_change_hysteresis:
                    logger.info("Workload changed: %s → %s (step %d)",
                                self._workload.value, detected.value, self._step_count)
                    self._workload = detected
                    self._workload_history[self._workload.value] += 1
                    self._candidate_workload = None
                    self._candidate_count = 0
            else:
                self._candidate_workload = None
                self._candidate_count = 0

        if self._last_snap and snap:
            reward, latency_penalty = self._compute_reward(self._last_snap, snap)
            self._bandit.update(reward)
            self._ema.update(self.controller.current_depth(), reward)
            regret = self._bandit.get_regret()
        else:
            reward, latency_penalty, regret = 0.5, 0.0, 0.0

        if self.cfg.policy_type == PolicyType.FIXED_DEPTH:
            final_depth, confidence = self._fixed.choose_depth()
        elif self.cfg.policy_type == PolicyType.PARALLEL_SD:
            final_depth, confidence = self._parallel.choose_depth()
        elif self.cfg.policy_type == PolicyType.DYNAMIC_LOOKAHEAD:
            if snap:
                self._lookahead.observe(snap.acceptance_ratio)
            final_depth, confidence = self._lookahead.choose_depth()
        elif self.cfg.policy_type == PolicyType.SPECDEC_PLUS_APPROX:
            ema_entropy = snap.token_entropy if snap else 0.0
            final_depth, confidence = self._specdec.choose_depth(ema_entropy)
        else:
            depths = {
                PolicyType.HEURISTIC: self._heuristic.choose_depth(self._workload, snap, self.controller),
                PolicyType.BANDIT: self._bandit.choose_depth(),
                PolicyType.EMA: self._ema.choose_depth(),
            }
            final_depth, _ = self._combine(depths, snap)
            confidence = self._compute_ensemble_confidence()
        final_depth = max(self.controller.cfg.min_depth, min(self.controller.cfg.max_depth, final_depth))

        tps = snap.tokens_per_sec if snap else 0.0
        acc = snap.acceptance_ratio if snap else 0.0
        self._step_history.append(PolicyStep(
            chosen_depth=final_depth, policy_type=self.cfg.policy_type,
            workload=self._workload, reward=reward, regret=regret,
            tps=tps, acceptance=acc, latency_penalty=latency_penalty,
        ))
        self._last_snap = snap
        return final_depth, confidence

    def _compute_ensemble_confidence(self) -> float:
        """
        IMPROVED: Agreement-based confidence calculation
        """
        if self.cfg.policy_type == PolicyType.HEURISTIC:
            return self._heuristic.confidence()
        if self.cfg.policy_type == PolicyType.BANDIT:
            return self._bandit.confidence()
        if self.cfg.policy_type == PolicyType.EMA:
            return self._ema.confidence()
        
        # Get individual confidences
        c_h = self._heuristic.confidence()
        c_b = self._bandit.confidence()
        c_e = self._ema.confidence()
        
        # Get recent depth suggestions
        recent_steps = list(self._step_history)[-5:] if self._step_history else []
        
        if len(recent_steps) >= 2:
            # Calculate agreement ratio
            depths_recent = [s.chosen_depth for s in recent_steps]
            unique_depths = len(set(depths_recent))
            agreement_ratio = 1.0 / unique_depths if unique_depths > 0 else 0.5
            
            # Weighted confidence with agreement bonus
            base_conf = (c_h + c_b + c_e) / 3.0
            agreement_bonus = 0.2 * agreement_ratio
            
            return min(1.0, base_conf + agreement_bonus)
        else:
            # Simple weighted average
            w_h = self.cfg.ensemble_heuristic_weight * c_h
            w_b = self.cfg.ensemble_bandit_weight * c_b
            w_e = self.cfg.ensemble_ema_weight * c_e
            total = self.cfg.ensemble_heuristic_weight + self.cfg.ensemble_bandit_weight + self.cfg.ensemble_ema_weight
            return (w_h + w_b + w_e) / total if total > 0 else 0.5
        

    def _analyze_prompt(self, prompt: str) -> WorkloadType:
        p = prompt.lower()
        if any(k in p for k in ["def ", "class ", "code", "function", "implement", "python", "algorithm"]):
            return WorkloadType.CODING
        if any(k in p for k in ["explain", "why", "how does", "theory", "because", "therefore"]):
            return WorkloadType.REASONING
        if any(k in p for k in ["hello", "hi", "chat", "tell me about", "what is"]):
            return WorkloadType.CHAT
        return WorkloadType.CREATIVE

    def _detect_workload(self) -> WorkloadType:
        if not self._token_buffer:
            return self._workload
        text = " ".join(self._token_buffer)
        words = text.split()
        n = max(len(words), 1)

        code_chars = sum(text.count(c) for c in "{}[]()=><;:#")
        code_score = (code_chars * 2 + text.count("\n")) / n

        reasoning_words = [
            "therefore", "because", "thus", "conclude", "hence", "implies",
            "valid", "hypothesis", "theorem", "axiom", "premise", "induction",
            "polynomial", "optimization", "bound", "solution", "continuous",
            "interval", "algorithm", "terminates", "non-empty", "consequently",
            "observe", "mathematical", "factors", "follows", "holds", "true"
        ]
        reasoning_count = sum(text.lower().count(w) for w in reasoning_words)
        reasoning_score = (reasoning_count / n) * 25

        unique = len(set(w.lower().strip(".,!?;:") for w in words))
        creative_score = (unique / n) * 3

        chat_words = ["hello", "hi", "how", "can", "help", "thanks", "please", "welcome", "chat", "assistant"]
        chat_count = sum(text.lower().count(w) for w in chat_words)
        chat_score = (chat_count / n) * 15

        scores = {
            WorkloadType.CODING: code_score,
            WorkloadType.REASONING: reasoning_score,
            WorkloadType.CREATIVE: creative_score,
            WorkloadType.CHAT: chat_score,
        }

        max_score = max(scores.values())
        if max_score < 0.3:
            return self._workload
        return max(scores, key=lambda k: scores[k])

    def _compute_reward(self, prev: InferenceSnapshot, curr: InferenceSnapshot) -> Tuple[float, float]:
        # FIX (reviewer-flagged bug): old formula = sigmoid(0.05*(latency_ms-50)).
        # For CPU/GGUF inference latency_ms is ~300-900, so 0.05*(700-50)=32.5
        # and sigmoid(32.5) saturates to ~1.0 regardless of whether latency
        # improves or worsens -> the latency term of the reward was dead.
        # Fix: use the *relative* change in latency (bounded, scale-free),
        # same construction as tps_reward below. Decrease in latency -> penalty -> 1.
        lat_prev = max(prev.latency_ms, 1e-6)
        lat_delta = (curr.latency_ms - lat_prev) / lat_prev
        latency_penalty = 1.0 / (1.0 + math.exp(max(-60.0, min(60.0, 3.0 * lat_delta))))
        tps_prev = max(prev.tokens_per_sec, 1e-6)
        tps_delta = (curr.tokens_per_sec - prev.tokens_per_sec) / tps_prev
        tps_reward = 1.0 / (1.0 + math.exp(max(-60.0, min(60.0, -3.0 * tps_delta))))
        cpu_penalty = 1.0 - max(0, curr.cpu_utilization - 0.5) * 2.0
        reward = (
            self.cfg.reward_tps_weight * tps_reward
            + self.cfg.reward_acceptance_weight * curr.acceptance_ratio
            + self.cfg.reward_latency_weight * latency_penalty
            + self.cfg.reward_cpu_weight * cpu_penalty
        )
        return max(0.0, min(1.0, reward)), latency_penalty

    def _combine(self, depths: Dict[PolicyType, Tuple[int, float]],
                snap: Optional[InferenceSnapshot]) -> Tuple[int, bool]:
        """
        IMPROVED: Dynamic weighting based on recent performance + agreement-based confidence
        """
        pt = self.cfg.policy_type
        if pt == PolicyType.HEURISTIC:
            return depths[PolicyType.HEURISTIC][0], True
        if pt == PolicyType.BANDIT:
            return depths[PolicyType.BANDIT][0], True
        if pt == PolicyType.EMA:
            return depths[PolicyType.EMA][0], True
        
        # ═══════════════════════════════════════════════════════════════
        # NEW: Performance-based dynamic weighting
        # ═══════════════════════════════════════════════════════════════
        
        # Calculate recent performance for each policy (last 10 steps)
        recent_steps = list(self._step_history)[-10:] if self._step_history else []
        
        if len(recent_steps) >= 3:
            # Track which policy suggestions led to good outcomes
            policy_performance = {"heuristic": [], "bandit": [], "ema": []}
            
            for step in recent_steps:
                if step.reward > 0.6:  # Good outcome
                    # We don't track which policy contributed, so use current confidence
                    policy_performance["heuristic"].append(self._heuristic.confidence())
                    policy_performance["bandit"].append(self._bandit.confidence())
                    policy_performance["ema"].append(self._ema.confidence())
            
            # Calculate dynamic weights based on confidence stability
            avg_conf = {
                "heuristic": sum(policy_performance["heuristic"]) / max(1, len(policy_performance["heuristic"])),
                "bandit": sum(policy_performance["bandit"]) / max(1, len(policy_performance["bandit"])),
                "ema": sum(policy_performance["ema"]) / max(1, len(policy_performance["ema"]))
            }
            
            # Normalize weights
            total_conf = sum(avg_conf.values())
            if total_conf > 0:
                w_h = avg_conf["heuristic"] / total_conf
                w_b = avg_conf["bandit"] / total_conf
                w_e = avg_conf["ema"] / total_conf
            else:
                w_h, w_b, w_e = 0.33, 0.33, 0.34
        else:
            # Fall back to static weights
            w_h = self.cfg.ensemble_heuristic_weight
            w_b = self.cfg.ensemble_bandit_weight
            w_e = self.cfg.ensemble_ema_weight
            total = w_h + w_b + w_e
            w_h, w_b, w_e = w_h/total, w_b/total, w_e/total
        
        # ═══════════════════════════════════════════════════════════════
        # NEW: Agreement-based decision (majority voting)
        # ═══════════════════════════════════════════════════════════════
        
        d_h, c_h = depths[PolicyType.HEURISTIC]
        d_b, c_b = depths[PolicyType.BANDIT]
        d_e, c_e = depths[PolicyType.EMA]
        
        # If policies agree, use that depth
        if d_h == d_b == d_e:
            return d_h, True
        
        # If two agree, use majority
        if d_h == d_b:
            return d_h, True
        if d_h == d_e:
            return d_h, True
        if d_b == d_e:
            return d_b, True
        
        # No agreement: use weighted average but with confidence penalty
        weighted = (w_h * d_h + w_b * d_b + w_e * d_e)
        final_depth = round(weighted)
        
        # Clamp to valid range
        final_depth = max(min(self.cfg.depth_choices), min(max(self.cfg.depth_choices), final_depth))
        
        return final_depth, True

    def workload(self) -> WorkloadType:
        return self._workload

    def get_workload_distribution(self) -> dict:
        total = sum(self._workload_history.values()) or 1
        # FIX: skip zero-count workloads to avoid cluttered output
        return {
            k: {"count": v, "percent": round(v / total * 100, 1)}
            for k, v in self._workload_history.items()
            if v > 0
        }

    def get_policy_contribution(self) -> dict:
        if self.cfg.policy_type != PolicyType.ENSEMBLE:
            return {self.cfg.policy_type.value: 1.0}
        w_h = self.cfg.ensemble_heuristic_weight * self._heuristic.confidence()
        w_b = self.cfg.ensemble_bandit_weight * self._bandit.confidence()
        w_e = self.cfg.ensemble_ema_weight * self._ema.confidence()
        total = w_h + w_b + w_e
        if total <= 0:
            return {"heuristic": 0.33, "bandit": 0.33, "ema": 0.34}
        return {"heuristic": round(w_h / total, 4), "bandit": round(w_b / total, 4), "ema": round(w_e / total, 4)}

    def summary(self) -> dict:
        recent = list(self._step_history)[-20:] if self._step_history else []
        avg_reward = sum(s.reward for s in recent) / len(recent) if recent else 0.0
        avg_regret = sum(s.regret for s in recent) / len(recent) if recent else 0.0
        avg_depth = sum(s.chosen_depth for s in recent) / len(recent) if recent else 0.0
        return {
            "workload": self._workload.value,
            "policy_type": self.cfg.policy_type.value,
            "avg_reward_last20": round(avg_reward, 4),
            "avg_regret_last20": round(avg_regret, 4),
            "avg_depth_last20": round(avg_depth, 2),
            "bandit_epsilon": round(self._bandit._epsilon, 4),
            "bandit_best_arm": self._bandit.best_arm(),
            "ema_rewards": self._ema.summary(),
            "policy_contribution": self.get_policy_contribution(),
            "total_steps": self._step_count,
        }