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
    depth_choices: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 6, 8])
    confidence_threshold: float = 0.6
    workload_change_hysteresis: int = 4

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

    def best_arm(self) -> int:
        pulled = [a for a in self._arms.values() if a.pulls > 0]
        return max(pulled, key=lambda a: a.avg_reward).depth if pulled else self.cfg.depth_choices[0]

    def arm_summary(self) -> dict:
        return {d: {"pulls": a.pulls, "avg_reward": round(a.avg_reward, 4),
                    "avg_regret": round(a.avg_regret(), 4)} for d, a in self._arms.items()}

    def confidence(self) -> float:
        return self._confidence

class EMAPolicy:
    def __init__(self, config: PolicyConfig):
        self.cfg = config
        self._ema_rewards: Dict[int, float] = {d: 0.5 for d in config.depth_choices}
        self._ema_variance: Dict[int, float] = {d: 0.1 for d in config.depth_choices}
        self._last_depth: Optional[int] = None

    def choose_depth(self) -> Tuple[int, float]:
        best = max(self._ema_rewards, key=self._ema_rewards.get)
        self._last_depth = best
        return best, 0.6

    def update(self, depth: int, reward: float):
        if depth in self._ema_rewards:
            alpha = self.cfg.ema_alpha
            old = self._ema_rewards[depth]
            self._ema_rewards[depth] = alpha * reward + (1.0 - alpha) * old
            self._ema_variance[depth] = alpha * ((reward - old) ** 2) + (1.0 - alpha) * self._ema_variance[depth]

    def summary(self) -> dict:
        return {d: {"reward": round(v, 4), "variance": round(self._ema_variance[d], 6)}
                for d, v in self._ema_rewards.items()}

    def confidence(self) -> float:
        return 0.6

class DynamicPolicyEngine:
    def __init__(self, monitor: RuntimeMonitor, controller: AdaptiveDraftController,
                 config: Optional[PolicyConfig] = None):
        self.monitor = monitor
        self.controller = controller
        self.cfg = config or PolicyConfig()
        self._heuristic = HeuristicPolicy(self.cfg)
        self._bandit = BanditPolicy(self.cfg)
        self._ema = EMAPolicy(self.cfg)
        
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

        depths = {
            PolicyType.HEURISTIC: self._heuristic.choose_depth(self._workload, snap, self.controller),
            PolicyType.BANDIT: self._bandit.choose_depth(),
            PolicyType.EMA: self._ema.choose_depth(),
        }
        final_depth, _ = self._combine(depths, snap)
        final_depth = max(self.controller.cfg.min_depth, min(self.controller.cfg.max_depth, final_depth))
        confidence = self._compute_ensemble_confidence()

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
        if self.cfg.policy_type == PolicyType.HEURISTIC:
            return self._heuristic.confidence()
        if self.cfg.policy_type == PolicyType.BANDIT:
            return self._bandit.confidence()
        if self.cfg.policy_type == PolicyType.EMA:
            return self._ema.confidence()
        w_h = self.cfg.ensemble_heuristic_weight * self._heuristic.confidence()
        w_b = self.cfg.ensemble_bandit_weight * self._bandit.confidence()
        w_e = self.cfg.ensemble_ema_weight * self._ema.confidence()
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
        latency_penalty = 1.0 / (1.0 + math.exp(0.05 * (curr.latency_ms - 50.0)))
        tps_prev = max(prev.tokens_per_sec, 1e-6)
        tps_delta = (curr.tokens_per_sec - prev.tokens_per_sec) / tps_prev
        tps_reward = 1.0 / (1.0 + math.exp(-3.0 * tps_delta))
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
        pt = self.cfg.policy_type
        if pt == PolicyType.HEURISTIC:
            return depths[PolicyType.HEURISTIC][0], True
        if pt == PolicyType.BANDIT:
            return depths[PolicyType.BANDIT][0], True
        if pt == PolicyType.EMA:
            return depths[PolicyType.EMA][0], True

        w_h = self.cfg.ensemble_heuristic_weight * self._heuristic.confidence()
        w_b = self.cfg.ensemble_bandit_weight * self._bandit.confidence()
        w_e = self.cfg.ensemble_ema_weight * self._ema.confidence()
        total = w_h + w_b + w_e
        if total <= 0:
            return depths[PolicyType.HEURISTIC][0], True
        weighted = (w_h * depths[PolicyType.HEURISTIC][0]
                    + w_b * depths[PolicyType.BANDIT][0]
                    + w_e * depths[PolicyType.EMA][0]) / total
        return round(weighted), True

    def workload(self) -> WorkloadType:
        return self._workload

    def get_workload_distribution(self) -> dict:
        total = sum(self._workload_history.values()) or 1
        return {k: {"count": v, "percent": round(v / total * 100, 1)} for k, v in self._workload_history.items()}

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