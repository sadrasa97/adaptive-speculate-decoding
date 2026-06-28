"""
Module 2: Adaptive Draft Controller
Scientifically correct oscillation detection and policy integration.
"""
import time
import math
import logging
import collections
from dataclasses import dataclass, field
from typing import Optional, List, Deque
from enum import Enum
from monitor import RuntimeMonitor, InferenceSnapshot

logger = logging.getLogger("AdaptiveSD.Controller")

class SpeculationMode(Enum):
    DISABLED = "disabled"
    CONSERVATIVE = "conservative"
    NORMAL = "normal"
    AGGRESSIVE = "aggressive"

@dataclass
class DraftControllerConfig:
    min_depth: int = 1
    max_depth: int = 4  # FIX: Reduced for CPU-constrained environments
    initial_depth: int = 2  # FIX: Start conservative
    acceptance_high: float = 0.65
    acceptance_low: float = 0.25  # FIX: Lower threshold for CPU
    acceptance_disable: float = 0.05  # FIX: More aggressive disable
    cpu_high: float = 0.85
    cpu_critical: float = 0.95
    entropy_high: float = 8.5
    entropy_low: float = 5.0
    ctx_long: int = 2048
    ctx_very_long: int = 6144
    bandwidth_threshold: float = 500.0
    depth_change_cooldown: int = 5  # FIX: Faster adaptation
    latency_spike_ms: float = 200.0  # FIX: Higher threshold for CPU
    latency_spike_multiplier: float = 1.8  # FIX: More tolerant
    bad_steps_to_disable: int = 10  # FIX: Faster disable
    oscillation_window: int = 10  # FIX: Shorter window
    oscillation_threshold: int = 3  # FIX: Lower threshold
    max_oscillation_suppressions: int = 3
    policy_confidence_threshold: float = 0.60

@dataclass
class ControlDecision:
    mode: SpeculationMode
    draft_depth: int
    verify_batch_size: int
    reason: str
    policy_suggestion_applied: bool = False
    timestamp: float = field(default_factory=time.perf_counter)

class AdaptiveDraftController:
    def __init__(self, monitor: RuntimeMonitor, config: Optional[DraftControllerConfig] = None):
        self.monitor = monitor
        self.cfg = config or DraftControllerConfig()
        self._depth = self.cfg.initial_depth
        self._mode = SpeculationMode.NORMAL
        self._steps_since_change = 0
        self._depth_history: Deque[int] = collections.deque(maxlen=self.cfg.oscillation_window)
        self._history: Deque[ControlDecision] = collections.deque(maxlen=200)
        
        self._ema_acceptance = 0.5
        self._ema_cpu = 0.0
        self._ema_entropy = 1.0
        self._ema_alpha = 0.15
        
        self._bad_steps_cpu = 0
        self._bad_steps_acceptance = 0
        self._oscillation_count = 0
        self._depth_change_count = 0
        self._oscillation_suppressions = 0
        
        self._decision_counts = {reason: 0 for reason in [
            "cpu_critical", "acceptance_collapsed", "ctx_very_long",
            "bandwidth_saturated", "latency_spike", "cpu_high",
            "entropy_spike_high", "low_acceptance", "conditions_favorable",
            "recovery", "steady_state", "no_data_yet",
            "oscillation_suppressed", "context_pressure",
            "policy_suggestion_rejected", "policy_suggestion_accepted"
        ]}
        
        self._calibrated_latency_threshold: float = self.cfg.latency_spike_ms
        self._warmup_itl_samples: List[float] = []
        self._threshold_calibrated: bool = False
        
        logger.info(
            "AdaptiveDraftController initialised (depth=%d, range=[%d,%d])",
            self._depth, self.cfg.min_depth, self.cfg.max_depth
        )
    
    def step(self, snap: Optional[InferenceSnapshot] = None,
             policy_suggestion: Optional[int] = None,
             policy_confidence: float = 0.5) -> ControlDecision:
        if snap is None:
            snap = self.monitor.get_current_snapshot()
        if snap is None:
            return self._make_decision(SpeculationMode.CONSERVATIVE, self.cfg.initial_depth, "no_data_yet")
        
        self._steps_since_change += 1
        self._depth_history.append(self._depth)
        
        self._ema_acceptance = self._ema(self._ema_acceptance, snap.acceptance_ratio)
        self._ema_cpu = self._ema(self._ema_cpu, snap.cpu_utilization)
        self._ema_entropy = self._ema(self._ema_entropy, snap.token_entropy)
        
        # Oscillation detection
        if len(self._depth_history) >= self.cfg.oscillation_window:
            oscillations = self._count_oscillations()
            if oscillations >= self.cfg.oscillation_threshold:
                self._oscillation_count += 1
                if self._oscillation_suppressions < self.cfg.max_oscillation_suppressions:
                    self._oscillation_suppressions += 1
                    return self._suppress_oscillation()
        
        # Rule 1: CPU critical
        if self._ema_cpu >= self.cfg.cpu_critical:
            self._bad_steps_cpu += 1
            return self._set(SpeculationMode.DISABLED, self.cfg.min_depth, "cpu_critical")
        self._bad_steps_cpu = 0
        
        # Rule 2: acceptance collapse
        if self._ema_acceptance < self.cfg.acceptance_disable:
            self._bad_steps_acceptance += 1
            if self._bad_steps_acceptance >= self.cfg.bad_steps_to_disable:
                return self._set(SpeculationMode.DISABLED, self.cfg.min_depth, "acceptance_collapsed")
        else:
            self._bad_steps_acceptance = 0
        
        # Rule 3: very long context
        if snap.context_length >= self.cfg.ctx_very_long:
            return self._set(SpeculationMode.CONSERVATIVE, max(self.cfg.min_depth, 2), "ctx_very_long")
        
        # Rule 4: bandwidth saturation
        if snap.memory_bandwidth_mb_s > self.cfg.bandwidth_threshold and self.monitor._total_steps > 15:
            new_depth = max(self.cfg.min_depth, self._depth - 1)
            if self._can_change():
                return self._adjust_depth(new_depth, "bandwidth_saturated")
        
        # Rule 5: latency spike
        if snap.latency_ms > 0 and self.monitor._total_steps <= 20:
            self._warmup_itl_samples.append(snap.latency_ms)
        if not self._threshold_calibrated and len(self._warmup_itl_samples) >= 10:
            median_itl = sorted(self._warmup_itl_samples)[len(self._warmup_itl_samples) // 2]
            if median_itl > 1.0:
                self._calibrated_latency_threshold = median_itl * self.cfg.latency_spike_multiplier
                logger.info(
                    "Latency threshold calibrated: median_itl=%.1fms → spike_threshold=%.1fms",
                    median_itl, self._calibrated_latency_threshold
                )
            self._threshold_calibrated = True
        
        if snap.latency_ms > self._calibrated_latency_threshold and self._can_change() and self.monitor._total_steps > 10:
            new_depth = max(self.cfg.min_depth, self._depth - 1)
            return self._adjust_depth(new_depth, "latency_spike")
        
        # Rule 6: high CPU
        if self._ema_cpu >= self.cfg.cpu_high and self._can_change():
            new_depth = max(self.cfg.min_depth, self._depth - 1)
            return self._adjust_depth(new_depth, "cpu_high")
        
        # Rule 7: high entropy
        if self._ema_entropy > self.cfg.entropy_high and self._can_change():
            new_depth = max(self.cfg.min_depth, self._depth - 1)
            return self._adjust_depth(new_depth, "entropy_spike_high")
        
        # Rule 8: low acceptance
        if self._ema_acceptance < self.cfg.acceptance_low and self._can_change():
            new_depth = max(self.cfg.min_depth, self._depth - 1)
            return self._adjust_depth(new_depth, "low_acceptance")
        
        # Rule 9: favourable conditions
        if (
            self._ema_acceptance > self.cfg.acceptance_high
            and self._ema_cpu < self.cfg.cpu_high * 0.8
            and self._can_change()
            and snap.context_length < self.cfg.ctx_long
            and self._ema_entropy < self.cfg.entropy_low
        ):
            new_depth = min(self.cfg.max_depth, self._depth + 1)
            return self._adjust_depth(new_depth, "conditions_favorable")
        
        # Rule 10: re-enable if disabled
        if self._mode == SpeculationMode.DISABLED:
            if self._ema_acceptance > self.cfg.acceptance_low and self._ema_cpu < self.cfg.cpu_high:
                self._bad_steps_cpu = 0
                self._bad_steps_acceptance = 0
                return self._set(SpeculationMode.CONSERVATIVE, self.cfg.initial_depth, "recovery")
        
        # Rule 11: policy suggestion
        if policy_suggestion is not None and self._can_change():
            clamped = max(self.cfg.min_depth, min(self.cfg.max_depth, policy_suggestion))
            if clamped != self._depth and policy_confidence >= self.cfg.policy_confidence_threshold:
                if self._is_safe_suggestion(clamped, snap):
                    return self._adjust_depth(clamped, "policy_suggestion_accepted")
                else:
                    # Count rejection in decision stats (via _make_decision below)
                    self._decision_counts["policy_suggestion_rejected"] = (
                        self._decision_counts.get("policy_suggestion_rejected", 0) + 1
                    )
        
        self._mode = self._depth_to_mode(self._depth)
        return self._make_decision(self._mode, self._depth, "steady_state")
    
    def _is_safe_suggestion(self, suggested_depth: int, snap: InferenceSnapshot) -> bool:
        if self._oscillation_count > 2:
            return False
        if snap.stability_cv > 0.25:
            return False
        if suggested_depth > self._depth:
            if self._ema_acceptance < self.cfg.acceptance_low:
                return False
            if self._ema_cpu > self.cfg.cpu_high:
                return False
        return True
    
    def _count_oscillations(self) -> int:
        if len(self._depth_history) < 4:
            return 0
        changes = 0
        for i in range(1, len(self._depth_history)):
            if self._depth_history[i] != self._depth_history[i - 1]:
                changes += 1
        return changes
    
    def _suppress_oscillation(self) -> ControlDecision:
        depths = list(self._depth_history)
        median_depth = sorted(depths)[len(depths) // 2]
        logger.info("Oscillation suppressed: settling at depth=%d", median_depth)
        return self._set(SpeculationMode.NORMAL, median_depth, "oscillation_suppressed")
    
    def recommended_verify_batch(self) -> int:
        if self._mode == SpeculationMode.DISABLED:
            return 1
        if self._ema_cpu > 0.85:
            return max(1, self._depth // 2)
        return max(1, self._depth)
    
    def current_depth(self) -> int:
        return self._depth
    
    def current_mode(self) -> SpeculationMode:
        return self._mode
    
    def is_speculation_enabled(self) -> bool:
        return self._mode != SpeculationMode.DISABLED
    
    def history(self) -> List[ControlDecision]:
        return list(self._history)
    
    def get_decision_stats(self) -> dict:
        return dict(self._decision_counts)
    
    def get_stability_metrics(self) -> dict:
        total_changes = self._depth_change_count
        oscillations = self._oscillation_count
        stability_score = max(0.0, 1.0 - (oscillations / max(total_changes, 1)))
        depth_dist = {}
        for d in self._depth_history:
            depth_dist[d] = depth_dist.get(d, 0) + 1
        return {
            "depth_changes": total_changes,
            "oscillations": oscillations,
            "stability_score": round(stability_score, 4),
            "mean_depth": round(sum(self._depth_history) / len(self._depth_history), 2) if self._depth_history else self._depth,
            "depth_distribution": depth_dist,
        }
    
    def summary(self) -> dict:
        return {
            "mode": self._mode.value,
            "draft_depth": self._depth,
            "ema_acceptance": round(self._ema_acceptance, 3),
            "ema_cpu": round(self._ema_cpu, 3),
            "ema_entropy": round(self._ema_entropy, 4),
            "bad_steps_cpu": self._bad_steps_cpu,
            "bad_steps_acceptance": self._bad_steps_acceptance,
            "steps_since_depth_change": self._steps_since_change,
            "oscillation_count": self._oscillation_count,
            "calibrated_latency_threshold_ms": round(self._calibrated_latency_threshold, 1),
        }
    
    def _ema(self, prev: float, new: float) -> float:
        return self._ema_alpha * new + (1.0 - self._ema_alpha) * prev
    
    def _can_change(self) -> bool:
        return self._steps_since_change >= self.cfg.depth_change_cooldown
    
    def _adjust_depth(self, new_depth: int, reason: str) -> ControlDecision:
        if new_depth != self._depth:
            old = self._depth
            self._depth = new_depth
            self._steps_since_change = 0
            self._depth_change_count += 1
            logger.info("Depth changed: %d → %d (reason: %s)", old, new_depth, reason)
        self._mode = self._depth_to_mode(self._depth)
        return self._make_decision(self._mode, self._depth, reason)
    
    def _set(self, mode: SpeculationMode, depth: int, reason: str) -> ControlDecision:
        old_mode, old_depth = self._mode, self._depth
        self._mode = mode
        self._depth = depth
        self._steps_since_change = 0
        if old_mode != mode or old_depth != depth:
            self._depth_change_count += 1
            logger.info("Mode: %s/%d → %s/%d (reason: %s)", old_mode.value, old_depth, mode.value, depth, reason)
        return self._make_decision(mode, depth, reason)
    
    def _make_decision(self, mode: SpeculationMode, depth: int, reason: str, policy_applied: bool = False) -> ControlDecision:
        decision = ControlDecision(
            mode=mode, draft_depth=depth,
            verify_batch_size=self.recommended_verify_batch(),
            reason=reason, policy_suggestion_applied=policy_applied,
        )
        self._history.append(decision)
        if reason in self._decision_counts:
            self._decision_counts[reason] += 1
        return decision
    
    def _depth_to_mode(self, depth: int) -> SpeculationMode:
        if depth <= 0:
            return SpeculationMode.DISABLED
        if depth <= 2:
            return SpeculationMode.CONSERVATIVE
        if depth >= self.cfg.max_depth - 1:
            return SpeculationMode.AGGRESSIVE
        return SpeculationMode.NORMAL