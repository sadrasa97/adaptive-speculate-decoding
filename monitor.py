"""
Module 1: Runtime Monitoring Engine
Scientifically correct implementation with proper simulation support.
"""
import time
import threading
import collections
import math
import logging
import psutil
import os
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Dict
logger = logging.getLogger("AdaptiveSD.Monitor")

@dataclass
class InferenceSnapshot:
    timestamp: float
    tokens_per_sec: float
    ema_tps: float
    acceptance_ratio: float
    rejection_count: int
    cpu_utilization: float
    system_cpu_percent: float
    memory_rss_mb: float
    memory_bandwidth_mb_s: float
    context_length: int
    token_entropy: float
    l1_miss_proxy: float
    latency_ms: float
    itl_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    tail_latency_ratio: float
    speculation_depth: int
    phase: str
    stability_cv: float

@dataclass
class MonitorConfig:
    window_size: int = 50
    poll_interval_s: float = 0.05
    entropy_ema_alpha: float = 0.1
    bandwidth_ema_alpha: float = 0.2
    latency_spike_threshold_ms: float = 150.0
    acceptance_low_threshold: float = 0.4
    cpu_overload_threshold: float = 0.90
    tps_ema_alpha: float = 0.1
    stability_window: int = 20

class RuntimeMonitor:
    def __init__(self, config: Optional[MonitorConfig] = None):
        self.cfg = config or MonitorConfig()
        self._lock = threading.Lock()
        self._process = psutil.Process(os.getpid())
        self._last_cpu_times = self._process.cpu_times()
        self._last_cpu_time = time.perf_counter()

        self._latencies: Deque[float] = collections.deque(maxlen=self.cfg.window_size)
        self._itl_history: Deque[float] = collections.deque(maxlen=self.cfg.window_size)
        self._acceptance_window: Deque[float] = collections.deque(maxlen=self.cfg.window_size)
        self._step_times: Deque[float] = collections.deque(maxlen=self.cfg.window_size)
        self._step_tokens: Deque[int] = collections.deque(maxlen=self.cfg.window_size)
        self._snapshots: Deque[InferenceSnapshot] = collections.deque(maxlen=self.cfg.window_size)
        self._tps_history: Deque[float] = collections.deque(maxlen=self.cfg.stability_window)

        self._phase_snapshots: Dict[str, List[InferenceSnapshot]] = {
            "warmup": [], "steady": [], "cooldown": []
        }

        self._total_drafted = 0
        self._total_accepted = 0
        self._total_tokens_committed = 0
        self._total_steps = 0

        self._context_length = 0
        self._speculation_depth = 4
        self._last_token_time: Optional[float] = None
        self._first_token_time: Optional[float] = None
        self._last_mem_bytes = self._process.memory_info().rss
        self._last_mem_time = time.perf_counter()

        self._entropy_ema = 0.0
        self._entropy_initialised = False
        self._bandwidth_ema = 0.0
        self._tps_ema = 0.0

        self._start_time = time.perf_counter()
        self._generation_start_time: Optional[float] = None   # FIX: track generation start
        self._warmup_end = 20
        self._cooldown_start = 180

        self._baseline_tps: Optional[float] = None
        self._phase = "warmup"

        self._cpu_percent = 0.0
        self._sys_cpu_percent = 0.0

        self._simulation_mode = False
        self._simulated_cpu = 0.0

        self._stop_event = threading.Event()
        self._poll_thread = threading.Thread(target=self._cpu_poller, daemon=True)
        self._poll_thread.start()
        logger.info("RuntimeMonitor initialised (window=%d)", self.cfg.window_size)

    def set_baseline_tps(self, tps: float):
        self._baseline_tps = tps
        logger.info("Baseline TPS set to %.2f", tps)

    def mark_generation_start(self):
        now = time.perf_counter()
        self._start_time = now
        self._generation_start_time = now   # FIX: separate tracker for overall_tps denominator
        self._first_token_time = None
        logger.info("Generation start time marked.")

    def set_simulation_mode(self, enabled: bool):
        self._simulation_mode = enabled

    def _detect_phase(self) -> str:
        if self._total_steps < self._warmup_end:
            return "warmup"
        if self._total_steps >= self._cooldown_start:
            return "cooldown"
        return "steady"

    def _compute_stability(self) -> float:
        if len(self._tps_history) < 3:
            return 0.0
        mean = sum(self._tps_history) / len(self._tps_history)
        if mean <= 0:
            return 0.0
        variance = sum((x - mean) ** 2 for x in self._tps_history) / len(self._tps_history)
        std = math.sqrt(variance)
        return min(1.0, std / mean)

    def _compute_percentile(self, data, p: float) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_data[int(k)]
        return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)

    def on_token_generated(
        self,
        logits: Optional[List[float]] = None,
        drafted: int = 0,
        accepted: int = 0,
        context_length: int = 0,
        speculation_depth: int = 4,
        sim_entropy: Optional[float] = None,
        simulated_cpu: Optional[float] = None,
    ) -> "InferenceSnapshot":
        now = time.perf_counter()
        self._total_steps += 1
        self._phase = self._detect_phase()

        latency_ms = (now - self._last_token_time) * 1000.0 if self._last_token_time is not None else 0.0
        itl_ms = latency_ms
        self._last_token_time = now

        if self._first_token_time is None:
            self._first_token_time = now

        self._latencies.append(latency_ms)
        self._itl_history.append(itl_ms)

        committed = accepted + 1
        self._total_tokens_committed += committed
        self._step_times.append(now)
        self._step_tokens.append(committed)

        self._total_drafted += drafted
        accepted_drafts = max(0, accepted)
        self._total_accepted += accepted_drafts
        step_ratio = (accepted_drafts / drafted) if drafted > 0 else 0.0
        self._acceptance_window.append(step_ratio)

        if logits:
            entropy = self._compute_entropy(logits)
        elif sim_entropy is not None:
            entropy = sim_entropy
        else:
            entropy = 0.0

        if not self._entropy_initialised:
            self._entropy_ema = entropy
            self._entropy_initialised = True
        else:
            alpha = self.cfg.entropy_ema_alpha
            self._entropy_ema = alpha * entropy + (1.0 - alpha) * self._entropy_ema

        mem_bytes = self._process.memory_info().rss
        dt = now - self._last_mem_time
        if dt > 0.001 and self._total_steps > 10:
            delta_mb = abs(mem_bytes - self._last_mem_bytes) / (1024.0 * 1024.0)
            bw = delta_mb / dt
            alpha = self.cfg.bandwidth_ema_alpha
            self._bandwidth_ema = alpha * bw + (1.0 - alpha) * self._bandwidth_ema
        self._last_mem_bytes = mem_bytes
        self._last_mem_time = now

        tps = self.rolling_tps()
        if self._tps_ema == 0.0:
            self._tps_ema = self._baseline_tps if self._baseline_tps else tps
        else:
            self._tps_ema = self.cfg.tps_ema_alpha * tps + (1.0 - self.cfg.tps_ema_alpha) * self._tps_ema
        self._tps_history.append(tps)

        p95 = self._compute_percentile(self._itl_history, 0.95)
        p99 = self._compute_percentile(self._itl_history, 0.99)
        mean_itl = sum(self._itl_history) / len(self._itl_history) if self._itl_history else 0.0
        tail_ratio = (p95 / mean_itl) if mean_itl > 0 else 1.0
        stability = self._compute_stability()

        if simulated_cpu is not None:
            self._simulated_cpu = simulated_cpu
            self._cpu_percent = simulated_cpu
            cpu_util = simulated_cpu
        elif self._simulation_mode:
            cpu_util = self._simulated_cpu
        else:
            cpu_util = self._cpu_percent

        with self._lock:
            snap = InferenceSnapshot(
                timestamp=now,
                tokens_per_sec=tps,
                ema_tps=self._tps_ema,
                acceptance_ratio=self._rolling_acceptance(),
                rejection_count=self._total_drafted - self._total_accepted,
                cpu_utilization=cpu_util,
                system_cpu_percent=self._sys_cpu_percent,
                memory_rss_mb=mem_bytes / (1024.0 * 1024.0),
                memory_bandwidth_mb_s=self._bandwidth_ema,
                context_length=context_length,
                token_entropy=self._entropy_ema,
                l1_miss_proxy=self._latency_variance(),
                latency_ms=latency_ms,
                itl_ms=itl_ms,
                p95_latency_ms=p95,
                p99_latency_ms=p99,
                tail_latency_ratio=tail_ratio,
                speculation_depth=speculation_depth,
                phase=self._phase,
                stability_cv=stability,
            )
            self._snapshots.append(snap)
            self._phase_snapshots[self._phase].append(snap)
            self._context_length = context_length
            self._speculation_depth = speculation_depth

        return snap

    def rolling_tps(self) -> float:
        n = len(self._step_times)
        if n < 2:
            elapsed = time.perf_counter() - self._start_time
            return (self._total_tokens_committed / elapsed) if elapsed > 0 else 0.0
        span = self._step_times[-1] - self._step_times[0]
        if span < 0.5:
            elapsed = time.perf_counter() - self._start_time
            return (self._total_tokens_committed / elapsed) if elapsed > 0 else 0.0
        tokens_in_window = sum(list(self._step_tokens)[1:])
        return tokens_in_window / span

    def rolling_latency_ms(self) -> float:
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies)

    def _rolling_acceptance(self) -> float:
        if not self._acceptance_window:
            return 0.0
        return sum(self._acceptance_window) / len(self._acceptance_window)

    def get_ttft_ms(self) -> float:
        if self._first_token_time is None:
            return 0.0
        return (self._first_token_time - self._start_time) * 1000.0

    def get_current_snapshot(self) -> Optional[InferenceSnapshot]:
        with self._lock:
            return self._snapshots[-1] if self._snapshots else None

    def get_all_snapshots(self) -> List[InferenceSnapshot]:
        with self._lock:
            return list(self._snapshots)

    def is_cpu_overloaded(self) -> bool:
        return self._cpu_percent > self.cfg.cpu_overload_threshold

    def is_acceptance_low(self) -> bool:
        return self._rolling_acceptance() < self.cfg.acceptance_low_threshold

    def is_latency_spiking(self) -> bool:
        return self.rolling_latency_ms() > self.cfg.latency_spike_threshold_ms

    def get_phase_breakdown(self) -> Dict[str, Dict]:
        has_speculation = self._total_drafted > 0
        result = {}
        for phase, snaps in self._phase_snapshots.items():
            if not snaps:
                result[phase] = {"steps": 0, "avg_tps": 0.0, "avg_acceptance": 0.0, "avg_depth": 0.0}
                continue
            valid_snaps = [s for s in snaps if s.itl_ms > 1.0]
            if valid_snaps:
                avg_itl = sum(s.itl_ms for s in valid_snaps) / len(valid_snaps)
                avg_depth_phase = sum(s.speculation_depth for s in snaps) / len(snaps)
                avg_acc_phase = sum(s.acceptance_ratio for s in snaps) / len(snaps)
                avg_tok_step = 1.0 + avg_depth_phase * avg_acc_phase
                avg_tps = (avg_tok_step * 1000.0) / avg_itl if avg_itl > 0 else 0.0
            else:
                avg_tps = 0.0
            avg_depth_phase_raw = sum(s.speculation_depth for s in snaps) / len(snaps)
            result[phase] = {
                "steps": len(snaps),
                "avg_tps": round(avg_tps, 2),
                "avg_acceptance": round(sum(s.acceptance_ratio for s in snaps) / len(snaps), 4),
                "avg_depth": round(avg_depth_phase_raw, 2) if has_speculation else 0.0,
            }
        return result

    def get_evaluation_metrics(self) -> dict:
        current_tps = self.rolling_tps()

        # FIX: use generation_start_time so overall_tps excludes model-load / baseline time
        t_ref = self._generation_start_time if self._generation_start_time is not None else self._start_time
        elapsed_total = time.perf_counter() - t_ref
        overall_tps = (self._total_tokens_committed / elapsed_total) if elapsed_total > 0 else 0.0
        
        has_speculation = self._total_drafted > 0
        avg_acceptance = (self._total_accepted / self._total_drafted) if has_speculation else 0.0
        
        # FIX: speedup is None when no speculative path was active (not 1.0)
        speedup = (
            overall_tps / self._baseline_tps
            if self._baseline_tps and self._baseline_tps > 0 and has_speculation else None
        )
        avg_depth = (
            sum(s.speculation_depth for s in self._snapshots) / len(self._snapshots)
            if self._snapshots else 1.0
        )
        
        total_wasted = self._total_drafted - self._total_accepted
        efficiency = (
            self._total_tokens_committed / (self._total_tokens_committed + max(0, total_wasted))
            if (self._total_tokens_committed + max(0, total_wasted)) > 0 else 0.0
        )

        sample_sufficient = self._total_tokens_committed >= 100

        ttft = self.get_ttft_ms()
        p95 = self._compute_percentile(self._itl_history, 0.95)
        p99 = self._compute_percentile(self._itl_history, 0.99)
        mean_itl = sum(self._itl_history) / len(self._itl_history) if self._itl_history else 0.0
        max_itl = max(self._itl_history) if self._itl_history else 0.0
        tail_ratio = (p95 / mean_itl) if mean_itl > 0 else 1.0
        stability = self._compute_stability()
        peak_tps = max(self._tps_history) if self._tps_history else current_tps
        min_tps = min(self._tps_history) if self._tps_history else current_tps

        # FIX: Correctly determine baseline source based on actual mode
        if self._simulation_mode:
            baseline_source = "estimated_from_model_size"
        elif self._baseline_tps and self._baseline_tps > 0:
            baseline_source = "measured_real_tps"
        else:
            baseline_source = "unknown"

        return {
            "current_tps": round(current_tps, 2),
            "overall_tps": round(overall_tps, 2),
            "baseline_tps": round(self._baseline_tps, 2) if self._baseline_tps else None,
            "peak_tps": round(peak_tps, 2),
            "min_tps": round(min_tps, 2),
            "speedup_ratio": round(speedup, 3) if speedup is not None else None,
            "efficiency_gain": round(efficiency, 3) if has_speculation else None,
            "stability_cv": round(stability, 4),
            "total_tokens": self._total_tokens_committed,
            "total_drafted": self._total_drafted,
            "total_accepted": self._total_accepted,
            "total_wasted": max(0, total_wasted),
            "avg_acceptance": round(avg_acceptance, 4),
            "avg_depth": round(avg_depth, 2) if has_speculation else 0.0,
            "speculative_efficiency": round(efficiency, 3) if has_speculation else None,
            "token_throughput_efficiency": round(avg_acceptance, 4) if has_speculation else None,
            "ttft_ms": round(ttft, 2),
            "mean_itl_ms": round(mean_itl, 2),
            "p95_itl_ms": round(p95, 2),
            "p99_itl_ms": round(p99, 2),
            "max_itl_ms": round(max_itl, 2),
            "tail_latency_ratio": round(tail_ratio, 2),
            "avg_latency_ms": round(self.rolling_latency_ms(), 2),
            "rejection_rate": round(1.0 - avg_acceptance, 4) if has_speculation else 0.0,
            "total_steps": self._total_steps,
            "total_speculative_steps": self._total_steps if has_speculation else 0,
            "speculation_active": has_speculation,
            "sample_sufficient": sample_sufficient,
            "scientific_validity": {
                "speculation_metrics_valid": has_speculation,
                "sample_size_sufficient": sample_sufficient,
                "sample_tokens": self._total_tokens_committed,
                "baseline_source": baseline_source,  # FIX: Now correctly reports source
            },
            "phase_breakdown": self.get_phase_breakdown(),
        }

    def summary(self) -> dict:
        snap = self.get_current_snapshot()
        p95 = self._compute_percentile(self._itl_history, 0.95)
        p99 = self._compute_percentile(self._itl_history, 0.99)
        mean_itl = sum(self._itl_history) / len(self._itl_history) if self._itl_history else 0.0

        if self._simulation_mode:
            cpu_report = self._simulated_cpu
        else:
            cpu_report = self._cpu_percent

        return {
            "rolling_tps": round(self.rolling_tps(), 2),
            "ema_tps": round(self._tps_ema, 2),
            "rolling_latency_ms": round(self.rolling_latency_ms(), 2),
            "rolling_acceptance": round(self._rolling_acceptance(), 3),
            "cpu_utilization": round(cpu_report, 3),
            "system_cpu_percent": round(self._sys_cpu_percent, 1),
            "memory_rss_mb": round(snap.memory_rss_mb, 1) if snap else 0,
            "bandwidth_mb_s": round(self._bandwidth_ema, 2),
            "entropy_ema": round(self._entropy_ema, 4),
            "context_length": self._context_length,
            "l1_miss_proxy": round(self._latency_variance(), 4),
            "total_drafted": self._total_drafted,
            "total_accepted": self._total_accepted,
            "phase": self._phase,
            "stability_cv": round(self._compute_stability(), 4),
            "ttft_ms": round(self.get_ttft_ms(), 2),
            "p95_itl_ms": round(p95, 2),
            "p99_itl_ms": round(p99, 2),
            "mean_itl_ms": round(mean_itl, 2),
        }

    def stop(self):
        self._stop_event.set()
        logger.info("RuntimeMonitor stopped")

    def _cpu_poller(self):
        while not self._stop_event.is_set():
            try:
                if not self._simulation_mode:
                    new_times = self._process.cpu_times()
                    now = time.perf_counter()
                    dt = now - self._last_cpu_time
                    if dt > 0:
                        proc_cpu = sum(new_times) - sum(self._last_cpu_times)
                        proc_cpu_pct = proc_cpu / dt / max(1, os.cpu_count() or 1)
                        with self._lock:
                            self._cpu_percent = min(1.0, proc_cpu_pct)
                    self._last_cpu_times = new_times
                    self._last_cpu_time = now

                sys_cpu = psutil.cpu_percent(interval=None)
                with self._lock:
                    self._sys_cpu_percent = sys_cpu
            except Exception:
                pass
            time.sleep(self.cfg.poll_interval_s)

    @staticmethod
    def _compute_entropy(logits: List[float]) -> float:
        max_l = max(logits)
        exps = [math.exp(x - max_l) for x in logits]
        s = sum(exps)
        if s <= 0.0:
            return 0.0
        probs = [e / s for e in exps]
        return -sum(p * math.log(p) for p in probs if p > 0.0)

    def _latency_variance(self) -> float:
        n = len(self._latencies)
        if n < 2:
            return 0.0
        mean = sum(self._latencies) / n
        return sum((x - mean) ** 2 for x in self._latencies) / (n - 1)