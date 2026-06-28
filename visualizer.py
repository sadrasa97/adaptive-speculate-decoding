"""
Module 6: Metrics Visualizer
Generates publication-quality PNG plots from benchmark/evaluation data.
Saves all plots to the output directory.
"""
import os
import math
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger("AdaptiveSD.Visualizer")

# ── Try to import matplotlib; degrade gracefully if unavailable ──────────────
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend (no display required)
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches
    import numpy as np
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False
    logger.warning("matplotlib not available — visualizations skipped")


@dataclass
class RunRecord:
    """Flat record for one benchmark run."""
    prompt_short: str
    elapsed_s: float
    overall_tps: float
    speedup_ratio: Optional[float]
    avg_acceptance: float
    avg_depth: float
    mean_itl_ms: float
    p95_itl_ms: float
    p99_itl_ms: float
    speculative_efficiency: Optional[float]
    total_tokens: int
    total_drafted: int
    total_accepted: int
    total_wasted: int
    baseline_tps: Optional[float]
    peak_tps: float
    min_tps: float
    stability_cv: float
    ttft_ms: float
    phase_breakdown: dict
    decision_stats: dict
    stability_metrics: dict
    workload_distribution: dict
    policy_summary: dict
    kv_stats: dict
    # per-step time series (appended by engine)
    tps_series: List[float] = None
    acceptance_series: List[float] = None
    depth_series: List[int] = None
    itl_series: List[float] = None
    cpu_series: List[float] = None
    entropy_series: List[float] = None

    def __post_init__(self):
        for attr in ("tps_series", "acceptance_series", "depth_series",
                     "itl_series", "cpu_series", "entropy_series"):
            if getattr(self, attr) is None:
                object.__setattr__(self, attr, [])


# ── Colour palette ────────────────────────────────────────────────────────────
_PAL = {
    "blue":   "#2563EB",
    "green":  "#16A34A",
    "red":    "#DC2626",
    "orange": "#EA580C",
    "purple": "#7C3AED",
    "teal":   "#0D9488",
    "gray":   "#6B7280",
    "bg":     "#F8FAFC",
    "grid":   "#E2E8F0",
}


def _style(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    ax.set_facecolor(_PAL["bg"])
    ax.grid(True, color=_PAL["grid"], linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)


def _save(fig, path: str):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Plot saved → %s", path)
    print(f"  [PLOT] Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot 1 – Overview Dashboard (multi-panel)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_overview_dashboard(runs: List[RunRecord], out_dir: str, tag: str = "") -> str:
    if not _MPL_AVAILABLE or not runs:
        return ""

    labels = [f"Run {i+1}" for i in range(len(runs))]
    tps_vals   = [r.overall_tps  for r in runs]
    spd_vals   = [r.speedup_ratio if r.speedup_ratio else 0.0 for r in runs]
    acc_vals   = [r.avg_acceptance * 100 for r in runs]
    itl_vals   = [r.mean_itl_ms  for r in runs]
    p95_vals   = [r.p95_itl_ms   for r in runs]
    eff_vals   = [(r.speculative_efficiency or 0.0) * 100 for r in runs]

    fig = plt.figure(figsize=(14, 9), facecolor="white")
    fig.suptitle(
        f"Adaptive Speculative Decoding — Overview Dashboard{' [' + tag + ']' if tag else ''}",
        fontsize=13, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    x = np.arange(len(runs))
    bar_w = 0.5

    # ── TPS ──
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(x, tps_vals, bar_w, color=_PAL["blue"], alpha=0.85, zorder=3)
    if runs[0].baseline_tps:
        ax1.axhline(runs[0].baseline_tps, color=_PAL["red"], ls="--", lw=1.2,
                    label=f"Baseline {runs[0].baseline_tps:.1f}")
        ax1.legend(fontsize=7)
    for bar, v in zip(bars, tps_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=7)
    _style(ax1, "Throughput (TPS)", "", "tokens/sec")

    # ── Speedup ──
    ax2 = fig.add_subplot(gs[0, 1])
    colors = [_PAL["green"] if s >= 1.0 else _PAL["red"] for s in spd_vals]
    bars2 = ax2.bar(x, spd_vals, bar_w, color=colors, alpha=0.85, zorder=3)
    ax2.axhline(1.0, color=_PAL["gray"], ls="--", lw=1.0, label="1.0× baseline")
    ax2.legend(fontsize=7)
    for bar, v in zip(bars2, spd_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{v:.3f}×", ha="center", va="bottom", fontsize=7)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=7)
    _style(ax2, "Speculative Speedup", "", "speedup ratio")

    # ── Acceptance Rate ──
    ax3 = fig.add_subplot(gs[0, 2])
    bars3 = ax3.bar(x, acc_vals, bar_w, color=_PAL["purple"], alpha=0.85, zorder=3)
    ax3.axhline(50, color=_PAL["gray"], ls="--", lw=0.8, label="50% line")
    ax3.legend(fontsize=7)
    for bar, v in zip(bars3, acc_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{v:.1f}%", ha="center", va="bottom", fontsize=7)
    ax3.set_ylim(0, 110)
    ax3.set_xticks(x); ax3.set_xticklabels(labels, fontsize=7)
    _style(ax3, "Draft Acceptance Rate", "", "%")

    # ── Latency ──
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.bar(x - bar_w/4, itl_vals, bar_w/2, label="Mean ITL", color=_PAL["teal"], alpha=0.85, zorder=3)
    ax4.bar(x + bar_w/4, p95_vals, bar_w/2, label="P95 ITL", color=_PAL["orange"], alpha=0.85, zorder=3)
    ax4.legend(fontsize=7)
    ax4.set_xticks(x); ax4.set_xticklabels(labels, fontsize=7)
    _style(ax4, "Inter-Token Latency", "", "ms")

    # ── Speculative Efficiency ──
    ax5 = fig.add_subplot(gs[1, 1])
    bars5 = ax5.bar(x, eff_vals, bar_w, color=_PAL["green"], alpha=0.85, zorder=3)
    ax5.set_ylim(0, 110)
    for bar, v in zip(bars5, eff_vals):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{v:.1f}%", ha="center", va="bottom", fontsize=7)
    ax5.set_xticks(x); ax5.set_xticklabels(labels, fontsize=7)
    _style(ax5, "Speculative Efficiency", "", "%")

    # ── Draft Token Budget ──
    ax6 = fig.add_subplot(gs[1, 2])
    total_d = [r.total_drafted for r in runs]
    total_a = [r.total_accepted for r in runs]
    total_w = [r.total_wasted  for r in runs]
    ax6.bar(x, total_a, bar_w, label="Accepted",  color=_PAL["green"],  alpha=0.85, zorder=3)
    ax6.bar(x, total_w, bar_w, bottom=total_a,   label="Wasted",    color=_PAL["red"],    alpha=0.85, zorder=3)
    ax6.legend(fontsize=7)
    ax6.set_xticks(x); ax6.set_xticklabels(labels, fontsize=7)
    _style(ax6, "Draft Token Budget", "", "tokens")

    path = os.path.join(out_dir, f"01_overview_dashboard{tag}.png")
    _save(fig, path)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot 2 – Time-series per run (TPS, depth, acceptance, ITL)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_time_series(runs: List[RunRecord], out_dir: str, tag: str = "") -> List[str]:
    if not _MPL_AVAILABLE:
        return []
    paths = []
    for idx, run in enumerate(runs):
        # Build synthetic step-series from cumulative stats if no real series
        tps   = run.tps_series or []
        acc   = run.acceptance_series or []
        depth = run.depth_series or []
        itl   = run.itl_series or []
        cpu   = run.cpu_series or []
        ent   = run.entropy_series or []

        if not tps:
            logger.debug("No time-series data for run %d — skipping", idx+1)
            continue

        steps = np.arange(len(tps))
        fig, axes = plt.subplots(3, 2, figsize=(13, 10), facecolor="white")
        fig.suptitle(
            f"Run {idx+1} Time Series — {run.prompt_short[:55]}",
            fontsize=11, fontweight="bold"
        )

        # TPS
        a = axes[0, 0]
        a.plot(steps, tps, color=_PAL["blue"], lw=1.2, label="TPS")
        if run.baseline_tps:
            a.axhline(run.baseline_tps, color=_PAL["red"], ls="--", lw=1.0,
                      label=f"Baseline {run.baseline_tps:.1f}")
        a.legend(fontsize=7); _style(a, "Tokens/sec over Steps", "step", "TPS")

        # ITL
        a2 = axes[0, 1]
        if itl:
            a2.plot(steps[:len(itl)], itl, color=_PAL["orange"], lw=1.0, alpha=0.8)
            _smooth = _ewma(itl, 0.15)
            a2.plot(steps[:len(_smooth)], _smooth, color=_PAL["red"], lw=1.4, label="EMA")
            a2.legend(fontsize=7)
        _style(a2, "Inter-Token Latency (ms)", "step", "ms")

        # Acceptance
        a3 = axes[1, 0]
        if acc:
            a3.plot(steps[:len(acc)], [v*100 for v in acc],
                    color=_PAL["purple"], lw=1.0, alpha=0.8)
            a3.axhline(50, color=_PAL["gray"], ls="--", lw=0.8)
        a3.set_ylim(0, 105)
        _style(a3, "Acceptance Rate (%)", "step", "%")

        # Depth
        a4 = axes[1, 1]
        if depth:
            a4.step(steps[:len(depth)], depth, color=_PAL["teal"], lw=1.2, where="post")
        _style(a4, "Draft Depth", "step", "depth")

        # CPU
        a5 = axes[2, 0]
        if cpu:
            a5.plot(steps[:len(cpu)], [v*100 for v in cpu],
                    color=_PAL["red"], lw=1.0, alpha=0.8)
            a5.axhline(85, color=_PAL["gray"], ls="--", lw=0.8, label="High threshold")
            a5.legend(fontsize=7)
        a5.set_ylim(0, 105)
        _style(a5, "CPU Utilisation (%)", "step", "%")

        # Entropy
        a6 = axes[2, 1]
        if ent:
            a6.plot(steps[:len(ent)], ent, color=_PAL["green"], lw=1.0, alpha=0.8)
        _style(a6, "Token Entropy (EMA)", "step", "H (nats)")

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(out_dir, f"02_timeseries_run{idx+1}{tag}.png")
        _save(fig, path)
        paths.append(path)
    return paths


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot 3 – Controller Decisions + Depth Distribution
# ═══════════════════════════════════════════════════════════════════════════════
def plot_controller_analysis(runs: List[RunRecord], out_dir: str, tag: str = "") -> str:
    if not _MPL_AVAILABLE or not runs:
        return ""

    # Aggregate decision stats across runs
    agg_decisions: Dict[str, int] = {}
    agg_depth_dist: Dict[int, int] = {}
    for r in runs:
        for k, v in r.decision_stats.items():
            agg_decisions[k] = agg_decisions.get(k, 0) + v
        dd = r.stability_metrics.get("depth_distribution", {})
        for d, cnt in dd.items():
            agg_depth_dist[int(d)] = agg_depth_dist.get(int(d), 0) + cnt

    # Filter non-zero
    decisions = {k: v for k, v in sorted(agg_decisions.items(), key=lambda x: -x[1]) if v > 0}
    depth_keys = sorted(agg_depth_dist.keys())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="white")
    fig.suptitle("Controller Decision Analysis", fontsize=12, fontweight="bold")

    # Decision bar
    ax1 = axes[0]
    if decisions:
        d_labels = list(decisions.keys())
        d_vals   = list(decisions.values())
        y_pos = np.arange(len(d_labels))
        colors = [_PAL["blue"] if "favorable" in k or "recovery" in k
                  else _PAL["red"] if "spike" in k or "critical" in k or "collapse" in k
                  else _PAL["gray"] for k in d_labels]
        ax1.barh(y_pos, d_vals, 0.6, color=colors, alpha=0.85, zorder=3)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([k.replace("_", " ") for k in d_labels], fontsize=8)
        for i, v in enumerate(d_vals):
            ax1.text(v + 0.1, i, str(v), va="center", fontsize=7)
    _style(ax1, "Controller Decisions (aggregated)", "count", "")

    # Depth histogram
    ax2 = axes[1]
    if depth_keys:
        vals = [agg_depth_dist[d] for d in depth_keys]
        bars = ax2.bar(depth_keys, vals, 0.6, color=_PAL["teal"], alpha=0.85, zorder=3)
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                     str(v), ha="center", fontsize=8)
    _style(ax2, "Draft Depth Distribution", "depth", "steps")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(out_dir, f"03_controller_analysis{tag}.png")
    _save(fig, path)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot 4 – Policy Engine & Bandit EMA Rewards
# ═══════════════════════════════════════════════════════════════════════════════
def plot_policy_analysis(runs: List[RunRecord], out_dir: str, tag: str = "") -> str:
    if not _MPL_AVAILABLE or not runs:
        return ""

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor="white")
    fig.suptitle("Policy Engine Analysis", fontsize=12, fontweight="bold")

    # EMA rewards per depth per run
    ax1 = axes[0]
    depth_labels = None
    for i, run in enumerate(runs):
        ema = run.policy_summary.get("ema_rewards", {})
        if not ema:
            continue
        ds = sorted(ema.keys())
        depth_labels = depth_labels or ds
        rewards = []
        for d in ds:
            info = ema[d]
            r = info.get("reward", info) if isinstance(info, dict) else info
            rewards.append(float(r))
        ax1.plot(ds, rewards, marker="o", label=f"Run {i+1}", lw=1.4)
    ax1.legend(fontsize=7)
    _style(ax1, "EMA Reward per Depth", "depth", "reward")

    # Policy contribution (pie per run or averaged)
    ax2 = axes[1]
    agg_contrib: Dict[str, float] = {}
    for run in runs:
        for k, v in run.policy_summary.get("policy_contribution", {}).items():
            agg_contrib[k] = agg_contrib.get(k, 0.0) + v
    if agg_contrib:
        total = sum(agg_contrib.values())
        labels_p = list(agg_contrib.keys())
        sizes_p  = [v / total * 100 for v in agg_contrib.values()]
        pie_colors = [_PAL["blue"], _PAL["green"], _PAL["orange"]][:len(labels_p)]
        ax2.pie(sizes_p, labels=labels_p, autopct="%1.1f%%",
                colors=pie_colors, textprops={"fontsize": 8})
    ax2.set_title("Policy Contribution (avg)", fontsize=10, fontweight="bold")

    # Reward & Regret per run
    ax3 = axes[2]
    x = np.arange(len(runs))
    rewards_r = [r.policy_summary.get("avg_reward_last20", 0.0) for r in runs]
    regrets_r = [r.policy_summary.get("avg_regret_last20", 0.0) for r in runs]
    ax3.bar(x - 0.2, rewards_r, 0.35, label="Avg Reward", color=_PAL["green"], alpha=0.85, zorder=3)
    ax3.bar(x + 0.2, regrets_r, 0.35, label="Avg Regret", color=_PAL["red"],   alpha=0.85, zorder=3)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"Run {i+1}" for i in range(len(runs))], fontsize=7)
    ax3.legend(fontsize=7)
    _style(ax3, "Policy Reward / Regret", "run", "value")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(out_dir, f"04_policy_analysis{tag}.png")
    _save(fig, path)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot 5 – KV Cache Statistics
# ═══════════════════════════════════════════════════════════════════════════════
def plot_kv_cache(runs: List[RunRecord], out_dir: str, tag: str = "") -> str:
    if not _MPL_AVAILABLE or not runs:
        return ""

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), facecolor="white")
    fig.suptitle("KV Cache Statistics", fontsize=12, fontweight="bold")

    x = np.arange(len(runs))
    labels = [f"Run {i+1}" for i in range(len(runs))]

    # Hit rate
    ax1 = axes[0]
    hr = [r.kv_stats.get("hit_rate", 0.0) * 100 for r in runs]
    bars = ax1.bar(x, hr, 0.5, color=_PAL["green"], alpha=0.85, zorder=3)
    ax1.set_ylim(0, 110)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=7)
    for bar, v in zip(bars, hr):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{v:.1f}%", ha="center", fontsize=7)
    _style(ax1, "KV Cache Hit Rate", "", "%")

    # Rollbacks (partial vs full)
    ax2 = axes[1]
    full_rb    = [r.kv_stats.get("full_rollbacks", 0)    for r in runs]
    partial_rb = [r.kv_stats.get("partial_rollbacks", 0) for r in runs]
    ax2.bar(x, full_rb,    0.5, label="Full",    color=_PAL["red"],    alpha=0.85, zorder=3)
    ax2.bar(x, partial_rb, 0.5, bottom=full_rb,  label="Partial", color=_PAL["orange"], alpha=0.85, zorder=3)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=7)
    ax2.legend(fontsize=7)
    _style(ax2, "KV Rollbacks", "", "count")

    # Compression ratio & memory pressure
    ax3 = axes[2]
    cr  = [r.kv_stats.get("compression_ratio", 1.0)  for r in runs]
    mp  = [r.kv_stats.get("memory_pressure", 0.0) * 100 for r in runs]
    ax3.bar(x - 0.15, cr,  0.28, label="Compression ×", color=_PAL["blue"],   alpha=0.85, zorder=3)
    ax3_r = ax3.twinx()
    ax3_r.bar(x + 0.15, mp, 0.28, label="Mem Pressure %", color=_PAL["purple"], alpha=0.6,  zorder=3)
    ax3_r.set_ylabel("Memory Pressure (%)", fontsize=8)
    ax3_r.tick_params(labelsize=7)
    ax3.set_xticks(x); ax3.set_xticklabels(labels, fontsize=7)
    ax3.legend(loc="upper left",  fontsize=7)
    ax3_r.legend(loc="upper right", fontsize=7)
    _style(ax3, "Compression & Memory", "", "ratio")

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = os.path.join(out_dir, f"05_kv_cache{tag}.png")
    _save(fig, path)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot 6 – Phase Breakdown heatmap + Latency CDF
# ═══════════════════════════════════════════════════════════════════════════════
def plot_phase_and_latency(runs: List[RunRecord], out_dir: str, tag: str = "") -> str:
    if not _MPL_AVAILABLE or not runs:
        return ""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")
    fig.suptitle("Phase Breakdown & Latency Distribution", fontsize=12, fontweight="bold")

    # Phase stacked bar
    ax1 = axes[0]
    phases = ["warmup", "steady", "cooldown"]
    phase_colors = {
        "warmup":   _PAL["blue"],
        "steady":   _PAL["green"],
        "cooldown": _PAL["gray"],
    }
    x = np.arange(len(runs))
    bottoms = np.zeros(len(runs))
    for phase in phases:
        vals = []
        for r in runs:
            pb = r.phase_breakdown or {}
            info = pb.get(phase, {})
            vals.append(info.get("steps", 0))
        vals_arr = np.array(vals, dtype=float)
        ax1.bar(x, vals_arr, 0.5, bottom=bottoms,
                label=phase, color=phase_colors[phase], alpha=0.85, zorder=3)
        bottoms += vals_arr
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Run {i+1}" for i in range(len(runs))], fontsize=7)
    ax1.legend(fontsize=7)
    _style(ax1, "Steps per Phase", "run", "steps")

    # Latency boxplot (mean / P95 / P99 / Max)
    ax2 = axes[1]
    metrics = ["mean_itl_ms", "p95_itl_ms", "p99_itl_ms"]
    metric_labels = ["Mean ITL", "P95 ITL", "P99 ITL"]
    metric_colors = [_PAL["blue"], _PAL["orange"], _PAL["red"]]
    n_m = len(metrics)
    width = 0.2
    for i, (m, c, lbl) in enumerate(zip(metrics, metric_colors, metric_labels)):
        vals = [getattr(r, m) for r in runs]
        offset = (i - n_m/2 + 0.5) * width
        bars = ax2.bar(x + offset, vals, width, label=lbl, color=c, alpha=0.85, zorder=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Run {i+1}" for i in range(len(runs))], fontsize=7)
    ax2.legend(fontsize=7)
    _style(ax2, "Latency Percentiles (ms)", "run", "ms")

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = os.path.join(out_dir, f"06_phase_latency{tag}.png")
    _save(fig, path)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot 7 – Comparative summary (if both gguf & simulation available)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_backend_comparison(
    gguf_runs: List[RunRecord],
    sim_runs: List[RunRecord],
    out_dir: str,
) -> str:
    if not _MPL_AVAILABLE or not gguf_runs or not sim_runs:
        return ""

    def _mean(lst, attr):
        vals = [getattr(r, attr) for r in lst if getattr(r, attr) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    metrics = ["overall_tps", "avg_acceptance", "mean_itl_ms", "speculative_efficiency"]
    labels  = ["TPS", "Acceptance", "Mean ITL (ms)", "Spec Efficiency"]
    gguf_v  = [_mean(gguf_runs, m) * (100 if "accept" in m or "efficiency" in m else 1) for m in metrics]
    sim_v   = [_mean(sim_runs,  m) * (100 if "accept" in m or "efficiency" in m else 1) for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    x = np.arange(len(metrics))
    ax.bar(x - 0.2, gguf_v, 0.35, label="GGUF (real)",  color=_PAL["blue"],  alpha=0.85, zorder=3)
    ax.bar(x + 0.2, sim_v,  0.35, label="Simulation",   color=_PAL["green"], alpha=0.85, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=9)
    _style(ax, "Backend Comparison: GGUF vs Simulation (averaged across runs)", "", "value")

    path = os.path.join(out_dir, "07_backend_comparison.png")
    _save(fig, path)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  Convenience: run all plots for a set of RunRecords
# ═══════════════════════════════════════════════════════════════════════════════
def generate_all_plots(runs: List[RunRecord], out_dir: str, tag: str = "") -> List[str]:
    if not _MPL_AVAILABLE:
        print("  [WARN] matplotlib unavailable — no plots generated")
        return []

    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'─'*60}")
    print(f"  Generating plots → {out_dir}")
    print(f"{'─'*60}")

    saved = []
    p = plot_overview_dashboard(runs, out_dir, tag)
    if p: saved.append(p)

    ts = plot_time_series(runs, out_dir, tag)
    saved.extend(ts)

    p = plot_controller_analysis(runs, out_dir, tag)
    if p: saved.append(p)

    p = plot_policy_analysis(runs, out_dir, tag)
    if p: saved.append(p)

    p = plot_kv_cache(runs, out_dir, tag)
    if p: saved.append(p)

    p = plot_phase_and_latency(runs, out_dir, tag)
    if p: saved.append(p)

    print(f"{'─'*60}")
    print(f"  ✓ {len(saved)} plot(s) saved")
    return saved


# ── helpers ───────────────────────────────────────────────────────────────────
def _ewma(data: list, alpha: float) -> list:
    out, ema = [], None
    for v in data:
        ema = v if ema is None else alpha * v + (1 - alpha) * ema
        out.append(ema)
    return out