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
    # NEW: efficiency-framing metrics
    itl_cv: float = 0.0
    itl_variance_ms2: float = 0.0
    wasted_compute_fraction: float = 0.0
    compute_efficiency: float = 0.0
    # NEW: mean wall-clock cost of policy.step()+controller.step() per
    # generation step, in ms (reviewer point 4 — complexity vs payoff).
    controller_overhead_ms_mean: float = 0.0
    # BUGFIX: fraction of this run's steps that fell back to plain AR
    # generation because main.py's disable-speculation guard tripped.
    # Surfaced so speedup numbers aren't silently dominated by AR fallback.
    ar_fallback_fraction: float = 0.0

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


def _set_run_xticks(ax, x, labels, max_labels: int = 12):
    """FIX (reviewer: overlapping column labels in multi-panel plots).
    Previously every panel called set_xticklabels(labels, fontsize=7) with no
    rotation, so once a benchmark had more than ~8 runs (e.g. repeats=5 x 3
    prompts = 15 bars) the labels overlapped into unreadable smears.
    This helper: (1) rotates labels 40 degrees, right-aligned, so each label
    has diagonal room instead of horizontal room; (2) thins labels to at most
    `max_labels` evenly spaced ticks when there are more runs than that,
    showing every Nth run instead of all of them; (3) uses a slightly smaller
    font for the thinned case."""
    n = len(labels)
    if n > max_labels:
        step = math.ceil(n / max_labels)
        shown_idx = list(range(0, n, step))
        ax.set_xticks([x[i] for i in shown_idx])
        ax.set_xticklabels([labels[i] for i in shown_idx], fontsize=6.5, rotation=40, ha="right")
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=40, ha="right")

def _add_bar_labels(ax, bars, values, fmt="{:.1f}", offset=0.5, suffix="", fontsize=6.5):
    """
    Add rotated labels to bars to prevent overlapping in dense/multi-panel plots.
    Uses 45-degree rotation and right-alignment to save horizontal space.
    """
    for bar, v in zip(bars, values):
        if v is None:
            continue
        label = fmt.format(v) + suffix
        # rotation=45 and ha="right" prevent horizontal overlap
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                label, ha="right", va="bottom", fontsize=fontsize, rotation=45)

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
#  Plot 00 – Memory-Efficiency Dashboard (PRIMARY — new narrative framing)
#  Replaces throughput-speedup as the lead figure. Shows:
#  (1) Wasted Compute Fraction  (2) ITL Coefficient of Variation
#  (3) Compute Efficiency %      (4) Tail-Latency Ratio (P95/Mean)
#  (5) Speculative Efficiency    (6) Acceptance Rate
#  Thesis: the framework minimises wasted computation and latency variance
#  under CPU memory-bandwidth constraints, even when absolute speedup < 1.
# ═══════════════════════════════════════════════════════════════════════════════
def plot_efficiency_dashboard(runs: List[RunRecord], out_dir: str, tag: str = "") -> str:
    """Plot 00: Memory-efficiency-framed primary dashboard."""
    if not _MPL_AVAILABLE or not runs:
        return ""
    
    labels = [f"Run {i+1}" for i in range(len(runs))]
    x = np.arange(len(runs))
    bar_w = 0.5
    
    fig = plt.figure(figsize=(15, 10), facecolor="white")
    fig.suptitle(
        f"AdaptiveSD — Memory-Efficiency Dashboard{' [' + tag + ']' if tag else ''}\n"
        "Primary metric: minimise wasted compute and latency variance under CPU memory-bandwidth constraints",
        fontsize=11, fontweight="bold", y=0.985
    )
    
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.65, wspace=0.42,
                           top=0.88, bottom=0.09, left=0.06, right=0.98)
    
    # ── Panel 1: Wasted Compute Fraction ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    wcf_vals = [r.wasted_compute_fraction * 100 for r in runs]
    bars1 = ax1.bar(x, wcf_vals, bar_w, color=_PAL["orange"], alpha=0.85, zorder=3)
    ax1.set_ylim(0, 60)
    ax1.axhline(25, color=_PAL["gray"], ls="--", lw=0.9, label="25% reference")
    ax1.legend(fontsize=7)
    _add_bar_labels(ax1, bars1, wcf_vals, fmt="{:.1f}", offset=0.5, suffix="%")
    _set_run_xticks(ax1, x, labels)
    _style(ax1, "Wasted Compute Fraction", "", "% of drafted tokens wasted")
    
    # ── Panel 2: Compute Efficiency % ────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    eff_vals = [r.compute_efficiency * 100 for r in runs]
    colors2 = [_PAL["green"] if e >= 75 else _PAL["orange"] for e in eff_vals]
    bars2 = ax2.bar(x, eff_vals, bar_w, color=colors2, alpha=0.85, zorder=3)
    ax2.set_ylim(0, 110)
    ax2.axhline(75, color=_PAL["gray"], ls="--", lw=0.9, label="75% target")
    ax2.legend(fontsize=7)
    _add_bar_labels(ax2, bars2, eff_vals, fmt="{:.1f}", offset=0.5, suffix="%")
    _set_run_xticks(ax2, x, labels)
    _style(ax2, "Compute Efficiency  (1 − WCF)", "", "%")
    
    # ── Panel 3: ITL Coefficient of Variation ─────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    cv_vals = [r.itl_cv for r in runs]
    colors3 = [_PAL["red"] if cv > 1.5 else _PAL["teal"] for cv in cv_vals]
    bars3 = ax3.bar(x, cv_vals, bar_w, color=colors3, alpha=0.85, zorder=3)
    ax3.axhline(1.5, color=_PAL["red"], ls="--", lw=0.9, label="CV=1.5 threshold")
    ax3.legend(fontsize=7)
    _add_bar_labels(ax3, bars3, cv_vals, fmt="{:.2f}", offset=0.01)
    _set_run_xticks(ax3, x, labels)
    _style(ax3, "ITL Coeff. of Variation  σ/μ", "", "CV  (lower = more stable)")
    
    # ── Panel 4: Tail-Latency Ratio P95/Mean ─────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    tlr_vals = [r.p95_itl_ms / r.mean_itl_ms if r.mean_itl_ms > 0 else 1.0 for r in runs]
    colors4 = [_PAL["red"] if t > 2.5 else _PAL["green"] for t in tlr_vals]
    bars4 = ax4.bar(x, tlr_vals, bar_w, color=colors4, alpha=0.85, zorder=3)
    ax4.axhline(2.5, color=_PAL["gray"], ls="--", lw=0.9, label="P95/Mean = 2.5×")
    ax4.legend(fontsize=7)
    _add_bar_labels(ax4, bars4, tlr_vals, fmt="{:.2f}", offset=0.02, suffix="×")
    _set_run_xticks(ax4, x, labels)
    _style(ax4, "Tail-Latency Ratio  P95 / Mean", "", "ratio  (lower = more stable)")
    
    # ── Panel 5: Speculative Efficiency % ────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    se_vals = [(r.speculative_efficiency or 0.0) * 100 for r in runs]
    bars5 = ax5.bar(x, se_vals, bar_w, color=_PAL["purple"], alpha=0.85, zorder=3)
    ax5.set_ylim(0, 110)
    ax5.axhline(68, color=_PAL["gray"], ls="--", lw=0.9, label="Sim baseline 68%")
    ax5.legend(fontsize=7)
    _add_bar_labels(ax5, bars5, se_vals, fmt="{:.1f}", offset=0.5, suffix="%")
    _set_run_xticks(ax5, x, labels)
    _style(ax5, "Speculative Efficiency  N_committed / N_total", "", "%")
    
    # ── Panel 6: Draft Acceptance Rate ────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    acc_vals = [r.avg_acceptance * 100 for r in runs]
    bars6 = ax6.bar(x, acc_vals, bar_w, color=_PAL["blue"], alpha=0.85, zorder=3)
    ax6.set_ylim(0, 110)
    ax6.axhline(50, color=_PAL["gray"], ls="--", lw=0.8, label="50% line")
    ax6.legend(fontsize=7)
    _add_bar_labels(ax6, bars6, acc_vals, fmt="{:.1f}", offset=0.5, suffix="%")
    _set_run_xticks(ax6, x, labels)
    _style(ax6, "Draft Acceptance Rate", "", "%")
    
    path = os.path.join(out_dir, f"00_efficiency_dashboard{tag}.png")
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    logger.info("Plot saved -> %s", path)
    print(f"  [PLOT] Saved: {path}")
    return path

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
    
    fig = plt.figure(figsize=(15, 10), facecolor="white")
    fig.suptitle(
        f"Adaptive Speculative Decoding — Overview Dashboard{' [' + tag + ']' if tag else ''}",
        fontsize=13, fontweight="bold", y=0.97
    )
    
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.60, wspace=0.40,
                           top=0.90, bottom=0.09, left=0.06, right=0.98)
    x = np.arange(len(runs))
    bar_w = 0.5
    
    # ── TPS ──
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(x, tps_vals, bar_w, color=_PAL["blue"], alpha=0.85, zorder=3)
    if runs[0].baseline_tps:
        ax1.axhline(runs[0].baseline_tps, color=_PAL["red"], ls="--", lw=1.2,
                    label=f"Baseline {runs[0].baseline_tps:.1f}")
        ax1.legend(fontsize=7)
    _add_bar_labels(ax1, bars, tps_vals, fmt="{:.1f}", offset=0.05)
    _set_run_xticks(ax1, x, labels)
    _style(ax1, "Throughput (TPS)", "", "tokens/sec")
    
    # ── Speedup ──
    ax2 = fig.add_subplot(gs[0, 1])
    colors = [_PAL["green"] if s >= 1.0 else _PAL["red"] for s in spd_vals]
    bars2 = ax2.bar(x, spd_vals, bar_w, color=colors, alpha=0.85, zorder=3)
    ax2.axhline(1.0, color=_PAL["gray"], ls="--", lw=1.0, label="1.0× baseline")
    ax2.legend(fontsize=7)
    _add_bar_labels(ax2, bars2, spd_vals, fmt="{:.3f}", offset=0.005, suffix="×")
    _set_run_xticks(ax2, x, labels)
    _style(ax2, "Speculative Speedup", "", "speedup ratio")
    
    # ── Acceptance Rate ──
    ax3 = fig.add_subplot(gs[0, 2])
    bars3 = ax3.bar(x, acc_vals, bar_w, color=_PAL["purple"], alpha=0.85, zorder=3)
    ax3.axhline(50, color=_PAL["gray"], ls="--", lw=0.8, label="50% line")
    ax3.legend(fontsize=7)
    _add_bar_labels(ax3, bars3, acc_vals, fmt="{:.1f}", offset=0.3, suffix="%")
    ax3.set_ylim(0, 110)
    _set_run_xticks(ax3, x, labels)
    _style(ax3, "Draft Acceptance Rate", "", "%")
    
    # ── Latency ──
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.bar(x - bar_w/4, itl_vals, bar_w/2, label="Mean ITL", color=_PAL["teal"], alpha=0.85, zorder=3)
    ax4.bar(x + bar_w/4, p95_vals, bar_w/2, label="P95 ITL", color=_PAL["orange"], alpha=0.85, zorder=3)
    ax4.legend(fontsize=7)
    _set_run_xticks(ax4, x, labels)
    _style(ax4, "Inter-Token Latency", "", "ms")
    
    # ── Speculative Efficiency ──
    ax5 = fig.add_subplot(gs[1, 1])
    bars5 = ax5.bar(x, eff_vals, bar_w, color=_PAL["green"], alpha=0.85, zorder=3)
    ax5.set_ylim(0, 110)
    _add_bar_labels(ax5, bars5, eff_vals, fmt="{:.1f}", offset=0.3, suffix="%")
    _set_run_xticks(ax5, x, labels)
    _style(ax5, "Speculative Efficiency", "", "%")
    
    # ── Draft Token Budget ──
    ax6 = fig.add_subplot(gs[1, 2])
    total_d = [r.total_drafted for r in runs]
    total_a = [r.total_accepted for r in runs]
    total_w = [r.total_wasted  for r in runs]
    ax6.bar(x, total_a, bar_w, label="Accepted",  color=_PAL["green"],  alpha=0.85, zorder=3)
    ax6.bar(x, total_w, bar_w, bottom=total_a,   label="Wasted",    color=_PAL["red"],    alpha=0.85, zorder=3)
    ax6.legend(fontsize=7)
    _set_run_xticks(ax6, x, labels)
    _style(ax6, "Draft Token Budget", "", "tokens")
    
    path = os.path.join(out_dir, f"01_overview_dashboard{tag}.png")
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    logger.info("Plot saved -> %s", path)
    print(f"  [PLOT] Saved: {path}")
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
    
    agg_decisions: Dict[str, int] = {}
    agg_depth_dist: Dict[int, int] = {}
    for r in runs:
        for k, v in r.decision_stats.items():
            agg_decisions[k] = agg_decisions.get(k, 0) + v
        dd = r.stability_metrics.get("depth_distribution", {})
        for d, cnt in dd.items():
            agg_depth_dist[int(d)] = agg_depth_dist.get(int(d), 0) + cnt
            
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
        # FIX: Rotated histogram labels to prevent overlap
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                     str(v), ha="right", va="bottom", fontsize=7, rotation=45)
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
    _set_run_xticks(ax3, x, [f"Run {i+1}" for i in range(len(runs))])
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
    _set_run_xticks(ax1, x, labels)
    # FIX: Rotated bar labels
    _add_bar_labels(ax1, bars, hr, fmt="{:.1f}", offset=0.5, suffix="%")
    _style(ax1, "KV Cache Hit Rate", "", "%")
    
    # Rollbacks (partial vs full)
    ax2 = axes[1]
    full_rb    = [r.kv_stats.get("full_rollbacks", 0)    for r in runs]
    partial_rb = [r.kv_stats.get("partial_rollbacks", 0) for r in runs]
    ax2.bar(x, full_rb,    0.5, label="Full",    color=_PAL["red"],    alpha=0.85, zorder=3)
    ax2.bar(x, partial_rb, 0.5, bottom=full_rb,  label="Partial", color=_PAL["orange"], alpha=0.85, zorder=3)
    _set_run_xticks(ax2, x, labels)
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
    _set_run_xticks(ax3, x, labels)
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
    _set_run_xticks(ax1, x, [f"Run {i+1}" for i in range(len(runs))])
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
    _set_run_xticks(ax2, x, [f"Run {i+1}" for i in range(len(runs))])
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
#  Plot: Cross-method comparison (AdaptiveSD policies vs baselines/SOTA-approx)
#  Requested by reviewer: no comparison against prior methods existed before.
#  One figure, 4 panels, each method = one bar group; 95% CI error bars where
#  n>=2 runs are available. Labels use _set_run_xticks-style rotation so method
#  names never overlap even with all 8 methods shown at once.
# ═══════════════════════════════════════════════════════════════════════════════
def plot_method_comparison(records_by_method: Dict[str, List["RunRecord"]], out_dir: str, tag: str = "",
                            paper_style: bool = False, dpi: int = 150) -> str:
    """paper_style=True: drop the in-image title/subtitle (papers use \\caption
    instead) and render at 300dpi, suitable for direct figure inclusion."""
    if not _MPL_AVAILABLE or not records_by_method:
        return ""
    os.makedirs(out_dir, exist_ok=True)

    methods = [m for m, recs in records_by_method.items() if recs]
    if not methods:
        return ""

    def _mean_ci(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return 0.0, 0.0
        if len(vals) < 2:
            return vals[0], 0.0
        m = sum(vals) / len(vals)
        sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
        sem = sd / (len(vals) ** 0.5)
        # 95% CI via normal approx (consistent with small-n exploratory framing
        # used elsewhere; main.py prints a low-n warning when repeats<5)
        return m, 1.96 * sem

    speedup_mc = [_mean_ci([r.speedup_ratio for r in records_by_method[m]]) for m in methods]
    wcf_mc     = [_mean_ci([r.wasted_compute_fraction * 100 for r in records_by_method[m]]) for m in methods]
    cv_mc      = [_mean_ci([r.itl_cv for r in records_by_method[m]]) for m in methods]
    acc_mc     = [_mean_ci([r.avg_acceptance * 100 for r in records_by_method[m]]) for m in methods]
    overhead_mc = [_mean_ci([r.controller_overhead_ms_mean for r in records_by_method[m]]) for m in methods]
    # BUGFIX: surface AR-fallback contamination directly in the figure —
    # this is what actually explains why every method's speedup looks
    # similar and sub-unity (see main.py's speculation-disable guard).
    ar_fb_mc = [_mean_ci([getattr(r, "ar_fallback_fraction", 0.0) * 100
                          for r in records_by_method[m]]) for m in methods]

    fixed_speedups = [r.speedup_ratio for r in records_by_method.get("fixed_depth", []) if r.speedup_ratio]
    fixed_mean = (sum(fixed_speedups) / len(fixed_speedups)) if fixed_speedups else None
    if fixed_mean:
        vs_fixed_mc = [_mean_ci([(v / fixed_mean) for v in
                                  [r.speedup_ratio for r in records_by_method[m] if r.speedup_ratio]])
                       for m in methods]
    else:
        vs_fixed_mc = [(1.0, 0.0) for _ in methods]

    fig = plt.figure(figsize=(15, 14), facecolor="white")
    if not paper_style:
        fig.suptitle(
            f"Method Comparison — AdaptiveSD policies vs. baselines / SOTA-style approximations{' [' + tag + ']' if tag else ''}\n"
            "specdec_plus_approx & dynamic_lookahead are rule-based re-implementations, not the original trained methods (see policy_engine.py)",
            fontsize=10.5, fontweight="bold", y=0.985
        )
        top_margin = 0.87
    else:
        top_margin = 0.96
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.75, wspace=0.35,
                            top=top_margin, bottom=0.08, left=0.06, right=0.98)

    x = np.arange(len(methods))
    colors = [_PAL["blue"] if m in ("ensemble", "bandit", "ema", "heuristic")
              else _PAL["gray"] for m in methods]

    def _panel(ax, values_ci, title, ylabel, ref_line=None):
        means = [v[0] for v in values_ci]
        errs = [v[1] for v in values_ci]
        ax.bar(x, means, 0.6, yerr=errs, capsize=4, color=colors, alpha=0.85, zorder=3,
               error_kw={"linewidth": 1.2, "ecolor": _PAL["gray"]})
        if ref_line is not None:
            ax.axhline(ref_line, color=_PAL["red"], ls="--", lw=0.9)
        _set_run_xticks(ax, x, methods, max_labels=8)
        _style(ax, title, "", ylabel)

    ax1 = fig.add_subplot(gs[0, 0])
    _panel(ax1, speedup_mc, "Speedup vs. no-speculation baseline", "Speedup (x)", ref_line=1.0)

    ax2 = fig.add_subplot(gs[0, 1])
    _panel(ax2, wcf_mc, "Wasted Compute Fraction", "WCF (%)")

    ax3 = fig.add_subplot(gs[0, 2])
    _panel(ax3, cv_mc, "Inter-Token-Latency Coefficient of Variation", "ITL CV")

    ax4 = fig.add_subplot(gs[1, 0])
    _panel(ax4, acc_mc, "Draft Acceptance Rate", "Acceptance (%)")

    # NEW panel (reviewer point 4 — is the complexity justified?): wall-clock
    # cost of the controller/policy machinery itself, per generation step.
    ax5 = fig.add_subplot(gs[1, 1])
    _panel(ax5, overhead_mc, "Controller/Policy Overhead per Step", "ms/step")

    # NEW panel (reviewer points 1 & 4): does AdaptiveSD reduce the slowdown
    # relative to the fixed-depth baseline, or just add complexity on top of
    # the same (or worse) sub-unity speedup?
    ax6 = fig.add_subplot(gs[1, 2])
    _panel(ax6, vs_fixed_mc, "Speedup relative to fixed_depth baseline", "ratio (x)", ref_line=1.0)

    # NEW panel (root-cause diagnostic): fraction of each run's steps that
    # fell back to plain AR generation because main.py's over-eager,
    # non-recoverable disable-speculation guard tripped. High bars here
    # explain why the other panels look similar across methods.
    ax7 = fig.add_subplot(gs[2, 0])
    _panel(ax7, ar_fb_mc, "AR-Fallback Fraction (contamination)", "% of steps")

    suffix = "_paper" if paper_style else ""
    path = os.path.join(out_dir, f"method_comparison{tag}{suffix}.png")
    fig.savefig(path, dpi=(300 if paper_style else dpi), facecolor="white")
    plt.close(fig)
    logger.info("Plot saved -> %s", path)
    print(f"  [PLOT] Saved: {path}")
    return path


def export_comparison_table(summary_rows: List[dict], out_dir: str, tag: str = "") -> Dict[str, str]:
    """Write the method-comparison table in both LaTeX (booktabs) and Markdown,
    ready to paste directly into the paper (Section 4.7 / new SOTA-comparison
    section). `summary_rows` is the same list produced by
    AdaptiveInferenceEngine.run_baseline_comparison() in main.py.
    """
    os.makedirs(out_dir, exist_ok=True)

    def _fmt(v, suffix=""):
        if v is None:
            return "N/A"
        return f"{v:.3f}{suffix}"

    def _fmt_ci(ci):
        if not ci:
            return "--"
        return f"[{ci[0]:.3f}, {ci[1]:.3f}]"

    # ── Markdown ────────────────────────────────────────────────────────
    md_lines = [
        "| Method | n | Mean TPS | Speedup | Speedup 95% CI | WCF | ITL CV | Ctrl ms/step | "
        "AR-fallback | Speedup vs fixed_depth | p (vs fixed_depth) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in summary_rows:
        overhead = _fmt(r.get("mean_controller_overhead_ms"))
        vs_fixed = _fmt(r.get("speedup_ratio_vs_fixed_depth"), "x")
        p_val = r.get("p_value_vs_fixed_depth")
        p_str = "ref" if r["method"] == "fixed_depth" else (f"{p_val:.4f}" if p_val is not None else "N/A")
        ar_fb = r.get("mean_ar_fallback_fraction")
        ar_fb_str = f"{ar_fb:.1%}" if ar_fb is not None else "N/A"
        md_lines.append(
            f"| {r['method']} | {r['n_runs']} | {r['mean_tps']:.2f} | "
            f"{_fmt(r['mean_speedup'], 'x')} | {_fmt_ci(r['speedup_ci95'])} | "
            f"{_fmt(r['mean_wasted_compute_fraction'])} | {_fmt(r['mean_itl_cv'])} | "
            f"{overhead} | {ar_fb_str} | {vs_fixed} | {p_str} |"
        )
    md_path = os.path.join(out_dir, f"comparison_table{tag}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    # ── LaTeX (booktabs) ────────────────────────────────────────────────
    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Comparison of AdaptiveSD's policies against baseline and "
        r"SOTA-style approximation methods, run on identical hardware, backend, "
        r"prompt set, and repeat count. specdec\_plus\_approx and "
        r"dynamic\_lookahead are rule-based re-implementations of each method's "
        r"core control idea (no trained/released components were available). "
        r"``Ctrl ms/step'' is the wall-clock cost of the policy/controller "
        r"machinery itself, measured per generation step, isolated so baselines "
        r"only pay for what they use (see policy\_engine.py). ``Speedup vs "
        r"fixed\_depth'' and its $p$-value (Welch's two-sided $t$-test) test "
        r"whether each method's speedup differs significantly from the static "
        r"depth-2 baseline; see Section~4.7/4.8 for discussion of the "
        r"consistently sub-unity speedup on the CPU/GGUF backend.}",
        r"\label{tab:method-comparison}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Method & $n$ & Mean TPS & Speedup & WCF & ITL CV & Ctrl ms/step & AR-fallback & $p$ (vs fixed) \\",
        r"\midrule",
    ]
    for r in summary_rows:
        method_tex = r["method"].replace("_", r"\_")
        speedup = f"{r['mean_speedup']:.3f}x" if r["mean_speedup"] is not None else "N/A"
        wcf = f"{r['mean_wasted_compute_fraction']:.3f}" if r["mean_wasted_compute_fraction"] is not None else "N/A"
        cv = f"{r['mean_itl_cv']:.3f}" if r["mean_itl_cv"] is not None else "N/A"
        overhead = f"{r['mean_controller_overhead_ms']:.4f}" if r.get("mean_controller_overhead_ms") is not None else "N/A"
        ar_fb = r.get("mean_ar_fallback_fraction")
        ar_fb_str = f"{ar_fb*100:.1f}\\%" if ar_fb is not None else "N/A"
        p_val = r.get("p_value_vs_fixed_depth")
        p_str = "ref" if r["method"] == "fixed_depth" else (f"{p_val:.4f}" if p_val is not None else "N/A")
        tex_lines.append(f"{method_tex} & {r['n_runs']} & {r['mean_tps']:.2f} & {speedup} & {wcf} & {cv} & {overhead} & {ar_fb_str} & {p_str} \\\\")
    tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_path = os.path.join(out_dir, f"comparison_table{tag}.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines) + "\n")

    print(f"  [TABLE] Markdown -> {md_path}")
    print(f"  [TABLE] LaTeX    -> {tex_path}")
    return {"markdown": md_path, "latex": tex_path}


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
    p = plot_efficiency_dashboard(runs, out_dir, tag)   # NEW — lead figure
    if p: saved.append(p)

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