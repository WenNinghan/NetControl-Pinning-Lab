"""Shared Print-Native Editorial Style & Statistical Helper (WRM / JoH Standard).

Enforces:
- Strict Times New Roman Bold + STIX Math
- 184-198 mm two-column print canvas width at 600 DPI
- Open top/right spines, thin #8D9FA8 spines (lw=0.75), #F3F6F8 tinted panels with white gridlines
  or clean horizontal forest/raincloud panels with alternating #F2F5F7 / #F4E6EB borderless washes
- Zero framed legends (frameon=False), zero FancyBboxPatch, zero floating text bboxes
- Every rendered text element >= 9.0 pt (verified programmatically by enforce_qa_gates)
- Holm-Bonferroni step-down multiplicity adjustment and paired bootstrap confidence intervals
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
import numpy as np

MM = 1.0 / 25.4

# Master Muted Academic Print Palette
INK = "#28323C"          # Charcoal ink for text, ticks, exact/oracle references
SPINE = "#8D9FA8"        # Thin open spines (lw=0.75)
REF_LINE = "#545E67"     # Dashed / dotted reference thresholds
SEP_LINE = "#C6CDD3"     # Hairline group separators
GRID_PALE = "#DBE2E7"    # Subtle gridlines on white canvas
PANEL_BG = "#F3F6F8"     # Quiet plot canvas paired with crisp white gridlines
ROW_STRIPE = "#F2F5F7"   # Alternating row wash on white canvas
HERO_WASH = "#F4E6EB"    # Borderless dusty-rose focal row highlight
SAND_WASH = "#F2ECE1"    # Borderless warm sand secondary control highlight
TEAL_WASH = "#E6F1F2"    # Borderless soft petrol highlight strip
GAP_SHADE = "#D9E2EA"    # Quiet dominance / interval fill

# Semantic Series Colors
C_BURGUNDY = "#A73859"   # Focal method / primary curve
C_TEAL = "#147E88"       # Key positive / structural control / Pareto frontier
C_BRONZE = "#B18B48"     # Strongest comparator / ablation control
C_BLUE = "#477FA7"       # External baselines / standard methods
C_SLATE = "#7893A4"      # Unconstrained / raw baseline
C_CHARCOAL = "#28323C"   # Ground truth / analytical curve
C_PLUM = "#7A517D"       # Secondary ablation variant

SEQ_WINE_HEX = ["#FBF9FA", "#F2E4E9", "#DFBCC7", "#C18599", "#9B4F6B", "#69253F"]
DIV_SLATE_ROSE_HEX = ["#477FA7", "#F8FAFB", "#A73859"]


def get_wine_cmap() -> LinearSegmentedColormap:
    stops = [0.00, 0.08, 0.28, 0.55, 0.80, 1.00]
    return LinearSegmentedColormap.from_list("wrm_wine", list(zip(stops, SEQ_WINE_HEX)), N=256)


def get_diverge_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("wrm_diverge", DIV_SLATE_ROSE_HEX, N=256)


def apply_editorial_rc() -> None:
    """Apply print-native Times New Roman Bold + STIX math editorial rcParams."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "font.size": 9.5,
        "axes.titlesize": 10.8,
        "axes.labelsize": 10.0,
        "xtick.labelsize": 9.2,
        "ytick.labelsize": 9.2,
        "legend.fontsize": 9.2,
        "legend.frameon": False,
        "axes.linewidth": 0.75,
        "axes.edgecolor": SPINE,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.72,
        "ytick.major.width": 0.72,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.pad": 2.8,
        "ytick.major.pad": 2.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def style_panel_tinted(ax: plt.Axes, grid_axis: str = "both") -> None:
    """Quiet #F3F6F8 canvas with pure white gridlines and thin #8D9FA8 bottom/left spines."""
    ax.set_facecolor(PANEL_BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_color(SPINE)
    ax.spines["bottom"].set_color(SPINE)
    ax.spines["left"].set_linewidth(0.75)
    ax.spines["bottom"].set_linewidth(0.75)
    ax.tick_params(colors=INK, length=3.0, width=0.72)
    ax.grid(True, axis=grid_axis, color="white", linestyle="-", linewidth=0.85, zorder=0)
    ax.set_axisbelow(True)


def style_panel_horizontal(ax: plt.Axes) -> None:
    """White canvas, left spine OFF, pale vertical gridlines (forest / dot-whisker / horizontal bar)."""
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(SPINE)
    ax.spines["bottom"].set_linewidth(0.75)
    ax.tick_params(axis="x", colors=INK, length=3.0, width=0.72)
    ax.tick_params(axis="y", colors=INK, length=0)
    ax.grid(True, axis="x", color=GRID_PALE, linestyle="-", linewidth=0.60, alpha=0.85, zorder=0)
    ax.grid(False, axis="y")
    ax.set_axisbelow(True)


def style_panel_clean_heatmap(ax: plt.Axes) -> None:
    """Frameless clean matrix/heatmap panel with zero spines and zero ticks."""
    ax.set_facecolor("white")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, colors=INK)
    ax.grid(False)


def add_panel_header(
    ax: plt.Axes,
    tag: str,
    title: str,
    x_title: float = 0.11,
    y: float = 1.035,
    tag_size: float = 11.5,
    title_size: float = 10.2,
) -> None:
    """Place bold panel tag '(a)' flush left and concise title beside it."""
    ax.text(
        0.0, y, f"({tag})", transform=ax.transAxes,
        fontsize=tag_size, fontweight="bold", color=INK, ha="left", va="bottom",
    )
    if title:
        ax.text(
            x_title, y + 0.002, title, transform=ax.transAxes,
            fontsize=title_size, fontweight="bold", color=INK, ha="left", va="bottom",
        )


def enforce_qa_gates(fig: plt.Figure, min_pt: float = 9.0) -> None:
    """Programmatic check of the 5 Hard WRM Editorial Gates."""
    w_mm = fig.get_figwidth() * 25.4
    h_mm = fig.get_figheight() * 25.4
    assert 180.0 <= w_mm <= 200.5, f"Canvas width {w_mm:.1f} mm outside [180, 200.5] mm!"

    for leg in fig.legends:
        assert not leg.get_frame_on(), "Figure legend has frameon=True!"
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is not None:
            assert not leg.get_frame_on(), f"Axes legend has frameon=True on {ax}!"

    min_found = 999.0
    for ax in fig.axes:
        for p in ax.patches:
            assert not isinstance(p, FancyBboxPatch), f"Forbidden FancyBboxPatch found on {ax}!"
        for txt in ax.texts:
            if not txt.get_text().strip():
                continue
            assert txt.get_bbox_patch() is None, f"Forbidden text bbox on: {txt.get_text()!r}"
            fs = float(txt.get_fontsize())
            min_found = min(min_found, fs)
            assert fs >= min_pt - 1e-6, f"Font size {fs} pt < {min_pt} pt on {txt.get_text()!r}"
        for lbl in (ax.get_xticklabels() + ax.get_yticklabels() + [ax.xaxis.label, ax.yaxis.label, ax.title]):
            if lbl.get_text().strip():
                fs = float(lbl.get_fontsize())
                min_found = min(min_found, fs)
                assert fs >= min_pt - 1e-6, f"Axis/tick font {fs} pt < {min_pt} pt on {lbl.get_text()!r}"
    for txt in fig.texts:
        if txt.get_text().strip():
            assert txt.get_bbox_patch() is None, f"Forbidden fig text bbox on {txt.get_text()!r}"
            fs = float(txt.get_fontsize())
            min_found = min(min_found, fs)
            assert fs >= min_pt - 1e-6, f"Fig text font {fs} pt < {min_pt} pt on {txt.get_text()!r}"

    # Pixel-level bounding-box non-overlap gate across all visible text & tick labels
    renderer = fig.canvas.get_renderer()
    boxes: list[tuple[str, mpl.transforms.Bbox]] = []

    def _collect(t_obj: mpl.text.Text) -> None:
        if not t_obj.get_visible():
            return
        s = t_obj.get_text().strip()
        if not s:
            return
        if isinstance(t_obj, mpl.text.Annotation):
            bb = mpl.text.Text.get_window_extent(t_obj, renderer).expanded(0.94, 0.90)
        else:
            bb = t_obj.get_window_extent(renderer).expanded(0.94, 0.90)
        if bb.width > 1.0 and bb.height > 1.0:
            boxes.append((s, bb))

    for txt in fig.texts:
        _collect(txt)
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        for txt in ax.texts:
            _collect(txt)
        xlo, xhi = sorted(ax.get_xlim())
        ylo, yhi = sorted(ax.get_ylim())
        for xt, lbl in zip(ax.get_xticks(), ax.get_xticklabels()):
            if xlo - 1e-9 <= xt <= xhi + 1e-9:
                _collect(lbl)
        for yt, lbl in zip(ax.get_yticks(), ax.get_yticklabels()):
            if ylo - 1e-9 <= yt <= yhi + 1e-9:
                _collect(lbl)
        for lbl in [ax.xaxis.label, ax.yaxis.label, ax.title]:
            _collect(lbl)
        leg = ax.get_legend()
        if leg is not None and leg.get_visible():
            for lt in leg.get_texts():
                _collect(lt)

    fig_w_px = fig.get_figwidth() * fig.dpi
    fig_h_px = fig.get_figheight() * fig.dpi
    for s_lbl, bb in boxes:
        if bb.x0 < 2.0 or bb.y0 < 2.0 or bb.x1 > fig_w_px - 2.0 or bb.y1 > fig_h_px - 2.0:
            raise AssertionError(
                f"Text clipped at canvas boundary ({bb.x0:.1f},{bb.y0:.1f})-({bb.x1:.1f},{bb.y1:.1f}) "
                f"outside [0,{fig_w_px:.1f}]x[0,{fig_h_px:.1f}]: {s_lbl!r}"
            )

    for i in range(len(boxes)):
        s1, b1 = boxes[i]
        for j in range(i + 1, len(boxes)):
            s2, b2 = boxes[j]
            if b1.overlaps(b2):
                ix = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
                iy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
                if ix > 2.5 and iy > 2.5:
                    raise AssertionError(f"Text overlap ({ix:.1f}x{iy:.1f} px): {s1!r} <-> {s2!r}")

    print(f"[QA PASS] {w_mm:.1f}x{h_mm:.1f} mm | min_font={min_found:.1f} pt | 0 overlaps | 0 boxes | 0 framed legends")


def save_editorial_png(fig: plt.Figure, out_path: str | Path, min_pt: float = 9.0, dpi: int = 600) -> None:
    """Enforce QA gates and save crisp 600-DPI white-canvas PNG."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Trigger tick label instantiation before checking QA gates
    fig.canvas.draw()
    enforce_qa_gates(fig, min_pt=min_pt)
    fig.savefig(p, format="png", dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"Saved: {p} ({p.stat().st_size:,} bytes)")


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Step-down Holm-Bonferroni multiplicity adjustment (Holm 1979)."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adj = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        val = min(1.0, float(p) * (m - rank))
        running_max = max(running_max, val)
        adj[orig_idx] = round(running_max, 6)
    return adj


def paired_bootstrap_ci(
    diff: Sequence[float] | np.ndarray,
    n_boot: int = 10000,
    seed: int = 2026,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Return (mean_diff, ci_low, ci_high) via percentile paired bootstrap."""
    arr = np.asarray(diff, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    lo = float(np.percentile(means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return float(arr.mean()), lo, hi
