"""Render print-native WRM-style editorial figures (600 DPI, 194 mm width) for NetControl-Pinning-Lab.

Enforces all 5 programmatic WRM Editorial QA Gates via netcontrol.editorial_style.save_editorial_png:
  1. Canvas width in [180.0, 200.5] mm (194.0 mm)
  2. Zero framed legends (frameon=False)
  3. Zero FancyBboxPatch
  4. Minimum rendered font size >= 9.0 pt across every text, tick, label, and legend
  5. Zero pixel-level text bounding-box overlaps
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from netcontrol import (
    analyze_structural_controllability,
    build_continuous_system_matrix,
    classify_control_edges,
     editorial_style as es,
    minimum_energy_trajectory,
    reaction_cascade_network,
    select_actuators_baseline,
    select_actuators_submodular,
    solve_lqr_controller,
    verify_pbh_controllability,
)
from netcontrol.gramian_lqr import build_input_matrix
from netcontrol.neural_pinning import rollout_pinning_torch
from scripts.compute_holdout_inference import compute_holdout_inference
from scripts.neural_pinning_comparison import (
    CAP,
    Q as Q_PIN,
    R as R_PIN,
    SEED as PIN_SEED,
    STEPS as PIN_STEPS,
    T as PIN_T,
    _model_set,
    finite_horizon_gains,
    set_or_train_weights,
)


def build_network_topology_figure(exp2_summary: dict | None, out_path: Path) -> None:
    """Figure 1: (a) 32-node cascade topology & edge criticality, (b) 4-node disjoint-cycle wiring & PBH rank."""
    es.apply_editorial_rc()
    fig = plt.figure(figsize=(194.0 * es.MM, 108.0 * es.MM))

    ax_a = fig.add_axes([0.025, 0.145, 0.535, 0.755])
    ax_b_top = fig.add_axes([0.625, 0.620, 0.345, 0.280])
    ax_b_mat = fig.add_axes([0.720, 0.165, 0.140, 0.355])
    ax_b_bar = fig.add_axes([0.875, 0.165, 0.095, 0.355])

    # --- (a) 32-node, 65-edge directed cascade network ---
    es.style_panel_clean_heatmap(ax_a)
    G = reaction_cascade_network(n=32, branch_factor=2.2, seed=42)
    sc_res = analyze_structural_controllability(G)
    edge_classes = classify_control_edges(G)
    _, pbh_rank = verify_pbh_controllability(G, sc_res.driver_nodes, seed=42)

    es.add_panel_header(
        ax_a,
        "a",
        f"32-node cascade (11 actuators; PBH {pbh_rank}/32)",
        x_title=0.085,
        y=1.025,
    )

    # Four staggered horizontal rows of 8 nodes each -> generous spacing, zero label overlap
    pos: dict[int, tuple[float, float]] = {}
    rows_nodes = [
        (3.05, 0.00, list(range(0, 8))),    # Precursors (8)
        (2.05, 0.36, list(range(8, 16))),   # Upper intermediates (8)
        (1.05, 0.00, list(range(16, 24))),  # Lower intermediates (8)
        (0.05, 0.36, list(range(24, 32))),  # Byproducts (8)
    ]
    for y_row, x_shift, nlist in rows_nodes:
        for idx, u in enumerate(nlist):
            pos[u] = (idx * 1.05 + x_shift, y_row)

    def _draw_edges(edgelist: list[tuple[int, int]], color: str, lw: float, ls: str, alpha: float, rad: float, z: int) -> None:
        for u, v in edgelist:
            p_u = pos[u]
            p_v = pos[v]
            arr = FancyArrowPatch(
                p_u,
                p_v,
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>,head_length=3.6,head_width=2.3",
                color=color,
                linewidth=lw,
                linestyle=ls,
                alpha=alpha,
                shrinkA=9.2,
                shrinkB=9.5,
                zorder=z,
            )
            ax_a.add_patch(arr)

    _draw_edges(edge_classes["redundant"], es.C_SLATE, 0.80, "--", 0.50, 0.09, 1)
    _draw_edges(edge_classes["ordinary"], es.C_BLUE, 0.85, "-", 0.38, -0.07, 2)
    _draw_edges(edge_classes["critical"], es.C_BURGUNDY, 2.05, "-", 0.96, 0.05, 3)

    driver_set = set(sc_res.driver_nodes)
    other_nodes = [u for u in G.nodes() if u not in driver_set]

    ax_a.scatter(
        [pos[u][0] for u in other_nodes],
        [pos[u][1] for u in other_nodes],
        s=225,
        facecolors=es.PANEL_BG,
        edgecolors=es.C_TEAL,
        linewidths=1.25,
        zorder=4,
    )
    ax_a.scatter(
        [pos[u][0] for u in sc_res.driver_nodes],
        [pos[u][1] for u in sc_res.driver_nodes],
        s=255,
        facecolors=es.HERO_WASH,
        edgecolors=es.C_BURGUNDY,
        linewidths=1.85,
        zorder=5,
    )
    for u in G.nodes():
        x_u, y_u = pos[u]
        ax_a.text(
            x_u,
            y_u,
            str(u),
            ha="center",
            va="center",
            fontsize=9.0,
            color=(es.C_BURGUNDY if u in driver_set else es.INK),
            zorder=6,
        )

    ax_a.set_xlim(-0.55, 8.25)
    ax_a.set_ylim(-0.42, 3.52)
    ax_a.set_xticks([])
    ax_a.set_yticks([])

    leg_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=es.HERO_WASH,
               markeredgecolor=es.C_BURGUNDY, markeredgewidth=1.8, markersize=7.5,
               label=f"Actuator ({sc_res.n_drivers})"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=es.PANEL_BG,
               markeredgecolor=es.C_TEAL, markeredgewidth=1.25, markersize=7.0,
               label=f"State ({len(other_nodes)})"),
        Line2D([0], [0], color=es.C_BURGUNDY, lw=2.0, ls="-",
               label=f"Critical ({len(edge_classes['critical'])})"),
        Line2D([0], [0], color=es.C_BLUE, lw=1.2, ls="-",
               label=f"Ordinary ({len(edge_classes['ordinary'])})"),
        Line2D([0], [0], color=es.C_SLATE, lw=1.1, ls="--",
               label=f"Redundant ({len(edge_classes['redundant'])})"),
    ]
    ax_a.legend(
        handles=leg_handles,
        loc="upper center",
        bbox_to_anchor=(0.49, -0.02),
        ncol=3,
        fontsize=9.0,
        columnspacing=0.75,
        handlelength=1.25,
        handletextpad=0.35,
    )

    # --- (b) Two-disconnected-cycle diagnostic (4 nodes: 0 <-> 1 and 2 <-> 3) ---
    es.style_panel_clean_heatmap(ax_b_top)
    es.add_panel_header(
        ax_b_top,
        "b",
        "Disjoint cycles: shared vs. dedicated",
        x_title=0.125,
        y=1.075,
    )

    c_xy = {0: (-0.62, 0.12), 1: (-0.20, 0.12), 2: (0.20, 0.12), 3: (0.62, 0.12)}
    for u_c, v_c in [(0, 1), (1, 0), (2, 3), (3, 2)]:
        arr_c = FancyArrowPatch(
            c_xy[u_c],
            c_xy[v_c],
            connectionstyle="arc3,rad=0.32",
            arrowstyle="-|>,head_length=3.8,head_width=2.4",
            color=es.INK,
            linewidth=1.35,
            shrinkA=10.5,
            shrinkB=10.5,
            zorder=3,
        )
        ax_b_top.add_patch(arr_c)

    for nid, (cx, cy) in c_xy.items():
        is_act = nid in (0, 2)
        ax_b_top.scatter(
            [cx], [cy],
            s=250,
            facecolors=(es.HERO_WASH if is_act else es.PANEL_BG),
            edgecolors=(es.C_BURGUNDY if is_act else es.C_TEAL),
            linewidths=(1.75 if is_act else 1.2),
            zorder=4,
        )
        ax_b_top.text(cx, cy, str(nid), ha="center", va="center", fontsize=9.2, color=es.INK, zorder=5)

    ax_b_top.text(-0.41, 0.62, r"Cycle $C_1$", ha="center", va="center", fontsize=9.2, color=es.INK)
    ax_b_top.text(0.41, 0.62, r"Cycle $C_2$", ha="center", va="center", fontsize=9.2, color=es.INK)
    ax_b_top.plot([0.0, 0.0], [-0.28, 0.68], color=es.SEP_LINE, ls=":", lw=1.0)
    ax_b_top.text(0.0, -0.34, "No coupling between cycles", ha="center", va="center", fontsize=9.0, color=es.REF_LINE)
    ax_b_top.set_xlim(-0.88, 0.88)
    ax_b_top.set_ylim(-0.48, 0.80)
    ax_b_top.set_xticks([])
    ax_b_top.set_yticks([])

    # Bottom-right coordinated wiring matrix + PBH minimum rank horizontal comparison
    cyc_ctrl = (
        exp2_summary["two_disjoint_cycles_control"]
        if exp2_summary is not None
        else {"pbh_rank_one_site": 3, "pbh_rank_dedicated": 4, "pbh_rank_shared": 4}
    )
    es.style_panel_clean_heatmap(ax_b_mat)
    # Matrix codes: 0 = unactuated, 1 = signal u1, 2 = independent signal u2
    w_mat = np.array([
        [1, 0, 0, 0],  # 1 site (node 0), m=1 -> rank 3/4
        [1, 0, 2, 0],  # 2 dedicated (0, 2), m=2 -> rank 4/4
        [1, 0, 1, 0],  # 1 shared broadcast (0 & 2), m=1 -> rank 4/4
    ])
    cell_colors = {0: es.PANEL_BG, 1: es.HERO_WASH, 2: es.TEAL_WASH}
    cell_text = {0: "0", 1: r"$u_1$", 2: r"$u_2$"}
    for r_i in range(3):
        for c_i in range(4):
            code = int(w_mat[r_i, c_i])
            rect = plt.Rectangle((c_i - 0.46, 2 - r_i - 0.40), 0.92, 0.80,
                                 facecolor=cell_colors[code], edgecolor="white", linewidth=1.2)
            ax_b_mat.add_patch(rect)
            ax_b_mat.text(
                c_i, 2 - r_i, cell_text[code],
                ha="center", va="center", fontsize=9.2,
                color=(es.C_BURGUNDY if code == 1 else (es.C_TEAL if code == 2 else es.C_SLATE)),
            )

    ax_b_mat.set_xlim(-0.5, 3.5)
    ax_b_mat.set_ylim(-0.55, 2.55)
    ax_b_mat.set_xticks([0, 1, 2, 3])
    ax_b_mat.set_xticklabels(["0", "1", "2", "3"], fontsize=9.0)
    ax_b_mat.set_xlabel("State node", fontsize=9.5)
    ax_b_mat.set_yticks([2, 1, 0])
    ax_b_mat.set_yticklabels([
        "Node 0 (m=1)",
        "Ded. 0,2 (m=2)",
        "Shared 0&2 (m=1)",
    ], fontsize=9.0)
    ax_b_mat.text(1.5, 2.72, "Input matrix B", ha="center", va="bottom", fontsize=9.2, color=es.INK)

    es.style_panel_horizontal(ax_b_bar)
    ranks = [
        cyc_ctrl["pbh_rank_shared"],
        cyc_ctrl["pbh_rank_dedicated"],
        cyc_ctrl["pbh_rank_one_site"],
    ]
    bar_cols = [es.C_BURGUNDY, es.C_TEAL, es.C_SLATE]
    ax_b_bar.axhspan(-0.40, 0.40, color=es.HERO_WASH, zorder=0)
    for y_pos, rk, col in zip([0, 1, 2], ranks, bar_cols):
        ax_b_bar.barh([y_pos], [rk], height=0.52, color=col, zorder=3)
        ax_b_bar.text(rk * 0.5, y_pos, f"{rk}/4", ha="center", va="center", fontsize=9.0, color="white", zorder=4)

    ax_b_bar.axvline(4.0, color=es.REF_LINE, ls="--", lw=0.85, zorder=2)
    ax_b_bar.set_yticks([])
    ax_b_bar.set_xlim(0, 4.85)
    ax_b_bar.set_ylim(-0.55, 2.55)
    ax_b_bar.set_xticks([0, 2, 4])
    ax_b_bar.set_xlabel("PBH rank", fontsize=9.5)
    ax_b_bar.text(2.8, 2.72, "Rank", ha="center", va="bottom", fontsize=9.2, color=es.INK)

    es.save_editorial_png(fig, out_path)


def build_driver_scaling_figure(exp1_summary: dict, inference: dict, out_path: Path) -> None:
    """Figure 2: (a) 120-node driver scaling law vs exp(-<k>/2), (b) Dedicated vs shared source-SCC overhead."""
    es.apply_editorial_rc()
    fig = plt.figure(figsize=(194.0 * es.MM, 88.0 * es.MM))

    ax_a = fig.add_axes([0.082, 0.175, 0.395, 0.675])
    ax_b = fig.add_axes([0.580, 0.175, 0.390, 0.675])

    # --- (a) Minimum independent input fraction n_D = N_D / N across mean out-degrees ---
    es.style_panel_tinted(ax_a)
    es.add_panel_header(ax_a, "a", "Shared input fraction (N = 120)", x_title=0.105, y=1.035)

    k_vals = np.array(exp1_summary["k_values"], dtype=float)
    n_nodes = float(exp1_summary["n_nodes"])
    fam_specs = [
        ("Scale-Free (γ=2.2, Heterogeneous)", r"Scale-Free ($\gamma = 2.2$)", es.C_BURGUNDY, "o", "-"),
        ("Scale-Free (γ=2.8, Moderate)", r"Scale-Free ($\gamma = 2.8$)", es.C_BLUE, "s", "-"),
        ("Erdős–Rényi (Random)", "Erdos-Renyi (ER)", es.C_TEAL, "^", "-"),
    ]
    for fam_key, disp_lbl, col, mk, ls in fam_specs:
        means, stds = [], []
        for kv in k_vals:
            vals = [
                t["minimum_input_channels"] / n_nodes
                for t in exp1_summary["trials"]
                if t["family"] == fam_key and float(t["target_mean_out_degree"]) == float(kv)
            ]
            means.append(float(np.mean(vals)) * 100.0)
            stds.append(float(np.std(vals)) * 100.0)
        mu_arr = np.array(means)
        sd_arr = np.array(stds)
        ax_a.plot(k_vals, mu_arr, color=col, marker=mk, ls=ls, lw=1.85, markersize=4.8, label=disp_lbl, zorder=4)
        ax_a.fill_between(
            k_vals,
            np.maximum(0.0, mu_arr - sd_arr),
            mu_arr + sd_arr,
            color=col,
            alpha=0.14,
            zorder=2,
        )

    ref_curve = np.array(exp1_summary["exponential_reference"], dtype=float) * 100.0
    ax_a.plot(k_vals, ref_curve, color=es.C_CHARCOAL, ls="--", lw=1.55, label=r"Ref. $\exp(-\langle k \rangle / 2)$", zorder=3)

    # Highlight finite-size ER deviation at <k> = 4: 3.00% vs 13.53%
    ax_a.annotate(
        "",
        xy=(4.0, 3.00),
        xytext=(4.0, 13.53),
        arrowprops={"arrowstyle": "<->", "color": es.C_CHARCOAL, "lw": 1.0},
        zorder=5,
    )
    ax_a.annotate(
        r"At $\langle k \rangle = 4$: ER 3.00%" + "\nvs. ref. 13.53%",
        xy=(4.04, 9.2),
        xytext=(4.75, 36.5),
        fontsize=9.0,
        color=es.INK,
        va="center",
        arrowprops={"arrowstyle": "->", "color": es.REF_LINE, "lw": 0.85},
        zorder=5,
    )

    ax_a.set_xlim(0.7, 9.3)
    ax_a.set_ylim(-1.5, 72.0)
    ax_a.set_xticks([1, 3, 5, 7, 9])
    ax_a.set_xlabel(r"Target mean out-degree $\langle k \rangle$")
    ax_a.set_ylabel(r"Shared input fraction $n_D$ (%)")
    ax_a.legend(loc="upper right", fontsize=9.0, handlelength=1.4, handletextpad=0.35, labelspacing=0.30)

    # --- (b) Dedicated actuator fraction (m + beta - alpha)/N vs shared input fraction m/N ---
    es.style_panel_tinted(ax_b)
    es.add_panel_header(ax_b, "b", "Source-SCC overhead (PBH 120/120)", x_title=0.105, y=1.035)

    ovh_series = inference["benchmark_exact_summaries"]["driver_scaling_120"]["cyclic_source_scc_overhead"]["series"]
    xs_b = np.array([r["n_cyclic_source_sccs"] for r in ovh_series], dtype=float)
    shr_pct = np.array([r["shared_fraction"] * 100.0 for r in ovh_series], dtype=float)
    ded_pct = np.array([r["dedicated_fraction"] * 100.0 for r in ovh_series], dtype=float)

    ax_b.fill_between(xs_b, shr_pct, ded_pct, color=es.GAP_SHADE, alpha=0.85, zorder=2, label=r"Overhead $(\beta - \alpha)/N$")
    ax_b.plot(xs_b, ded_pct, color=es.C_BURGUNDY, marker="o", lw=1.95, markersize=5.0,
              label=r"Dedicated $(m + \beta - \alpha)/N$", zorder=4)
    ax_b.plot(xs_b, shr_pct, color=es.C_TEAL, marker="s", ls="--", lw=1.80, markersize=4.8,
              label=r"Shared $\max(1, m)/N$", zorder=4)

    # Mark ER/SF single-root regime at beta - alpha = 0
    ax_b.scatter([0.0], [3.00], s=42, color=es.C_BRONZE, marker="D", zorder=5)
    ax_b.text(0.55, 2.55, r"ER/SF roots: $\beta - \alpha = 0$", fontsize=9.0, color=es.INK, va="center")
    ax_b.text(10.2, 9.1, "+15 sites\nat 16 cycles", fontsize=9.0, color=es.C_BURGUNDY, va="center")

    ax_b.set_xlim(-0.6, 16.8)
    ax_b.set_ylim(0.5, 22.8)
    ax_b.set_xticks([0, 4, 8, 12, 16])
    ax_b.set_xlabel(r"Upstream cyclic source SCCs $\beta_c$")
    ax_b.set_ylabel("Required actuator fraction (%)")
    ax_b.legend(loc="upper left", fontsize=9.0, handlelength=1.4, handletextpad=0.35, labelspacing=0.30)

    es.save_editorial_png(fig, out_path)


def build_actuator_submodular_figure(exp3_summary: dict, out_path: Path) -> None:
    """Figure 3: (a) Gramian log-volume vs budget K, (b) Energy vs terminal error at K=10, (c) Tolerance crossover."""
    es.apply_editorial_rc()
    fig = plt.figure(figsize=(194.0 * es.MM, 140.0 * es.MM))

    ax_a = fig.add_axes([0.082, 0.560, 0.390, 0.325])
    ax_b = fig.add_axes([0.585, 0.560, 0.385, 0.325])
    ax_c = fig.add_axes([0.082, 0.095, 0.888, 0.345])

    strat_specs = [
        ("Submodular Greedy", "CELF Greedy", es.C_BURGUNDY, "o", "-", 2.05),
        ("Out-Degree Hubs", "Weighted Out-Degree", es.C_TEAL, "s", "-", 1.75),
        ("PageRank", "Reverse PageRank", es.C_BLUE, "^", "-", 1.65),
        ("Random", "Random", es.C_SLATE, "D", "--", 1.50),
        ("Betweenness", "Betweenness", es.C_BRONZE, "v", ":", 1.60),
    ]

    # --- (a) Controllability Gramian log-volume across budgets K in {4,6,8,10,12,15} ---
    es.style_panel_tinted(ax_a)
    es.add_panel_header(ax_a, "a", "Gramian log-volume vs. budget K", x_title=0.105, y=1.04)

    budgets = exp3_summary["budgets"]
    runs = exp3_summary["runs"]
    for key, disp, col, mk, ls, lw in strat_specs:
        ys = [next(r["normalized_log_det"] for r in runs if r["strategy"] == key and r["budget"] == b) for b in budgets]
        ax_a.plot(budgets, ys, color=col, marker=mk, ls=ls, lw=lw, markersize=4.8, label=disp, zorder=4 if key == "Submodular Greedy" else 3)

    ax_a.set_xticks(budgets)
    ax_a.set_xlim(3.4, 15.6)
    ax_a.set_ylim(54, 258)
    ax_a.set_yticks([75, 125, 175, 225])
    ax_a.set_xlabel("Actuator budget K")
    ax_a.set_ylabel(r"$\log\det(I + W_c / \epsilon)$")
    ax_a.legend(
        loc="lower center",
        bbox_to_anchor=(1.18, 1.12),
        ncol=5,
        fontsize=9.0,
        columnspacing=0.85,
        handlelength=1.35,
        handletextpad=0.35,
    )

    # --- (b) Applied input energy E vs terminal error ||x(T) - x_f||_2 at K = 10 ---
    es.style_panel_tinted(ax_b)
    es.add_panel_header(ax_b, "b", r"Energy vs. error at K = 10 ($\lambda = 10^{-4}$)", x_title=0.105, y=1.04)

    k10 = exp3_summary["budget_10_comparison"]
    lbl_pos_b = {
        "Submodular Greedy": (0.205, 392.0, "left"),
        "PageRank": (0.436, 362.0, "right"),
        "Random": (0.445, 276.0, "right"),
        "Out-Degree Hubs": (0.436, 172.0, "right"),
        "Betweenness": (0.525, 102.0, "right"),
    }
    for key, disp, col, mk, _, _ in strat_specs:
        rec = k10[key]
        err = float(rec["terminal_error"])
        eng = float(rec["input_energy"])
        rk = int(rec["reachability_rank"])
        ax_b.scatter([err], [eng], s=54, color=col, marker=mk, zorder=5)
        tx, ty, ha_align = lbl_pos_b[key]
        short = disp.replace("Weighted ", "").replace("Reverse ", "")
        ax_b.text(tx, ty, f"{short} ({rk}/30)", fontsize=9.0, color=es.INK, ha=ha_align, va="center")

    ax_b.set_xlim(0.14, 0.57)
    ax_b.set_ylim(65, 445)
    ax_b.set_xticks([0.2, 0.3, 0.4, 0.5])
    ax_b.set_xlabel(r"Terminal error $||x(T) - x_f||_2$")
    ax_b.set_ylabel("Applied input energy E")

    # --- (c) Equal-tolerance minimum input energy crossover at K = 10 ---
    es.style_panel_tinted(ax_c)
    es.add_panel_header(
        ax_c,
        "c",
        "Minimum input energy under matched endpoint tolerance (K = 10)",
        x_title=0.045,
        y=1.04,
    )

    G30 = reaction_cascade_network(30, 2.3, seed=7)
    A30 = build_continuous_system_matrix(G30, damping=1.15, coupling=0.38)
    x0_30 = np.random.default_rng(42).uniform(0.8, 2.0, 30)
    xf_30 = np.zeros(30)
    tau_grid = tuple(float(v) for v in np.linspace(0.36, 0.525, 65))

    for key, disp, col, mk, ls, lw in strat_specs:
        sel_nodes = k10[key]["selected_nodes"]
        B30 = build_input_matrix(30, sel_nodes)
        traj = minimum_energy_trajectory(A30, B30, x0_30, xf_30, horizon=2.0, reg=1e-4, endpoint_tolerances=tau_grid)
        xs_t, ys_e = [], []
        for item in traj["tolerance_comparison"]:
            if item["feasible"] and item["input_energy"] is not None and item["input_energy"] > 0:
                xs_t.append(item["tolerance"])
                ys_e.append(item["input_energy"])
        ax_c.plot(xs_t, ys_e, color=col, ls=ls, lw=lw, zorder=4 if key in ("Submodular Greedy", "Out-Degree Hubs") else 3)

        # Overlay exact benchmarked points at tau = 0.45 and tau = 0.50
        for item in k10[key]["tolerance_comparison"]:
            if item["feasible"] and item["input_energy"] is not None:
                ax_c.scatter([item["tolerance"]], [item["input_energy"]], s=42, color=col, marker=mk, zorder=5)

    ax_c.set_yscale("log")
    ax_c.axvline(0.45, color=es.REF_LINE, ls="--", lw=0.9, zorder=2)
    ax_c.axvline(0.50, color=es.REF_LINE, ls="--", lw=0.9, zorder=2)

    # Exact crossover and audited comparison callouts
    ax_c.scatter([0.483111], [32.2623], s=52, facecolors="white", edgecolors=es.C_CHARCOAL, linewidths=1.5, zorder=6)
    ax_c.text(0.363, 210.0, "Greedy 38.06 < Out-Deg. 151.15\n(at tol. = 0.45)", fontsize=9.0, color=es.C_BURGUNDY, va="center")
    ax_c.text(0.528, 12.8, "Out-Deg. 25.88 < Greedy 29.74 (at 0.50)", fontsize=9.0, color=es.C_TEAL, va="center", ha="right")
    ax_c.text(0.496, 36000.0, "Betweenness: 32,852 (at 0.45)", fontsize=9.0, color=es.C_BRONZE, va="center", ha="right")
    ax_c.annotate(
        r"Crossover $\tau^* = 0.483$",
        xy=(0.483111, 38.0),
        xytext=(0.476, 560.0),
        fontsize=9.0,
        color=es.INK,
        ha="center",
        va="center",
        arrowprops={"arrowstyle": "->", "color": es.REF_LINE, "lw": 0.85},
        zorder=6,
    )

    ax_c.set_xlim(0.358, 0.535)
    ax_c.set_ylim(8.5, 80000.0)
    ax_c.set_xticks([0.38, 0.42, 0.45, 0.48, 0.50, 0.52])
    ax_c.set_xticklabels(["0.38", "0.42", "0.45", "0.48", "0.50", "0.52"])
    ax_c.set_xlabel(r"Allowed terminal error tolerance $\tau$ ($||x(T) - x_f||_2 \leq \tau$)")
    ax_c.set_ylabel("Minimum input energy E (log scale)")

    es.save_editorial_png(fig, out_path)


def build_neural_pinning_figure(
    holdouts: dict,
    inference: dict,
    results_dir: Path,
    out_path: Path,
) -> None:
    """Figure 4: (a) 80-holdout trajectories (mean + 10-90%), (b) Forest plot vs Local linear, (c) Energy vs state cost."""
    es.apply_editorial_rc()
    fig = plt.figure(figsize=(194.0 * es.MM, 144.0 * es.MM))

    ax_a = fig.add_axes([0.080, 0.555, 0.375, 0.335])
    ax_b = fig.add_axes([0.630, 0.555, 0.340, 0.335])
    ax_c = fig.add_axes([0.085, 0.090, 0.885, 0.355])

    # --- (a) Closed-loop state norm ||x(t)||_2 trajectories (80 holdouts, mean + 10-90% band) ---
    es.style_panel_tinted(ax_a)
    es.add_panel_header(ax_a, "a", "80-holdout state norm (10-90%)", x_title=0.11, y=1.04)

    g28 = reaction_cascade_network(n=28, branch_factor=2.3, seed=PIN_SEED)
    A28 = build_continuous_system_matrix(g28, damping=0.55, coupling=0.36)
    A28_t = torch.tensor(A28, dtype=torch.float32)
    drv28 = sorted(v for v in g28 if g28.in_degree(v) == 0)
    pins28, _ = select_actuators_submodular(A28, budget_k=8, horizon=2.5, mandatory_nodes=drv28[:4])
    B28 = build_input_matrix(len(g28), pins28)
    K28_np, _ = solve_lqr_controller(A28, B28, q_weight=Q_PIN, r_weight=R_PIN)
    K28_t = torch.tensor(K28_np, dtype=torch.float32)
    schedule28 = finite_horizon_gains(A28, B28)
    models28 = _model_set(g28, drv28, pins28, K28_t)
    set_or_train_weights(models28, A28_t, pins28, torch.zeros((1, 28)), results_dir, replay=True)

    s80 = holdouts["splits"]["additional_80"]
    x80_t = torch.tensor([r["x0"] for r in s80], dtype=torch.float32)

    def _bounded(raw_u: torch.Tensor) -> torch.Tensor:
        return CAP * torch.tanh(raw_u / CAP)

    gains_iter = iter(schedule28)
    traj_specs = [
        ("self_only", "Self-only NN", models28["self_only"], es.C_SLATE, "--", 1.55, True),
        ("finite_lqr", "Finite LQR", lambda x: _bounded(-x @ next(gains_iter).T), es.C_BRONZE, ":", 1.75, False),
        ("local_linear", "Local linear", models28["local_linear"], es.C_BLUE, "-.", 1.75, True),
        ("gnn", "1-hop GNN", models28["gnn"], es.C_BURGUNDY, "-", 2.15, True),
        ("full_linear", "Full linear", models28["full_linear"], es.C_TEAL, "--", 1.85, False),
    ]
    with torch.no_grad():
        for m_key, m_lbl, ctrl_fn, col, ls, lw, show_band in traj_specs:
            ts_t, xs_t, _ = rollout_pinning_torch(A28_t, x80_t, pins28, ctrl_fn, PIN_T, PIN_STEPS)
            norms = np.linalg.norm(xs_t.numpy(), axis=2)
            t_arr = ts_t.numpy()
            mu = np.mean(norms, axis=1)
            p10 = np.percentile(norms, 10, axis=1)
            p90 = np.percentile(norms, 90, axis=1)
            ax_a.plot(t_arr, mu, color=col, ls=ls, lw=lw, label=m_lbl, zorder=5 if m_key == "gnn" else 4)
            if show_band:
                ax_a.fill_between(t_arr, p10, p90, color=col, alpha=0.13, zorder=2)

    ax_a.set_xlim(0.0, 4.0)
    ax_a.set_ylim(2.6, 11.8)
    ax_a.set_xticks([0, 1, 2, 3, 4])
    ax_a.set_xlabel("Time t (s)")
    ax_a.set_ylabel(r"State norm $||x(t)||_2$")
    ax_a.legend(
        loc="upper right",
        fontsize=9.0,
        ncol=1,
        handlelength=1.4,
        handletextpad=0.35,
        labelspacing=0.22,
    )

    # --- (b) Horizontal forest / paired contrast plot vs Local linear (N = 80) ---
    es.style_panel_horizontal(ax_b)
    es.add_panel_header(ax_b, "b", r"Paired $\Delta J$ vs. Local linear", x_title=0.125, y=1.04)

    f_map = {r["policy_a"]: r for r in inference["forest_vs_local_linear_80"]}
    p_sum = inference["policy_summaries"]["additional_80"]
    forest_rows = [
        ("full_linear", f"Full linear ({p_sum['full_linear']['total_j_mean']:.1f})", es.C_TEAL, es.SAND_WASH),
        ("gnn", f"1-hop GNN ({p_sum['gnn']['total_j_mean']:.1f})", es.C_BURGUNDY, es.HERO_WASH),
        ("finite_lqr", f"Finite LQR ({p_sum['finite_lqr']['total_j_mean']:.1f})", es.C_BRONZE, "white"),
        ("care", f"CARE-LQR ({p_sum['care']['total_j_mean']:.1f})", es.C_BLUE, es.ROW_STRIPE),
        ("self_only", f"Self-only ({p_sum['self_only']['total_j_mean']:.1f})", es.C_SLATE, "white"),
    ]

    ax_b.set_xscale("symlog", linthresh=35.0)
    nf = len(forest_rows)
    for idx, (pk, ylbl, col, bg_col) in enumerate(forest_rows):
        yp = nf - 1 - idx
        rec = f_map[pk]
        mv = rec["mean_diff"]
        lo, hi = rec["bootstrap_95ci"]
        ax_b.axhspan(yp - 0.42, yp + 0.42, color=bg_col, zorder=0)
        ax_b.errorbar(
            [mv], [yp],
            xerr=[[mv - lo], [hi - mv]],
            fmt="o",
            color=col,
            markersize=5.0,
            lw=1.5,
            capsize=3.0,
            zorder=4,
        )
        if mv < 0:
            ax_b.text(mv + 3.2, yp, f"{mv:.1f}*", fontsize=9.0, color=es.INK, va="center", ha="left")
        elif mv < 120:
            ax_b.text(hi * 1.16, yp, f"+{mv:.1f}*", fontsize=9.0, color=es.INK, va="center", ha="left")
        else:
            ax_b.text(lo * 0.84, yp, f"+{mv:.1f}*", fontsize=9.0, color=es.INK, va="center", ha="right")

    ax_b.axvline(0.0, color=es.REF_LINE, ls="--", lw=0.95, zorder=2)
    ax_b.set_yticks(range(nf))
    ax_b.set_yticklabels([r[1] for r in reversed(forest_rows)], fontsize=9.0)
    ax_b.set_xlim(-38.0, 950.0)
    ax_b.set_xticks([-30, 0, 30, 100, 500])
    ax_b.set_xticklabels(["-30", "0", "30", "100", "500"])
    ax_b.set_ylim(-0.55, nf - 0.45)
    ax_b.set_xlabel(r"$\Delta J$ vs. Local 689.3 (*Holm $p < 10^{-13}$)")

    # --- (c) Applied input energy vs state running + terminal cost across all 9 policies ---
    es.style_panel_tinted(ax_c)
    es.add_panel_header(
        ax_c,
        "c",
        "Applied input energy vs. state running + terminal cost across all 9 policies (N = 80)",
        x_title=0.045,
        y=1.04,
    )

    c_specs = [
        ("uncontrolled", "Uncontrolled", es.C_CHARCOAL, "X", (7.5, 2820.0, "left", None)),
        ("random_proportional", "Random + Prop.", es.C_SLATE, "v", (76.0, 1785.0, "left", None)),
        ("proportional", "Greedy + Prop.", es.C_BLUE, "P", (50.0, 1260.0, "left", None)),
        ("self_only", "Self-only NN (0-hop)", es.C_PLUM, "D", (210.5, 1430.0, "center", None)),
        ("care", "Steady-state CARE", es.C_BRONZE, "^", (123.0, 1020.0, "center", (133.3, 795.0))),
        ("finite_lqr", "Finite-horizon LQR", es.C_BRONZE, "s", (146.0, 520.0, "center", (149.1, 700.0))),
        ("local_linear", "Local linear (1-hop)", es.C_BLUE, "o", (190.0, 910.0, "center", (198.9, 700.0))),
        ("full_linear", "Full-state linear", es.C_TEAL, "*", (211.0, 505.0, "center", (210.7, 625.0))),
        ("gnn", "1-hop GNN", es.C_BURGUNDY, "o", (241.0, 845.0, "center", (234.8, 675.0))),
    ]

    # Connect empirically undominated policies (Pareto frontier)
    pareto_keys = ["uncontrolled", "proportional", "care", "finite_lqr", "local_linear", "full_linear"]
    px = [p_sum[k]["applied_input_energy_mean"] for k in pareto_keys]
    py = [p_sum[k]["state_cost_plus_terminal_mean"] for k in pareto_keys]
    ax_c.plot(px, py, color=es.C_TEAL, ls="--", lw=1.15, alpha=0.75, zorder=2, label="Empirical Pareto frontier")

    for pk, disp, col, mk, (tx, ty, ha_c, ptr_xy) in c_specs:
        rec = p_sum[pk]
        ex = rec["applied_input_energy_mean"]
        esd = rec["applied_input_energy_sd"]
        sy = rec["state_cost_plus_terminal_mean"]
        ssd = rec["state_cost_plus_terminal_sd"]
        ax_c.errorbar(
            [ex], [sy],
            xerr=[esd], yerr=[ssd],
            fmt=mk,
            color=col,
            markersize=(7.5 if mk == "*" else 5.5),
            lw=1.1,
            capsize=2.5,
            zorder=5 if pk in ("gnn", "full_linear") else 4,
        )
        t_col = es.C_BURGUNDY if pk == "gnn" else es.INK
        if ptr_xy is not None:
            ax_c.annotate(
                disp,
                xy=ptr_xy,
                xytext=(tx, ty),
                fontsize=9.0,
                color=t_col,
                ha=ha_c,
                va="center",
                arrowprops={"arrowstyle": "-", "color": es.SEP_LINE, "lw": 0.8},
                zorder=6,
            )
        else:
            ax_c.text(tx, ty, disp, fontsize=9.0, color=t_col, ha=ha_c, va="center", zorder=6)

    ax_c.set_xlim(-6.0, 272.0)
    ax_c.set_ylim(410.0, 3080.0)
    ax_c.set_xlabel(r"Mean applied input energy $E_{\mathrm{app}}$ ($\pm 1$ SD)")
    ax_c.set_ylabel(r"State cost $3\,\mathrm{ISE} + 12\,||x(T)||_2^2$")
    ax_c.legend(loc="upper right", fontsize=9.0, handlelength=1.5)

    es.save_editorial_png(fig, out_path)


def main() -> None:
    results_dir = ROOT / "results"
    assets_dir = ROOT / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    bench = json.loads((results_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
    holdouts = json.loads((results_dir / "neural_pinning_holdouts.json").read_text(encoding="utf-8"))
    inf_path = results_dir / "holdout_paired_inference.json"
    inference = (
        json.loads(inf_path.read_text(encoding="utf-8"))
        if inf_path.exists()
        else compute_holdout_inference(results_dir)
    )

    build_network_topology_figure(
        bench["experiment_2_reaction_cascade_structural"],
        assets_dir / "network_control_topology.png",
    )
    build_driver_scaling_figure(
        bench["experiment_1_scaling_law"],
        inference,
        assets_dir / "driver_scaling_law.png",
    )
    build_actuator_submodular_figure(
        bench["experiment_3_submodular_gramian"],
        assets_dir / "actuator_submodular_benchmark.png",
    )
    build_neural_pinning_figure(
        holdouts,
        inference,
        results_dir,
        assets_dir / "neural_pinning_trajectories.png",
    )


if __name__ == "__main__":
    main()
