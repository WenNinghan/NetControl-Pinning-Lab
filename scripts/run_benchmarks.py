"""
Automated benchmark runner and 300-DPI publication-grade figure generator for NetControl-Pinning-Lab.

Styled to CCF-A / Top-Journal (IEEE TAC / Automatica / Nature Physics / Nature Communications)
camera-ready standards: bold Times New Roman serif typography, STIX vector mathematics,
4-sided #111111 boxed frames, inward ticks, and high-contrast academic color hierarchy.

Generates:
1. assets/driver_scaling_law.png: Driver node fraction n_D = N_D / N vs. average degree <k>
   across Erdos-Renyi, Scale-Free (gamma=2.2, 2.8), and asymptotic theory.
2. assets/network_control_topology.png: Directed reaction-cascade network control map highlighting
   Matched vs. Minimum Driver Nodes and Critical / Ordinary / Redundant control links.
3. assets/actuator_submodular_benchmark.png: Volumetric Controllability log det(W_c) and
   Minimum Control Energy comparison across Submodular Greedy, Out-Degree, PageRank, Betweenness, and Random.
4. assets/neural_pinning_trajectories.png: 4-panel comparison of Uncontrolled, Linear Pinning,
   Gramian-LQR Pinning, and Differentiable Graph Neural Pinning Controller on nonlinear pollutant cascades.
5. results/benchmark_summary.json & results/benchmark_table.csv.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from netcontrol import (
    analyze_structural_controllability,
    build_continuous_system_matrix,
    classify_control_edges,
    erdos_renyi_directed,
    minimum_energy_trajectory,
    reaction_cascade_network,
    scale_free_directed,
    select_actuators_baseline,
    select_actuators_submodular,
    simulate_nonlinear_pinning,
    small_world_directed,
    solve_lqr_controller,
    train_neural_pinning_controller,
    verify_pbh_controllability,
)
from netcontrol.gramian_lqr import build_input_matrix

# High-contrast academic journal palette (IEEE TAC / Nature Physics)
PALETTE_JOURNAL = {
    "navy": "#1B4F72",
    "emerald": "#0B6E4F",
    "crimson": "#B03A2E",
    "ochre": "#C55A11",
    "plum": "#5B2C6F",
    "slate": "#5D6D7E",
    "teal": "#117864",
    "gold": "#7D6608",
    "charcoal": "#111111",
}


def configure_mpl_style() -> None:
    """Configure global Matplotlib rcParams for 300-DPI IEEE TAC / Automatica figures."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "font.size": 10.5,
            "axes.titlesize": 11.4,
            "axes.labelsize": 10.6,
            "xtick.labelsize": 9.6,
            "ytick.labelsize": 9.6,
            "legend.fontsize": 8.5,
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
            "axes.edgecolor": "#111111",
            "axes.linewidth": 1.30,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.spines.bottom": True,
            "axes.spines.left": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.minor.size": 2.4,
            "ytick.minor.size": 2.4,
            "xtick.minor.width": 0.9,
            "ytick.minor.width": 0.9,
            "xtick.color": "#111111",
            "ytick.color": "#111111",
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.35,
            "grid.color": "#94A3B8",
            "legend.fancybox": False,
            "legend.edgecolor": "#111111",
            "legend.facecolor": "#FFFFFF",
            "legend.framealpha": 0.96,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )


def _finalize_journal_figure(
    fig: plt.Figure,
    categorical_axes: Iterable[plt.Axes] = (),
    topology_axes: Iterable[plt.Axes] = (),
) -> None:
    """Enforce Times New Roman bold typography, 4-sided frames, and tick geometry across all axes."""
    cat_set = set(categorical_axes)
    topo_set = set(topology_axes)

    fig.patch.set_facecolor("#FFFFFF")
    fig.canvas.draw()

    for ax in fig.axes:
        ax.set_facecolor("#FFFFFF")
        ax.set_axisbelow(True)

        spine_lw = 1.20 if ax in topo_set else 1.30
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("#111111")
            spine.set_linewidth(spine_lw)

        if ax in topo_set:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
        elif ax in cat_set:
            ax.tick_params(
                axis="y",
                which="major",
                direction="in",
                left=True,
                right=True,
                length=4.5,
                width=1.2,
                colors="#111111",
            )
            ax.tick_params(
                axis="y",
                which="minor",
                direction="in",
                left=True,
                right=True,
                length=2.4,
                width=0.9,
                colors="#111111",
            )
            ax.tick_params(
                axis="x",
                which="both",
                direction="out",
                bottom=True,
                top=False,
                length=4.0,
                width=1.2,
                colors="#111111",
            )
            ax.grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.35, color="#94A3B8")
            ax.grid(False, axis="x")
        else:
            ax.tick_params(
                axis="both",
                which="major",
                direction="in",
                top=True,
                right=True,
                bottom=True,
                left=True,
                length=4.5,
                width=1.2,
                colors="#111111",
            )
            ax.tick_params(
                axis="both",
                which="minor",
                direction="in",
                top=True,
                right=True,
                bottom=True,
                left=True,
                length=2.4,
                width=0.9,
                colors="#111111",
            )
            ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.35, color="#94A3B8")

        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontfamily("Times New Roman")
            lbl.set_fontweight("bold")

        for item in (ax.title, ax.xaxis.label, ax.yaxis.label):
            item.set_fontfamily("Times New Roman")
            item.set_fontweight("bold")

        for txt in ax.texts:
            txt.set_fontfamily("Times New Roman")
            txt.set_fontweight("bold")

        leg = ax.get_legend()
        if leg is not None:
            for ltxt in leg.get_texts():
                ltxt.set_fontfamily("Times New Roman")
                ltxt.set_fontweight("bold")
            if leg.get_title() is not None:
                leg.get_title().set_fontfamily("Times New Roman")
                leg.get_title().set_fontweight("bold")
            frame = leg.get_frame()
            frame.set_linewidth(1.1)
            frame.set_edgecolor("#111111")
            frame.set_facecolor("#FFFFFF")
            frame.set_alpha(0.96)


def run_scaling_law_experiment(assets_dir: Path) -> dict:
    """Experiment 1: Driver node fraction n_D vs average degree <k> across network families."""
    k_values = [1.0, 1.5, 2.0, 3.0, 4.0, 5.5, 7.0, 9.0]
    n_nodes = 120
    n_trials = 5

    curves = {
        "Scale-Free (γ=2.2, Heterogeneous)": [],
        "Scale-Free (γ=2.8, Moderate)": [],
        "Erdős–Rényi (Random)": [],
        "Theory: ER Asymptotic exp(-<k>/2)": [],
    }

    for k in k_values:
        sf22, sf28, er_list = [], [], []
        for trial in range(n_trials):
            seed = 1000 + int(k * 10) + trial
            g_sf22 = scale_free_directed(n_nodes, mean_degree=k, gamma_exp=2.2, seed=seed)
            g_sf28 = scale_free_directed(n_nodes, mean_degree=k, gamma_exp=2.8, seed=seed + 50)
            g_er = erdos_renyi_directed(n_nodes, mean_degree=k, seed=seed + 100)

            sf22.append(analyze_structural_controllability(g_sf22).driver_fraction)
            sf28.append(analyze_structural_controllability(g_sf28).driver_fraction)
            er_list.append(analyze_structural_controllability(g_er).driver_fraction)

        curves["Scale-Free (γ=2.2, Heterogeneous)"].append((float(np.mean(sf22)), float(np.std(sf22))))
        curves["Scale-Free (γ=2.8, Moderate)"].append((float(np.mean(sf28)), float(np.std(sf28))))
        curves["Erdős–Rényi (Random)"].append((float(np.mean(er_list)), float(np.std(er_list))))
        curves["Theory: ER Asymptotic exp(-<k>/2)"].append(
            (float(max(1.0 / n_nodes, np.exp(-0.5 * k))), 0.0)
        )

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    palette = {
        "Scale-Free (γ=2.2, Heterogeneous)": (
            PALETTE_JOURNAL["crimson"],
            "o",
            "-",
            "Scale-Free ($\\gamma = 2.2$, Heterogeneous)",
        ),
        "Scale-Free (γ=2.8, Moderate)": (
            PALETTE_JOURNAL["plum"],
            "s",
            "-",
            "Scale-Free ($\\gamma = 2.8$, Moderate)",
        ),
        "Erdős–Rényi (Random)": (
            PALETTE_JOURNAL["emerald"],
            "^",
            "-",
            "Erdős–Rényi (Random)",
        ),
        "Theory: ER Asymptotic exp(-<k>/2)": (
            PALETTE_JOURNAL["charcoal"],
            None,
            "--",
            "Theory: ER Asymptotic $\\exp(-\\langle k \\rangle / 2)$",
        ),
    }

    for key, stats in curves.items():
        means = np.array([m for m, _ in stats])
        stds = np.array([s for _, s in stats])
        color, marker, ls, disp_label = palette[key]
        ax.plot(
            k_values,
            means,
            label=disp_label,
            color=color,
            marker=marker,
            linestyle=ls,
            linewidth=2.3,
            markersize=6.2,
            markeredgecolor="#111111",
            markeredgewidth=0.6,
        )
        if np.any(stds > 0):
            ax.fill_between(
                k_values,
                np.maximum(0.0, means - stds),
                np.minimum(1.0, means + stds),
                color=color,
                alpha=0.14,
            )

    ax.set_xlabel("Average Directed Degree $\\langle k \\rangle$")
    ax.set_ylabel("Minimum Driver Node Fraction $n_D = N_D / N$")
    ax.set_title("Structural Controllability Scaling Law: Heterogeneity vs. Randomness", pad=8)
    ax.set_xlim(0.65, 9.35)
    ax.set_ylim(-0.02, 0.74)
    ax.legend(loc="upper right", frameon=True, fancybox=False, fontsize=9.0)

    fig.tight_layout(pad=1.10)
    _finalize_journal_figure(fig)
    fig.savefig(assets_dir / "driver_scaling_law.png")
    plt.close(fig)

    return {
        "k_values": k_values,
        "sf_gamma_2.2_mean_nD": [round(m, 4) for m, _ in curves["Scale-Free (γ=2.2, Heterogeneous)"]],
        "er_mean_nD": [round(m, 4) for m, _ in curves["Erdős–Rényi (Random)"]],
    }


def run_topology_and_link_classification(assets_dir: Path) -> dict:
    """Experiment 2: Reaction cascade structural controllability, PBH verification, and edge criticality."""
    G = reaction_cascade_network(n=32, branch_factor=2.2, seed=42)
    sc_res = analyze_structural_controllability(G)
    edge_classes = classify_control_edges(G)
    pbh_ok, pbh_rank = verify_pbh_controllability(G, sc_res.driver_nodes, seed=42)

    fig, axes = plt.subplots(
        1, 2, figsize=(13.4, 5.7), gridspec_kw={"width_ratios": [1.58, 1.0]}
    )

    # Panel (a): Hierarchical layout of Reaction Cascade Network
    ax = axes[0]
    pos = {}
    tiers = {"precursor": [], "intermediate": [], "byproduct": []}
    for u, data in G.nodes(data=True):
        tiers[data.get("tier", "intermediate")].append(u)

    y_map = {"precursor": 2.0, "intermediate": 1.0, "byproduct": 0.0}
    rng = np.random.default_rng(42)
    for tier_name, node_list in tiers.items():
        xs = np.linspace(0.0, 10.0, len(node_list))
        for idx, u in enumerate(node_list):
            jitter_y = rng.uniform(-0.18, 0.18)
            pos[u] = np.array([xs[idx], y_map[tier_name] + jitter_y])

    driver_set = set(sc_res.driver_nodes)
    matched_nodes = [u for u in G.nodes() if u not in driver_set]

    # Draw edges by criticality class
    edge_colors = {
        "critical": PALETTE_JOURNAL["crimson"],
        "ordinary": PALETTE_JOURNAL["navy"],
        "redundant": PALETTE_JOURNAL["slate"],
    }
    for cls_name, e_list in [
        ("redundant", edge_classes["redundant"]),
        ("ordinary", edge_classes["ordinary"]),
        ("critical", edge_classes["critical"]),
    ]:
        if e_list:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=e_list,
                ax=ax,
                edge_color=edge_colors[cls_name],
                width=2.3 if cls_name == "critical" else (1.3 if cls_name == "ordinary" else 1.0),
                style="solid" if cls_name != "redundant" else "dashed",
                alpha=0.92 if cls_name == "critical" else (0.58 if cls_name == "ordinary" else 0.50),
                arrows=True,
                arrowsize=11,
                node_size=280,
            )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=matched_nodes,
        ax=ax,
        node_color="#D4EFDF",
        node_size=265,
        edgecolors=PALETTE_JOURNAL["emerald"],
        linewidths=1.8,
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=sc_res.driver_nodes,
        ax=ax,
        node_color="#FDEBD0",
        node_size=360,
        node_shape="^",
        edgecolors=PALETTE_JOURNAL["ochre"],
        linewidths=2.0,
    )
    lbl_artists = nx.draw_networkx_labels(
        G,
        pos,
        ax=ax,
        font_size=7.8,
        font_color="#111111",
        font_family="Times New Roman",
        font_weight="bold",
    )
    for txt_obj in lbl_artists.values():
        txt_obj.set_fontfamily("Times New Roman")
        txt_obj.set_fontweight("bold")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#D4EFDF",
            markeredgecolor=PALETTE_JOURNAL["emerald"],
            markeredgewidth=1.7,
            markersize=8.2,
            label=f"Matched Nodes ({len(matched_nodes)})",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor="#FDEBD0",
            markeredgecolor=PALETTE_JOURNAL["ochre"],
            markeredgewidth=1.8,
            markersize=9.0,
            label=f"Driver Nodes $N_D$ ({sc_res.n_drivers})",
        ),
        Line2D(
            [0],
            [0],
            color=PALETTE_JOURNAL["crimson"],
            linewidth=2.3,
            linestyle="-",
            label=f"Critical Links ({len(edge_classes['critical'])})",
        ),
        Line2D(
            [0],
            [0],
            color=PALETTE_JOURNAL["navy"],
            linewidth=1.5,
            linestyle="-",
            label=f"Ordinary Links ({len(edge_classes['ordinary'])})",
        ),
        Line2D(
            [0],
            [0],
            color=PALETTE_JOURNAL["slate"],
            linewidth=1.2,
            linestyle="--",
            label=f"Redundant Links ({len(edge_classes['redundant'])})",
        ),
    ]
    ax.set_xlim(-0.85, 10.85)
    ax.set_ylim(-0.52, 2.96)
    ax.set_title(
        f"(a) Ozone–Pollutant Cascade ($N = {sc_res.n_nodes}$, $N_D = {sc_res.n_drivers}$, "
        f"$\\mathrm{{rank}}_{{\\mathrm{{PBH}}}} = {pbh_rank}/{sc_res.n_nodes}$)",
        pad=8,
    )
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        fontsize=8.0,
        columnspacing=0.9,
        handletextpad=0.45,
        frameon=True,
        fancybox=False,
    )

    # Panel (b): Edge Classification & Topology Summary Comparison
    ax2 = axes[1]
    topo_names = [
        "Reaction\nCascade",
        "Scale-Free\n($\\gamma = 2.3$)",
        "Small-World\n(WS)",
        "Erdős–Rényi\n(Random)",
    ]
    graphs = [
        G,
        scale_free_directed(32, mean_degree=2.8, gamma_exp=2.3, seed=42),
        small_world_directed(32, k_out=2, p_rewire=0.25, seed=42),
        erdos_renyi_directed(32, mean_degree=2.8, seed=42),
    ]
    crit_fracs, ord_fracs, red_fracs, nd_fracs = [], [], [], []
    for g_item in graphs:
        ec = classify_control_edges(g_item)
        tot_e = max(1, g_item.number_of_edges())
        crit_fracs.append(len(ec["critical"]) / tot_e)
        ord_fracs.append(len(ec["ordinary"]) / tot_e)
        red_fracs.append(len(ec["redundant"]) / tot_e)
        nd_fracs.append(analyze_structural_controllability(g_item).driver_fraction)

    x_idx = np.arange(len(topo_names))
    w = 0.50
    ax2.bar(
        x_idx,
        crit_fracs,
        width=w,
        color=PALETTE_JOURNAL["crimson"],
        edgecolor="#111111",
        linewidth=0.95,
        label="Critical",
    )
    ax2.bar(
        x_idx,
        ord_fracs,
        width=w,
        bottom=crit_fracs,
        color=PALETTE_JOURNAL["navy"],
        edgecolor="#111111",
        linewidth=0.95,
        label="Ordinary",
    )
    ax2.bar(
        x_idx,
        red_fracs,
        width=w,
        bottom=np.array(crit_fracs) + np.array(ord_fracs),
        color="#CBD5E1",
        edgecolor="#111111",
        linewidth=0.95,
        hatch="//",
        label="Redundant",
    )
    for idx, nd_val in enumerate(nd_fracs):
        ax2.text(
            idx,
            1.025,
            f"$n_D = {nd_val:.2f}$",
            ha="center",
            va="bottom",
            fontsize=8.6,
            color=PALETTE_JOURNAL["crimson"],
        )

    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(topo_names, fontsize=9.0)
    ax2.set_ylim(0.0, 1.28)
    ax2.set_ylabel("Fraction of Directed Edges")
    ax2.set_title("(b) Structural Control Link Classification", pad=8)
    ax2.legend(
        loc="upper center",
        ncol=3,
        fontsize=8.0,
        columnspacing=0.8,
        handletextpad=0.4,
        frameon=True,
        fancybox=False,
    )

    fig.tight_layout(pad=1.15)
    _finalize_journal_figure(fig, categorical_axes=(ax2,), topology_axes=(ax,))
    fig.savefig(assets_dir / "network_control_topology.png")
    plt.close(fig)

    return {
        "n_nodes": sc_res.n_nodes,
        "n_edges": sc_res.n_edges,
        "n_drivers": sc_res.n_drivers,
        "driver_fraction": round(sc_res.driver_fraction, 4),
        "pbh_full_rank_verified": pbh_ok,
        "pbh_rank": pbh_rank,
        "edge_counts": {k: len(v) for k, v in edge_classes.items()},
    }


def run_submodular_gramian_experiment(assets_dir: Path) -> dict:
    """Experiment 3: Submodular greedy actuator selection vs centrality baselines."""
    G = reaction_cascade_network(n=30, branch_factor=2.3, seed=7)
    A = build_continuous_system_matrix(G, damping=1.15, coupling=0.38)
    budgets = [4, 6, 8, 10, 12, 15]

    methods = ["Submodular Greedy (Ours)", "Out-Degree Hubs", "PageRank", "Betweenness", "Random"]
    logdet_records = {m: [] for m in methods}
    energy_records = {m: [] for m in methods}

    rng = np.random.default_rng(42)
    x0 = rng.uniform(0.8, 2.0, size=G.number_of_nodes())
    xf = np.zeros(G.number_of_nodes())

    for k in budgets:
        sel_sub, m_sub = select_actuators_submodular(A, budget_k=k, horizon=2.0, reg=1e-5)
        B_sub = build_input_matrix(G.number_of_nodes(), sel_sub)
        traj_sub = minimum_energy_trajectory(A, B_sub, x0, xf, horizon=2.0, reg=1e-4)
        logdet_records["Submodular Greedy (Ours)"].append(m_sub["log_det"])
        energy_records["Submodular Greedy (Ours)"].append(float(traj_sub["energy"]))

        for m_label, strat_key in [
            ("Out-Degree Hubs", "out_degree"),
            ("PageRank", "pagerank"),
            ("Betweenness", "betweenness"),
            ("Random", "random"),
        ]:
            sel_b, m_b = select_actuators_baseline(
                G, A, budget_k=k, strategy=strat_key, horizon=2.0, reg=1e-5, seed=42 + k
            )
            B_b = build_input_matrix(G.number_of_nodes(), sel_b)
            traj_b = minimum_energy_trajectory(A, B_b, x0, xf, horizon=2.0, reg=1e-4)
            logdet_records[m_label].append(m_b["log_det"])
            energy_records[m_label].append(float(traj_b["energy"]))

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))
    style_map = {
        "Submodular Greedy (Ours)": (PALETTE_JOURNAL["crimson"], "o", "-", 2.6),
        "Out-Degree Hubs": (PALETTE_JOURNAL["navy"], "s", "--", 2.0),
        "PageRank": (PALETTE_JOURNAL["emerald"], "^", "-.", 2.0),
        "Betweenness": (PALETTE_JOURNAL["plum"], "D", ":", 2.0),
        "Random": (PALETTE_JOURNAL["slate"], "X", ":", 1.7),
    }

    for m_label in methods:
        c, mk, ls, lw = style_map[m_label]
        axes[0].plot(
            budgets,
            logdet_records[m_label],
            label=m_label,
            color=c,
            marker=mk,
            markersize=6.0,
            markeredgecolor="#111111",
            markeredgewidth=0.55,
            linestyle=ls,
            linewidth=lw,
        )
        axes[1].plot(
            budgets,
            energy_records[m_label],
            label=m_label,
            color=c,
            marker=mk,
            markersize=6.0,
            markeredgecolor="#111111",
            markeredgewidth=0.55,
            linestyle=ls,
            linewidth=lw,
        )

    axes[0].set_xlabel("Actuator Node Budget $K = |S_{\\mathrm{pin}}|$")
    axes[0].set_ylabel("Volumetric Controllability $\\log\\det(\\mathbf{W}_c + \\varepsilon \\mathbf{I})$")
    axes[0].set_title("(a) Submodular Gramian Maximization", pad=8)
    axes[0].set_xlim(3.4, 15.6)
    axes[0].set_ylim(-292.0, -82.0)
    axes[0].legend(loc="upper left", fontsize=8.4, frameon=True, fancybox=False)

    axes[1].set_yscale("log")
    axes[1].set_xlabel("Actuator Node Budget $K = |S_{\\mathrm{pin}}|$")
    axes[1].set_ylabel("Minimum Steering Energy $E^*(\\mathbf{x}_0 \\to \\mathbf{0})$ (Log Scale)")
    axes[1].set_title("(b) Pontryagin Minimum Control Energy vs. Budget", pad=8)
    axes[1].set_xlim(3.4, 15.6)
    axes[1].set_ylim(1.2e2, 2.4e4)
    axes[1].legend(loc="upper right", fontsize=8.4, frameon=True, fancybox=False)

    fig.tight_layout(pad=1.15)
    _finalize_journal_figure(fig)
    fig.savefig(assets_dir / "actuator_submodular_benchmark.png")
    plt.close(fig)

    k_idx = budgets.index(10)
    return {
        "budgets": budgets,
        "budget_10_comparison": {
            m: {
                "log_det": round(logdet_records[m][k_idx], 3),
                "min_energy": round(energy_records[m][k_idx], 3),
            }
            for m in methods
        },
    }


def run_neural_pinning_experiment(assets_dir: Path) -> Tuple[dict, list]:
    """Experiment 4: Nonlinear Pollutant Cascade Suppression with Graph Neural Pinning Control."""
    G = reaction_cascade_network(n=28, branch_factor=2.3, seed=2026)
    A = build_continuous_system_matrix(G, damping=0.55, coupling=0.36)
    sc = analyze_structural_controllability(G)

    budget_k = 8
    pinned_sub, _ = select_actuators_submodular(
        A, budget_k=budget_k, horizon=2.5, mandatory_nodes=sc.driver_nodes[:4]
    )
    B_pin = build_input_matrix(G.number_of_nodes(), pinned_sub)
    # Exact fair LQR baseline using the identical (q_weight=3.0, r_weight=0.08) weights
    K_lqr, _ = solve_lqr_controller(A, B_pin, q_weight=3.0, r_weight=0.08)

    rng = np.random.default_rng(2026)
    x0 = rng.uniform(1.2, 2.8, size=G.number_of_nodes())

    gnn_ctrl, train_losses = train_neural_pinning_controller(
        G=G,
        A=A,
        pinned_nodes=pinned_sub,
        driver_nodes=sc.driver_nodes,
        x0=x0,
        horizon=4.0,
        n_steps=80,
        epochs=55,
        lr=0.035,
        q_state=3.0,
        r_ctrl=0.08,
        q_term=12.0,
        n_train_inits=8,
        regime="pollutant_cascade",
        seed=2026,
    )

    # Also compare random actuator placement with linear pinning
    pinned_rand, _ = select_actuators_baseline(G, A, budget_k=budget_k, strategy="random", seed=99)

    sim_uncontrolled = simulate_nonlinear_pinning(
        A, x0, pinned_sub, mode="uncontrolled", horizon=4.0, n_steps=80
    )
    sim_rand_lin = simulate_nonlinear_pinning(
        A, x0, pinned_rand, mode="linear_pinning", kp_linear=2.4, horizon=4.0, n_steps=80
    )
    sim_sub_lin = simulate_nonlinear_pinning(
        A, x0, pinned_sub, mode="linear_pinning", kp_linear=2.4, horizon=4.0, n_steps=80
    )
    sim_lqr = simulate_nonlinear_pinning(
        A, x0, pinned_sub, mode="lqr_pinning", K_lqr=K_lqr, horizon=4.0, n_steps=80
    )
    sim_gnn = simulate_nonlinear_pinning(
        A, x0, pinned_sub, mode="gnn_pinning", gnn_controller=gnn_ctrl, horizon=4.0, n_steps=80
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.8))

    # Panel (a): State Norm ||x(t)||_2 Trajectory
    ax = axes[0, 0]
    ts = sim_gnn["t"]
    ax.plot(
        ts,
        sim_uncontrolled["state_norms"],
        label="Uncontrolled Cascade",
        color=PALETTE_JOURNAL["slate"],
        linestyle=":",
        linewidth=2.2,
    )
    ax.plot(
        ts,
        sim_rand_lin["state_norms"],
        label="Random Pinning + Linear Gain",
        color=PALETTE_JOURNAL["ochre"],
        linestyle="--",
        linewidth=2.0,
    )
    ax.plot(
        ts,
        sim_sub_lin["state_norms"],
        label="Submodular Pinning + Linear Gain",
        color=PALETTE_JOURNAL["navy"],
        linestyle="-.",
        linewidth=2.1,
    )
    ax.plot(
        ts,
        sim_lqr["state_norms"],
        label="Submodular Pinning + CARE LQR",
        color=PALETTE_JOURNAL["emerald"],
        linestyle="-",
        linewidth=2.2,
    )
    ax.plot(
        ts,
        sim_gnn["state_norms"],
        label="Submodular + Graph Neural Pinning (Ours)",
        color=PALETTE_JOURNAL["crimson"],
        linestyle="-",
        linewidth=2.7,
    )
    ax.axhline(
        0.15 * float(sim_gnn["state_norms"][0]),
        color="#111111",
        linestyle="--",
        linewidth=1.3,
        alpha=0.75,
        label="15% Safe Threshold",
    )
    ax.set_xlabel("Time $t$ ($\\mathrm{s}$)")
    ax.set_ylabel("Network Pollutant Norm $||\\mathbf{x}(t)||_2$")
    ax.set_title("(a) Nonlinear Cascade State Suppression", pad=8)
    ax.set_xlim(-0.1, 4.1)
    ax.set_ylim(0.8, 12.1)
    # Place legend strictly between y=2.0 (above 1.62 threshold) and y=6.3 (below trajectories at t < 1.9)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.02, 0.11),
        fontsize=7.9,
        frameon=True,
        fancybox=False,
    )

    # Panel (b): Individual Node Trajectories under Graph Neural Pinning
    ax = axes[0, 1]
    x_gnn = sim_gnn["x"]
    ax.axhline(0.0, color="#111111", linestyle=":", linewidth=1.0, alpha=0.55)
    for i in range(G.number_of_nodes()):
        is_pin = i in set(pinned_sub)
        ax.plot(
            ts,
            x_gnn[:, i],
            color=PALETTE_JOURNAL["crimson"] if is_pin else PALETTE_JOURNAL["navy"],
            alpha=0.88 if is_pin else 0.30,
            linewidth=1.65 if is_pin else 1.05,
        )
    ax.plot(
        [],
        [],
        color=PALETTE_JOURNAL["crimson"],
        linewidth=1.9,
        label=f"Pinned Actuator Nodes ($K = {budget_k}$)",
    )
    ax.plot(
        [],
        [],
        color=PALETTE_JOURNAL["navy"],
        alpha=0.65,
        linewidth=1.3,
        label=f"Unpinned Network Nodes ($N - K = {G.number_of_nodes() - budget_k}$)",
    )
    ax.set_xlabel("Time $t$ ($\\mathrm{s}$)")
    ax.set_ylabel("Node Concentration $x_i(t)$")
    ax.set_title("(b) Node-Level Synchronization to Safe Equilibrium", pad=8)
    ax.set_xlim(-0.1, 4.1)
    ax.set_ylim(-2.90, 3.45)
    ax.legend(loc="upper right", fontsize=8.4, frameon=True, fancybox=False)

    # Panel (c): Control Input Signals u_P(t) for Graph Neural Pinning
    ax = axes[1, 0]
    u_gnn = sim_gnn["u"]
    node_colors = [
        PALETTE_JOURNAL["navy"],
        PALETTE_JOURNAL["emerald"],
        PALETTE_JOURNAL["crimson"],
        PALETTE_JOURNAL["ochre"],
        PALETTE_JOURNAL["plum"],
        PALETTE_JOURNAL["slate"],
        PALETTE_JOURNAL["teal"],
        PALETTE_JOURNAL["gold"],
    ]
    ax.axhline(0.0, color="#111111", linestyle=":", linewidth=1.0, alpha=0.55)
    for j, node_id in enumerate(pinned_sub):
        ax.plot(
            ts,
            u_gnn[:, j],
            color=node_colors[j % len(node_colors)],
            linewidth=1.75,
            label=f"Node #{node_id}",
        )
    ax.set_xlabel("Time $t$ ($\\mathrm{s}$)")
    ax.set_ylabel("Pinning Control Input $u_i(t)$")
    ax.set_title("(c) Bounded Graph Neural Control Actions $\\mathbf{u}_{\\mathrm{pin}}(t)$", pad=8)
    ax.set_xlim(-0.1, 4.1)
    ax.set_ylim(-6.55, 2.95)
    ax.legend(loc="lower right", ncol=2, fontsize=8.0, frameon=True, fancybox=False)

    # Panel (d): Differentiable ODE Training Convergence
    ax = axes[1, 1]
    epochs_arr = np.arange(1, len(train_losses) + 1)
    ax.plot(
        epochs_arr,
        train_losses,
        color=PALETTE_JOURNAL["plum"],
        marker="o",
        markersize=3.8,
        markeredgecolor="#111111",
        markeredgewidth=0.45,
        linewidth=2.2,
        label="RK4 Adjoint Cost $J(\\boldsymbol{\\theta})$",
    )
    loss_drop_pct = 100.0 * (train_losses[-1] - train_losses[0]) / max(1e-9, train_losses[0])
    ax.text(
        0.44,
        0.54,
        f"Initial Loss: ${train_losses[0]:.1f}$\n"
        f"Converged Loss: ${train_losses[-1]:.1f}$\n"
        f"Reduction: ${loss_drop_pct:.1f}\\%$",
        transform=ax.transAxes,
        fontsize=8.8,
        bbox=dict(boxstyle="square,pad=0.34", fc="#F8F9F9", ec="#111111", lw=1.0, alpha=0.96),
    )
    ax.set_xlabel("Differentiable RK4 Training Epoch")
    ax.set_ylabel("Total Trajectory Cost $J(\\boldsymbol{\\theta})$")
    ax.set_title("(d) Differentiable Graph Neural Controller Optimization", pad=8)
    ax.set_xlim(0.0, len(train_losses) + 1.5)
    y_lo = min(train_losses) * 0.93
    y_hi = max(train_losses) * 1.07
    ax.set_ylim(y_lo, y_hi)
    ax.legend(loc="upper right", fontsize=8.4, frameon=True, fancybox=False)

    fig.tight_layout(pad=1.15)
    _finalize_journal_figure(fig)
    fig.savefig(assets_dir / "neural_pinning_trajectories.png")
    plt.close(fig)

    # Evaluate zero-shot generalization across 10 unseen random initial states
    rng_test = np.random.default_rng(9090)
    unseen_lqr_j, unseen_gnn_j, unseen_lqr_ise, unseen_gnn_ise = [], [], [], []
    for _ in range(10):
        x0_unseen = rng_test.uniform(1.0, 3.0, size=G.number_of_nodes())
        s_lqr_u = simulate_nonlinear_pinning(
            A, x0_unseen, pinned_sub, mode="lqr_pinning", K_lqr=K_lqr, horizon=4.0, n_steps=80
        )
        s_gnn_u = simulate_nonlinear_pinning(
            A, x0_unseen, pinned_sub, mode="gnn_pinning", gnn_controller=gnn_ctrl, horizon=4.0, n_steps=80
        )
        unseen_lqr_j.append(float(s_lqr_u["total_j"]))
        unseen_gnn_j.append(float(s_gnn_u["total_j"]))
        unseen_lqr_ise.append(float(s_lqr_u["ise_state"]))
        unseen_gnn_ise.append(float(s_gnn_u["ise_state"]))

    rows = []
    for name, sim in [
        ("Uncontrolled Cascade", sim_uncontrolled),
        ("Random Pinning + Linear Feedback", sim_rand_lin),
        ("Submodular Pinning + Linear Feedback", sim_sub_lin),
        ("Submodular Pinning + Algebraic Riccati LQR", sim_lqr),
        ("Submodular Pinning + Graph Neural Controller (Ours)", sim_gnn),
    ]:
        rows.append(
            {
                "strategy": name,
                "ise_state": round(float(sim["ise_state"]), 4),
                "control_energy": round(float(sim["control_energy"]), 4),
                "running_cost_J": round(float(sim["running_j"]), 4),
                "total_cost_J": round(float(sim["total_j"]), 4),
                "terminal_norm": round(float(sim["terminal_norm"]), 4),
                "settling_time_s": round(float(sim["settling_time"]), 3),
            }
        )

    return {
        "n_nodes": G.number_of_nodes(),
        "budget_k": budget_k,
        "selected_pinned_nodes": pinned_sub,
        "training_initial_loss": round(train_losses[0], 4),
        "training_final_loss": round(train_losses[-1], 4),
        "unseen_initial_states_eval": {
            "n_unseen_trials": 10,
            "lqr_mean_ise": round(float(np.mean(unseen_lqr_ise)), 4),
            "gnn_mean_ise": round(float(np.mean(unseen_gnn_ise)), 4),
            "lqr_mean_total_J": round(float(np.mean(unseen_lqr_j)), 4),
            "gnn_mean_total_J": round(float(np.mean(unseen_gnn_j)), 4),
        },
        "comparison": rows,
    }, rows


def main() -> None:
    configure_mpl_style()
    assets_dir = ROOT / "assets"
    results_dir = ROOT / "results"
    assets_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Running Structural Controllability Scaling Law benchmark...")
    exp1 = run_scaling_law_experiment(assets_dir)

    print("[2/4] Running Reaction Cascade Topology & Link Criticality analysis...")
    exp2 = run_topology_and_link_classification(assets_dir)

    print("[3/4] Running Submodular Controllability Gramian & Minimum Energy benchmark...")
    exp3 = run_submodular_gramian_experiment(assets_dir)

    print("[4/4] Training & evaluating Differentiable Graph Neural Pinning Controller...")
    exp4, table_rows = run_neural_pinning_experiment(assets_dir)

    summary = {
        "experiment_1_scaling_law": exp1,
        "experiment_2_reaction_cascade_structural": exp2,
        "experiment_3_submodular_gramian": exp3,
        "experiment_4_neural_pinning_control": exp4,
    }

    with open(results_dir / "benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(results_dir / "benchmark_table.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "strategy",
                "ise_state",
                "control_energy",
                "running_cost_J",
                "total_cost_J",
                "terminal_norm",
                "settling_time_s",
            ],
        )
        writer.writeheader()
        writer.writerows(table_rows)

    print("All benchmarks completed successfully. Results saved to results/ and assets/.")


if __name__ == "__main__":
    main()
