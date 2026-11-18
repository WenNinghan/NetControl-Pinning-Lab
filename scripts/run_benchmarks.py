"""Reproduce four experiments on synthetic directed networks.

The figures and machine-readable records cover structural input wiring,
finite-horizon actuator selection, and bounded feedback on held-out initial states.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
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
    small_world_directed,
    verify_pbh_controllability,
    verify_pbh_input_placement,
)
from netcontrol.gramian_lqr import build_input_matrix

from scripts.neural_pinning_comparison import LABELS, RECORDS, WEIGHTS, evaluate_feedback, mean_metric

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
    """Finite graph sample: input-channel fraction and illustrative exponential reference."""
    k_values = [1.0, 1.5, 2.0, 3.0, 4.0, 5.5, 7.0, 9.0]
    n_nodes = 120
    n_trials = 5
    families = ["Scale-Free (γ=2.2, Heterogeneous)", "Scale-Free (γ=2.8, Moderate)",
                "Erdős–Rényi (Random)"]
    reference = "Reference: exp(-<k>/2)"
    curves = {key: [] for key in [*families, reference]}
    trials = []
    for k in k_values:
        summaries = {key: [] for key in families}
        for trial in range(n_trials):
            seed = 1000 + int(k * 10) + trial
            samples = [
                (families[0], seed, scale_free_directed(n_nodes, mean_degree=k, gamma_exp=2.2, seed=seed)),
                (families[1], seed + 50, scale_free_directed(n_nodes, mean_degree=k, gamma_exp=2.8, seed=seed + 50)),
                (families[2], seed + 100, erdos_renyi_directed(n_nodes, mean_degree=k, seed=seed + 100)),
            ]
            for family, graph_seed, graph in samples:
                result = analyze_structural_controllability(graph)
                summaries[family].append(result.minimum_input_channels / n_nodes)
                trials.append({
                    "family": family, "target_mean_out_degree": k, "seed": graph_seed,
                    "n_edges": graph.number_of_edges(),
                    "minimum_input_channels": result.minimum_input_channels,
                    "minimum_dedicated_sites": result.n_drivers,
                })
        for family, values in summaries.items():
            curves[family].append((float(np.mean(values)), float(np.std(values))))
        curves[reference].append((float(np.exp(-0.5 * k)), 0.0))

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
        reference: (
            PALETTE_JOURNAL["charcoal"],
            None,
            "--",
            "Reference $\\exp(-\\langle k \\rangle / 2)$",
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

    ax.set_xlabel("Target Mean Out-Degree $\\langle k \\rangle$")
    ax.set_ylabel("Minimum independent inputs / number of states")
    ax.set_title("Finite-network comparison (120 states; 5 graphs per point)", pad=8)
    ax.set_xlim(0.65, 9.35)
    ax.set_ylim(-0.02, 0.74)
    ax.legend(loc="upper right", frameon=True, fancybox=False, fontsize=9.0)

    fig.tight_layout(pad=1.10)
    _finalize_journal_figure(fig)
    fig.savefig(assets_dir / "driver_scaling_law.png")
    plt.close(fig)

    return {
        "k_values": k_values,
        "n_nodes": n_nodes,
        "n_trials_per_family_per_degree": n_trials,
        "plotted_quantity": "Minimum independent input channels / n_nodes (shared wiring permitted)",
        "shading": "One descriptive standard deviation across the five sampled graphs",
        "exponential_reference": [m for m, _ in curves[reference]],
        "sf_gamma_2.8_mean_nD": [round(m, 4) for m, _ in curves[families[1]]],
        "trials": trials,
        "sf_gamma_2.2_mean_nD": [round(m, 4) for m, _ in curves["Scale-Free (γ=2.2, Heterogeneous)"]],
        "er_mean_nD": [round(m, 4) for m, _ in curves["Erdős–Rényi (Random)"]],
    }


def run_topology_and_link_classification(assets_dir: Path) -> dict:
    """Experiment 2: Reaction cascade structural controllability, PBH verification, and edge criticality."""
    G = reaction_cascade_network(n=32, branch_factor=2.2, seed=42)
    sc_res = analyze_structural_controllability(G)
    edge_classes = classify_control_edges(G)
    pbh_ok, pbh_rank = verify_pbh_controllability(G, sc_res.driver_nodes, seed=42)

    # Two disconnected cyclic subsystems distinguish input channels from physical sites.
    cycles = nx.DiGraph([(0, 1), (1, 0), (2, 3), (3, 2)])
    cycle_layout = analyze_structural_controllability(cycles)
    disjoint_cycles_control = {
        "edges": list(cycles.edges()),
        "dedicated_sites": cycle_layout.driver_nodes,
        "minimum_input_channels": cycle_layout.minimum_input_channels,
        "shared_input_sites": cycle_layout.shared_input_sites,
        "pbh_rank_one_site": verify_pbh_controllability(cycles, [0])[1],
        "pbh_rank_dedicated": verify_pbh_controllability(cycles, cycle_layout.driver_nodes)[1],
        "pbh_rank_shared": verify_pbh_input_placement(cycles, cycle_layout.shared_input_sites)[1],
    }

    fig, axes = plt.subplots(
        1, 2, figsize=(13.4, 6.2), gridspec_kw={"width_ratios": [1.58, 1.0]}
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
    other_nodes = [u for u in G.nodes() if u not in driver_set]

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
        nodelist=other_nodes,
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
            label=f"Other nodes ({len(other_nodes)})",
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
            label=f"Dedicated sites ({sc_res.n_drivers})",
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
        f"(a) Synthetic cascade ($N = {sc_res.n_nodes}$, dedicated sites = {sc_res.n_drivers}, "
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
            f"sites/N = {nd_val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.6,
            color=PALETTE_JOURNAL["crimson"],
        )

    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(topo_names, fontsize=9.0)
    ax2.set_ylim(0.0, 1.28)
    ax2.set_ylabel("Fraction of Directed Edges")
    ax2.set_title("(b) Matching-based edge classification", pad=8)
    ax2.legend(
        loc="upper center",
        ncol=3,
        fontsize=8.0,
        columnspacing=0.8,
        handletextpad=0.4,
        frameon=True,
        fancybox=False,
    )

    fig.tight_layout(pad=1.15, rect=(0, 0.14, 1, 1))
    fig.text(0.50, 0.115, "(c) Two disconnected cycles: where the input is wired", ha="center", fontsize=11)
    checks = [
        ("One signal at node 0", "pbh_rank_one_site"),
        ("Separate signals at nodes 0 and 2", "pbh_rank_dedicated"),
        ("One shared signal at nodes 0 and 2", "pbh_rank_shared"),
    ]
    for x, (label, rank_key) in zip((0.17, 0.50, 0.83), checks):
        rank = disjoint_cycles_control[rank_key]
        fig.text(x, 0.065, f"{label}\nMinimum PBH rank {rank}/{len(cycles)}", ha="center", va="center", fontsize=9.4)
    _finalize_journal_figure(fig, categorical_axes=(ax2,), topology_axes=(ax,))
    fig.savefig(assets_dir / "network_control_topology.png")
    plt.close(fig)

    return {
        "n_nodes": sc_res.n_nodes,
        "n_edges": sc_res.n_edges,
        "n_drivers": sc_res.n_drivers,
        "dedicated_sites": sc_res.driver_nodes,
        "minimum_input_channels": sc_res.minimum_input_channels,
        "shared_input_sites": sc_res.shared_input_sites,
        "source_components": sc_res.source_components,
        "maximum_matching": sc_res.matching_pairs,
        "graph_seed": 42,
        "pbh_realization": "seed 42, generic edge weights on graph support, dedicated inputs",
        "edge_classification_basis": "Matching membership and increase of minimum independent channel count",
        "edges_by_class": edge_classes,
        "two_disjoint_cycles_control": disjoint_cycles_control,
        "driver_fraction": round(sc_res.driver_fraction, 4),
        "pbh_full_rank_verified": pbh_ok,
        "pbh_rank": pbh_rank,
        "edge_counts": {k: len(v) for k, v in edge_classes.items()},
    }


def run_submodular_gramian_experiment(assets_dir: Path) -> dict:
    """Compare actuator coverage, effort and endpoint error on a seeded linear cascade."""
    G = reaction_cascade_network(n=30, branch_factor=2.3, seed=7)
    A = build_continuous_system_matrix(G, damping=1.15, coupling=0.38)
    budgets = [4, 6, 8, 10, 12, 15]
    horizon, selection_ridge, steering_ridge = 2.0, 1e-5, 1e-4
    tolerances = (0.45, 0.50)
    x0 = np.random.default_rng(42).uniform(0.8, 2.0, size=len(A))
    xf = np.zeros_like(x0)
    methods = {
        "Submodular Greedy": "greedy", "Out-Degree Hubs": "out_degree",
        "PageRank": "pagerank", "Betweenness": "betweenness", "Random": "random",
    }
    rows = []
    for k in budgets:
        for name, strategy in methods.items():
            audit = {}
            if strategy == "greedy":
                selected, metrics = select_actuators_submodular(A, k, horizon, selection_ridge)
                naive, naive_metrics = select_actuators_submodular(
                    A, k, horizon, selection_ridge, use_lazy=False
                )
                audit = {
                    "naive_selected_nodes": naive,
                    "same_order_as_naive_greedy": selected == naive,
                    "celf_marginal_evaluations": int(metrics["num_marginal_evals"]),
                    "naive_marginal_evaluations": int(naive_metrics["num_marginal_evals"]),
                }
            else:
                selected, metrics = select_actuators_baseline(
                    G, A, k, strategy=strategy, horizon=horizon,
                    reg=selection_ridge, seed=42 + k,
                )
            B = build_input_matrix(len(A), selected)
            trajectory = minimum_energy_trajectory(
                A, B, x0, xf, horizon=horizon, reg=steering_ridge,
                endpoint_tolerances=tolerances,
            )
            row = {
                "budget": k, "strategy": name, "selected_nodes": selected,
                "log_det": metrics["log_det"], "normalized_log_det": metrics["normalized_log_det"],
                "terminal_state": trajectory["x"][-1].tolist(), **audit,
            }
            for key in ("input_energy", "terminal_error", "terminal_penalty",
                        "regularized_objective", "reachability_rank", "rank_tolerance",
                        "minimum_terminal_error", "exact_target_reachable", "exact_target_energy",
                        "endpoint_numerical_atol", "tolerance_comparison"):
                row[key] = trajectory[key]
            rows.append(row)

    styles = [
        ("crimson", "o"), ("navy", "s"), ("emerald", "^"), ("plum", "D"), ("slate", "X"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6))
    panels = [
        ("normalized_log_det", "Normalized log determinant", "(a) Reachable volume"),
        ("input_energy", "Continuous input energy", "(b) Input energy, fixed terminal penalty"),
        ("terminal_error", "Terminal state error (L2 norm)", "(c) Residual, fixed terminal penalty"),
    ]
    for (name, (color, marker)) in zip(methods, styles):
        series = [row for row in rows if row["strategy"] == name]
        options = {"color": PALETTE_JOURNAL[color], "marker": marker, "label": name}
        for ax, (metric, _, _) in zip(axes.flat, panels):
            ax.plot(budgets, [row[metric] for row in series], **options)
        ref = next(row for row in series if row["budget"] == 10)
        matches = ref["tolerance_comparison"]
        axes[1, 1].plot(tolerances, [
            v["input_energy"] if v["feasible"] else np.nan for v in matches
        ], **options)
    for ax, (_, ylabel, title) in zip(axes.flat, panels):
        ax.set(xlabel="Actuator budget", ylabel=ylabel, title=title, xticks=budgets)
    axes[1, 1].set(xlabel="Allowed terminal error (L2 norm)", ylabel="Minimum input energy",
                   title="(d) Matched endpoint tolerance, 10 actuators", yscale="log",
                   xticks=tolerances, xticklabels=["0.45", "0.50"])
    axes[1, 1].margins(x=0.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc="lower center", fontsize=9)
    fig.tight_layout(rect=(0, 0.045, 1, 1), pad=1.1)
    _finalize_journal_figure(fig)
    fig.savefig(assets_dir / "actuator_submodular_benchmark.png")
    plt.close(fig)

    return {
        "budgets": budgets,
        "protocol": {
            "nodes": len(A), "node_order": list(range(len(A))), "graph_seed": 7,
            "branch_factor": 2.3, "damping": 1.15, "coupling": 0.38, "laplacian_weight": 0.5,
            "initial_state_seed": 42, "random_selection_seed": "42 + budget",
            "horizon": horizon, "selection_ridge": selection_ridge, "steering_ridge": steering_ridge,
            "trajectory_steps": 80, "endpoint_tolerances": list(tolerances),
            "inputs": "Unbounded continuous inputs on selected coordinates",
            "energy": "Integral of squared input over [0,T]",
            "regularized_objective": "input_energy + terminal_error^2 / steering_ridge",
            "exact_target_energy": "Minimum energy when target displacement lies in numerical reachable subspace; null otherwise",
            "rank": "SVD of a square-root finite-horizon Gramian; absolute singular-value cutoff saved for each row",
        },
        "system_matrix": A.tolist(), "initial_state": x0.tolist(), "target_state": xf.tolist(),
        "graph_edges": [[int(u), int(v), float(attrs["weight"])] for u, v, attrs in G.edges(data=True)],
        "budget_10_comparison": {row["strategy"]: row for row in rows if row["budget"] == 10},
        "runs": rows,
    }


def run_neural_pinning_experiment(assets_dir: Path) -> Tuple[dict, list]:
    """Compare feedback on the same synthetic graph; train each learned policy once."""
    results_dir = assets_dir.parent / "results"
    record = evaluate_feedback(results_dir)
    splits = record["splits"]
    seen = splits["reference_in_training"][0]["scores"]
    table = []
    for method, label in LABELS.items():
        values = seen[method]
        settled = values["settling_time"]
        table.append({
            "strategy": label,
            "ise_state": round(values["ise_state"], 4),
            "control_energy": round(values["control_energy"], 4),
            "applied_input_energy": round(values["applied_input_energy"], 4),
            "running_cost_J": round(values["running_j"], 4),
            "total_cost_J": round(values["total_j"], 4),
            "terminal_norm": round(values["terminal_norm"], 4),
            "settling_time_s": None if settled is None else round(settled, 3),
        })
    stats = {}
    for split in ("original_10", "additional_80"):
        trials = splits[split]
        stats[split] = {}
        for method in LABELS:
            values = {field: mean_metric(trials, method, field)
                      for field in ("ise_state", "control_energy", "applied_input_energy",
                                    "running_j", "total_j", "terminal_norm")}
            negatives = [r["scores"][method]["minimum_coordinate"] for r in trials]
            values.update(n=len(trials), minimum_coordinate=min(negatives),
                          negative_trajectories=sum(v < -1e-6 for v in negatives))
            stats[split][method] = values

    methods = ("care", "finite_lqr", "local_linear", "full_linear", "self_only", "gnn")
    y = np.arange(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.35), sharey=True)
    for ax, field, caption in zip(axes, ("total_j", "applied_input_energy"),
                                  ("(a) Mean total cost $J$", "(b) Mean applied input energy")):
        for shift, split, label, marker, color in [
            (-0.13, "original_10", "Original 10 initial states", "o", PALETTE_JOURNAL["navy"]),
            (0.13, "additional_80", "Additional 80 initial states", "D", PALETTE_JOURNAL["ochre"]),
        ]:
            means = [stats[split][m][field] for m in methods]
            ax.scatter(means, y + shift, marker=marker, color=color, s=36, label=label, zorder=3)
        ax.set_yticks(y, [LABELS[m] for m in methods])
        ax.set_xlabel(caption)
        ax.set_ylim(len(methods) - 0.5, -0.5)
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.93, 1.04), ncol=2, frameon=False)
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.18, top=0.84, wspace=0.08)
    _finalize_journal_figure(fig)
    fig.savefig(assets_dir / "neural_pinning_trajectories.png")
    plt.close(fig)

    primary = stats["original_10"]
    history = record["reference_loss_history"]["gnn"]
    return {
        "n_nodes": 28, "budget_k": 8, "selected_pinned_nodes": record["protocol"]["pinned_nodes"],
        "training_initial_loss": round(history[0], 4), "training_final_loss": round(history[-1], 4),
        "unseen_initial_states_eval": {
            "n_unseen_trials": len(splits["original_10"]),
            "lqr_mean_ise": round(primary["care"]["ise_state"], 4),
            "gnn_mean_ise": round(primary["gnn"]["ise_state"], 4),
            "lqr_mean_total_J": round(primary["care"]["total_j"], 4),
            "gnn_mean_total_J": round(primary["gnn"]["total_j"], 4),
        },
        "per_initial_state_file": RECORDS, "trained_weights_file": WEIGHTS,
        "replay_command": "python -m scripts.neural_pinning_comparison --replay",
        "held_out_mean_metrics": stats, "comparison": table,
    }, table


def run_all(output_dir: Path) -> None:
    configure_mpl_style()
    assets_dir = output_dir / "assets"
    results_dir = output_dir / "results"
    assets_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Comparing minimum input channels across finite graph samples...")
    exp1 = run_scaling_law_experiment(assets_dir)

    print("[2/4] Checking cascade input wiring and matching-based edge classes...")
    exp2 = run_topology_and_link_classification(assets_dir)

    print("[3/4] Comparing actuator reachability, input energy and terminal error...")
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
        json.dump(summary, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")

    with open(results_dir / "benchmark_table.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "strategy",
                "ise_state",
                "control_energy",
                "applied_input_energy",
                "running_cost_J",
                "total_cost_J",
                "terminal_norm",
                "settling_time_s",
            ],
        )
        writer.writeheader()
        writer.writerows(table_rows)


def main() -> None:
    with TemporaryDirectory(dir=ROOT) as temp:
        output = Path(temp)
        run_all(output)
        for folder in ("assets", "results"):
            destination = ROOT / folder
            destination.mkdir(exist_ok=True)
            for result in (output / folder).iterdir():
                result.replace(destination / result.name)
    print("All benchmarks completed successfully. Results saved to results/ and assets/.")


if __name__ == "__main__":
    main()
