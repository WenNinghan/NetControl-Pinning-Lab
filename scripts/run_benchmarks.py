"""
Automated benchmark runner and publication-grade figure generator for NetControl-Pinning-Lab.

Generates:
1. assets/driver_scaling_law.png: Driver node fraction n_D = N_D / N vs. average degree <k>
   across Erdos-Renyi, Scale-Free (gamma=2.2, 2.7), Small-World, and Reaction Cascade networks.
2. assets/network_control_topology.png: Directed reaction-cascade network control map highlighting
   Matched vs. Unmatched Driver Nodes and Critical / Ordinary / Redundant control links.
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

import matplotlib
matplotlib.use("Agg")
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


plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Arial", "Microsoft YaHei"],
    "axes.unicode_minus": False,
    "figure.dpi": 160,
    "savefig.dpi": 240,
})


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
        curves["Theory: ER Asymptotic exp(-<k>/2)"].append((float(max(1.0 / n_nodes, np.exp(-0.5 * k))), 0.0))

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    palette = {
        "Scale-Free (γ=2.2, Heterogeneous)": ("#d95f02", "o", "-"),
        "Scale-Free (γ=2.8, Moderate)": ("#7570b3", "s", "-"),
        "Erdős–Rényi (Random)": ("#1b9e77", "^", "-"),
        "Theory: ER Asymptotic exp(-<k>/2)": ("#333333", None, "--"),
    }

    for label, stats in curves.items():
        means = np.array([m for m, _ in stats])
        stds = np.array([s for _, s in stats])
        color, marker, ls = palette[label]
        ax.plot(k_values, means, label=label, color=color, marker=marker, linestyle=ls, linewidth=2.2, markersize=6)
        if np.any(stds > 0):
            ax.fill_between(k_values, np.maximum(0, means - stds), np.minimum(1, means + stds), color=color, alpha=0.14)

    ax.set_xlabel("Average Directed Degree <k>", fontsize=11, fontweight="bold")
    ax.set_ylabel("Minimum Driver Node Fraction $n_D = N_D / N$", fontsize=11, fontweight="bold")
    ax.set_title("Structural Controllability Scaling Law: Degree Heterogeneity vs. Randomness", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=9.5)
    fig.tight_layout()
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

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), gridspec_kw={"width_ratios": [1.55, 1.0]})

    # Panel A: Hierarchical layout of Reaction Cascade Network
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
        "critical": "#e41a1c",
        "ordinary": "#377eb8",
        "redundant": "#bdbdbd",
    }
    for cls_name, e_list in [("redundant", edge_classes["redundant"]),
                             ("ordinary", edge_classes["ordinary"]),
                             ("critical", edge_classes["critical"])]:
        if e_list:
            nx.draw_networkx_edges(
                G, pos, edgelist=e_list, ax=ax,
                edge_color=edge_colors[cls_name],
                width=2.1 if cls_name == "critical" else (1.3 if cls_name == "ordinary" else 0.9),
                style="solid" if cls_name != "redundant" else "dashed",
                alpha=0.85 if cls_name == "critical" else 0.55,
                arrows=True, arrowsize=11, node_size=260,
            )

    nx.draw_networkx_nodes(
        G, pos, nodelist=matched_nodes, ax=ax,
        node_color="#4daf4a", node_size=240, edgecolors="#1b4332", linewidths=1.2,
        label=f"Matched State Nodes ({len(matched_nodes)})",
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=sc_res.driver_nodes, ax=ax,
        node_color="#ff7f00", node_size=340, node_shape="^", edgecolors="#7f2704", linewidths=1.5,
        label=f"Minimum Driver Nodes N_D ({sc_res.n_drivers})",
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7.5, font_color="#111111", font_weight="bold")
    ax.set_title(
        f"A. Ozone-Pollutant Reaction Cascade Network (N={sc_res.n_nodes}, N_D={sc_res.n_drivers}, PBH Rank={pbh_rank}/{sc_res.n_nodes})",
        fontsize=10.5, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=8.5, frameon=True)
    ax.axis("off")

    # Panel B: Edge Classification & Topology Summary Comparison
    ax2 = axes[1]
    topo_names = ["Reaction\nCascade", "Scale-Free\n(γ=2.3)", "Small-World\n(WS)", "Erdős–Rényi\n(Random)"]
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
    w = 0.52
    ax2.bar(x_idx, crit_fracs, width=w, color="#e41a1c", label="Critical Links")
    ax2.bar(x_idx, ord_fracs, width=w, bottom=crit_fracs, color="#377eb8", label="Ordinary Links")
    ax2.bar(
        x_idx, red_fracs, width=w,
        bottom=np.array(crit_fracs) + np.array(ord_fracs),
        color="#bdbdbd", label="Redundant Links",
    )
    for idx, nd_val in enumerate(nd_fracs):
        ax2.text(idx, 1.03, f"$n_D$={nd_val:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#d95f02")

    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(topo_names, fontsize=9)
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("Fraction of Directed Edges", fontsize=10, fontweight="bold")
    ax2.set_title("B. Structural Control Link Classification", fontsize=10.5, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=8.5, frameon=True)
    ax2.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
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
            sel_b, m_b = select_actuators_baseline(G, A, budget_k=k, strategy=strat_key, horizon=2.0, reg=1e-5, seed=42 + k)
            B_b = build_input_matrix(G.number_of_nodes(), sel_b)
            traj_b = minimum_energy_trajectory(A, B_b, x0, xf, horizon=2.0, reg=1e-4)
            logdet_records[m_label].append(m_b["log_det"])
            energy_records[m_label].append(float(traj_b["energy"]))

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))
    style_map = {
        "Submodular Greedy (Ours)": ("#e41a1c", "o", "-", 2.4),
        "Out-Degree Hubs": ("#377eb8", "s", "--", 1.8),
        "PageRank": ("#4daf4a", "^", "-.", 1.8),
        "Betweenness": ("#984ea3", "D", ":", 1.8),
        "Random": ("#7f7f7f", "x", ":", 1.5),
    }

    for m_label in methods:
        c, mk, ls, lw = style_map[m_label]
        axes[0].plot(budgets, logdet_records[m_label], label=m_label, color=c, marker=mk, linestyle=ls, linewidth=lw)
        axes[1].plot(budgets, energy_records[m_label], label=m_label, color=c, marker=mk, linestyle=ls, linewidth=lw)

    axes[0].set_xlabel("Actuator Node Budget $K = |\\mathcal{P}|$", fontsize=10.5, fontweight="bold")
    axes[0].set_ylabel("Volumetric Controllability $\\log\\det(W_c + \\epsilon I)$", fontsize=10.5, fontweight="bold")
    axes[0].set_title("A. Submodular Gramian Maximization", fontsize=11, fontweight="bold")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=8.5, frameon=True)

    axes[1].set_yscale("log")
    axes[1].set_xlabel("Actuator Node Budget $K = |\\mathcal{P}|$", fontsize=10.5, fontweight="bold")
    axes[1].set_ylabel("Minimum Steering Energy $\\mathcal{E}^*(x_0 \\to 0)$ [log scale]", fontsize=10.5, fontweight="bold")
    axes[1].set_title("B. Pontryagin Minimum Control Energy vs. Budget", fontsize=11, fontweight="bold")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=8.5, frameon=True)

    fig.tight_layout()
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
    pinned_sub, _ = select_actuators_submodular(A, budget_k=budget_k, horizon=2.5, mandatory_nodes=sc.driver_nodes[:4])
    B_pin = build_input_matrix(G.number_of_nodes(), pinned_sub)
    K_lqr, _ = solve_lqr_controller(A, B_pin, q_weight=3.0, r_weight=0.12)

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
        epochs=50,
        lr=0.035,
        q_state=3.0,
        r_ctrl=0.08,
        regime="pollutant_cascade",
        seed=2026,
    )

    # Also compare random actuator placement with linear pinning
    pinned_rand, _ = select_actuators_baseline(G, A, budget_k=budget_k, strategy="random", seed=99)

    sim_uncontrolled = simulate_nonlinear_pinning(A, x0, pinned_sub, mode="uncontrolled", horizon=4.0, n_steps=80)
    sim_rand_lin = simulate_nonlinear_pinning(A, x0, pinned_rand, mode="linear_pinning", kp_linear=2.4, horizon=4.0, n_steps=80)
    sim_sub_lin = simulate_nonlinear_pinning(A, x0, pinned_sub, mode="linear_pinning", kp_linear=2.4, horizon=4.0, n_steps=80)
    sim_lqr = simulate_nonlinear_pinning(A, x0, pinned_sub, mode="lqr_pinning", K_lqr=K_lqr, horizon=4.0, n_steps=80)
    sim_gnn = simulate_nonlinear_pinning(
        A, x0, pinned_sub, mode="gnn_pinning", gnn_controller=gnn_ctrl, horizon=4.0, n_steps=80
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2))

    # Panel A: State Norm ||x(t)||_2 Trajectory
    ax = axes[0, 0]
    ts = sim_gnn["t"]
    ax.plot(ts, sim_uncontrolled["state_norms"], label="Uncontrolled Cascade", color="#7f7f7f", linestyle=":", linewidth=2.0)
    ax.plot(ts, sim_rand_lin["state_norms"], label="Random Pinning + Linear Gain", color="#ff7f00", linestyle="--", linewidth=1.9)
    ax.plot(ts, sim_sub_lin["state_norms"], label="Submodular Pinning + Linear Gain", color="#377eb8", linestyle="-.", linewidth=2.0)
    ax.plot(ts, sim_lqr["state_norms"], label="Submodular Pinning + CARE LQR", color="#4daf4a", linestyle="-", linewidth=2.1)
    ax.plot(ts, sim_gnn["state_norms"], label="Submodular + Graph Neural Pinning (Ours)", color="#e41a1c", linestyle="-", linewidth=2.6)
    ax.axhline(0.15 * float(sim_gnn["state_norms"][0]), color="#222222", linestyle="--", alpha=0.45, label="15% Safe Threshold")
    ax.set_xlabel("Time $t$ (s)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Network Pollutant Norm $\\|x(t)\\|_2$", fontsize=10, fontweight="bold")
    ax.set_title("A. Nonlinear Cascade State Suppression", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Panel B: Individual Node Trajectories under Graph Neural Pinning
    ax = axes[0, 1]
    x_gnn = sim_gnn["x"]
    for i in range(G.number_of_nodes()):
        is_pin = i in set(pinned_sub)
        ax.plot(
            ts, x_gnn[:, i],
            color="#e41a1c" if is_pin else "#457b9d",
            alpha=0.85 if is_pin else 0.35,
            linewidth=1.6 if is_pin else 1.0,
        )
    ax.plot([], [], color="#e41a1c", linewidth=1.8, label=f"Pinned Actuator Nodes (K={budget_k})")
    ax.plot([], [], color="#457b9d", alpha=0.6, linewidth=1.2, label=f"Unpinned Network Nodes ({G.number_of_nodes() - budget_k})")
    ax.set_xlabel("Time $t$ (s)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Node Concentration $x_i(t)$", fontsize=10, fontweight="bold")
    ax.set_title("B. Node-Level Synchronization to Safe Equilibrium", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5, frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Panel C: Control Input Signals u_P(t) for Graph Neural Pinning
    ax = axes[1, 0]
    u_gnn = sim_gnn["u"]
    for j, node_id in enumerate(pinned_sub):
        ax.plot(ts, u_gnn[:, j], linewidth=1.6, label=f"Node #{node_id}")
    ax.set_xlabel("Time $t$ (s)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Pinning Control Input $u_i(t)$", fontsize=10, fontweight="bold")
    ax.set_title("C. Bounded Graph Neural Control Actions $u_{\\mathcal{P}}(t)$", fontsize=11, fontweight="bold")
    ax.legend(ncol=2, fontsize=7.8, frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Panel D: Differentiable ODE Training Convergence & Method Comparison Bar
    ax = axes[1, 1]
    ax.plot(range(1, len(train_losses) + 1), train_losses, color="#6a3d9a", marker="o", markersize=3.5, linewidth=2.0)
    ax.set_xlabel("Differentiable RK4 Training Epoch", fontsize=10, fontweight="bold")
    ax.set_ylabel("Total Trajectory Loss $\\mathcal{L}(\\theta)$", fontsize=10, fontweight="bold")
    ax.set_title("D. Differentiable Graph Neural Controller Optimization", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(assets_dir / "neural_pinning_trajectories.png")
    plt.close(fig)

    rows = []
    for name, sim in [
        ("Uncontrolled Cascade", sim_uncontrolled),
        ("Random Pinning + Linear Feedback", sim_rand_lin),
        ("Submodular Pinning + Linear Feedback", sim_sub_lin),
        ("Submodular Pinning + Algebraic Riccati LQR", sim_lqr),
        ("Submodular Pinning + Graph Neural Controller (Ours)", sim_gnn),
    ]:
        rows.append({
            "strategy": name,
            "ise_state": round(float(sim["ise_state"]), 4),
            "control_energy": round(float(sim["control_energy"]), 4),
            "total_cost_J": round(float(sim["total_j"]), 4),
            "terminal_norm": round(float(sim["terminal_norm"]), 4),
            "settling_time_s": round(float(sim["settling_time"]), 3),
        })

    return {
        "n_nodes": G.number_of_nodes(),
        "budget_k": budget_k,
        "selected_pinned_nodes": pinned_sub,
        "training_initial_loss": round(train_losses[0], 4),
        "training_final_loss": round(train_losses[-1], 4),
        "comparison": rows,
    }, rows


def main() -> None:
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
            fieldnames=["strategy", "ise_state", "control_energy", "total_cost_J", "terminal_norm", "settling_time_s"],
        )
        writer.writeheader()
        writer.writerows(table_rows)

    print("All benchmarks completed successfully. Results saved to results/ and assets/.")


if __name__ == "__main__":
    main()
