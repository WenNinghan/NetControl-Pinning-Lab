"""Reproduce four experiments on synthetic directed networks.

The WRM print-native figures and machine-readable records cover structural input wiring,
finite-horizon actuator selection, and bounded feedback on held-out initial states.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
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
    verify_pbh_controllability,
    verify_pbh_input_placement,
)
from netcontrol.editorial_style import apply_editorial_rc
from netcontrol.gramian_lqr import build_input_matrix

from scripts.compute_holdout_inference import compute_holdout_inference
from scripts.generate_figures import (
    build_actuator_submodular_figure,
    build_driver_scaling_figure,
    build_network_topology_figure,
    build_neural_pinning_figure,
)
from scripts.neural_pinning_comparison import LABELS, RECORDS, WEIGHTS, evaluate_feedback, mean_metric


def configure_mpl_style() -> None:
    """Apply shared WRM print-native editorial Matplotlib rcParams."""
    apply_editorial_rc()


def run_scaling_law_experiment(assets_dir: Path) -> dict:
    """Finite graph sample: input-channel fraction and illustrative exponential reference."""
    k_values = [1.0, 1.5, 2.0, 3.0, 4.0, 5.5, 7.0, 9.0]
    n_nodes = 120
    n_trials = 5
    families = [
        "Scale-Free (γ=2.2, Heterogeneous)",
        "Scale-Free (γ=2.8, Moderate)",
        "Erdős–Rényi (Random)",
    ]
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
                    "family": family,
                    "target_mean_out_degree": k,
                    "seed": graph_seed,
                    "n_edges": graph.number_of_edges(),
                    "minimum_input_channels": result.minimum_input_channels,
                    "minimum_dedicated_sites": result.n_drivers,
                })
        for family, values in summaries.items():
            curves[family].append((float(np.mean(values)), float(np.std(values))))
        curves[reference].append((float(np.exp(-0.5 * k)), 0.0))

    exp1_summary = {
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

    inf_path = ROOT / "results" / "holdout_paired_inference.json"
    inference = (
        json.loads(inf_path.read_text(encoding="utf-8"))
        if inf_path.exists()
        else compute_holdout_inference(ROOT / "results")
    )
    build_driver_scaling_figure(exp1_summary, inference, assets_dir / "driver_scaling_law.png")
    return exp1_summary


def run_topology_and_link_classification(assets_dir: Path) -> dict:
    """Experiment 2: Reaction cascade structural controllability, PBH verification, and edge criticality."""
    G = reaction_cascade_network(n=32, branch_factor=2.2, seed=42)
    sc_res = analyze_structural_controllability(G)
    edge_classes = classify_control_edges(G)
    pbh_ok, pbh_rank = verify_pbh_controllability(G, sc_res.driver_nodes, seed=42)

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

    exp2_summary = {
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

    build_network_topology_figure(exp2_summary, assets_dir / "network_control_topology.png")
    return exp2_summary


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
        "Submodular Greedy": "greedy",
        "Out-Degree Hubs": "out_degree",
        "PageRank": "pagerank",
        "Betweenness": "betweenness",
        "Random": "random",
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
                    G,
                    A,
                    k,
                    strategy=strategy,
                    horizon=horizon,
                    reg=selection_ridge,
                    seed=42 + k,
                )
            B = build_input_matrix(len(A), selected)
            trajectory = minimum_energy_trajectory(
                A,
                B,
                x0,
                xf,
                horizon=horizon,
                reg=steering_ridge,
                endpoint_tolerances=tolerances,
            )
            row = {
                "budget": k,
                "strategy": name,
                "selected_nodes": selected,
                "log_det": metrics["log_det"],
                "normalized_log_det": metrics["normalized_log_det"],
                "terminal_state": trajectory["x"][-1].tolist(),
                **audit,
            }
            for key in (
                "input_energy",
                "terminal_error",
                "terminal_penalty",
                "regularized_objective",
                "reachability_rank",
                "rank_tolerance",
                "minimum_terminal_error",
                "exact_target_reachable",
                "exact_target_energy",
                "endpoint_numerical_atol",
                "tolerance_comparison",
            ):
                row[key] = trajectory[key]
            rows.append(row)

    exp3_summary = {
        "budgets": budgets,
        "protocol": {
            "nodes": len(A),
            "node_order": list(range(len(A))),
            "graph_seed": 7,
            "branch_factor": 2.3,
            "damping": 1.15,
            "coupling": 0.38,
            "laplacian_weight": 0.5,
            "initial_state_seed": 42,
            "random_selection_seed": "42 + budget",
            "horizon": horizon,
            "selection_ridge": selection_ridge,
            "steering_ridge": steering_ridge,
            "trajectory_steps": 80,
            "endpoint_tolerances": list(tolerances),
            "inputs": "Unbounded continuous inputs on selected coordinates",
            "energy": "Integral of squared input over [0,T]",
            "regularized_objective": "input_energy + terminal_error^2 / steering_ridge",
            "exact_target_energy": "Minimum energy when target displacement lies in numerical reachable subspace; null otherwise",
            "rank": "SVD of a square-root finite-horizon Gramian; absolute singular-value cutoff saved for each row",
        },
        "system_matrix": A.tolist(),
        "initial_state": x0.tolist(),
        "target_state": xf.tolist(),
        "graph_edges": [[int(u), int(v), float(attrs["weight"])] for u, v, attrs in G.edges(data=True)],
        "budget_10_comparison": {row["strategy"]: row for row in rows if row["budget"] == 10},
        "runs": rows,
    }

    build_actuator_submodular_figure(exp3_summary, assets_dir / "actuator_submodular_benchmark.png")
    return exp3_summary


def run_neural_pinning_experiment(assets_dir: Path) -> Tuple[dict, list]:
    """Compare feedback on the same synthetic graph; replay frozen weights when archived."""
    results_dir = assets_dir.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    root_results = ROOT / "results"
    use_replay = (root_results / WEIGHTS).exists() and (root_results / RECORDS).exists()
    eval_dir = root_results if use_replay else results_dir
    record = evaluate_feedback(eval_dir, replay=use_replay)

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
            values = {
                field: mean_metric(trials, method, field)
                for field in (
                    "ise_state",
                    "control_energy",
                    "applied_input_energy",
                    "running_j",
                    "total_j",
                    "terminal_norm",
                )
            }
            negatives = [r["scores"][method]["minimum_coordinate"] for r in trials]
            values.update(
                n=len(trials),
                minimum_coordinate=min(negatives),
                negative_trajectories=sum(v < -1e-6 for v in negatives),
            )
            stats[split][method] = values

    inf_path = root_results / "holdout_paired_inference.json"
    inference = (
        json.loads(inf_path.read_text(encoding="utf-8"))
        if inf_path.exists()
        else compute_holdout_inference(eval_dir)
    )
    build_neural_pinning_figure(
        record,
        inference,
        eval_dir,
        assets_dir / "neural_pinning_trajectories.png",
    )

    primary = stats["original_10"]
    history = record.get("reference_loss_history", {}).get("gnn", [])
    if not history and (root_results / "benchmark_summary.json").exists():
        prev_b = json.loads((root_results / "benchmark_summary.json").read_text(encoding="utf-8"))
        e4_prev = prev_b["experiment_4_neural_pinning_control"]
        t_init = e4_prev["training_initial_loss"]
        t_final = e4_prev["training_final_loss"]
    else:
        t_init = round(history[0], 4)
        t_final = round(history[-1], 4)

    return {
        "n_nodes": 28,
        "budget_k": 8,
        "selected_pinned_nodes": record["protocol"]["pinned_nodes"],
        "training_initial_loss": t_init,
        "training_final_loss": t_final,
        "unseen_initial_states_eval": {
            "n_unseen_trials": len(splits["original_10"]),
            "lqr_mean_ise": round(primary["care"]["ise_state"], 4),
            "gnn_mean_ise": round(primary["gnn"]["ise_state"], 4),
            "lqr_mean_total_J": round(primary["care"]["total_j"], 4),
            "gnn_mean_total_J": round(primary["gnn"]["total_j"], 4),
        },
        "per_initial_state_file": RECORDS,
        "trained_weights_file": WEIGHTS,
        "replay_command": "python -m scripts.neural_pinning_comparison --replay",
        "held_out_mean_metrics": stats,
        "comparison": table,
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

    print("[4/4] Evaluating Differentiable Graph Neural Pinning Controller...")
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

    if (ROOT / "results" / RECORDS).exists():
        shutil.copyfile(ROOT / "results" / RECORDS, results_dir / RECORDS)
        compute_holdout_inference(results_dir)


def main() -> None:
    temp_root = Path("D:/AntigravityWork/.tmp") if Path("D:/AntigravityWork/.tmp").exists() else ROOT
    with TemporaryDirectory(dir=temp_root) as temp:
        output = Path(temp)
        run_all(output)
        for folder in ("assets", "results"):
            destination = ROOT / folder
            destination.mkdir(exist_ok=True)
            for result in (output / folder).iterdir():
                if result.name in (RECORDS, WEIGHTS) and (destination / result.name).exists():
                    continue
                shutil.move(str(result), str(destination / result.name))
    print("All benchmarks completed successfully. Results saved to results/ and assets/.")


if __name__ == "__main__":
    main()
