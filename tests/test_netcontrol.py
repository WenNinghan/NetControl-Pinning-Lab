"""
Unit tests verifying mathematical and algorithmic properties of NetControl-Pinning-Lab.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from netcontrol import (
    analyze_structural_controllability,
    build_continuous_system_matrix,
    classify_control_edges,
    compute_controllability_gramian,
    minimum_energy_trajectory,
    reaction_cascade_network,
    select_actuators_submodular,
    simulate_nonlinear_pinning,
    solve_lqr_controller,
    train_neural_pinning_controller,
    verify_pbh_controllability,
)
from netcontrol.gramian_lqr import build_input_matrix


def test_directed_path_and_star_structural_controllability():
    # A simple directed path 0 -> 1 -> 2 -> 3 requires exactly 1 driver node (node 0)
    G_path = nx.DiGraph([(0, 1), (1, 2), (2, 3)])
    res_path = analyze_structural_controllability(G_path)
    assert res_path.n_drivers == 1
    assert res_path.driver_nodes == [0]
    ok_path, rk_path = verify_pbh_controllability(G_path, res_path.driver_nodes, seed=1)
    assert ok_path and rk_path == 4

    # A directed out-star from 0 to {1, 2, 3, 4} can only match 1 leaf in maximum matching,
    # so N_D = 5 - 1 = 4 driver nodes
    G_star = nx.DiGraph([(0, 1), (0, 2), (0, 3), (0, 4)])
    res_star = analyze_structural_controllability(G_star)
    assert res_star.n_drivers == 4
    ok_star, rk_star = verify_pbh_controllability(G_star, res_star.driver_nodes, seed=2)
    assert ok_star and rk_star == 5


def test_edge_classification_partition():
    G = reaction_cascade_network(n=20, branch_factor=2.0, seed=12)
    classes = classify_control_edges(G)
    total_classified = len(classes["critical"]) + len(classes["ordinary"]) + len(classes["redundant"])
    assert total_classified == G.number_of_edges()

    # Removing any critical edge must strictly increase N_D
    base_nd = analyze_structural_controllability(G).n_drivers
    for u, v in classes["critical"][:3]:
        G_rem = G.copy()
        G_rem.remove_edge(u, v)
        assert analyze_structural_controllability(G_rem).n_drivers > base_nd


def test_gramian_symmetry_and_minimum_energy_steering():
    G = reaction_cascade_network(n=12, branch_factor=2.2, seed=21)
    A = build_continuous_system_matrix(G, damping=1.2, coupling=0.35)
    sc = analyze_structural_controllability(G)
    actuators, _ = select_actuators_submodular(A, budget_k=max(4, sc.n_drivers), mandatory_nodes=sc.driver_nodes)
    B = build_input_matrix(G.number_of_nodes(), actuators)

    W = compute_controllability_gramian(A, B, horizon=1.5, n_steps=60)
    assert np.allclose(W, W.T, atol=1e-10)
    eigvals = np.linalg.eigvalsh(W)
    assert np.all(eigvals >= -1e-9)

    x0 = np.ones(G.number_of_nodes())
    xf = np.zeros(G.number_of_nodes())
    traj = minimum_energy_trajectory(A, B, x0, xf, horizon=1.5, n_steps=80, reg=1e-7)
    assert traj["energy"] > 0.0
    assert traj["terminal_error"] < 0.05


def test_neural_pinning_beats_uncontrolled_and_lqr():
    G = reaction_cascade_network(n=16, branch_factor=2.1, seed=99)
    A = build_continuous_system_matrix(G, damping=0.55, coupling=0.35)
    sc = analyze_structural_controllability(G)
    pinned, _ = select_actuators_submodular(A, budget_k=5, mandatory_nodes=sc.driver_nodes[:2])
    B = build_input_matrix(G.number_of_nodes(), pinned)
    K_lqr, _ = solve_lqr_controller(A, B, q_weight=3.0, r_weight=0.12)

    x0 = np.linspace(1.2, 2.4, G.number_of_nodes())
    gnn_ctrl, losses = train_neural_pinning_controller(
        G=G, A=A, pinned_nodes=pinned, driver_nodes=sc.driver_nodes,
        x0=x0, horizon=3.0, n_steps=50, epochs=25, lr=0.04, seed=99,
    )
    assert losses[-1] < losses[0]

    sim_unc = simulate_nonlinear_pinning(A, x0, pinned, mode="uncontrolled", horizon=3.0, n_steps=50)
    sim_lqr = simulate_nonlinear_pinning(A, x0, pinned, mode="lqr_pinning", K_lqr=K_lqr, horizon=3.0, n_steps=50)
    sim_gnn = simulate_nonlinear_pinning(A, x0, pinned, mode="gnn_pinning", gnn_controller=gnn_ctrl, horizon=3.0, n_steps=50)
    assert sim_gnn["ise_state"] < 0.45 * sim_unc["ise_state"]
    assert sim_gnn["ise_state"] < sim_lqr["ise_state"]
    assert sim_gnn["terminal_norm"] < 0.35 * sim_unc["terminal_norm"]
