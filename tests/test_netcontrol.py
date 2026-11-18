"""
Comprehensive unit and adversarial verification tests for NetControl-Pinning-Lab.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from netcontrol import (
    analyze_structural_controllability,
    build_continuous_system_matrix,
    classify_control_edges,
    compute_controllability_gramian,
    compute_infinite_horizon_gramian,
    minimum_energy_trajectory,
    reaction_cascade_network,
    select_actuators_submodular,
    simulate_nonlinear_pinning,
    solve_lqr_controller,
    stabilize_hurwitz_matrix,
    train_neural_pinning_controller,
    verify_pbh_controllability,
    verify_pbh_input_placement,
)
from netcontrol.gramian_lqr import build_input_matrix


def test_directed_path_star_cycle_and_string_labels_controllability():
    # 1. Directed path 0 -> 1 -> 2 -> 3 requires N_D = 1 driver node (root 0)
    G_path = nx.DiGraph([(0, 1), (1, 2), (2, 3)])
    res_path = analyze_structural_controllability(G_path)
    assert res_path.n_drivers == 1
    assert res_path.driver_nodes == [0]
    ok_path, rk_path = verify_pbh_controllability(G_path, res_path.driver_nodes, seed=1)
    assert ok_path and rk_path == 4
    # Adversarial check: driving non-root node [2] instead of [0] MUST fail PBH rank test
    ok_wrong, rk_wrong = verify_pbh_controllability(G_path, [2], seed=1)
    assert not ok_wrong and rk_wrong < 4

    # 2. Directed out-star 0 -> {1, 2, 3, 4}: |M*| = 1 => N_D = 5 - 1 = 4 driver nodes
    G_star = nx.DiGraph([(0, 1), (0, 2), (0, 3), (0, 4)])
    res_star = analyze_structural_controllability(G_star)
    assert res_star.n_drivers == 4
    ok_star, rk_star = verify_pbh_controllability(G_star, res_star.driver_nodes, seed=2)
    assert ok_star and rk_star == 5
    # Adversarial check (no fake self-loops): fewer than N_D=4 drivers MUST strictly fail PBH!
    ok_under, rk_under = verify_pbh_controllability(G_star, res_star.driver_nodes[:-1], seed=2)
    assert not ok_under and rk_under == 4
    ok_one, rk_one = verify_pbh_controllability(G_star, [0], seed=2)
    assert not ok_one and rk_one == 2

    # 3. Disjoint cycles + string node IDs with underscores
    G_str = nx.DiGraph([("chem_O3", "chem_NO"), ("chem_NO", "chem_O3"), ("rad_OH", "rad_HO2"), ("rad_HO2", "rad_OH")])
    res_str = analyze_structural_controllability(G_str)
    assert res_str.n_drivers == 2
    assert res_str.minimum_input_channels == 1
    assert res_str.shared_input_sites == [res_str.driver_nodes]
    ok_one, rank_one = verify_pbh_controllability(G_str, [res_str.driver_nodes[0]], seed=3)
    assert not ok_one and rank_one == 3
    ok_str, rk_str = verify_pbh_controllability(G_str, res_str.driver_nodes, seed=3)
    assert ok_str and rk_str == 4
    assert verify_pbh_input_placement(G_str, res_str.shared_input_sites, seed=3) == (True, 4)


def test_minimum_dedicated_layout_resolves_matching_ties_and_handles_mixed_labels():
    # The cycle-cover matching leaves the downstream node unmatched and would
    # need one extra source site. The other maximum matching leaves 0 unmatched.
    edges = [(0, "partner"), ("partner", 0), ("partner", ("sink", 1))]
    for order in (edges, edges[::-1]):
        graph = nx.DiGraph(order)
        layout = analyze_structural_controllability(graph)
        assert layout.driver_nodes == [0]
        assert layout.source_components == [[0, "partner"]]
        assert len(layout.matching_pairs) == 2
        assert verify_pbh_controllability(graph, layout.driver_nodes) == (True, 3)


def test_structural_cascade_dedicated_layout_and_edge_counts():
    graph = reaction_cascade_network(n=32, branch_factor=2.2, seed=42)
    layout = analyze_structural_controllability(graph)
    assert (graph.number_of_edges(), len(layout.matching_pairs)) == (65, 21)
    assert layout.n_drivers == layout.minimum_input_channels == 11
    assert verify_pbh_controllability(graph, layout.driver_nodes) == (True, 32)
    counts = {kind: len(edges) for kind, edges in classify_control_edges(graph).items()}
    assert counts == {"critical": 5, "ordinary": 47, "redundant": 13}


def test_edge_classification_critical_ordinary_redundant_distinction():
    # 1. Path + feedforward shortcut: 0 -> 1 -> 2 plus chord 0 -> 2
    # Maximum matching is uniquely {(0->1), (1->2)} of size 2 (N_D = 1).
    # Forcing (0->2) leaves node 1 isolated => size 1 < 2 => (0->2) is REDUNDANT!
    # Removing (0->1) or (1->2) drops |M*| to 1 (N_D = 2 > 1) => both are CRITICAL!
    G_ff = nx.DiGraph([(0, 1), (1, 2), (0, 2)])
    ec_ff = classify_control_edges(G_ff)
    assert set(ec_ff["critical"]) == {(0, 1), (1, 2)}
    assert set(ec_ff["redundant"]) == {(0, 2)}
    assert ec_ff["ordinary"] == []

    # 2. Directed diamond: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
    # Two maximum matchings of size 2: {(0,1),(2,3)} and {(0,2),(1,3)} => all 4 edges are ORDINARY!
    G_diam = nx.DiGraph([(0, 1), (0, 2), (1, 3), (2, 3)])
    ec_diam = classify_control_edges(G_diam)
    assert ec_diam["critical"] == []
    assert ec_diam["redundant"] == []
    assert len(ec_diam["ordinary"]) == 4

    # 3. Directed 3-cycle: 0 -> 1 -> 2 -> 0 (|M*| = 3, N_D = 1)
    # Removing any single edge leaves a path with |M*| = 2 and N_D = max(1, 3-2) = 1 (unchanged!),
    # while every edge belongs to the unique perfect matching => all 3 edges are ORDINARY (LSB 2011)!
    G_cyc = nx.DiGraph([(0, 1), (1, 2), (2, 0)])
    ec_cyc = classify_control_edges(G_cyc)
    assert ec_cyc["critical"] == []
    assert ec_cyc["redundant"] == []
    assert len(ec_cyc["ordinary"]) == 3

    # 4. Full reaction cascade partition & N_D strict increase on every critical edge
    G = reaction_cascade_network(n=20, branch_factor=2.0, seed=12)
    classes = classify_control_edges(G)
    total_classified = len(classes["critical"]) + len(classes["ordinary"]) + len(classes["redundant"])
    assert total_classified == G.number_of_edges()

    base_nd = analyze_structural_controllability(G).minimum_input_channels
    for u, v in classes["critical"]:
        G_rem = G.copy()
        G_rem.remove_edge(u, v)
        assert analyze_structural_controllability(G_rem).minimum_input_channels > base_nd
    for u, v in classes["ordinary"] + classes["redundant"]:
        G_rem = G.copy()
        G_rem.remove_edge(u, v)
        assert analyze_structural_controllability(G_rem).minimum_input_channels == base_nd


def test_van_loan_vs_simpson_and_hurwitz_lyapunov_gramian():
    G = reaction_cascade_network(n=12, branch_factor=2.2, seed=21)
    A = build_continuous_system_matrix(G, damping=1.2, coupling=0.35)
    sc = analyze_structural_controllability(G)
    actuators, _ = select_actuators_submodular(A, budget_k=max(4, sc.n_drivers), mandatory_nodes=sc.driver_nodes)
    B = build_input_matrix(G.number_of_nodes(), actuators)

    # 1. Exact Van Loan block-exponential vs fine Simpson quadrature (including odd n_steps handling)
    W_vl = compute_controllability_gramian(A, B, horizon=1.5, method="van_loan")
    W_sp = compute_controllability_gramian(A, B, horizon=1.5, n_steps=121, method="simpson")
    assert np.allclose(W_vl, W_vl.T, atol=1e-12)
    assert np.allclose(W_vl, W_sp, rtol=1e-5, atol=1e-7)
    assert np.all(np.linalg.eigvalsh(W_vl) >= -1e-12)

    # 2. For Hurwitz A, finite-horizon W_c(T) converges to infinite-horizon Lyapunov solution as T -> inf
    W_inf, shift_zero = compute_infinite_horizon_gramian(A, B, stability_margin=1e-3)
    assert shift_zero == 0.0
    W_long = compute_controllability_gramian(A, B, horizon=30.0, method="van_loan")
    assert np.allclose(W_long, W_inf, rtol=1e-5, atol=1e-6)

    # 3. For unstable A (positive eigenvalue), Hurwitz stability shift guarantees PSD Lyapunov Gramian
    A_unstable = A + 3.0 * np.eye(A.shape[0])
    assert np.max(np.real(np.linalg.eigvals(A_unstable))) > 0.0
    A_shifted, shift_pos = stabilize_hurwitz_matrix(A_unstable, stability_margin=0.05)
    assert shift_pos > 0.0
    assert np.max(np.real(np.linalg.eigvals(A_shifted))) <= -0.05 + 1e-10
    W_inf_stab, _ = compute_infinite_horizon_gramian(A_unstable, B, stability_margin=0.05)
    assert np.all(np.linalg.eigvalsh(W_inf_stab) >= -1e-10)

    # 4. Pontryagin minimum-energy steering
    x0 = np.ones(G.number_of_nodes())
    xf = np.zeros(G.number_of_nodes())
    traj = minimum_energy_trajectory(A, B, x0, xf, horizon=1.5, n_steps=80, reg=0)
    assert traj["exact_target_reachable"]
    assert float(traj["energy"]) > 0.0
    assert np.allclose(traj["x"][-1], xf, atol=1e-7)


def test_celf_lazy_greedy_exact_equivalence_and_submodularity():
    G = reaction_cascade_network(n=18, branch_factor=2.2, seed=33)
    A = build_continuous_system_matrix(G, damping=1.1, coupling=0.35)

    sel_lazy, m_lazy = select_actuators_submodular(A, budget_k=7, horizon=2.0, reg=1e-5, use_lazy=True)
    sel_naive, m_naive = select_actuators_submodular(A, budget_k=7, horizon=2.0, reg=1e-5, use_lazy=False)

    assert sel_lazy == sel_naive
    assert abs(m_lazy["log_det"] - m_naive["log_det"]) < 1e-10
    assert m_lazy["normalized_log_det"] > 0.0
    assert m_lazy["num_marginal_evals"] < m_naive["num_marginal_evals"]

    # Verify diminishing marginal returns: F(S_sub u {v}) - F(S_sub) >= F(S_sup u {v}) - F(S_sup)
    S_sub = [0, 2]
    S_sup = [0, 2, 5, 8]
    cand = 11
    reg = 1e-5
    I_n = np.eye(A.shape[0])

    def f_norm(subset: list[int]) -> float:
        W = compute_controllability_gramian(A, build_input_matrix(A.shape[0], subset), horizon=2.0)
        return float(np.linalg.slogdet(W + reg * I_n)[1] - A.shape[0] * np.log(reg))

    gain_sub = f_norm(S_sub + [cand]) - f_norm(S_sub)
    gain_sup = f_norm(S_sup + [cand]) - f_norm(S_sup)
    assert gain_sub >= gain_sup - 1e-10


def test_seeded_neural_pinning_cost_against_care():
    G = reaction_cascade_network(n=16, branch_factor=2.1, seed=99)
    A = build_continuous_system_matrix(G, damping=0.55, coupling=0.35)
    sc = analyze_structural_controllability(G)
    pinned, _ = select_actuators_submodular(A, budget_k=5, mandatory_nodes=sc.driver_nodes[:2])
    B = build_input_matrix(G.number_of_nodes(), pinned)
    # CARE-LQR using the same running (q_weight=3.0, r_weight=0.08) weights
    K_lqr, _ = solve_lqr_controller(A, B, q_weight=3.0, r_weight=0.08)

    x0 = np.linspace(1.2, 2.4, G.number_of_nodes())
    gnn_ctrl, losses = train_neural_pinning_controller(
        G=G,
        A=A,
        pinned_nodes=pinned,
        driver_nodes=sc.driver_nodes,
        x0=x0,
        horizon=3.0,
        n_steps=50,
        epochs=35,
        lr=0.035,
        q_state=3.0,
        r_ctrl=0.08,
        q_term=12.0,
        n_train_inits=6,
        seed=99,
    )
    assert losses[-1] < losses[0]

    sim_unc = simulate_nonlinear_pinning(A, x0, pinned, mode="uncontrolled", horizon=3.0, n_steps=50)
    sim_lqr = simulate_nonlinear_pinning(A, x0, pinned, mode="lqr_pinning", K_lqr=K_lqr, horizon=3.0, n_steps=50)
    sim_gnn = simulate_nonlinear_pinning(A, x0, pinned, mode="gnn_pinning", gnn_controller=gnn_ctrl, horizon=3.0, n_steps=50)
    assert abs(float(sim_gnn["total_j"]) - losses[-1]) < 1e-3
    assert float(sim_gnn["total_j"]) < float(sim_lqr["total_j"])
    assert float(sim_gnn["ise_state"]) < 0.45 * float(sim_unc["ise_state"])

    # Unseen out-of-sample initial condition check
    rng = np.random.default_rng(777)
    x0_unseen = rng.uniform(1.1, 2.5, size=G.number_of_nodes())
    sim_lqr_u = simulate_nonlinear_pinning(A, x0_unseen, pinned, mode="lqr_pinning", K_lqr=K_lqr, horizon=3.0, n_steps=50)
    sim_gnn_u = simulate_nonlinear_pinning(A, x0_unseen, pinned, mode="gnn_pinning", gnn_controller=gnn_ctrl, horizon=3.0, n_steps=50)
    assert float(sim_gnn_u["total_j"]) < float(sim_lqr_u["total_j"])
