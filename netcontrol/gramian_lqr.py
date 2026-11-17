"""
Controllability Gramian, Minimum-Energy Optimal Control Trajectory,
Submodular Greedy Actuator Selection, and Infinite-Horizon LQR.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from scipy.linalg import expm, solve_continuous_are


def build_input_matrix(n_nodes: int, actuator_nodes: List[int]) -> np.ndarray:
    """Create B matrix in R^{N x M} with unit columns at selected actuator nodes."""
    m = len(actuator_nodes)
    B = np.zeros((n_nodes, m), dtype=float)
    for j, node in enumerate(actuator_nodes):
        B[int(node), j] = 1.0
    return B


def compute_controllability_gramian(
    A: np.ndarray,
    B: np.ndarray,
    horizon: float = 2.0,
    n_steps: int = 80,
) -> np.ndarray:
    """
    Compute the finite-horizon Controllability Gramian:
        W_c(T) = int_0^T exp(A t) B B^T exp(A^T t) dt
    using Simpson's rule over matrix exponentials for numerical stability on general directed graphs.
    """
    n = A.shape[0]
    if B.size == 0:
        return np.zeros((n, n), dtype=float)

    ts = np.linspace(0.0, horizon, n_steps + 1)
    dt = horizon / float(n_steps)
    BBT = B @ B.T

    W = np.zeros((n, n), dtype=float)
    for idx, t in enumerate(ts):
        weight = 1.0
        if 0 < idx < n_steps:
            weight = 4.0 if (idx % 2 == 1) else 2.0
        eat = expm(A * t)
        W += weight * (eat @ BBT @ eat.T)

    W *= dt / 3.0
    # Symmetrize numerical round-off
    W = 0.5 * (W + W.T)
    return W


def gramian_metrics(W: np.ndarray, reg: float = 1e-6) -> Dict[str, float]:
    """Compute energy-related spectral metrics of the Controllability Gramian."""
    eigvals = np.linalg.eigvalsh(W)
    eigvals_clipped = np.maximum(eigvals, 0.0)
    lam_min = float(eigvals_clipped[0])
    lam_max = float(eigvals_clipped[-1])
    trace_val = float(np.sum(eigvals_clipped))
    log_det = float(np.sum(np.log(eigvals_clipped + reg)))
    avg_energy = float(np.mean(1.0 / (eigvals_clipped + reg)))
    cond_num = float((lam_max + reg) / (lam_min + reg))
    return {
        "lambda_min": lam_min,
        "lambda_max": lam_max,
        "trace": trace_val,
        "log_det": log_det,
        "mean_inverse_eig": avg_energy,
        "condition_number": cond_num,
    }


def minimum_energy_trajectory(
    A: np.ndarray,
    B: np.ndarray,
    x0: np.ndarray,
    xf: np.ndarray,
    horizon: float = 2.0,
    n_steps: int = 80,
    reg: float = 1e-6,
) -> Dict[str, np.ndarray | float]:
    """
    Synthesize the Pontryagin minimum-energy open-loop control input u*(t) steering
    x(0) = x0 to x(T) = xf in horizon T, and simulate the state trajectory x(t).
    """
    n = A.shape[0]
    W = compute_controllability_gramian(A, B, horizon=horizon, n_steps=n_steps)
    eAT = expm(A * horizon)
    delta = xf - eAT @ x0

    W_reg = W + reg * np.eye(n)
    v_opt = np.linalg.solve(W_reg, delta)
    min_energy = float(delta.T @ v_opt)

    ts = np.linspace(0.0, horizon, n_steps + 1)
    dt = horizon / float(n_steps)

    us = np.zeros((n_steps + 1, B.shape[1]), dtype=float)
    for idx, t in enumerate(ts):
        eat_rem = expm(A.T * (horizon - t))
        us[idx] = B.T @ (eat_rem @ v_opt)

    # Forward RK4 integration of dx/dt = A x + B u(t)
    xs = np.zeros((n_steps + 1, n), dtype=float)
    xs[0] = x0.copy()
    for idx in range(n_steps):
        u_cur = us[idx]
        u_next = us[idx + 1]
        u_mid = 0.5 * (u_cur + u_next)
        x_cur = xs[idx]

        k1 = A @ x_cur + B @ u_cur
        k2 = A @ (x_cur + 0.5 * dt * k1) + B @ u_mid
        k3 = A @ (x_cur + 0.5 * dt * k2) + B @ u_mid
        k4 = A @ (x_cur + dt * k3) + B @ u_next
        xs[idx + 1] = x_cur + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    terminal_error = float(np.linalg.norm(xs[-1] - xf))
    return {
        "t": ts,
        "x": xs,
        "u": us,
        "energy": min_energy,
        "terminal_error": terminal_error,
        "gramian_metrics": gramian_metrics(W, reg=reg),
    }


def select_actuators_submodular(
    A: np.ndarray,
    budget_k: int,
    horizon: float = 2.0,
    reg: float = 1e-5,
    mandatory_nodes: List[int] | None = None,
) -> Tuple[List[int], Dict[str, float]]:
    """
    Submodular greedy actuator selection maximizing the Volumetric Controllability
    objective f(S) = log det(W_c(S) + reg * I).

    Because W_c(S) = sum_{i in S} W_c({i}) is modular in the actuator set S,
    f(S) = log det(sum_{i in S} W_i + reg * I) is monotone submodular and enjoys
    a (1 - 1/e) optimality guarantee.
    """
    n = A.shape[0]
    budget_k = min(max(1, budget_k), n)

    # Precompute single-node Gramians W_i
    single_gramians = []
    for i in range(n):
        B_i = build_input_matrix(n, [i])
        single_gramians.append(compute_controllability_gramian(A, B_i, horizon=horizon, n_steps=50))

    selected: List[int] = []
    W_accum = reg * np.eye(n)

    if mandatory_nodes:
        for node in mandatory_nodes:
            if len(selected) < budget_k and node not in selected:
                selected.append(int(node))
                W_accum = W_accum + single_gramians[int(node)]

    remaining = [i for i in range(n) if i not in selected]
    while len(selected) < budget_k and remaining:
        best_node = remaining[0]
        best_score = -np.inf
        for cand in remaining:
            sign, logdet = np.linalg.slogdet(W_accum + single_gramians[cand])
            score = float(logdet) if sign > 0 else -1e9
            if score > best_score:
                best_score = score
                best_node = cand
        selected.append(best_node)
        W_accum = W_accum + single_gramians[best_node]
        remaining.remove(best_node)

    W_final = W_accum - reg * np.eye(n)
    return selected, gramian_metrics(W_final, reg=reg)


def select_actuators_baseline(
    G: nx.DiGraph,
    A: np.ndarray,
    budget_k: int,
    strategy: str = "out_degree",
    horizon: float = 2.0,
    reg: float = 1e-5,
    seed: int = 42,
) -> Tuple[List[int], Dict[str, float]]:
    """
    Select k actuator nodes using classical network centrality heuristics
    ('out_degree', 'betweenness', 'pagerank', 'random') and evaluate Gramian metrics.
    """
    n = G.number_of_nodes()
    budget_k = min(max(1, budget_k), n)
    nodes = sorted(G.nodes())

    if strategy == "out_degree":
        scores = dict(G.out_degree(weight="weight"))
        ranked = sorted(nodes, key=lambda u: scores[u], reverse=True)
        selected = ranked[:budget_k]
    elif strategy == "betweenness":
        scores = nx.betweenness_centrality(G)
        ranked = sorted(nodes, key=lambda u: scores[u], reverse=True)
        selected = ranked[:budget_k]
    elif strategy == "pagerank":
        scores = nx.pagerank(G.reverse(copy=True))
        ranked = sorted(nodes, key=lambda u: scores[u], reverse=True)
        selected = ranked[:budget_k]
    elif strategy == "random":
        rng = np.random.default_rng(seed)
        selected = rng.choice(nodes, size=budget_k, replace=False).tolist()
    else:
        raise ValueError(f"Unknown baseline strategy: {strategy}")

    B = build_input_matrix(n, selected)
    W = compute_controllability_gramian(A, B, horizon=horizon, n_steps=50)
    return [int(x) for x in selected], gramian_metrics(W, reg=reg)


def solve_lqr_controller(
    A: np.ndarray,
    B: np.ndarray,
    q_weight: float = 2.0,
    r_weight: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the continuous-time Algebraic Riccati Equation (CARE):
        A^T P + P A - P B R^{-1} B^T P + Q = 0
    and return feedback gain matrix K = R^{-1} B^T P (where u = -K x) and cost matrix P.
    """
    n = A.shape[0]
    m = B.shape[1]
    Q = q_weight * np.eye(n)
    R = r_weight * np.eye(m)
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return K, P
