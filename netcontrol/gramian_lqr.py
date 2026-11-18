"""
Controllability Gramian (Exact Van Loan Finite-Horizon & Hurwitz-Shifted Infinite-Horizon
Lyapunov), Minimum-Energy Optimal Control Trajectory, CELF Lazy Greedy Submodular Actuator
Selection, and Continuous-Time Algebraic Riccati LQR.
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Literal, Tuple

import networkx as nx
import numpy as np
from scipy.linalg import expm, solve_continuous_are, solve_continuous_lyapunov


def build_input_matrix(n_nodes: int, actuator_nodes: List[int]) -> np.ndarray:
    """Create B matrix in R^{N x M} with unit columns at selected actuator nodes."""
    m = len(actuator_nodes)
    B = np.zeros((n_nodes, m), dtype=float)
    for j, node in enumerate(actuator_nodes):
        B[int(node), j] = 1.0
    return B


def stabilize_hurwitz_matrix(
    A: np.ndarray,
    stability_margin: float = 1e-2,
) -> Tuple[np.ndarray, float]:
    """
    Ensure continuous-time Hurwitz stability (`max_k Re lambda_k(A_shift) <= -stability_margin`)
    prior to solving infinite-horizon Lyapunov equations `A W_c + W_c A^T + B B^T = 0`.

    Returns `(A_shifted, shift_amount)` where `A_shifted = A - shift_amount * I`.
    If `A` is already strictly Hurwitz with spectral abscissa `<= -stability_margin`,
    `shift_amount = 0.0` and `A` is returned unshifted.
    """
    if A.size == 0:
        return A.copy(), 0.0
    spectral_abscissa = float(np.max(np.real(np.linalg.eigvals(A))))
    if spectral_abscissa > -stability_margin:
        shift = spectral_abscissa + stability_margin
        return A - shift * np.eye(A.shape[0], dtype=float), float(shift)
    return A.copy(), 0.0


def compute_infinite_horizon_gramian(
    A: np.ndarray,
    B: np.ndarray,
    stability_margin: float = 1e-2,
) -> Tuple[np.ndarray, float]:
    """
    Compute the infinite-horizon Controllability Gramian solving the continuous-time
    Lyapunov equation:
        A_s W_c + W_c A_s^T + B B^T = 0
    after applying a minimal Hurwitz stability shift `A_s = A - shift * I` if `A` is
    marginally stable or unstable (since `W_c` is positive semidefinite iff `A_s` is Hurwitz).
    """
    n = A.shape[0]
    if n == 0 or B.size == 0:
        return np.zeros((n, n), dtype=float), 0.0
    A_s, shift = stabilize_hurwitz_matrix(A, stability_margin=stability_margin)
    W = solve_continuous_lyapunov(A_s, -(B @ B.T))
    W = 0.5 * (W + W.T)
    return W, shift


def compute_controllability_gramian(
    A: np.ndarray,
    B: np.ndarray,
    horizon: float = 2.0,
    n_steps: int = 80,
    method: Literal["van_loan", "simpson"] = "van_loan",
) -> np.ndarray:
    """
    Compute the finite-horizon Controllability Gramian:
        W_c(T) = int_0^T exp(A t) B B^T exp(A^T t) dt

    By default (`method="van_loan"`), uses Van Loan's (IEEE TAC, 1978) exact 2N x 2N block
    matrix exponential identity for machine-precision (~1e-15) quadrature-free evaluation:
        exp([[ -A, B B^T ], [ 0, A^T ]] * T) = [[ exp(-A T), exp(-A T) W_c(T) ], [ 0, exp(A^T T) ]]
        => W_c(T) = exp(A^T T)^T @ M_{12}.
    Also supports composite Simpson's 1/3 rule (`method="simpson"`, enforcing even `n_steps`).
    """
    n = A.shape[0]
    if n == 0 or B.size == 0 or horizon <= 0.0:
        return np.zeros((n, n), dtype=float)

    BBT = B @ B.T
    if method == "van_loan":
        # Van Loan (IEEE TAC, 1978, Algorithm 1): Scaling-and-squaring block exponential
        # Keeps ||A * h||_1 <= 0.5 on the base step to prevent exp(-A*T) conditioning loss
        norm_A = float(np.linalg.norm(A, ord=1))
        n_squarings = max(0, int(np.ceil(np.log2(max(1.0, 2.0 * norm_A * float(horizon))))))
        h = float(horizon) / float(2 ** n_squarings)

        block = np.zeros((2 * n, 2 * n), dtype=float)
        block[:n, :n] = -A
        block[:n, n:] = BBT
        block[n:, n:] = A.T
        exp_block = expm(block * h)
        m12 = exp_block[:n, n:]
        phi = exp_block[n:, n:].T  # exp(A * h)
        W = phi @ m12
        W = 0.5 * (W + W.T)

        for _ in range(n_squarings):
            W = W + phi @ W @ phi.T
            W = 0.5 * (W + W.T)
            phi = phi @ phi

        return W

    if method == "simpson":
        steps = max(2, int(n_steps))
        if steps % 2 == 1:
            steps += 1
        ts = np.linspace(0.0, horizon, steps + 1)
        dt = horizon / float(steps)

        W = np.zeros((n, n), dtype=float)
        for idx, t in enumerate(ts):
            weight = 1.0
            if 0 < idx < steps:
                weight = 4.0 if (idx % 2 == 1) else 2.0
            eat = expm(A * t)
            W += weight * (eat @ BBT @ eat.T)

        W *= dt / 3.0
        return 0.5 * (W + W.T)

    raise ValueError(f"Unsupported Gramian integration method: {method}")


def gramian_metrics(W: np.ndarray, reg: float = 1e-6) -> Dict[str, float]:
    """
    Compute energy-related spectral metrics of the Controllability Gramian, including both
    raw volumetric `log_det = log det(W_c + reg * I)` and the normalized non-negative
    monotone submodular gain `normalized_log_det = log det(I + reg^{-1} W_c) >= 0`.
    """
    n = W.shape[0]
    eigvals = np.linalg.eigvalsh(W)
    eigvals_clipped = np.maximum(eigvals, 0.0)
    lam_min = float(eigvals_clipped[0])
    lam_max = float(eigvals_clipped[-1])
    trace_val = float(np.sum(eigvals_clipped))
    log_det = float(np.sum(np.log(eigvals_clipped + reg)))
    normalized_log_det = float(log_det - n * np.log(reg))
    avg_energy = float(np.mean(1.0 / (eigvals_clipped + reg)))
    cond_num = float((lam_max + reg) / (lam_min + reg))
    return {
        "lambda_min": lam_min,
        "lambda_max": lam_max,
        "trace": trace_val,
        "log_det": log_det,
        "normalized_log_det": normalized_log_det,
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
) -> Dict[str, np.ndarray | float | Dict[str, float]]:
    """
    Synthesize the Pontryagin minimum-energy open-loop control input u*(t) steering
    x(0) = x0 to x(T) = xf in horizon T, and simulate the state trajectory x(t) via
    exact-midpoint 4th-order Runge-Kutta integration.
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

    # Forward RK4 integration of dx/dt = A x + B u*(t) with exact midpoint u*(t + 0.5*dt)
    xs = np.zeros((n_steps + 1, n), dtype=float)
    xs[0] = x0.copy()
    for idx in range(n_steps):
        u_cur = us[idx]
        u_next = us[idx + 1]
        t_mid = ts[idx] + 0.5 * dt
        u_mid = B.T @ (expm(A.T * (horizon - t_mid)) @ v_opt)
        x_cur = xs[idx]

        k1 = A @ x_cur + B @ u_cur
        k2 = A @ (x_cur + 0.5 * dt * k1) + B @ u_mid
        k3 = A @ (x_cur + 0.5 * dt * k2) + B @ u_mid
        k4 = A @ (x_cur + dt * k3) + B @ u_next
        xs[idx + 1] = x_cur + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

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
    use_lazy: bool = True,
) -> Tuple[List[int], Dict[str, float]]:
    """
    Submodular greedy actuator selection maximizing the normalized Volumetric Controllability
    objective:
        tilde{F}(S) = log det(sum_{i in S} W_i(T) + reg * I) - log det(reg * I)
                    = log det(I + reg^{-1} W_c(S, T)) >= 0,   tilde{F}(emptyset) = 0.

    Because W_c(S, T) = sum_{i in S} W_i(T) is modular in S and X -> log det(X) is operator
    concave on the positive-definite cone, tilde{F}(S) is normalized, non-negative, monotone
    non-decreasing, and submodular (Summers, Cortesi & Lygeros, IEEE TCNS 2016), guaranteeing
    tilde{F}(S_greedy) >= (1 - 1/e) tilde{F}(S*).

    Uses CELF Lazy Greedy acceleration (Minoux, 1978) with a max-heap priority queue of
    marginal-gain upper bounds `Delta(i | S)` when `use_lazy=True`, returning the exact same
    maximizer as exhaustive greedy (`use_lazy=False`) in far fewer determinant evaluations.
    """
    n = A.shape[0]
    budget_k = min(max(1, budget_k), n)

    single_gramians = [
        compute_controllability_gramian(A, build_input_matrix(n, [i]), horizon=horizon)
        for i in range(n)
    ]

    selected: List[int] = []
    W_accum = reg * np.eye(n, dtype=float)

    if mandatory_nodes:
        for node in mandatory_nodes:
            if len(selected) < budget_k and int(node) not in selected:
                selected.append(int(node))
                W_accum = W_accum + single_gramians[int(node)]

    def _logdet_psd(mat: np.ndarray) -> float:
        sign, val = np.linalg.slogdet(mat)
        return float(val) if sign > 0 else -1e9

    curr_val = _logdet_psd(W_accum)
    remaining = [i for i in range(n) if i not in selected]
    n_evals = 0

    if use_lazy and remaining and len(selected) < budget_k:
        # Min-heap over (-marginal_gain_upper_bound, candidate_node_idx, step_evaluated)
        heap: List[Tuple[float, int, int]] = []
        for cand in remaining:
            gain = _logdet_psd(W_accum + single_gramians[cand]) - curr_val
            n_evals += 1
            heapq.heappush(heap, (-gain, cand, 0))

        step_idx = 0
        while len(selected) < budget_k and heap:
            neg_gain, cand, eval_step = heapq.heappop(heap)
            if eval_step == step_idx:
                # Upper bound is exact at current step -> guaranteed argmax by submodularity!
                selected.append(cand)
                W_accum = W_accum + single_gramians[cand]
                curr_val = curr_val - neg_gain
                step_idx += 1
            else:
                new_val = _logdet_psd(W_accum + single_gramians[cand])
                n_evals += 1
                exact_gain = new_val - curr_val
                heapq.heappush(heap, (-exact_gain, cand, step_idx))
    else:
        while len(selected) < budget_k and remaining:
            best_node = remaining[0]
            best_val = -np.inf
            for cand in remaining:
                val = _logdet_psd(W_accum + single_gramians[cand])
                n_evals += 1
                if val > best_val:
                    best_val = val
                    best_node = cand
            selected.append(best_node)
            W_accum = W_accum + single_gramians[best_node]
            curr_val = best_val
            remaining.remove(best_node)

    W_final = W_accum - reg * np.eye(n, dtype=float)
    metrics = gramian_metrics(W_final, reg=reg)
    metrics["num_marginal_evals"] = float(n_evals)
    return selected, metrics


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
    Note: 'pagerank' runs on G.reverse() to rank upstream broadcast influence authority.
    """
    n = G.number_of_nodes()
    budget_k = min(max(1, budget_k), n)
    nodes = sorted(G.nodes())

    if strategy == "out_degree":
        scores = dict(G.out_degree(weight="weight"))
        ranked = sorted(nodes, key=lambda u: (-scores[u], u))
        selected = ranked[:budget_k]
    elif strategy == "betweenness":
        scores = nx.betweenness_centrality(G)
        ranked = sorted(nodes, key=lambda u: (-scores[u], u))
        selected = ranked[:budget_k]
    elif strategy == "pagerank":
        scores = nx.pagerank(G.reverse(copy=True))
        ranked = sorted(nodes, key=lambda u: (-scores[u], u))
        selected = ranked[:budget_k]
    elif strategy == "random":
        rng = np.random.default_rng(seed)
        selected = rng.choice(nodes, size=budget_k, replace=False).tolist()
    else:
        raise ValueError(f"Unknown baseline strategy: {strategy}")

    B = build_input_matrix(n, selected)
    W = compute_controllability_gramian(A, B, horizon=horizon)
    return [int(x) for x in selected], gramian_metrics(W, reg=reg)


def solve_lqr_controller(
    A: np.ndarray,
    B: np.ndarray,
    q_weight: float = 3.0,
    r_weight: float = 0.08,
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
