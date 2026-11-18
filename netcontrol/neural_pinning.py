"""
Nonlinear Complex Network Pinning Control & Differentiable Graph Neural Controller.

The two synthetic dynamics regimes combine a linear network term with a smooth
node-wise saturation or a cubic term. States are signed coordinates; the simulated
cascade does not enforce the positivity required of chemical concentrations.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import trapezoid


class GraphNeuralPinningController(nn.Module):
    """
    Topology-aware Graph Message-Passing Neural Pinning Controller.

    Given current network state error e(t) = x(t) - x_target of shape `(N,)` or `(B, N)`
    and static structural node features (normalized in/out strength, driver indicator),
    computes localized message passing over the directed network and outputs bounded
    control actions u_P(t) on the pinned actuator subset P.
    """

    def __init__(
        self,
        n_nodes: int,
        pinned_nodes: List[int],
        adj_norm: torch.Tensor,
        struct_feats: torch.Tensor,
        hidden_dim: int = 32,
        u_max: float = 6.0,
    ) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.pinned_nodes = list(pinned_nodes)
        self.u_max = float(u_max)

        self.register_buffer("adj_norm", adj_norm)
        self.register_buffer("struct_feats", struct_feats)
        self.register_buffer("pinned_idx", torch.tensor(self.pinned_nodes, dtype=torch.long))

        in_dim = 2 + struct_feats.shape[1]  # [e_i, tanh(e_i), struct_feats...]
        self.node_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
        )
        self.msg_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.self_layer = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU()
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Direct proportional gain residual initialized for stable warm start
        self.kp_residual = nn.Parameter(torch.full((len(self.pinned_nodes),), 2.2))

    def forward(self, state_error: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_error: Tensor of shape `(N,)` or `(B, N)` representing `x(t) - x_target`.
        Returns:
            u_pinned: Tensor of shape `(|P|,)` or `(B, |P|)` with control signals at pinned nodes.
        """
        squeeze_batch = False
        if state_error.ndim == 1:
            state_error = state_error.unsqueeze(0)
            squeeze_batch = True

        bsz = state_error.shape[0]
        e_col = state_error.unsqueeze(-1)  # (B, N, 1)
        s_feats = self.struct_feats.unsqueeze(0).expand(bsz, -1, -1)  # (B, N, F)
        x_in = torch.cat([e_col, torch.tanh(e_col), s_feats], dim=-1)  # (B, N, 2+F)
        h0 = self.node_encoder(x_in)  # (B, N, H)
        h_msg = torch.matmul(self.adj_norm.unsqueeze(0), self.msg_layer(h0))  # (B, N, H)
        h1 = self.act(self.self_layer(h0) + h_msg)

        h_cat = torch.cat([h0, h1], dim=-1)  # (B, N, 2H)
        h_pinned = h_cat[:, self.pinned_idx, :]  # (B, |P|, 2H)
        e_pinned = state_error[:, self.pinned_idx]  # (B, |P|)

        u_nn = self.readout(h_pinned).squeeze(-1)  # (B, |P|)
        u_raw = -torch.relu(self.kp_residual).unsqueeze(0) * e_pinned + u_nn * torch.tanh(
            e_pinned.abs()
        )
        u_out = self.u_max * torch.tanh(u_raw / self.u_max)
        return u_out.squeeze(0) if squeeze_batch else u_out


def build_structural_features(
    G: nx.DiGraph, driver_nodes: List[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute normalized adjacency matrix and structural node features for GNN controller."""
    n = G.number_of_nodes()
    nodes = list(range(n))
    W = nx.to_numpy_array(G, nodelist=nodes, weight="weight")
    # Row-normalize incoming + outgoing neighborhood for message passing
    A_sym = W + W.T + np.eye(n)
    deg = A_sym.sum(axis=1, keepdims=True)
    adj_norm = torch.tensor(A_sym / np.maximum(deg, 1e-6), dtype=torch.float32)

    in_deg = np.array([G.in_degree(u, weight="weight") for u in nodes], dtype=np.float32)
    out_deg = np.array([G.out_degree(u, weight="weight") for u in nodes], dtype=np.float32)
    in_deg /= max(float(in_deg.max()), 1.0)
    out_deg /= max(float(out_deg.max()), 1.0)
    driver_set = set(driver_nodes)
    is_driver = np.array([1.0 if u in driver_set else 0.0 for u in nodes], dtype=np.float32)

    feats = np.stack([in_deg, out_deg, is_driver], axis=1)
    return adj_norm, torch.tensor(feats, dtype=torch.float32)


def _nonlinear_rhs_torch(
    x: torch.Tensor,
    u_full: torch.Tensor,
    A_lin: torch.Tensor,
    regime: str = "pollutant_cascade",
) -> torch.Tensor:
    """
    Differentiable right-hand side dx/dt for nonlinear coupled network dynamics.
    Supports `x` of shape `(N,)` or `(B, N)`. The target equilibrium is `x_target = 0`.
    """
    lin_term = torch.matmul(x, A_lin.T)
    if regime == "pollutant_cascade":
        nonlinear_source = 0.85 * (x ** 2) / (1.0 + x ** 2) + 0.25 * torch.sin(1.5 * x) * torch.tanh(x)
        return lin_term + nonlinear_source + u_full
    elif regime == "bistable_sync":
        cubic_term = 1.15 * x * (1.0 - (x / 2.2) ** 2)
        return lin_term + cubic_term + u_full
    else:
        raise ValueError(f"Unknown dynamics regime: {regime}")


def rollout_pinning_torch(
    A_lin: torch.Tensor,
    x0: torch.Tensor,
    pinned_nodes: List[int],
    controller_fn,
    horizon: float = 4.0,
    n_steps: int = 80,
    regime: str = "pollutant_cascade",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable RK4 rollout of nonlinear pinning control (supports `(N,)` or `(B, N)`)."""
    n = A_lin.shape[0]
    dt = horizon / float(n_steps)
    ts = torch.linspace(0.0, horizon, n_steps + 1, dtype=x0.dtype, device=x0.device)
    pin_idx = torch.tensor(pinned_nodes, dtype=torch.long, device=x0.device)

    xs = [x0]
    us = []
    x_cur = x0
    for _ in range(n_steps):
        u_pin = controller_fn(x_cur)
        us.append(u_pin)
        if x_cur.ndim == 1:
            u_full = torch.zeros(n, dtype=x_cur.dtype, device=x_cur.device)
            u_full[pin_idx] = u_pin
        else:
            u_full = torch.zeros(x_cur.shape[0], n, dtype=x_cur.dtype, device=x_cur.device)
            u_full[:, pin_idx] = u_pin

        k1 = _nonlinear_rhs_torch(x_cur, u_full, A_lin, regime=regime)
        k2 = _nonlinear_rhs_torch(x_cur + 0.5 * dt * k1, u_full, A_lin, regime=regime)
        k3 = _nonlinear_rhs_torch(x_cur + 0.5 * dt * k2, u_full, A_lin, regime=regime)
        k4 = _nonlinear_rhs_torch(x_cur + dt * k3, u_full, A_lin, regime=regime)
        x_cur = x_cur + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        xs.append(x_cur)

    us.append(controller_fn(x_cur))
    return ts, torch.stack(xs, dim=0), torch.stack(us, dim=0)


def _compute_bolza_cost_torch(
    ts: torch.Tensor,
    xs: torch.Tensor,
    us: torch.Tensor,
    q_state: float,
    r_ctrl: float,
    q_term: float,
) -> torch.Tensor:
    """
    Trapezoidal estimate of the Bolza objective from sampled states and commands:
        J = int_0^T (q_state ||x(t)||_2^2 + r_ctrl ||u(t)||_2^2) dt + q_term ||x(T)||_2^2
    averaged across batch dimension if `xs` has shape `(T+1, B, N)`.
    """
    x_sq = torch.sum(xs ** 2, dim=-1)  # (T+1,) or (T+1, B)
    u_sq = torch.sum(us ** 2, dim=-1)  # (T+1,) or (T+1, B)
    ise = torch.trapezoid(x_sq, ts, dim=0)
    ctrl_e = torch.trapezoid(u_sq, ts, dim=0)
    term = x_sq[-1]
    total = q_state * ise + r_ctrl * ctrl_e + q_term * term
    return torch.mean(total)


def training_initial_states(x0: np.ndarray, count: int, seed: int) -> torch.Tensor:
    """Build a fixed ensemble, including the reference initial condition."""
    if count < 1:
        raise ValueError("At least one training initial condition is required")
    ref = np.asarray(x0, dtype=np.float32)
    lo = max(0.3, float(ref.min()) * 0.75)
    hi = max(1.0, float(ref.max()) * 1.20)
    rng = np.random.default_rng(seed)
    others = rng.uniform(lo, hi, size=(count - 1, ref.size)).astype(np.float32)
    return torch.from_numpy(np.vstack([ref, others]))


def fit_pinning_feedback(
    controller: nn.Module, A: torch.Tensor, pinned_nodes: List[int], train_x: torch.Tensor,
    horizon: float, n_steps: int, epochs: int, lr: float, q_state: float, r_ctrl: float,
    q_term: float, regime: str,
) -> List[float]:
    """Fit a bounded feedback model on the fixed ensemble; track its reference loss."""
    optimizer = torch.optim.Adam(controller.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        rollout = rollout_pinning_torch(A, train_x, pinned_nodes, controller, horizon, n_steps, regime)
        loss = _compute_bolza_cost_torch(*rollout, q_state, r_ctrl, q_term)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm=5.0)
        optimizer.step()
        with torch.no_grad():
            ref = rollout_pinning_torch(A, train_x[0], pinned_nodes, controller, horizon, n_steps, regime)
            losses.append(float(_compute_bolza_cost_torch(*ref, q_state, r_ctrl, q_term).item()))
    controller.eval()
    return losses


def train_neural_pinning_controller(
    G: nx.DiGraph,
    A: np.ndarray,
    pinned_nodes: List[int],
    driver_nodes: List[int],
    x0: np.ndarray,
    horizon: float = 4.0,
    n_steps: int = 80,
    epochs: int = 50,
    lr: float = 0.035,
    q_state: float = 3.0,
    r_ctrl: float = 0.08,
    q_term: float = 12.0,
    n_train_inits: int = 8,
    regime: str = "pollutant_cascade",
    seed: int = 42,
) -> Tuple[GraphNeuralPinningController, List[float]]:
    """Train bounded one-hop neural feedback by differentiating the simulated trajectories."""
    torch.manual_seed(seed)
    adj, features = build_structural_features(G, driver_nodes)
    controller = GraphNeuralPinningController(G.number_of_nodes(), pinned_nodes, adj, features)
    train_x = training_initial_states(x0, n_train_inits, seed)
    losses = fit_pinning_feedback(controller, torch.as_tensor(A, dtype=torch.float32), pinned_nodes,
                                  train_x, horizon, n_steps, epochs, lr, q_state, r_ctrl, q_term, regime)
    return controller, losses


def sampled_pinning_metrics(t, xs, us, q_state, r_ctrl, q_term):
    """Score (time, batch, channel) trajectories using the benchmark's sampled-cost convention.

    RK4 holds a command fixed during each step. For historical comparability the
    reported cost samples the policy at both endpoints and uses trapezoidal sums.
    'applied_input_energy' separately integrates the zero-order-held commands exactly.
    """
    x = np.asarray(xs, dtype=np.float64)
    u = np.asarray(us, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    x_sq = (x * x).sum(axis=-1)
    u_sq = (u * u).sum(axis=-1)
    ise = trapezoid(x_sq, t, axis=0)
    energy = trapezoid(u_sq, t, axis=0)
    applied_energy = (np.diff(t)[:, None] * u_sq[:-1]).sum(axis=0)
    running = q_state * ise + r_ctrl * energy
    held_below = np.logical_and.accumulate((x_sq <= 0.15 ** 2 * x_sq[0])[::-1], axis=0)[::-1]
    settled = np.where(held_below.any(axis=0), t[held_below.argmax(axis=0)], np.nan)
    return {
        "ise_state": ise, "control_energy": energy, "applied_input_energy": applied_energy,
        "running_j": running, "total_j": running + q_term * x_sq[-1],
        "terminal_norm": np.sqrt(x_sq[-1]), "minimum_coordinate": x.min(axis=(0, 2)),
        "settling_time": settled,
    }


def simulate_nonlinear_pinning(
    A: np.ndarray,
    x0: np.ndarray,
    pinned_nodes: List[int],
    mode: str = "uncontrolled",
    K_lqr: np.ndarray | None = None,
    gnn_controller: GraphNeuralPinningController | None = None,
    kp_linear: float = 2.5,
    u_max: float = 6.0,
    horizon: float = 4.0,
    n_steps: int = 80,
    q_state: float = 3.0,
    r_ctrl: float = 0.08,
    q_term: float = 12.0,
    regime: str = "pollutant_cascade",
) -> Dict[str, np.ndarray | float]:
    """
    Evaluate a pinning control law on the nonlinear complex network and return
    trajectories and sampled trapezoidal Bolza cost metrics:
        J = int_0^T (q_state ||x(t)||_2^2 + r_ctrl ||u(t)||_2^2) dt + q_term ||x(T)||_2^2.
    """
    A_t = torch.tensor(A, dtype=torch.float32)
    x0_t = torch.tensor(x0, dtype=torch.float32)
    pin_idx = torch.tensor(pinned_nodes, dtype=torch.long)

    if mode == "uncontrolled":
        def ctrl_fn(x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(len(pinned_nodes), dtype=torch.float32)
    elif mode == "linear_pinning":
        def ctrl_fn(x: torch.Tensor) -> torch.Tensor:
            u_raw = -kp_linear * x[pin_idx]
            return u_max * torch.tanh(u_raw / u_max)
    elif mode == "lqr_pinning":
        assert K_lqr is not None, "K_lqr required for lqr_pinning mode"
        K_t = torch.tensor(K_lqr, dtype=torch.float32)

        def ctrl_fn(x: torch.Tensor) -> torch.Tensor:
            u_raw = -torch.matmul(K_t, x)
            return u_max * torch.tanh(u_raw / u_max)
    elif mode == "gnn_pinning":
        assert gnn_controller is not None, "gnn_controller required for gnn_pinning mode"
        gnn_controller.eval()

        def ctrl_fn(x: torch.Tensor) -> torch.Tensor:
            return gnn_controller(x)
    else:
        raise ValueError(f"Unknown simulation mode: {mode}")

    with torch.no_grad():
        ts, xs, us = rollout_pinning_torch(
            A_lin=A_t,
            x0=x0_t,
            pinned_nodes=pinned_nodes,
            controller_fn=ctrl_fn,
            horizon=horizon,
            n_steps=n_steps,
            regime=regime,
        )

    t_np = ts.numpy().astype(np.float64)
    x_np = xs.numpy().astype(np.float64)
    u_np = us.numpy().astype(np.float64)

    state_norms = np.linalg.norm(x_np, axis=1)
    metrics = sampled_pinning_metrics(t_np, x_np[:, None, :], u_np[:, None, :], q_state, r_ctrl, q_term)
    return {
        "t": t_np, "x": x_np, "u": u_np, "state_norms": state_norms,
        **{k: float(value[0]) for k, value in metrics.items()},
    }
