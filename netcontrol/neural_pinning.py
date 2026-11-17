"""
Nonlinear Complex Network Pinning Control & Differentiable Graph Neural Controller.

Supports two realistic nonlinear coupled dynamics regimes:
1. 'pollutant_cascade': Nonlinear reaction-diffusion ozone/pollutant cascade suppression
   with Michaelis-Menten saturation and autocatalytic radical cross-activation.
2. 'bistable_sync': Coupled Stuart-Landau / double-well bistable network steering from
   an undesired high-excitation equilibrium to a target synchronized safe state.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn


class GraphNeuralPinningController(nn.Module):
    """
    Topology-aware Graph Attention / Message-Passing Neural Pinning Controller.

    Given current network state error e(t) = x(t) - x_target and static structural
    node features (normalized in/out strength, driver indicator), computes localized
    message passing over the directed network and outputs bounded control actions
    u_P(t) on the pinned actuator subset P.
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
        self.u_max = u_max

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
            state_error: Tensor of shape (N,) representing x(t) - x_target.
        Returns:
            u_pinned: Tensor of shape (|P|,) with control signals at pinned nodes.
        """
        e_col = state_error.unsqueeze(-1)
        x_in = torch.cat([e_col, torch.tanh(e_col), self.struct_feats], dim=-1)
        h0 = self.node_encoder(x_in)
        h_msg = torch.matmul(self.adj_norm, self.msg_layer(h0))
        h1 = self.act(self.self_layer(h0) + h_msg)

        h_cat = torch.cat([h0, h1], dim=-1)
        h_pinned = h_cat[self.pinned_idx]
        e_pinned = state_error[self.pinned_idx]

        u_nn = self.readout(h_pinned).squeeze(-1)
        u_raw = -torch.relu(self.kp_residual) * e_pinned + u_nn * torch.tanh(e_pinned.abs())
        return self.u_max * torch.tanh(u_raw / self.u_max)


def build_structural_features(G: nx.DiGraph, driver_nodes: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
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
    is_driver = np.array([1.0 if u in set(driver_nodes) else 0.0 for u in nodes], dtype=np.float32)

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
    Target safe equilibrium is x_target = 0.
    """
    if regime == "pollutant_cascade":
        # Nonlinear autocatalytic/saturation load + diffusive/advective network coupling
        nonlinear_source = 0.85 * (x ** 2) / (1.0 + x ** 2) + 0.25 * torch.sin(1.5 * x) * torch.tanh(x)
        return torch.matmul(A_lin, x) + nonlinear_source + u_full
    elif regime == "bistable_sync":
        # Bistable Schlögl / Stuart-Landau cubic reaction term driving unforced states away from 0
        cubic_term = 1.15 * x * (1.0 - (x / 2.2) ** 2)
        return torch.matmul(A_lin, x) + cubic_term + u_full
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
    """Differentiable RK4 rollout of nonlinear pinning control."""
    n = A_lin.shape[0]
    dt = horizon / float(n_steps)
    ts = torch.linspace(0.0, horizon, n_steps + 1)
    pin_idx = torch.tensor(pinned_nodes, dtype=torch.long)

    xs = [x0]
    us = []
    x_cur = x0
    for _ in range(n_steps):
        u_pin = controller_fn(x_cur)
        us.append(u_pin)
        u_full = torch.zeros(n, dtype=x_cur.dtype, device=x_cur.device)
        u_full[pin_idx] = u_pin

        k1 = _nonlinear_rhs_torch(x_cur, u_full, A_lin, regime=regime)
        k2 = _nonlinear_rhs_torch(x_cur + 0.5 * dt * k1, u_full, A_lin, regime=regime)
        k3 = _nonlinear_rhs_torch(x_cur + 0.5 * dt * k2, u_full, A_lin, regime=regime)
        k4 = _nonlinear_rhs_torch(x_cur + dt * k3, u_full, A_lin, regime=regime)
        x_cur = x_cur + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        xs.append(x_cur)

    us.append(controller_fn(x_cur))
    return ts, torch.stack(xs, dim=0), torch.stack(us, dim=0)


def train_neural_pinning_controller(
    G: nx.DiGraph,
    A: np.ndarray,
    pinned_nodes: List[int],
    driver_nodes: List[int],
    x0: np.ndarray,
    horizon: float = 4.0,
    n_steps: int = 80,
    epochs: int = 45,
    lr: float = 0.035,
    q_state: float = 3.0,
    r_ctrl: float = 0.08,
    regime: str = "pollutant_cascade",
    seed: int = 42,
) -> Tuple[GraphNeuralPinningController, List[float]]:
    """
    Train the GraphNeuralPinningController via differentiable ODE trajectory optimization
    to minimize integrated quadratic state error + control effort + terminal error.
    """
    torch.manual_seed(seed)
    adj_norm, struct_feats = build_structural_features(G, driver_nodes)
    controller = GraphNeuralPinningController(
        n_nodes=G.number_of_nodes(),
        pinned_nodes=pinned_nodes,
        adj_norm=adj_norm,
        struct_feats=struct_feats,
        hidden_dim=32,
        u_max=6.0,
    )

    A_t = torch.tensor(A, dtype=torch.float32)
    x0_t = torch.tensor(x0, dtype=torch.float32)
    dt = horizon / float(n_steps)

    optimizer = torch.optim.Adam(controller.parameters(), lr=lr)
    loss_history: List[float] = []

    for _ in range(epochs):
        optimizer.zero_grad()
        _, xs, us = rollout_pinning_torch(
            A_lin=A_t,
            x0=x0_t,
            pinned_nodes=pinned_nodes,
            controller_fn=controller,
            horizon=horizon,
            n_steps=n_steps,
            regime=regime,
        )
        state_cost = q_state * torch.sum(xs ** 2) * dt
        ctrl_cost = r_ctrl * torch.sum(us ** 2) * dt
        term_cost = 12.0 * torch.sum(xs[-1] ** 2)
        loss = state_cost + ctrl_cost + term_cost
        loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm=5.0)
        optimizer.step()
        loss_history.append(float(loss.item()))

    return controller, loss_history


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
    regime: str = "pollutant_cascade",
) -> Dict[str, np.ndarray | float]:
    """
    Evaluate a pinning control law on the nonlinear complex network and return
    trajectories and quantitative control performance metrics.
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

    t_np = ts.numpy()
    x_np = xs.numpy()
    u_np = us.numpy()
    dt = horizon / float(n_steps)

    state_norms = np.linalg.norm(x_np, axis=1)
    ise_state = float(np.sum(x_np ** 2) * dt)
    control_energy = float(np.sum(u_np ** 2) * dt)
    terminal_norm = float(state_norms[-1])

    # Settling time T_s: first time step after which ||x(t)||_2 stays below 15% of ||x(0)||_2
    thresh = 0.15 * float(state_norms[0])
    below = state_norms <= thresh
    settling_time = float(horizon)
    for i in range(len(t_np)):
        if np.all(below[i:]):
            settling_time = float(t_np[i])
            break

    return {
        "t": t_np,
        "x": x_np,
        "u": u_np,
        "state_norms": state_norms,
        "ise_state": ise_state,
        "control_energy": control_energy,
        "total_j": 3.0 * ise_state + 0.08 * control_energy,
        "terminal_norm": terminal_norm,
        "settling_time": settling_time,
    }
