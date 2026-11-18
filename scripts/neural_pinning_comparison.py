"""Train or replay the fixed synthetic-graph feedback comparison for benchmark 4.

    python -m scripts.neural_pinning_comparison
    python -m scripts.neural_pinning_comparison --replay
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
import torch
from torch import nn

from netcontrol import (
    build_continuous_system_matrix, reaction_cascade_network, select_actuators_baseline,
    select_actuators_submodular, solve_lqr_controller,
)
from netcontrol.gramian_lqr import build_input_matrix
from netcontrol.neural_pinning import (
    GraphNeuralPinningController, build_structural_features, fit_pinning_feedback,
    rollout_pinning_torch, sampled_pinning_metrics, training_initial_states,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"
RECORDS = "neural_pinning_holdouts.json"
WEIGHTS = "neural_feedback_weights.npz"
T, STEPS = 4.0, 80
Q, R, QF = 3.0, 0.08, 12.0
CAP, SEED = 6.0, 2026
SPLITS = {"original_10": (9090, 10), "additional_80": (16219, 80)}
LABELS = {
    "uncontrolled": "Uncontrolled cascade",
    "random_proportional": "Random actuators + proportional",
    "proportional": "Gramian selection + proportional",
    "care": "CARE LQR",
    "finite_lqr": "Finite-horizon LQR",
    "local_linear": "Trained one-hop linear",
    "full_linear": "Trained full-state linear",
    "self_only": "Self-only neural",
    "gnn": "One-hop GNN",
}


class BoundedLinear(nn.Module):
    def __init__(self, gain: torch.Tensor, mask: torch.Tensor):
        super().__init__()
        self.gain = nn.Parameter(gain.clone())
        self.register_buffer("mask", mask)

    def forward(self, x):
        return CAP * torch.tanh(-(x @ (self.gain * self.mask).T) / CAP)


def finite_horizon_gains(A, B):
    """Riccati gains at the controller sample times, integrating backwards from Qf."""
    n = A.shape[0]
    brb = B @ B.T / R
    eye = np.eye(n)

    def reverse_riccati(_, flat):
        P = flat.reshape(n, n)
        return (A.T @ P + P @ A - P @ brb @ P + Q * eye).ravel()

    solution = solve_ivp(reverse_riccati, (0, T), (QF * eye).ravel(), method="DOP853",
                         t_eval=np.linspace(0., T, STEPS + 1), rtol=1e-10, atol=1e-11)
    if not solution.success:
        raise RuntimeError(solution.message)
    gains = [B.T @ flat.reshape(n, n) / R for flat in solution.y.T[::-1]]
    return torch.tensor(np.stack(gains), dtype=torch.float32)


def set_or_train_weights(models, A, pins, train_x, results_dir, replay):
    weights = results_dir / WEIGHTS
    if replay:
        with np.load(weights, allow_pickle=False) as archive:
            for name, model in models.items():
                model.load_state_dict({k: torch.from_numpy(archive[f"{name}.{k}"])
                                       for k in model.state_dict()})
                model.eval()
        return {}
    histories = {}
    arrays = {}
    for name, model in models.items():
        histories[name] = fit_pinning_feedback(model, A, pins, train_x, T, STEPS, 55, .035,
                                               Q, R, QF, "pollutant_cascade")
        for k, value in model.state_dict().items():
            arrays[f"{name}.{k}"] = value.detach().numpy()
    with weights.with_suffix(".tmp").open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    weights.with_suffix(".tmp").replace(weights)
    return histories


def _model_set(graph, drivers, pins, gain):
    adj, features = build_structural_features(graph, drivers)
    masks = (adj[pins, :] > 0).float()
    local = BoundedLinear(gain, masks)
    full = BoundedLinear(gain, torch.ones_like(masks))
    models = {"local_linear": local, "full_linear": full}
    for name, messages in [("self_only", torch.eye(len(graph))), ("gnn", adj)]:
        torch.manual_seed(SEED)
        models[name] = GraphNeuralPinningController(len(graph), pins, messages, features, u_max=CAP)
    return models


def _evaluate_policy(controller, A, x0s, pins):
    with torch.no_grad():
        ts, xs, us = rollout_pinning_torch(A, x0s, pins, controller, T, STEPS)
    return sampled_pinning_metrics(ts.numpy(), xs.numpy(), us.numpy(), Q, R, QF)


def evaluate_feedback(results_dir: Path, replay=False):
    """Evaluate every policy on the same initial states with per-policy data saved by seed."""
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / RECORDS
    previous = json.loads(path.read_text(encoding="utf8")) if replay else None
    weight_path = results_dir / WEIGHTS
    if replay and hashlib.sha256(weight_path.read_bytes()).hexdigest() != previous["protocol"]["weights_sha256"]:
        raise ValueError("The feedback weights do not match the recorded holdout experiment")
    torch.set_num_threads(1)
    graph = reaction_cascade_network(n=28, branch_factor=2.3, seed=SEED)
    n = len(graph)
    A = build_continuous_system_matrix(graph, damping=.55, coupling=.36)
    A32 = torch.tensor(A, dtype=torch.float32)
    # For this archived graph all seven drivers are zero-in-degree roots. Use their
    # ordered identifiers to avoid choosing a different maximum-matching representative.
    drivers = sorted(v for v in graph if graph.in_degree(v) == 0)
    pins, _ = select_actuators_submodular(A, budget_k=8, horizon=2.5, mandatory_nodes=drivers[:4])
    if drivers != list(range(7)) or list(pins) != [0, 1, 2, 3, 4, 6, 13, 18]:
        raise ValueError("The graph or selection differs from the frozen 28-node comparison")
    rand_pins, _ = select_actuators_baseline(graph, A, budget_k=8, strategy="random", seed=99)
    B = build_input_matrix(n, pins)
    K, _ = solve_lqr_controller(A, B, q_weight=Q, r_weight=R)
    K = torch.tensor(K, dtype=torch.float32)
    schedule = finite_horizon_gains(A, B)
    xref = np.random.default_rng(SEED).uniform(1.2, 2.8, n)
    train_x = training_initial_states(xref, 8, SEED)
    models = _model_set(graph, drivers, pins, K)
    histories = set_or_train_weights(models, A32, pins, train_x, results_dir, replay)
    split_states = {"reference_in_training": train_x[:1]}
    for name, (seed, count) in SPLITS.items():
        states = np.random.default_rng(seed).uniform(1., 3., (count, n))
        split_states[name] = torch.tensor(states, dtype=torch.float32)
    # A new iterator advances once per controller sample, including the final scored command.
    states = torch.cat(list(split_states.values()))
    gains_at_time = iter(schedule)
    def bounded(raw):
        return CAP * torch.tanh(raw / CAP)
    policies = {
        "uncontrolled": lambda x: torch.zeros(x.shape[0], len(pins)),
        "random_proportional": lambda x: bounded(-2.4 * x[:, rand_pins]),
        "proportional": lambda x: bounded(-2.4 * x[:, pins]),
        "care": lambda x: bounded(-x @ K.T),
        "finite_lqr": lambda x: bounded(-x @ next(gains_at_time).T),
        **models,
    }
    metrics = {name: _evaluate_policy(ctrl, A32, states, rand_pins if name == "random_proportional" else pins)
               for name, ctrl in policies.items()}
    rows = {}
    start = 0
    for name, inputs in split_states.items():
        rows[name] = []
        for index, state in enumerate(inputs):
            scores = {}
            for method, values in metrics.items():
                scores[method] = {}
                for key, series in values.items():
                    val = float(series[start + index])
                    scores[method][key] = None if key == "settling_time" and np.isnan(val) else val
            rows[name].append({"index": index, "x0": state.tolist(), "scores": scores})
        start += len(inputs)
    protocol = {
        "graph": "reaction_cascade_network(n=28, branch_factor=2.3, seed=2026)",
        "linear_dynamics": {"damping": .55, "coupling": .36}, "regime": "pollutant_cascade",
        "driver_indicator_nodes": drivers, "pinned_nodes": pins, "random_pinned_nodes": rand_pins,
        "selection": "Greedy logdet at T=2.5 with mandatory roots 0,1,2,3; sorted input columns.",
        "training_seed": SEED, "training_initial_states": train_x.tolist(),
        "test_uniform_interval": [1., 3.],
        "test_seeds": {k: seed for k, (seed, _) in SPLITS.items()},
        "horizon": T, "steps": STEPS, "q_state": Q, "r_control": R, "q_terminal": QF, "u_max": CAP,
        "training": "55 Adam epochs, lr=0.035, gradient norm clipped to 5; final epoch used for all four trained policies.",
        "linear_training": "Initialized from CARE; one-hop mask or full-state gain; zero bias and tanh actuation.",
        "neural_training": "Identical initialization, architecture and static node features; self_only replaces the message matrix by I before training.",
        "weights": WEIGHTS, "weights_sha256": hashlib.sha256(weight_path.read_bytes()).hexdigest(),
        "precision": "CPU float32 fixed-step RK4 with zero-order-held controls / policy evaluation; metrics accumulated in float64",
        "quadrature": "J uses trapezoidal sums of state and sampled policy command squares, including the endpoint. applied_input_energy integrates the piecewise-constant applied commands.",
        "settling_time": "null when the state norm never settles below 15% of its initial value before the horizon.",
        "torch_version": str(torch.__version__), "numpy_version": np.__version__,
    }
    record = {"protocol": protocol, "reference_loss_history": histories, "splits": rows}
    if replay:
        verify_replay(previous["splits"], rows)
        return previous
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf8")
    temp.replace(path)
    return record


def verify_replay(archived, recomputed):
    """Verify the full split, policy and metric inventories before comparing numbers."""
    if archived.keys() != recomputed.keys():
        raise ValueError("Replay split names differ")
    for name, actual_rows in recomputed.items():
        prior_rows = archived[name]
        if len(prior_rows) != len(actual_rows):
            raise ValueError(f"Replay split size differs: {name}")
        for prior, actual in zip(prior_rows, actual_rows):
            index = actual["index"]
            if prior.keys() != actual.keys() or prior["index"] != index:
                raise ValueError(f"Replay row identity differs: {name}/{index}")
            np.testing.assert_array_equal(actual["x0"], prior["x0"])
            if prior["scores"].keys() != actual["scores"].keys():
                raise ValueError(f"Replay policy names differ: {name}/{index}")
            for method, values in actual["scores"].items():
                previous = prior["scores"][method]
                if values.keys() != previous.keys():
                    raise ValueError(f"Replay metric fields differ: {name}/{index}/{method}")
                for field, value in values.items():
                    old = previous[field]
                    where = f"{name}/{index}/{method}/{field}"
                    if old is None or value is None:
                        if value is not old:
                            raise ValueError(f"Replay missing value differs: {where}")
                    else:
                        np.testing.assert_allclose(value, old, rtol=2e-5, atol=1e-5,
                                                   equal_nan=False, err_msg=where)


def mean_metric(rows, method, field):
    return float(np.mean([row["scores"][method][field] for row in rows]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true", help="load archived policy weights and verify all per-initial-state costs")
    args = parser.parse_args()
    evaluate_feedback(RESULTS, replay=args.replay)
    print("Replayed archived trials." if args.replay else f"Saved weights and trials in {RESULTS}.")
