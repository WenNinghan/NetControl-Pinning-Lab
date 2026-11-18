import numpy as np
import pytest
from scipy.integrate import trapezoid

from netcontrol.gramian_lqr import (
    build_input_matrix,
    minimum_energy_trajectory,
    select_actuators_baseline,
    select_actuators_submodular,
)
from netcontrol.topologies import build_continuous_system_matrix, reaction_cascade_network


def test_regularized_scalar_integrator_accounts_for_input_and_terminal_penalty():
    result = minimum_energy_trajectory(
        np.zeros((1, 1)), np.ones((1, 1)), np.array([1.0]), np.array([0.0]),
        horizon=2, reg=0.5, endpoint_tolerances=(0.0, 0.45, 1.0),
    )
    assert result["reachability_rank"] == 1
    assert result["exact_target_energy"] == pytest.approx(0.5)
    assert result["u"][:, 0] == pytest.approx(-0.4)
    assert result["x"][-1, 0] == pytest.approx(0.2)
    assert result["terminal_error"] == pytest.approx(0.2)
    assert result["input_energy"] == pytest.approx(0.32)
    assert result["energy"] == result["input_energy"]
    assert result["terminal_penalty"] == pytest.approx(0.08)
    assert result["regularized_objective"] == pytest.approx(0.4)
    energies = [entry["input_energy"] for entry in result["tolerance_comparison"]]
    assert energies == pytest.approx([0.5, (1 - 0.45)**2 / 2, 0.0])


def test_target_reachability_and_tolerance_boundary_on_an_unactuated_state():
    A = np.zeros((2, 2))
    B = np.array([[1.0], [0.0]])
    xf = np.zeros(2)
    result = minimum_energy_trajectory(
        A, B, np.array([1.0, 2.0]), xf, horizon=1.0,
        endpoint_tolerances=(1.99, 2.0, 2.1),
    )
    assert result["reachability_rank"] == 1
    assert result["minimum_terminal_error"] == pytest.approx(2.0)
    assert not result["exact_target_reachable"]
    assert result["exact_target_energy"] is None
    below, equal, above = result["tolerance_comparison"]
    assert not below["feasible"] and below["input_energy"] is None
    assert equal["feasible"] and equal["input_energy"] == pytest.approx(1.0)
    assert above["input_energy"] == pytest.approx((1 - np.sqrt(2.1**2 - 4))**2)
    with pytest.raises(ValueError, match="outside"):
        minimum_energy_trajectory(A, B, np.array([1.0, 2.0]), xf, reg=0.0)
    reachable = minimum_energy_trajectory(A, B, np.array([1.0, 0.0]), xf, horizon=1, reg=0)
    assert reachable["exact_target_reachable"]
    assert reachable["exact_target_energy"] == pytest.approx(1)
    assert reachable["x"][-1] == pytest.approx(xf, abs=1e-12)


def test_seeded_cascade_compares_energy_at_equal_terminal_accuracy():
    graph = reaction_cascade_network(30, 2.3, seed=7)
    A = build_continuous_system_matrix(graph, damping=1.15, coupling=0.38)
    x0 = np.random.default_rng(42).uniform(0.8, 2.0, 30)
    greedy, _ = select_actuators_submodular(A, budget_k=10)
    hubs, _ = select_actuators_baseline(graph, A, budget_k=10, strategy="out_degree")
    for nodes, rank, expected, matched in [
        (greedy, 30, (389.9414, 0.184614, 730.76385), (38.06298, 29.73717)),
        (hubs, 24, (127.1553, 0.452371, 2173.55063), (151.14699, 25.88140)),
    ]:
        result = minimum_energy_trajectory(
            A, build_input_matrix(30, nodes), x0, np.zeros(30), horizon=2, n_steps=1000,
            reg=1e-4, endpoint_tolerances=(0.45, 0.50),
        )
        assert result["reachability_rank"] == rank
        assert result["exact_target_reachable"] == (rank == 30)
        actual = [result[k] for k in ("input_energy", "terminal_error", "regularized_objective")]
        assert actual == pytest.approx(expected, rel=2e-6)
        assert np.linalg.norm(result["x"][-1]) == pytest.approx(result["terminal_error"], abs=1e-9)
        integrated = trapezoid(np.sum(result["u"]**2, axis=1), result["t"])
        assert integrated == pytest.approx(result["input_energy"], rel=2e-5)
        assert [v["input_energy"] for v in result["tolerance_comparison"]] == pytest.approx(matched)
