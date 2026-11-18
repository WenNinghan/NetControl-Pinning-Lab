from copy import deepcopy

import numpy as np
import pytest
import torch

from netcontrol.neural_pinning import sampled_pinning_metrics, training_initial_states
from scripts.neural_pinning_comparison import CAP, QF, R, BoundedLinear, finite_horizon_gains, verify_replay


def test_sampled_and_applied_energy_settling_and_terminal_penalty():
    t = np.array([0., 1., 2.])
    states = np.array([[[1.], [1.]], [[0.1], [1.]], [[0.], [0.16]]])
    controls = np.array([[[2.], [2.]], [[1.], [1.]], [[0.], [0.]]])
    m = sampled_pinning_metrics(t, states, controls, 3., .08, 12.)
    np.testing.assert_allclose(m["control_energy"], [3., 3.])
    np.testing.assert_allclose(m["applied_input_energy"], [5., 5.])
    np.testing.assert_allclose(m["total_j"], m["running_j"] + 12. * m["terminal_norm"]**2)
    assert m["settling_time"][0] == 1.
    assert np.isnan(m["settling_time"][1])


def test_bounded_linear_respects_its_observation_mask():
    model = BoundedLinear(torch.ones(1, 3), torch.tensor([[1., 0., 1.]]))
    x = torch.tensor([[1., 100., -1.], [1., -100., -1.], [9., -100., 9.]])
    output = model(x).detach().flatten()
    assert output[0] == output[1] == 0.
    assert -CAP < output[2] < 0.


def test_reference_initial_state_and_terminal_riccati_gain():
    x0 = np.array([1.5, 2.5])
    batch = training_initial_states(x0, 8, seed=2026)
    np.testing.assert_array_equal(batch[0].numpy(), x0)
    assert torch.equal(batch, training_initial_states(x0, 8, seed=2026))
    A = -np.eye(2)
    B = np.array([[1.], [0.]])
    schedule = finite_horizon_gains(A, B)
    np.testing.assert_allclose(schedule[-1], B.T * QF / R)
    assert schedule[0, 0, 0] < schedule[-1, 0, 0]


def test_replay_compares_split_policy_and_metric_keys_in_both_directions():
    complete = {"heldout": [{"index": 0, "x0": [1.], "scores": {"a": {"J": 2., "settling": None}}}]}
    verify_replay(complete, deepcopy(complete))
    for missing in ("split", "policy", "metric"):
        subset = deepcopy(complete)
        if missing == "split":
            del subset["heldout"]
        elif missing == "policy":
            del subset["heldout"][0]["scores"]["a"]
        else:
            del subset["heldout"][0]["scores"]["a"]["settling"]
        for source, target in [(subset, complete), (complete, subset)]:
            with pytest.raises(ValueError):
                verify_replay(source, target)
