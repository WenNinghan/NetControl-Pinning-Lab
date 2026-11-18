"""
NetControl-Pinning-Lab: Complex Network Structural Controllability,
Gramian-based Actuator Selection, and Graph Neural Pinning Control.
"""

from .topologies import (
    erdos_renyi_directed,
    scale_free_directed,
    small_world_directed,
    reaction_cascade_network,
    build_continuous_system_matrix,
)
from .structural import (
    StructuralControllabilityResult,
    analyze_structural_controllability,
    classify_control_edges,
    verify_pbh_controllability,
    verify_pbh_input_placement,
)
from .gramian_lqr import (
    compute_controllability_gramian,
    compute_infinite_horizon_gramian,
    minimum_energy_trajectory,
    select_actuators_submodular,
    select_actuators_baseline,
    solve_lqr_controller,
    stabilize_hurwitz_matrix,
)
from .neural_pinning import (
    GraphNeuralPinningController,
    simulate_nonlinear_pinning,
    train_neural_pinning_controller,
)

__all__ = [
    "erdos_renyi_directed",
    "scale_free_directed",
    "small_world_directed",
    "reaction_cascade_network",
    "build_continuous_system_matrix",
    "StructuralControllabilityResult",
    "analyze_structural_controllability",
    "classify_control_edges",
    "verify_pbh_controllability",
    "verify_pbh_input_placement",
    "compute_controllability_gramian",
    "compute_infinite_horizon_gramian",
    "stabilize_hurwitz_matrix",
    "minimum_energy_trajectory",
    "select_actuators_submodular",
    "select_actuators_baseline",
    "solve_lqr_controller",
    "GraphNeuralPinningController",
    "simulate_nonlinear_pinning",
    "train_neural_pinning_controller",
]
