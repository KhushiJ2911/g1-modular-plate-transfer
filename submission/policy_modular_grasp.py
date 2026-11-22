"""Sol's calibrated placement controller with public observations only."""

from method.modular.observed_controller import CalibratedPlacementController

from .policy_modular import make_policy as make_baseline


def make_policy():
    policy = make_baseline()
    policy.controller = CalibratedPlacementController(policy.controller.kin, policy.task)
    return policy
