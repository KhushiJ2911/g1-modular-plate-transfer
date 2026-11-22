"""Retain corroborated pinches and recognize a loaded plate dropping into its pocket."""

from method.modular.base_clearance import ClearancePolicy
from method.modular.fk_seating import make_policy as make_seated_policy
from method.modular.seated_drop import SeatedDropMixin
from submission.policy_grasp_retry import RetainedPinchController


class RoundFourController(SeatedDropMixin, RetainedPinchController):
    """Use the seating event during LOWER and pinch corroboration during pickup."""

    # Both component A/Bs used the original 414f15a pickup estimates.
    use_wrist_localization = False
    use_lift_xy_fallback = False


def make_policy():
    return ClearancePolicy(make_seated_policy(RoundFourController))
