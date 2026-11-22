"""Merged pickup policy with optional continuous placement corrections."""

import os

from method.modular.pickup import PickupController
from method.modular.placement_release import SeatedReleaseController
from method.modular.site_estimator import PlacementEstimate

from .policy_modular_heads import make_policy as make_pickup


def _make(groups):
    policy = make_pickup()
    policy.task = PlacementEstimate(policy.task, policy.controller, os.environ["ADV_HEADS"], groups)
    policy.controller.task_state_source = policy.task
    return policy


def make_site():
    return _make(("site",))


def make_plate():
    return _make(("plate",))


def make_policy():
    return _make(("site", "plate"))


class PickupAndRelease(SeatedReleaseController, PickupController):
    # SeatedReleaseController wraps LOWER, then delegates to PickupController
    # for the shared phase machine and its two-view pinch confirmation.
    def __init__(self, kin, source):
        super().__init__(kin, source, normal_source=None)


def make_fk_release():
    """Calibrated FK release and the verified pickup, using continuous placement corrections."""
    from submission.policy_modular import make_policy as make_baseline
    from submission.policy_modular_heads import PerceptionPolicy

    checkpoint = os.environ["ADV_HEADS"]
    policy = PerceptionPolicy(make_baseline(), checkpoint, PickupAndRelease)
    policy.task = PlacementEstimate(policy.task, policy.controller, checkpoint, ("site", "plate"))
    policy.controller.task_state_source = policy.task
    return policy


def make_visual_release():
    """Merged pickup plus Sol's seated visual calibration of the FK release normal."""
    from method.modular.placement_release import SeatedNormal
    from submission.policy_modular import make_policy as make_baseline
    from submission.policy_modular_heads import PerceptionPolicy

    checkpoint = os.environ["ADV_HEADS"]
    policy = PerceptionPolicy(make_baseline(), checkpoint, PickupAndRelease)
    policy.controller.normal_source = SeatedNormal(policy.task, checkpoint)
    return policy


def make_visual_geometry():
    """Apply continuous position corrections while calibrating orientation on its original inputs."""
    from method.modular.placement_release import SeatedNormal

    class RawSeatedNormal(SeatedNormal):
        def measure(self, state, task, hand, rotation):
            return super().measure(state, self.adapter.snapshot(state), hand, rotation)

    policy = make_visual_release()
    checkpoint = os.environ["ADV_HEADS"]
    policy.controller.normal_source = RawSeatedNormal(policy.task, checkpoint)
    policy.task = PlacementEstimate(policy.task, policy.controller, checkpoint, ("site", "plate"))
    policy.controller.task_state_source = policy.task
    return policy
