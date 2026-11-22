"""Permit a late release from a stationary, near-site grasp."""

from collections import deque

import numpy as np

from . import layout
from .controller import LOWER_RAMP_STEPS
from .kinematics import pelvis_rotation
from .phases import Phase
from .pickup import PickupController

WINDOW = 24
SITE_RADIUS_M = 0.060
MAX_TILT_DEG = 15
MAX_DRIFT_M = 0.008


class StationaryReleaseController(PickupController):
    """Use measured arm geometry after the nominal descent ramp."""

    def reset(self):
        super().reset()
        self.seating_samples = deque(maxlen=WINDOW)
        self.fallback_releases = []
        self.placement_state = None

    def _enter(self, task, hand_world, hand_rot):
        super()._enter(task, hand_world, hand_rot)
        if self.phase == Phase.LOWER:
            self.seating_samples.clear()

    def act(self, obs_state, task_state=None):
        self.placement_state = np.asarray(obs_state) if self.phase == Phase.LOWER else None
        return super().act(obs_state, task_state)

    def _seating_sample(self, task):
        state = self.placement_state
        if (
            state is None
            or self.hand_to_center is None
            or not task["held"]
            or state[layout.GRIPPERS][0] < 0.80
            or np.mean(state[layout.LEFT_FINGERS]) < -0.0185
            or task["contacts"]["table"]
            or task["contacts"]["floor"]
        ):
            return None
        pos, rot = self.kin.hand_pose("left")
        basis = pelvis_rotation(state)
        hand = state[layout.PELVIS_XYZ] + basis @ pos
        plate = hand + basis @ rot @ self.hand_to_center
        delta = plate - self.target_xyz
        if (
            np.linalg.norm(delta) > SITE_RADIUS_M
            or delta[2] > -0.016
            or task["tilt_deg"] > MAX_TILT_DEG
        ):
            return None
        return np.stack((hand, plate, task["plate_xyz"]))

    def _release_confirmed(self, task):
        if super()._release_confirmed(task):
            return True

        sample = self._seating_sample(task)
        if sample is None:
            self.seating_samples.clear()
            return False
        self.seating_samples.append(sample)
        if (
            self.machine.elapsed < LOWER_RAMP_STEPS
            or self.placement_streak > 0
            or len(self.seating_samples) < WINDOW
        ):
            return False

        spans = np.linalg.norm(np.ptp(self.seating_samples, axis=0), axis=1)
        if min(spans) > MAX_DRIFT_M:
            return False
        self.fallback_releases.append({"step": self.machine.step})
        return True


def make_policy():
    """Build the bd951bc calibrated head policy with the release-only fallback."""
    import os

    from submission.policy_modular import make_policy as make_baseline
    from submission.policy_modular_heads import PerceptionPolicy

    checkpoint = os.environ["ADV_HEADS"]
    policy = PerceptionPolicy(make_baseline(), checkpoint, StationaryReleaseController)
    with np.load(checkpoint, allow_pickle=False) as data:
        policy.controller.set_loaded_grasp_normal(data["loaded_normal"])
    return policy
