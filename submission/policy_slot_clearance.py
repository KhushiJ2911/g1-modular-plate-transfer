"""Hold the hand's cross-rack position as the loaded fingers open."""

import numpy as np

from method.modular import layout
from method.modular.kinematics import pelvis_rotation
from method.modular.phases import Phase
from submission.policy_modular_calibrated import make_policy as calibrated_policy


class OpeningPlaneFeedback:
    def __init__(self, policy):
        self.policy = policy
        self.controller = policy.controller
        self.reset()

    def reset(self):
        self.policy.reset()
        self.origin = None
        self.reference = None

    def __call__(self, obs):
        ctl = self.controller
        action = self.policy(obs)
        if ctl.phase not in (Phase.RELEASE, Phase.SETTLE):
            self.origin = None
            return action

        state = obs["state"]
        basis = pelvis_rotation(state)
        normal = ctl.target_rot[:, 2]
        kin = ctl.kin
        kin.set_state(state)
        hand = state[layout.PELVIS_XYZ] + basis @ kin.hand_pose("left")[0]
        position = float(normal @ hand)
        if self.origin is None:
            self.origin = position
            self.reference = ctl.last_raw[layout.LEFT_ARM_ACTION].copy()
            return action

        # Counter transverse motion seen in FK. Keep the loaded joint target's
        # posture and other axes, with a bounded correction along the Jacobian.
        row = (normal @ basis) @ kin.jacobian("left")[:3]
        error = np.clip(self.origin - position, -0.025, 0.025)
        correction = row * error / (row @ row + 0.025**2)
        previous = ctl.last_raw[layout.LEFT_ARM_ACTION]
        desired = self.reference + np.clip(correction, -0.12, 0.12)
        previous += np.clip(desired - previous, -0.012, 0.012)
        return layout.normalize_action(ctl.last_raw)


def make_policy():
    return OpeningPlaneFeedback(calibrated_policy())
