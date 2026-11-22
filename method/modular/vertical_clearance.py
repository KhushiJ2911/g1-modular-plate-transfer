"""Lift an open hand out of a supported rack after the normal retreat has stopped."""

import numpy as np

from . import layout
from .base_clearance import QUIET_STEPS, ClearancePolicy
from .kinematics import pelvis_rotation
from .phases import TIMEOUTS, Phase

# Reuse the placement hover height and the rate of the ordinary Cartesian retreat.
HOVER_HEIGHT = 0.090
RETREAT_INCREMENT = 0.0015


class VerticalClearance:
    def __init__(self):
        self.quiet_steps = 0
        self.command_pose = None
        self.start_z = None
        self.steps = 0
        self.finished = False

    def apply(self, action, state, task, controller):
        self.quiet_steps += 1
        if self.finished or self.quiet_steps < QUIET_STEPS:
            return action
        delta = task["plate_xyz"] - controller.target_xyz
        contacts = task["contacts"]
        near = np.linalg.norm(delta[:2]) < 0.065
        supported = contacts["target_rack"] or (near and -0.060 < delta[2] < -0.012)
        if not (
            np.all(state[layout.GRIPPERS] <= 0.08)
            and near
            and supported
            and not (contacts["table"] or contacts["floor"])
            and task["tilt_deg"] <= 18
        ):
            if self.command_pose is not None:
                self.finished = True
            return action

        kin = controller.kin
        basis = pelvis_rotation(state)
        origin = state[layout.PELVIS_XYZ]
        kin.set_state(state)
        hand, _ = kin.hand_pose("left")
        actual_z = float((origin + basis @ hand)[2])
        if self.command_pose is None:
            # Start at the FK of the *commanded* joints. Using measured FK as
            # an absolute target would introduce a lateral step under gravity.
            commanded = state.copy()
            commanded[layout.LEFT_ARM] = controller.last_raw[layout.LEFT_ARM_ACTION]
            kin.set_state(commanded)
            hand, rotation = kin.hand_pose("left")
            self.command_pose = origin + basis @ hand, basis @ rotation
            self.start_z = actual_z
            kin.set_state(state)

        if actual_z - self.start_z >= HOVER_HEIGHT or self.steps >= TIMEOUTS[Phase.RETREAT]:
            self.finished = True
            return action
        self.steps += 1
        position, rotation = self.command_pose
        rise = min(self.steps * RETREAT_INCREMENT, HOVER_HEIGHT)
        return controller._action(state, position + [0.0, 0.0, rise], rotation, close=False)


class VerticalClearancePolicy(ClearancePolicy):
    def __init__(self, policy):
        super().__init__(policy)
        self.lift = VerticalClearance()

    def reset(self):
        super().reset()
        self.lift = VerticalClearance()

    def __call__(self, obs):
        action = super().__call__(obs)
        # The incumbent creates recovery.origin only after an open supported
        # placement. Let that recovery finish before moving the arm.
        if (
            self.controller.phase != Phase.HOLD
            or self.recovery.origin is None
            or self.recovery.reason is None
            or self.lift.finished
        ):
            return action
        state = obs["state"]
        return self.lift.apply(action, state, self.task.snapshot(state), self.controller)


def make_policy():
    from submission.policy_modular_round4 import MeasuredReleaseController

    from .fk_seating import make_policy as make_seated_policy

    return VerticalClearancePolicy(make_seated_policy(MeasuredReleaseController))
