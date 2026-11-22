"""Withdraw the open hand with the base if a supported placement remains stalled."""

from collections import deque

import numpy as np

from . import layout
from .kinematics import pelvis_rotation, pelvis_to_world
from .phases import Phase

# Successful delayed placements in the 3001600 generation comparison finished
# in under one second of HOLD. Allow two seconds before disturbing the plate.
QUIET_STEPS = 100


class LateClearance:
    def __init__(self, quiet_steps=QUIET_STEPS):
        self.quiet_steps = quiet_steps
        self.samples = deque(maxlen=5)
        self.origin = None
        self.axis = None
        self.plate = None
        self.first_tilt = None
        self.elapsed = 0
        self.reason = None

    def apply(self, action, state, kin, task, site, site_rot, age):
        if age < self.quiet_steps - 5 or self.reason:
            return action
        self.samples.append(task["plate_xyz"])
        if age < self.quiet_steps:
            return action
        plate = np.median(self.samples, axis=0)
        gap = plate - site
        open_hand = max(state[layout.GRIPPERS]) <= 0.08
        near = np.linalg.norm(gap[:2]) < 0.065
        supported = task["contacts"]["target_rack"] or (near and -0.060 < gap[2] < -0.012)
        if self.origin is None:
            if not (open_hand and supported and near and task["tilt_deg"] <= 18):
                # Permit several fresh views when a single observation is noisy.
                if age > self.quiet_steps + 25:
                    self.reason = "outside guard"
                return action
            kin.set_state(state)
            hand = pelvis_to_world(state, kin.hand_pose("left")[0])
            axis = site_rot[:2, 0].copy()
            axis /= np.linalg.norm(axis)
            side = np.dot(hand[:2] - plate[:2], axis)
            if abs(side) < 0.01:
                side = np.dot(state[layout.PELVIS_XY] - plate[:2], axis)
            self.axis = axis if side >= 0 else -axis
            self.origin = state[layout.PELVIS_XY].copy()
            self.plate = plate.copy()
            self.first_tilt = task["tilt_deg"]
        travel = (state[layout.PELVIS_XY] - self.origin) @ self.axis
        if not open_hand or not supported:
            self.reason = "support or opening lost"
        elif np.linalg.norm(plate - self.plate) > 0.014 or task["tilt_deg"] > min(
            22, self.first_tilt + 6
        ):
            self.reason = "plate motion"
        elif travel >= 0.055 or self.elapsed >= 130:
            self.reason = "clearance bound"
        if self.reason:
            return action
        # Base xy limits are ±1 m/s, so velocities in m/s are already normalized.
        cmd = action.copy()
        cmd[layout.BASE_VELOCITY_ACTION] = pelvis_rotation(state)[:2, :2].T @ (0.075 * self.axis)
        self.elapsed += 1
        return cmd


class ClearancePolicy:
    def __init__(self, policy):
        self.policy = policy
        self.controller = policy.controller
        self.task = policy.task
        self.recovery = LateClearance()

    def reset(self):
        self.policy.reset()
        self.recovery = LateClearance()

    def __call__(self, obs):
        ctl = self.controller
        phase, age = ctl.phase, ctl.machine.elapsed
        action = self.policy(obs)
        if phase != Phase.HOLD or not ctl.transitions:
            return action
        source = ctl.transitions[-1]["from"]
        if source not in ("RETREAT", "SETTLE"):
            return action
        # SETTLE already exhausted its supported, stationary timeout. A
        # guarded RETREAT just moved the wrist, so give it a full quiet window.
        self.recovery.quiet_steps = 0 if source == "SETTLE" else QUIET_STEPS
        state = obs["state"]
        return self.recovery.apply(
            action, state, ctl.kin, self.task.snapshot(state), ctl.target_xyz, ctl.target_rot, age
        )


def make_policy():
    from submission.policy_modular_calibrated import make_policy as calibrated

    return ClearancePolicy(calibrated())


def make_fk_policy():
    from .fk_seating import make_policy as seated

    return ClearancePolicy(seated())
