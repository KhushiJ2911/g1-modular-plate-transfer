"""Held-plate height and rack seating from public wrist and arm measurements."""

from collections import deque

import numpy as np

from . import layout
from .kinematics import pelvis_rotation
from .phases import Phase
from .pickup import PickupController

HELD_PLACEMENT = (Phase.ALIGN, Phase.LOWER)


class DescentMonitor:
    """Distinguish contact from a hand waiting at a stationary command."""

    def __init__(self):
        self.samples = deque(maxlen=12)

    def update(self, measured_z, commanded_z):
        self.samples.append((float(measured_z), float(commanded_z)))

    def seated(self):
        if len(self.samples) != self.samples.maxlen:
            return False
        samples = np.asarray(self.samples)
        before, after = np.median(samples[:3], axis=0), np.median(samples[-3:], axis=0)
        measured_descent, commanded_descent = before - after
        lag_increase = commanded_descent - measured_descent
        # The unloaded IK target can have a large fixed Cartesian error.
        # Compare changes so that bias and static sag cancel.
        return (
            commanded_descent >= 0.0012
            and -0.0025 <= measured_descent <= 0.003
            and lag_increase >= max(0.0007, 0.35 * commanded_descent)
        )


class FKSeatingController(PickupController):
    """Replace biased vision height during placement and corroborate seating by motion."""

    loaded_center_bias = None

    def reset(self):
        super().reset()
        self.descent = DescentMonitor()
        self.commanded_hand_z = None
        self.lower_start_z = None
        self.measured_hand_z = None

    def set_loaded_center_bias(self, bias):
        """Set one optional calibration of measured pinch centers for this pickup method."""
        bias = np.asarray(bias, dtype=float)
        if bias.shape != (3,) or not np.isfinite(bias).all() or np.linalg.norm(bias) >= 0.03:
            raise ValueError("Expected a finite 3D hand-frame bias smaller than 3 cm")
        self.loaded_center_bias = bias.copy()

    def _enter(self, task, hand_world, hand_rot):
        super()._enter(task, hand_world, hand_rot)
        if self.phase == Phase.LOWER:
            self.lower_start_z = float(hand_world[2])
            self.commanded_hand_z = None
            self.descent.samples.clear()

    def act(self, obs_state, task_state=None):
        state = np.asarray(obs_state)
        task = self._read_task(state, task_state)
        if self.phase in HELD_PLACEMENT and self.hand_to_center is not None:
            self.kin.set_state(state)
            position, rotation = self.kin.hand_pose("left")
            basis = pelvis_rotation(state)
            pelvis = state[layout.PELVIS_XYZ]
            hand = pelvis + basis @ position
            self.measured_hand_z = float(hand[2])
            if self.phase == Phase.LOWER and self.commanded_hand_z is not None:
                self.descent.update(self.measured_hand_z, self.commanded_hand_z)
            if task["held"] and state[layout.GRIPPERS][0] >= 0.65:
                offset = self.hand_to_center
                if self.loaded_center_bias is not None:
                    offset = offset - self.loaded_center_bias
                fk_plate = hand + basis @ rotation @ offset
                plate = task["plate_xyz"].copy()
                # Keep observed lateral centering. Only height was corrupted
                # by the loaded view in the supported, unopened failures.
                plate[2] = fk_plate[2]
                task = {**task, "plate_xyz": plate, "plate_rel": basis.T @ (plate - pelvis)}
        return super().act(state, task)

    def _action(self, state, position=None, rotation=None, base=None, close=False):
        if self.phase == Phase.LOWER and position is not None:
            self.commanded_hand_z = float(position[2])
        return super()._action(state, position, rotation, base, close)

    def _ready_to_release(self, task):
        if (
            self.hand_to_center is None
            or self.measured_hand_z is None
            or self.lower_start_z is None
        ):
            return super()._ready_to_release(task)
        delta = task["plate_xyz"] - self.target_xyz
        contacts = task["contacts"]
        near = np.linalg.norm(delta[:2]) < 0.032
        lower_travel = self.lower_start_z - self.measured_hand_z
        if (
            not task["held"]
            or not near
            or contacts["table"]
            or contacts["floor"]
            or task["tilt_deg"] > 12
            or self.machine.elapsed < 84
            or lower_travel < 0.065
        ):
            return False
        if contacts["target_rack"] and -0.058 <= delta[2] <= -0.018:
            return True
        # Keep a bounded height window for the mechanical path. It extends
        # above the visual seating band when the locked site estimate is low.
        return -0.070 <= delta[2] <= 0.008 and self.descent.seated()


def make_policy():
    """Build the calibrated contact-head policy with FK placement height."""
    import os

    from submission.policy_modular import make_policy as make_baseline
    from submission.policy_modular_heads import PerceptionPolicy

    checkpoint = os.environ["ADV_HEADS"]
    policy = PerceptionPolicy(make_baseline(), checkpoint, FKSeatingController)
    with np.load(checkpoint, allow_pickle=False) as data:
        policy.controller.set_loaded_grasp_normal(data["loaded_normal"])
        if "loaded_center_bias" in data:
            policy.controller.set_loaded_center_bias(data["loaded_center_bias"])
    return policy
