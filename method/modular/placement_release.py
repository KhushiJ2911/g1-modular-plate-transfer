"""Level a seated plate with FK before unloading the fingers."""

import os
from collections import deque
from pathlib import Path

import numpy as np

from . import layout
from .controller import _rotation_from_vector
from .kinematics import pelvis_rotation
from .observed_controller import CalibratedPlacementController
from .orientation_feedback import LEFT_GRASP_NORMAL
from .phases import Phase
from .refined import inputs
from .robust_controller import _with_phase_trace


class SeatedNormal:
    """Use a signed visual measurement to calibrate the grasp's FK normal."""

    def __init__(self, adapter, path):
        self.adapter = adapter
        with np.load(path, allow_pickle=False) as weights:
            self.mean = weights["normal_mean"]
            self.scale = weights["normal_scale"]
            self.coef = weights["normal_coef"]
            self.bias = weights["normal_bias"]

    def measure(self, state, task, hand, rotation):
        adapter = self.adapter
        features = inputs(
            adapter.visual,
            state,
            np.r_[hand, rotation.ravel()],
            np.r_[
                task["plate_xyz"],
                task["site_xyz"],
                task["plate_rot"][:, 2],
                task["tilt_deg"],
                task["held"],
            ],
            adapter.first_plate,
            adapter.first_site,
        )
        normal = task["plate_rot"][:, 2] + (features - self.mean) / self.scale @ self.coef
        return rotation.T @ normal / max(np.linalg.norm(normal), 1e-8)


class SeatedReleaseController(CalibratedPlacementController):
    """Keep the fingers closed through a bounded, measured leveling maneuver."""

    def __init__(self, kin, task_state_source, normal_source):
        self.normal_source = normal_source
        super().__init__(kin, task_state_source)

    def reset(self):
        super().reset()
        self._reset_release()

    def _reset_release(self):
        self.normals = deque(maxlen=9)
        self.heights = deque(maxlen=12)
        self.level_pose = None
        self.level_normal = None
        self.level_step = None
        self.level_tilt = None
        self.level_target = None

    def _enter(self, task, hand_world, hand_rot):
        super()._enter(task, hand_world, hand_rot)
        if self.phase == Phase.LOWER:
            self._reset_release()

    def _read_lowering(self, state, visual):
        self.kin.set_state(state)
        position, rotation = self.kin.hand_pose("left")
        basis = pelvis_rotation(state)
        hand = state[layout.PELVIS_XYZ] + basis @ position
        hand_rot = basis @ rotation
        if visual["held"] and state[layout.GRIPPERS][0] >= 0.65:
            self.normals.append(self.normal_source.measure(state, visual, hand, hand_rot))
        self.heights.append(hand[2])
        self.hand = hand, hand_rot

    def _start_leveling(self, reference):
        hand, rotation = self.hand
        normal = LEFT_GRASP_NORMAL.copy()
        if len(self.normals) == self.normals.maxlen:
            measured = np.median(self.normals, axis=0)
            # Limit corrections to plausible grasp variation and keep the
            # calibrated prior when the wrist view is heavily occluded.
            tangent = measured - (measured @ normal) * normal
            spread = np.median(np.linalg.norm(np.array(self.normals) - measured, axis=1))
            if spread < 0.12:
                normal += tangent * min(0.9, 0.25 / max(1e-8, np.linalg.norm(tangent)))
        self.level_normal = normal / np.linalg.norm(normal)
        self.level_pose = hand, rotation
        self.level_target = reference
        self.level_step = self.machine.step

    def _ready_to_release(self, task):
        elapsed = self.machine.elapsed
        x, y, depth = task["plate_xyz"] - self.target_xyz
        contacts = task["contacts"]
        near = abs(x) < 0.032 and abs(y) < 0.030 and -0.062 < depth < -0.018
        stalled = (
            len(self.heights) == self.heights.maxlen
            and abs(self.heights[-1] - self.heights[0]) < 0.003
        )
        supported = contacts["target_rack"] or (stalled and elapsed >= 104 and depth < -0.027)
        firm = task["held"] and near and (supported or self.level_pose is not None)
        firm = firm and not (contacts["table"] or contacts["floor"])
        if self.level_pose is None:
            if firm and elapsed >= 90:
                self._start_leveling(self.target_rot[:, 2])
            return False
        normal = self.hand[1] @ self.level_normal
        self.level_tilt = float(np.rad2deg(np.arccos(np.clip(normal @ self.level_target, -1, 1))))
        return firm and self.machine.step - self.level_step >= 16 and self.level_tilt <= 5.5

    def _action(self, state, position=None, rotation=None, base=None, close=False):
        if self.phase == Phase.LOWER and self.level_pose is not None:
            position, original_rotation = self.level_pose
            normal = self.hand[1] @ self.level_normal
            error = np.cross(normal, self.level_target)
            correction = 0.8 * error
            if np.linalg.norm(correction) > 0.12:
                correction *= 0.12 / np.linalg.norm(correction)
            rotation = _rotation_from_vector(correction) @ self.hand[1]
            if self.level_tilt is not None and self.level_tilt <= 2.0:
                rotation = self.hand[1]
            elif self.machine.step - self.level_step < 2:
                rotation = _rotation_from_vector(correction) @ original_rotation
        return super()._action(state, position, rotation, base, close)

    def act(self, obs_state, task_state=None):
        if self.phase == Phase.LOWER:
            state = np.asarray(obs_state)
            task_state = self._read_task(state, task_state)
            self._read_lowering(state, task_state)
        return super().act(obs_state, task_state)


def make_previous_heads():
    from submission.policy_modular_heads import make_policy

    return _with_phase_trace(make_policy())


def make_policy():
    from submission.policy_modular_heads import make_policy as make_heads

    baseline = make_heads()
    default = Path(__file__).resolve().parents[2] / "checkpoints/astra/normal_v1.npz"
    normal = SeatedNormal(baseline.task, os.environ.get("ADV_PLACEMENT_NORMAL") or default)

    class PlacementPolicy(type(baseline)):
        def __init__(self):
            self.task = baseline.task
            self.controller = SeatedReleaseController(baseline.controller.kin, self.task, normal)
            self.task.controller = self.controller
            self.reset()

    return _with_phase_trace(PlacementPolicy())
