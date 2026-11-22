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
        if self.normal_source is not None and visual["held"] and state[layout.GRIPPERS][0] >= 0.65:
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
        error = np.cross(rotation @ self.level_normal, reference)
        correction = error * min(1.0, 0.22 / max(1e-8, np.linalg.norm(error)))
        self.level_pose = hand, _rotation_from_vector(correction) @ rotation
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
            # At the support point, hold hand translation while correcting
            # rotation. Higher rotation weight is local to this nudge.
            position, rotation = self.level_pose
            basis = pelvis_rotation(state).T
            arm = self.last_raw[layout.LEFT_ARM_ACTION]
            goal = self.kin.ik(
                "left",
                basis @ (position - state[layout.PELVIS_XYZ]),
                basis @ rotation,
                q_init=arm,
                rot_weight=0.22,
                iters=16,
            )
            arm += np.clip(goal - arm, -0.02, 0.02)
            position, rotation = None, None
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


def _make_policy(*, visual_calibration):
    from submission.policy_modular_heads import make_policy as make_heads

    baseline = make_heads()
    default = Path(__file__).resolve().parents[2] / "checkpoints/astra/normal_v1.npz"
    normal = (
        SeatedNormal(baseline.task, os.environ.get("ADV_PLACEMENT_NORMAL") or default)
        if visual_calibration
        else None
    )

    class PlacementPolicy(type(baseline)):
        def __init__(self):
            self.task = baseline.task
            self.controller = SeatedReleaseController(baseline.controller.kin, self.task, normal)
            self.task.controller = self.controller
            self.reset()

    return _with_phase_trace(PlacementPolicy())


def make_policy():
    return _make_policy(visual_calibration=True)


def make_fk_policy():
    return _make_policy(visual_calibration=False)


class GuardedClearance:
    """Move away from the rim once supported recontact is measured."""

    def __init__(self, detector):
        self.detector = detector
        self.reset()

    def reset(self):
        self.detector.reset()
        self.samples = deque(maxlen=5)
        self.start = None
        self.plate = None
        self.tilt = None
        self.direction = None
        self.travel = 0.0
        self.clear_frames = 0
        self.finished = False

    def update(self, state, task, elapsed, hand, rotation):
        contact = self.detector.update(state)
        self.samples.append(task["plate_xyz"])
        plate = np.median(self.samples, axis=0)
        error = plate - task["site_xyz"]
        contacts = task["contacts"]
        grippers_open = bool(np.all(state[layout.GRIPPERS] <= 0.08))
        supported = (
            contacts["target_rack"]
            and not (contacts["table"] or contacts["floor"])
            and abs(error[0]) < 0.026
            and abs(error[1]) < 0.024
            and -0.048 < error[2] < -0.018
        )
        if self.finished or elapsed < 8:
            return None
        if self.start is None:
            if not (contact and supported and grippers_open and task["tilt_deg"] < 17):
                return None
            # Slide radially away from the center in the rack plane. Preserve
            # wrist angle so the pads do not sweep across the plate faces.
            rack_axis = -task["site_rot"][:, 0]
            radial = hand - plate
            horizontal = rack_axis @ radial
            if horizontal <= 0.04:
                return None
            self.start = hand.copy()
            self.plate = plate
            self.tilt = task["tilt_deg"]
            self.direction = rack_axis
            self.rotation = rotation.copy()
        self.clear_frames = 0 if contact else self.clear_frames + 1
        if (
            not supported
            or np.linalg.norm(plate - self.plate) > 0.008
            or task["tilt_deg"] > self.tilt + 4
            or self.clear_frames >= 3
            or self.travel >= 0.025
            or not grippers_open
        ):
            self.finished = True
            return None
        self.travel = min(self.travel + 0.0008, 0.025)
        return self.start + self.travel * self.direction, self.rotation


class GuardedClearanceController(SeatedReleaseController):
    """Optional open-finger recovery with one bounded clearance attempt."""

    def reset(self):
        from .recontact import OpenFingerContact

        super().reset()
        self.clearance = GuardedClearance(OpenFingerContact())
        self.clearance_pose = None

    def _enter(self, task, hand_world, hand_rot):
        super()._enter(task, hand_world, hand_rot)
        if self.phase == Phase.RELEASE:
            self.clearance.reset()

    def act(self, obs_state, task_state=None):
        self.clearance_pose = None
        if self.phase in (Phase.RELEASE, Phase.SETTLE):
            state = np.asarray(obs_state)
            task_state = self._read_task(state, task_state)
            if self.phase == Phase.RELEASE:
                self.clearance.detector.update(state)
            else:
                self.kin.set_state(state)
                hand, rotation = self.kin.hand_pose("left")
                basis = pelvis_rotation(state)
                self.clearance_pose = self.clearance.update(
                    state,
                    {**task_state, "site_xyz": self.target_xyz, "site_rot": self.target_rot},
                    self.machine.elapsed,
                    state[layout.PELVIS_XYZ] + basis @ hand,
                    basis @ rotation,
                )
        return super().act(obs_state, task_state)

    def _action(self, state, position=None, rotation=None, base=None, close=False):
        if self.phase == Phase.SETTLE and self.clearance_pose is not None:
            position, rotation = self.clearance_pose
        return super()._action(state, position, rotation, base, close)
