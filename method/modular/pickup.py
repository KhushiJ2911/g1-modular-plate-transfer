"""Observation-only source localization and pinch/lift verification."""

import mujoco
import numpy as np

from . import layout
from .kinematics import pelvis_rotation
from .observed_controller import CalibratedPlacementController, LockedSiteController
from .phases import Phase

WINDOW = 6

# Wrist camera mount and 60 degree vertical FOV from the fixed G1 geometry.
_camera_q, _camera_mat = np.empty(4), np.empty(9)
mujoco.mju_euler2Quat(_camera_q, np.array([0.0, -0.87, -1.57]), "xyz")
mujoco.mju_quat2Mat(_camera_mat, _camera_q)
WRIST_CAMERA = _camera_mat.reshape(3, 3)
FOCAL_PIXELS = 42 / np.tan(np.deg2rad(30))


def rim_lateral(rgb, hand, rotation, height):
    """World y of an upright plate from the bright front lip at PREGRASP."""
    # A continuous vertical highlight distinguishes the rim from isolated
    # highlights on rack pins. Discard cropped and ambiguous wrist views.
    strip = rgb[:, 16:49, 5:79].min(axis=0)
    u = strip.argmax(axis=1)
    v = np.arange(len(u))
    bright = strip[v, u] > 195
    if bright.sum() < 9:
        return None
    u, v = u[bright] + 5, v[bright] + 16
    center = np.median(u)
    if np.median(np.abs(u - center)) > 3:
        return None
    ray = np.array([(center - 41.5) / FOCAL_PIXELS, (41.5 - np.median(v)) / FOCAL_PIXELS, -1])
    # The reached standoff fixes the observed lip at about 10.5 cm depth.
    # Account for the site-to-camera mount, nominal source lean, and the
    # single lateral calibration shared by all continuous source positions.
    point = hand + rotation @ ([-0.028, 0, 0.040] + 0.105 * (WRIST_CAMERA @ ray))
    return point[1] - 0.12 * (point[2] - height) + 0.020


class PickupController(CalibratedPlacementController):
    """Stop at two camera viewpoints and verify attachment before moving the base."""

    def reset(self):
        super().reset()
        self.initial_samples = []
        self.near_samples = []
        self.rim_samples = []
        self.source = None
        self.hand_to_center = None
        self.fk_rise = 0.0

    def _enter(self, task, hand_world, hand_rot):
        super()._enter(task, hand_world, hand_rot)
        if self.phase == Phase.PREGRASP:
            # A failed pinch can displace the plate. Discard the old near view.
            self.near_samples.clear()
            self.rim_samples.clear()
            self.hand_to_center = None
            self.fk_rise = 0.0
            if self.machine.grasp_retries:
                self.source = None
        elif self.phase == Phase.LIFT:
            self.hand_to_center = hand_rot.T @ (self.lift_origin - hand_world)

    def _action(self, state, position=None, rotation=None, base=None, close=False):
        # The GRASP entry is the reached pregrasp pose. Hold the arm targets
        # that got it there while observing, before starting the final reach.
        sampling = (self.phase == Phase.APPROACH and len(self.initial_samples) < WINDOW) or (
            self.phase == Phase.GRASP and len(self.near_samples) < WINDOW
        )
        if sampling:
            position = base = None
        return super()._action(state, position, rotation, base, close)

    def _pickup_task(self, state, task):
        phase = self.phase
        plate = task["plate_xyz"]
        if phase == Phase.APPROACH and len(self.initial_samples) < WINDOW:
            self.initial_samples.append(plate.copy())
            self.source = np.median(self.initial_samples, axis=0)
        elif phase == Phase.GRASP and len(self.near_samples) < WINDOW:
            self.near_samples.append(plate.copy())
            near = np.median(self.near_samples, axis=0)
            if self.machine.grasp_retries:
                self.source = near
            else:
                # Average the independent views in xy. The initial clear view
                # gives a more accurate height before the wrist occludes the rim.
                initial = np.median(self.initial_samples, axis=0)
                self.source = initial.copy()
                self.source[:2] = (initial[:2] + near[:2]) / 2

            if "left_wrist_rgb" in task and task["tilt_deg"] < 18:
                basis = pelvis_rotation(state)
                self.kin.set_state(state)
                pos, rot = self.kin.hand_pose("left")
                y = rim_lateral(
                    task["left_wrist_rgb"],
                    state[layout.PELVIS_XYZ] + basis @ pos,
                    basis @ rot,
                    self.source[2],
                )
                if y is not None:
                    self.rim_samples.append(y)
            if len(self.rim_samples) >= 3:
                offset = np.median(self.rim_samples) - self.source[1]
                self.source[1] += 0.90 * np.clip(offset, -0.035, 0.035)

        basis = pelvis_rotation(state)
        if phase == Phase.GRASP and not self.closing:
            task = {**task, "held": False}
        elif phase in (Phase.GRASP, Phase.LIFT):
            self.kin.set_state(state)
            pos, rot = self.kin.hand_pose("left")
            hand = state[layout.PELVIS_XYZ] + basis @ pos
            fingers = np.mean(state[layout.LEFT_FINGERS])
            pinched = state[layout.GRIPPERS][0] >= 0.80 and fingers >= -0.0185
            if phase == Phase.GRASP:
                target, _ = self._grasp_target({**task, "plate_xyz": self.source}, phase)
                consistent = np.linalg.norm(hand - target) < 0.035
                settled = abs(np.mean(state[layout.LEFT_FINGERS_VEL])) < 0.012
                held = (
                    pinched
                    and settled
                    and consistent
                    and np.linalg.norm(plate - self.source) < 0.050
                )
            else:
                rigid = hand + basis @ rot @ self.hand_to_center
                self.fk_rise = rigid[2] - self.lift_origin[2]
                # Fingers slipping shut or a plate that stays behind for ten
                # consecutive observations trips the phase machine's bounded retry.
                held = pinched and np.linalg.norm(rigid - plate) < 0.060
            task = {**task, "held": bool(held)}

        if phase in (Phase.APPROACH, Phase.PREGRASP, Phase.GRASP) and self.source is not None:
            task = {
                **task,
                "plate_xyz": self.source,
                "plate_rel": basis.T @ (self.source - state[layout.PELVIS_XYZ]),
            }
        return task

    def _lift_ready(self, task):
        return self.fk_rise >= 0.085 and super()._lift_ready(task)

    def act(self, obs_state, task_state=None):
        state = np.asarray(obs_state)
        task = self._pickup_task(state, self._read_task(state, task_state))
        task = self.orientation.update(state, task, self.phase)
        return LockedSiteController.act(self, state, task)
