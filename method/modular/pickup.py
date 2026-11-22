"""Observation-only source localization and pinch/lift verification."""

import numpy as np

from . import layout
from .estimated import PLATE_RADIUS
from .kinematics import pelvis_rotation
from .observed_controller import CalibratedPlacementController, LockedSiteController
from .phases import Phase
from .pickup_vision import SourceView

WINDOW = 6


class PickupController(CalibratedPlacementController):
    """Stop at two camera viewpoints and verify attachment before moving the base."""

    def reset(self):
        super().reset()
        self.initial_samples = []
        self.near_samples = []
        self.view = SourceView()
        self.source = None
        self.hand_to_center = None
        self.fk_rise = 0.0

    def _enter(self, task, hand_world, hand_rot):
        super()._enter(task, hand_world, hand_rot)
        if self.phase == Phase.PREGRASP:
            # A failed pinch can displace the plate. Discard the old near view.
            self.near_samples.clear()
            self.view = SourceView()
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

            if "left_wrist_rgb" in task:
                basis = pelvis_rotation(state)
                self.kin.set_state(state)
                pos, rot = self.kin.hand_pose("left")
                self.source = self.view.update(
                    task["left_wrist_rgb"],
                    state[layout.PELVIS_XYZ] + basis @ pos,
                    basis @ rot,
                    self.source,
                    task["tilt_deg"],
                )

        basis = pelvis_rotation(state)
        if phase == Phase.GRASP and self.view.rotation is not None:
            task = {
                **task,
                "plate_rot": self.view.rotation,
                "plate_rot_rel": basis.T @ self.view.rotation,
            }
        if phase == Phase.GRASP and not self.closing:
            task = {**task, "held": False}
        elif phase in (Phase.GRASP, Phase.LIFT):
            self.kin.set_state(state)
            pos, rot = self.kin.hand_pose("left")
            hand = state[layout.PELVIS_XYZ] + basis @ pos
            fingers = np.mean(state[layout.LEFT_FINGERS])
            pinched = state[layout.GRIPPERS][0] >= 0.80 and fingers >= -0.0185
            if phase == Phase.GRASP:
                # The loaded hand can deflect several cm along the plate and
                # still hold the rim. Check physical compatibility with the
                # observed plate plane and radius during settled closure.
                delta = hand - self.source
                normal = task["plate_rot"][:, 2]
                depth = delta @ normal
                radius = np.linalg.norm(delta - depth * normal)
                consistent = abs(depth) < 0.04 and 0.055 < radius < PLATE_RADIUS + 0.025
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
