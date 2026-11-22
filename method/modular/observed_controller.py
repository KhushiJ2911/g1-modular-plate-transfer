"""Observation-only refinements to J's phase inputs."""

from collections import deque

import numpy as np

from .controller import ModularController
from .estimated import PLATE_RADIUS
from .kinematics import pelvis_rotation
from .orientation_feedback import CalibratedGraspOrientation
from .phases import Phase

SITE_FRAMES = 6
POSE_FRAMES = 5


class LockedSiteContacts:
    """Estimate seating and table clearance relative to one stationary site."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._site_frames = []
        self._plate_frames = deque(maxlen=POSE_FRAMES)
        self.site = None

    def update(self, state, task):
        if len(self._site_frames) < SITE_FRAMES:
            self._site_frames.append(np.array(task["site_xyz"], dtype=float))
            self.site = np.median(self._site_frames, axis=0)
        return self._with_site(state, task)

    def _with_site(self, state, task):
        self._plate_frames.append(np.array(task["plate_xyz"], dtype=float))
        plate = np.median(self._plate_frames, axis=0)
        delta = plate - self.site
        # The release gate independently checks tighter center, depth and tilt.
        # Keep enough margin here to retain a seated plate while it rocks.
        target_contact = np.linalg.norm(delta) < 0.065 and -0.063 < delta[2] < -0.012
        normal_z = np.clip(task["plate_rot"][2, 2], -1, 1)
        lower_edge = plate[2] - PLATE_RADIUS * np.sqrt(1 - normal_z**2)
        contacts = {
            **task["contacts"],
            "target_rack": bool(target_contact),
            "table": bool(lower_edge <= self.site[2] - 0.17 + 0.006),
        }
        raw_delta = task["plate_xyz"] - self.site
        return {
            **task,
            "site_xyz": self.site.copy(),
            "target_site_rel": pelvis_rotation(state).T @ (self.site - state[18:21]),
            "site_distance": float(np.linalg.norm(raw_delta)),
            "site_xy_distance": float(np.linalg.norm(raw_delta[:2])),
            "contacts": contacts,
        }


class LockedSiteController(ModularController):
    """Share one site estimate between navigation, insertion and contact gates."""

    def reset(self):
        super().reset()
        self.site_contacts = LockedSiteContacts()

    def _read_task(self, state, task):
        if task is not None:
            return task
        if self.task_state_source is None:
            raise ValueError("An observation-derived task state is required")
        return self.task_state_source(state)

    def act(self, obs_state, task_state=None):
        state = np.asarray(obs_state)
        task = self.site_contacts.update(state, self._read_task(state, task_state))
        # J's first 24 approach steps have zero commanded base velocity.
        # Complete the initial six-sample lock in that same stationary window.
        if self.last_raw is not None and self.machine.step < SITE_FRAMES:
            self.target_xyz = task["site_xyz"].copy()
        return super().act(state, task)


class StableSourcePlate:
    """Use the stationary head view for walking and wrist views for pickup."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._initial = []
        self._recent = deque(maxlen=POSE_FRAMES)
        self._anchor = None

    def update(self, state, task, phase, elapsed, closing=False):
        if phase not in (Phase.APPROACH, Phase.PREGRASP, Phase.GRASP):
            return task
        position = np.array(task["plate_xyz"], dtype=float)
        if len(self._initial) < SITE_FRAMES:
            self._initial.append(position)
            self._anchor = np.median(self._initial, axis=0)
        self._recent.append(position)
        wrist = np.median(self._recent, axis=0)
        if phase == Phase.APPROACH:
            weight = 0.0
        elif phase == Phase.PREGRASP:
            weight = min(elapsed / 18, 1.0)
        else:
            weight = 1.0
        plate = (1 - weight) * self._anchor + weight * wrist
        if closing:
            # The source supports the plate up to initial pinch; the estimate
            # can suddenly dip when the hand occludes the center.
            plate[2] = max(plate[2], self._anchor[2] - 0.005)
        return {
            **task,
            "plate_xyz": plate,
            "plate_rel": pelvis_rotation(state).T @ (plate - state[18:21]),
        }


class CalibratedPlacementController(LockedSiteController):
    """J with observation-only pickup, grasp geometry and seating feedback."""

    def reset(self):
        super().reset()
        self.source_plate = StableSourcePlate()
        self.orientation = CalibratedGraspOrientation(self.kin)

    def act(self, obs_state, task_state=None):
        state = np.asarray(obs_state)
        task = self.source_plate.update(
            state,
            self._read_task(state, task_state),
            self.phase,
            self.machine.elapsed,
            self.closing,
        )
        task = self.orientation.update(state, task, self.phase)
        return super().act(state, task)
