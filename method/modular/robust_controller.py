"""Observation-only pickup, rigid grasp tracking, and rack seating feedback."""

import json
import os
from collections import deque

import numpy as np

from . import layout
from .controller import ModularController
from .kinematics import pelvis_rotation
from .observed_controller import (
    CalibratedPlacementController,
    LockedSiteContacts,
    LockedSiteController,
)
from .phases import TIMEOUTS, Phase

SOURCE_FRAMES = 20
STANDOFF_FRAMES = 20
ACTIVE_GRASP = (Phase.LIFT, Phase.CARRY, Phase.ALIGN, Phase.LOWER, Phase.RELEASE)


class StationaryPickup:
    """Take separate world-frame medians before walking and before grasping."""

    def __init__(self):
        self.initial = []
        self.near = []

    def update(self, state, task, phase, elapsed, closing):
        if phase == Phase.APPROACH:
            samples = self.initial
        elif phase in (Phase.PREGRASP, Phase.GRASP):
            if phase == Phase.PREGRASP and elapsed == 1:
                self.near.clear()
            samples = self.near
        else:
            return task
        if phase != Phase.GRASP and len(samples) < SOURCE_FRAMES:
            samples.append(task["plate_xyz"].copy())
        if not samples:
            return task
        plate = np.median(samples, axis=0)
        return {
            **task,
            "plate_xyz": plate,
            "plate_rel": pelvis_rotation(state).T @ (plate - state[layout.PELVIS_XYZ]),
        }


class RigidGrasp:
    """Capture the measured plate-to-hand transform at confirmed closure."""

    def __init__(self, kin):
        self.kin = kin
        self.centers = deque(maxlen=5)
        self.offset = None
        self.rotation = None
        self.slip_frames = 0

    def _hand(self, state, basis):
        self.kin.set_state(state)
        position, rotation = self.kin.hand_pose("left")
        return state[layout.PELVIS_XYZ] + basis @ position, basis @ rotation

    def update(self, state, task, visual, phase):
        aperture = state[layout.GRIPPERS][0]
        if phase not in (*ACTIVE_GRASP, Phase.GRASP):
            self.centers.clear()
            self.offset = None
            self.slip_frames = 0
            return task
        if aperture < 0.65:
            if phase == Phase.GRASP:
                self.centers.clear()
            return task
        basis = pelvis_rotation(state)
        hand, hand_rotation = self._hand(state, basis)
        if phase == Phase.GRASP:
            self.centers.append(hand_rotation.T @ (task["plate_xyz"] - hand))
            return task
        if self.offset is None:
            return task
        plate = hand + hand_rotation @ self.offset
        rotation = hand_rotation @ self.rotation
        tilt = np.rad2deg(np.arccos(np.clip(rotation[:, 2] @ task["site_rot"][:, 2], -1, 1)))
        # The visual prediction can be biased under arm occlusion. Require
        # visual disagreement about both attachment and position to flag slip.
        slipped = not visual["held"] and np.linalg.norm(visual["plate_xyz"] - plate) > 0.065
        self.slip_frames = self.slip_frames + 1 if slipped else max(0, self.slip_frames - 1)
        return {
            **task,
            "held": self.slip_frames < 6,
            "plate_xyz": plate,
            "plate_rel": basis.T @ (plate - state[layout.PELVIS_XYZ]),
            "plate_rot": rotation,
            "plate_rot_rel": basis.T @ rotation,
            "tilt_deg": float(tilt),
        }

    def capture(self, task, hand, rotation):
        self.offset = (
            np.median(self.centers, axis=0)
            if self.centers
            else rotation.T @ (task["plate_xyz"] - hand)
        )
        self.rotation = rotation.T @ task["plate_rot"]
        self.slip_frames = 0


class StandoffSite(LockedSiteContacts):
    """Navigate on a provisional estimate, then lock observed rack geometry in ALIGN."""

    def reset(self):
        super().reset()
        self.approach_sites = deque(maxlen=STANDOFF_FRAMES)
        self.standoff_sites = []
        self.locked = False

    def update_at_phase(self, state, task, phase):
        if not self.locked:
            if phase == Phase.ALIGN:
                self.standoff_sites.append(task["site_xyz"].copy())
                if len(self.standoff_sites) == STANDOFF_FRAMES:
                    self.site = np.median(self.standoff_sites, axis=0)
                    self.locked = True
                    self._plate_frames.clear()
            elif phase == Phase.APPROACH and len(self.approach_sites) < STANDOFF_FRAMES:
                self.approach_sites.append(task["site_xyz"].copy())
                self.site = np.median(self.approach_sites, axis=0)
        return self._with_site(state, task)


class SeatingFeedback:
    """Gate release on FK depth or a near-rack stall under commanded descent."""

    def __init__(self):
        self.depth_history = deque(maxlen=12)

    def ready(self, task, site, elapsed):
        # Use the same final insertion depth as the nominal controller path.
        # A contact stall must occur while the trajectory is asking for descent.
        x, y, depth = task["plate_xyz"] - site
        self.depth_history.append((elapsed, float(depth)))
        old_step, old_depth = self.depth_history[0]
        lowering = 0.135 * (min(elapsed, 125) - min(old_step, 125)) / 125
        stalled = lowering >= 0.009 and abs(depth - old_depth) < 0.003
        centered = abs(x) < 0.032 and abs(y) < 0.027
        near_rack = -0.059 < depth < -0.015
        at_depth = depth <= -0.032
        late = elapsed >= TIMEOUTS[Phase.LOWER] - 16
        # The late gate accepts a slightly shallower seat, with the same
        # horizontal clearance and tilt guard as the ordinary depth check.
        return (
            task["held"]
            and centered
            and near_rack
            and task["tilt_deg"] <= 18.0
            and (at_depth or stalled or late)
        )


class RobustPlacementController(CalibratedPlacementController):
    """Apply cumulative changes with each earlier configuration retained for A/B."""

    def __init__(self, kin, task_state_source, *, rigid=False, late_site=False, seating=False):
        self.use_rigid = rigid
        self.use_late_site = late_site
        self.use_seating = seating
        super().__init__(kin, task_state_source)

    def reset(self):
        super().reset()
        self.source_plate = StationaryPickup()
        self.grasp = RigidGrasp(self.kin) if self.use_rigid else None
        if self.use_late_site:
            self.site_contacts = StandoffSite()
        self.seating = SeatingFeedback() if self.use_seating else None

    def _enter(self, task, hand_world, hand_rot):
        super()._enter(task, hand_world, hand_rot)
        if self.phase == Phase.LIFT and self.grasp is not None:
            self.grasp.capture(task, hand_world, hand_rot)
            # Keep the control target and observed rigid transform reciprocal.
            self.plate_to_hand = (
                -self.grasp.rotation.T @ self.grasp.offset,
                self.grasp.rotation.T,
            )
            self.lift_origin = hand_world + hand_rot @ self.grasp.offset
        if self.phase == Phase.LOWER and self.seating is not None:
            self.seating.depth_history.clear()

    def _ready_to_lower(self, task, desired):
        site_ready = not self.use_late_site or self.site_contacts.locked
        return site_ready and super()._ready_to_lower(task, desired)

    def _ready_to_release(self, task):
        if self.seating is None:
            return super()._ready_to_release(task)
        return self.seating.ready(task, self.target_xyz, self.machine.elapsed)

    def _action(self, state, position=None, rotation=None, base=None, close=False):
        # Keep the camera poses fixed while collecting each source median.
        pickup_sampling = (
            self.phase in (Phase.APPROACH, Phase.PREGRASP) and self.machine.elapsed <= SOURCE_FRAMES
        )
        site_sampling = (
            self.use_late_site and self.phase == Phase.ALIGN and not self.site_contacts.locked
        )
        if pickup_sampling or site_sampling:
            position, rotation = None, None
        return super()._action(state, position, rotation, base, close)

    def act(self, obs_state, task_state=None):
        state = np.asarray(obs_state)
        visual = self._read_task(state, task_state)
        task = self.source_plate.update(
            state, visual, self.phase, self.machine.elapsed, self.closing
        )
        task = self.orientation.update(state, task, self.phase)
        if self.grasp is not None:
            task = self.grasp.update(state, task, visual, self.phase)
        if not self.use_late_site:
            return LockedSiteController.act(self, state, task)

        was_locked = self.site_contacts.locked
        task = self.site_contacts.update_at_phase(state, task, self.phase)
        if self.last_raw is not None:
            self.target_xyz = task["site_xyz"].copy()
        if not was_locked and self.site_contacts.locked:
            self.place_correction.fill(0)
            self.place_rotation_bias.fill(0)
        return ModularController.act(self, state, task)


def _make_policy(*, rigid=False, late_site=False, seating=False):
    from submission.policy_modular import make_policy

    policy = make_policy()
    policy.controller = RobustPlacementController(
        policy.controller.kin, policy.task, rigid=rigid, late_site=late_site, seating=seating
    )
    return _with_phase_trace(policy)


class _PhaseTrace:
    def __init__(self, policy):
        self.policy = policy
        self.seen = 0

    def reset(self):
        self.policy.reset()
        self.seen = 0

    def __call__(self, obs):
        action = self.policy(obs)
        transitions = self.policy.controller.transitions
        for transition in transitions[self.seen :]:
            print("SOL_PHASE " + json.dumps(transition), flush=True)
        self.seen = len(transitions)
        return action


def _with_phase_trace(policy):
    return _PhaseTrace(policy) if os.environ.get("SOL_PHASE_TRACE") == "1" else policy


def make_previous():
    from submission.policy_modular_grasp import make_policy

    return _with_phase_trace(make_policy())


def make_pickup():
    return _make_policy()


def make_rigid():
    return _make_policy(rigid=True)


def make_late_site():
    return _make_policy(rigid=True, late_site=True)


def make_seating():
    return _make_policy(rigid=True, late_site=True, seating=True)
