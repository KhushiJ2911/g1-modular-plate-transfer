"""Diagnostic intervention on the demonstration-replaying teacher v1."""

from __future__ import annotations

import numpy as np

FREEZE_STEPS = 75
OPEN_TOLERANCE = 0.08


class TeacherFreeze:
    """Pause replay with open fingers, then withdraw only while placement holds."""

    def __init__(self, teacher):
        self.teacher = teacher

    def reset(self, env):
        self.teacher.reset(env)
        self.freeze_action = None
        self.reference = None
        self.frozen_steps = 0
        self.open_steps = 0
        self.retreat_steps = 0
        self.guarded_steps = 0
        self.last_retreat = None
        self.guard_hold = None
        self.guard_tripped = False
        self.phase = "TEACHER"

    def act(self, ts):
        teacher = self.teacher
        if self.freeze_action is None:
            previous = teacher.last_action
            action = teacher.act(ts)
            if teacher.release_snapshot is not None and np.all(action[18:] <= 0):
                self.freeze_action = action.copy()
                self.freeze_action[[0, 1, 3]] = 0
                self.freeze_action[4:18] = previous[4:18]
                self.freeze_action[18:] = -1
                teacher.last_action = self.freeze_action.copy()
                teacher.previous_gripper = -1
                self.phase = "FREEZE"
                self.frozen_steps += 1
                return self.freeze_action.copy()
            return action

        snap = teacher.observe()
        fully_open = np.all(np.abs(ts.low_dim_obs[44:46]) <= OPEN_TOLERANCE)
        if self.open_steps < FREEZE_STEPS:
            self.open_steps = self.open_steps + 1 if fully_open else 0
            self.phase = "FREEZE"
            self.frozen_steps += 1
            teacher.physical_step += 1
            return self.freeze_action.copy()

        if self.reference is None:
            self.reference = snap
            self.last_retreat = self.freeze_action.copy()
        drift = np.linalg.norm(
            np.asarray(snap["pose"][:3]) - np.asarray(self.reference["pose"][:3])
        )
        self.guard_tripped |= bool(
            not fully_open
            or snap["held"]
            or not snap["contacts"]["target_rack"]
            or drift > 0.015
            or snap["tilt_deg"] > self.reference["tilt_deg"] + 5
        )
        if self.guard_tripped:
            if self.guard_hold is None:
                self.guard_hold = self.last_retreat.copy()
            self.phase = "RETREAT_GUARD"
            self.guarded_steps += 1
            teacher.physical_step += 1
            return self.guard_hold.copy()

        action = teacher.act(ts)
        action[[0, 1, 3]] = 0
        action[4:18] = self.last_retreat[4:18] + np.clip(
            action[4:18] - self.last_retreat[4:18], -0.012, 0.012
        )
        action[18:] = -1
        self.last_retreat = action.copy()
        teacher.last_action = action.copy()
        self.phase = "RETREAT"
        self.retreat_steps += 1
        return action
