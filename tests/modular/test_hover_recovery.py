from types import SimpleNamespace

import numpy as np

from method.modular import layout
from method.modular.controller import TARGET_STANDOFF_XY
from method.modular.phases import Phase
from submission.policy_hover import RetryStandoff


class ExistingPlacement:
    def reset(self):
        self.phase = Phase.LOWER
        self.target_xyz = np.array([0.52, -0.12, 0.85])
        self.kin = self
        self.machine = SimpleNamespace(transitions=[])

    def _enter(self, task, hand, rotation):
        pass

    def set_state(self, state):
        self.height = state[0]

    def hand_pose(self, side):
        return np.array([0, 0, self.height]), np.eye(3)

    def act(self, state, task):
        ready = self._ready_to_lower(task, None) if self.phase == Phase.ALIGN else None
        return ready, self._action(state)

    def _ready_to_lower(self, task, desired):
        return task["align_ready"]

    def _action(self, state, position=None, rotation=None, base=None, close=False):
        return base

    def _ready_to_release(self, task):
        return task["fallback"]


class Combined(RetryStandoff, ExistingPlacement):
    pass


def test_restore_standoff_only_after_clearing_failed_lower():
    ctl = Combined()
    ctl.reset()
    state = np.zeros(50)
    goal = ctl.target_xyz[:2] - TARGET_STANDOFF_XY
    state[layout.PELVIS_XY] = goal - [0.16, 0.0]
    state[layout.PELVIS_HEIGHT] = 0.75
    task = {"align_ready": True, "held": True, "contacts": {"target_rack": False}, "fallback": True}
    assert ctl.act(state, task) == (None, None)
    assert ctl._ready_to_release(task)

    ctl.phase = Phase.ALIGN
    ctl.machine.transitions.append({"from": "CARRY"})
    ctl._enter(task, [0, 0, 0.81], None)
    assert ctl.act(state, task) == (True, None)
    ctl.machine.transitions.append({"from": "LOWER"})
    ctl._enter(task, [0, 0, 0.81], None)
    state[0] = 0.10
    assert ctl.act(state, task) == (True, None)
    state[0] = 0.15
    task["contacts"]["target_rack"] = True
    assert ctl.act(state, task) == (True, None)
    task["contacts"]["target_rack"] = False
    ready, base = ctl.act(state, task)
    assert not ready
    np.testing.assert_array_equal(base, goal)

    state[layout.PELVIS_XY] = goal - [0.02, 0.0]
    state[layout.PELVIS_LINEAR_VEL.start] = 0.10
    ready, base = ctl.act(state, task)
    assert not ready
    np.testing.assert_array_equal(base, state[layout.PELVIS_XY])
    state[layout.PELVIS_LINEAR_VEL.start] = 0.0
    assert ctl.act(state, task) == (True, None)
    ctl.reset()
    assert ctl.retry_hand_z is None and ctl.stance_goal is None
