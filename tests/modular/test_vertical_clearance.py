from types import SimpleNamespace

import numpy as np
import pytest

from method.modular import layout
from method.modular.base_clearance import QUIET_STEPS
from method.modular.kinematics import pelvis_rotation
from method.modular.phases import TIMEOUTS, Phase
from method.modular.vertical_clearance import (
    HOVER_HEIGHT,
    RETREAT_INCREMENT,
    VerticalClearance,
    VerticalClearancePolicy,
)


class Hand:
    def set_state(self, state):
        self.state = state

    def hand_pose(self, side):
        assert side == "left"
        return self.state[:3].copy(), np.eye(3)


class Controller:
    def __init__(self):
        self.kin = Hand()
        self.last_raw = np.zeros(20)
        self.last_raw[layout.LEFT_ARM_ACTION] = [0.12, 0.20, 0.03, 0, 0, 0, 0]
        self.target_xyz = np.array([0.25, 0.5, 0.75])
        self.target_rot = np.eye(3)
        self.calls = []

    def _action(self, state, position, rotation, close):
        self.calls.append((position, rotation, close))
        return np.ones(20)


@pytest.fixture
def inputs():
    ctl = Controller()
    state = np.zeros(50)
    state[layout.PELVIS_YAW] = np.pi / 2
    state[layout.PELVIS_XYZ] = [0.1, -0.5, 0.6]
    task = {
        "contacts": {"target_rack": False, "table": False, "floor": False},
        "plate_xyz": ctl.target_xyz - [0, 0, 0.03],
        "tilt_deg": 10,
    }
    return np.zeros(20), state, task, ctl


def test_vertical_retreat_preserves_commanded_world_xy_and_rotation(inputs):
    original, state, task, ctl = inputs
    lift = VerticalClearance()
    for _ in range(QUIET_STEPS - 1):
        assert lift.apply(*inputs) is original

    lift.apply(*inputs)
    start = state[layout.PELVIS_XYZ] + pelvis_rotation(state) @ ctl.last_raw[4:7]
    pos, rot, close = ctl.calls[-1]
    np.testing.assert_allclose(pos, start + [0, 0, RETREAT_INCREMENT])
    np.testing.assert_allclose(rot, pelvis_rotation(state))
    assert close is False and ctl.kin.state is state
    state[layout.PELVIS_XY] += 0.02
    state[layout.PELVIS_YAW] += 0.02
    lift.apply(*inputs)
    np.testing.assert_allclose(ctl.calls[-1][0], start + [0, 0, 2 * RETREAT_INCREMENT])
    np.testing.assert_array_equal(ctl.calls[-1][1], rot)
    task["contacts"]["table"] = True
    assert lift.apply(*inputs) is original
    task["contacts"]["table"] = False
    assert lift.apply(*inputs) is original and lift.finished


@pytest.mark.parametrize("guard", ["closed", "high", "far", "tilted", "floor"])
def test_vertical_retreat_requires_open_supported_hand(inputs, guard):
    original, state, task, ctl = inputs
    lift = VerticalClearance()
    lift.quiet_steps = QUIET_STEPS
    if guard == "closed":
        state[layout.GRIPPERS] = [0.5, 0]
    elif guard == "high":
        task["plate_xyz"][2] += 0.1
    elif guard == "far":
        task["plate_xyz"][0] += 0.1
        task["contacts"]["target_rack"] = True
    elif guard == "tilted":
        task["tilt_deg"] = 30
    else:
        task["contacts"]["floor"] = True
    assert lift.apply(*inputs) is original and not ctl.calls


@pytest.mark.parametrize("limit", ["travel", "timeout"])
def test_vertical_retreat_stops_at_travel_or_timeout(inputs, limit):
    original, state, task, ctl = inputs
    lift = VerticalClearance()
    lift.quiet_steps = QUIET_STEPS
    lift.apply(*inputs)
    if limit == "travel":
        state[2] += HOVER_HEIGHT + 1e-4
    else:
        lift.steps = TIMEOUTS[Phase.RETREAT]
    assert lift.apply(*inputs) is original and lift.finished


def test_wrapper_gives_incumbent_priority_and_resets(inputs):
    original, state, task, ctl = inputs
    ctl.phase = Phase.HOLD
    ctl.machine = SimpleNamespace(elapsed=0)
    ctl.transitions = [{"from": "RETREAT"}]

    class Incumbent:
        def __init__(self):
            self.controller = ctl
            self.task = SimpleNamespace(snapshot=lambda state: task)
            self.resets = 0

        def __call__(self, obs):
            return original

        def reset(self):
            self.resets += 1

    policy = VerticalClearancePolicy(Incumbent())
    obs = {"state": state}
    assert policy(obs) is original
    policy.recovery.origin = np.zeros(2)
    assert policy(obs) is original and policy.lift.quiet_steps == 0
    policy.recovery.reason = "clearance bound"
    assert policy(obs) is original and policy.lift.quiet_steps == 1
    policy.reset()
    assert policy.policy.resets == 1 and policy.recovery.origin is None
    assert vars(policy.lift) == vars(VerticalClearance())
