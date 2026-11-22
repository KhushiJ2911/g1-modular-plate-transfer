import numpy as np
import pytest

from method.modular import layout
from method.modular.controller import ModularController
from method.modular.phases import SETTLE_STEPS, TIMEOUTS, Phase, PhaseMachine
from method.modular.run_privileged import failure_bucket


class FixedKinematics:
    def set_state(self, state):
        pass

    def hand_pose(self, side):
        return np.array([0.38, 0.20, 0.12]), np.eye(3)

    def ik(self, side, target_pos, target_rot, **kwargs):
        self.ik_target = target_pos
        return kwargs["q_init"] + 0.1


def placement():
    return {
        "site_xyz": np.array([0.47, 0.2, 0.85]),
        "site_rot": np.eye(3),
        "plate_xyz": np.array([0.47, 0.2, 0.82]),
        "tilt_deg": 6.0,
        "held": False,
        "contacts": {"target_rack": True, "source_rack": False, "table": False, "floor": False},
    }


def test_release_freezes_until_both_grippers_open_and_settled():
    kin = FixedKinematics()
    controller = ModularController(kin)
    state = np.zeros(50, dtype=np.float32)
    state[layout.PELVIS_HEIGHT] = 0.75
    state[layout.LEFT_ARM] = 0.1
    task = placement()
    controller.act(state, task)
    controller.machine.phase = Phase.RELEASE
    task["held"] = True
    state[layout.GRIPPERS] = 0.9
    release = controller.act(state, task)
    np.testing.assert_array_equal(release[layout.GRIPPER_ACTION], [-1, -1])
    for _ in range(10):
        controller.act(state, task)
    assert controller.phase == Phase.RELEASE

    state[layout.GRIPPERS] = 0
    task["held"] = False
    controller.act(state, task)
    for _ in range(SETTLE_STEPS - 1):
        action = controller.act(state, task)
        assert controller.phase == Phase.SETTLE
        np.testing.assert_array_equal(action[4:18], release[4:18])
    controller.act(state, task)
    assert controller.phase == Phase.RETREAT
    controller.act(state, task)
    np.testing.assert_allclose(kin.ik_target, [0.377, 0.20, 0.12], atol=1e-7)

    task["plate_xyz"] += [0, 0.02, 0]
    guarded = controller.act(state, task)
    assert controller.phase == Phase.HOLD
    np.testing.assert_array_equal(guarded, controller.act(state, task))


def test_grasp_timeout_retries_are_bounded():
    machine = PhaseMachine()
    machine.phase = Phase.GRASP
    for count in (1, 2):
        machine.elapsed = TIMEOUTS[machine.phase]
        machine.advance(False)
        assert machine.phase == Phase.PREGRASP
        assert machine.grasp_retries == count
    machine.elapsed = TIMEOUTS[machine.phase]
    machine.advance(False)
    assert machine.phase == Phase.HOLD


@pytest.mark.parametrize(
    ("end_held", "bucket"),
    [(True, "D1 on rack, never released"), (False, "D2 on rack, not settled")],
)
def test_failure_buckets(end_held, bucket):
    record = {
        "success": False,
        "termination": "timeout",
        "first_held": 12,
        "m2": True,
        "min_target_xy_while_lifted": 0.01,
        "end_contacts": {"target_rack": True},
        "end_held": end_held,
    }
    assert failure_bucket(record) == bucket
