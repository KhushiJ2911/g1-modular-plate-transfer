import numpy as np

from method.modular.base_clearance import QUIET_STEPS, LateClearance


class Hand:
    def set_state(self, state):
        self.state = state

    def hand_pose(self, side):
        assert side == "left"
        # At yaw +90°, a hand on the positive local y side is at world −x.
        return np.array([0.0, 0.09, 0.03]), np.eye(3)


def test_clearance_waits_then_withdraws_in_pelvis_axes_and_stops_on_motion():
    state = np.zeros(50)
    state[21] = np.pi / 2
    kin = Hand()
    original = np.zeros(20, dtype=np.float32)
    original[18:] = -1
    task = {"plate_xyz": np.zeros(3), "tilt_deg": 5.0, "contacts": {"target_rack": True}}
    release = LateClearance()

    def act(i):
        return release.apply(original, state, kin, task, np.array([0, 0, 0.032]), np.eye(3), i)

    for age in range(1, QUIET_STEPS):
        assert act(age) is original
    issued = act(QUIET_STEPS)
    np.testing.assert_allclose(issued[:2], [0, 0.075], atol=1e-7)
    np.testing.assert_array_equal(issued[2:], original[2:])
    assert np.all(original[:2] == 0)

    task["plate_xyz"] = np.array([0.025, 0, 0])
    for age in range(QUIET_STEPS + 1, QUIET_STEPS + 4):
        issued = act(age)
    assert release.reason == "plate motion"
    assert issued is original
    task["plate_xyz"] = np.zeros(3)
    assert act(QUIET_STEPS + 25) is original
