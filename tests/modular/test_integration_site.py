"""Placement correction respects yaw during both the six-frame lock and late perception."""

from types import SimpleNamespace

import numpy as np

from method.modular.kinematics import pelvis_rotation
from method.modular.site_estimator import PlacementEstimate, plate_features, site_features


def test_stationary_site_and_late_plate(tmp_path):
    state = np.zeros(50)
    visual = np.zeros(368)

    class Source:
        def reset(self):
            self.visual = visual

        def update(self, obs):
            basis = pelvis_rotation(obs["state"])
            self.task = dict(
                plate_rel=np.ones(3),
                site_xyz=np.zeros(3),
                plate_xyz=basis @ np.ones(3),
                target_site_rel=np.zeros(3),
            )
            return self.task

        def snapshot(self, state):
            return self.task

    weights = {}
    for name, x, correction in (
        ("site", site_features(visual, np.zeros(3)), [0.01, 0, 0]),
        ("plate", plate_features(visual, state, np.ones(3), np.zeros(3)), [0, 0, 0.02]),
    ):
        weights.update(
            {
                name + "_mean": np.zeros(len(x)),
                name + "_scale": np.ones(len(x)),
                name + "_coef": np.zeros((len(x), 3)),
                name + "_offset": correction,
            }
        )
    path = tmp_path / "weights.npz"
    np.savez(path, **weights)
    ctl = SimpleNamespace(phase=SimpleNamespace(value="APPROACH"))
    wrapped = PlacementEstimate(Source(), ctl, path, ("site", "plate"))
    for _ in range(6):
        initial = wrapped.update({"state": state})
    np.testing.assert_allclose(initial["site_xyz"], [0.01, 0, 0])
    np.testing.assert_allclose(initial["plate_rel"], np.ones(3))
    ctl.phase.value = "LOWER"
    state[21] = np.pi / 2
    task = wrapped.update({"state": state})
    np.testing.assert_allclose(task["target_site_rel"], [0, -0.01, 0], atol=1e-8)
    np.testing.assert_allclose(task["plate_xyz"], [-1, 1, 1.02])
    assert wrapped.snapshot(state) is task
    wrapped.reset()
    assert wrapped.site_delta is None


def test_pinch_center_requires_closed_held_grasp_before_release(tmp_path):
    state = np.zeros(50)
    state[18:22] = [0.20, -0.10, 0.80, np.pi / 2]
    state[44] = 1

    class Source:
        def reset(self):
            self.held = True

        def update(self, obs):
            return dict(
                held=self.held, plate_rel=np.zeros(3), plate_xyz=state[18:21], site_xyz=np.zeros(3)
            )

    class Hand:
        def set_state(self, observed):
            np.testing.assert_array_equal(observed, state)

        def hand_pose(self, side):
            assert side == "left"
            return np.array([0.10, 0.04, 0.02]), np.eye(3)

    source = Source()
    phase = SimpleNamespace(value="CARRY")
    ctl = SimpleNamespace(phase=phase, kin=Hand(), hand_to_center=np.array([0.01, 0.03, -0.08]))
    path = tmp_path / "pinch.npz"
    np.savez(path, pinch_center_bias=[0.001, 0.003, 0.004])
    adapter = PlacementEstimate(source, ctl, path, ("pinch",))
    obs = {"state": state}
    observed = adapter.update(obs)
    np.testing.assert_allclose(observed["plate_rel"], [0.109, 0.067, -0.064])
    np.testing.assert_allclose(observed["plate_xyz"], [0.133, 0.009, 0.736])
    for phase.value, source.held, state[44] in [
        ("RELEASE", True, 1),
        ("LOWER", False, 1),
        ("ALIGN", True, 0.5),
    ]:
        np.testing.assert_array_equal(adapter.update(obs)["plate_xyz"], state[18:21])
