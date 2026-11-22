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
