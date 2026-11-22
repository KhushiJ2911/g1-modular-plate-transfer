"""Regression checks for signed plate geometry and an observation-only state source."""

from types import SimpleNamespace

import numpy as np
import pytest

from method.modular.estimated import SITE_ROTATION
from method.modular.refined import CorrectedTaskState, align_normal, inputs


@pytest.mark.parametrize("normal", ([0.14, -0.96, -0.24], [1, 0, 0], [0, 0, 0]))
def test_normal_frame(normal):
    normal = np.array(normal)
    r = align_normal(SITE_ROTATION, normal)
    np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-10)
    assert np.linalg.det(r) == pytest.approx(1)
    if np.any(normal):
        np.testing.assert_allclose(r[:, 2], normal / np.linalg.norm(normal))


def test_public_state_source_uses_same_signed_normal_for_tilt(tmp_path):
    class Head:
        def register_forward_pre_hook(self, hook):
            self.hook = hook

    class Estimator:
        model = SimpleNamespace(head=[Head()])

        def __call__(self, obs):
            assert set(obs) == {"state", "rgb"}
            tensor = SimpleNamespace(
                detach=lambda: SimpleNamespace(
                    cpu=lambda: SimpleNamespace(numpy=lambda: np.zeros(368))
                )
            )
            self.model.head[0].hook(None, [[tensor]])
            return dict(
                plate_rel=np.array([0.4, 0.2, 0.1]),
                target_site_rel=np.array([0.4, -0.2, 0.12]),
                tilt_deg=5.0,
                held=0.0,
            )

    kin = SimpleNamespace(
        set_state=lambda state: None, hand_pose=lambda side: (np.zeros(3), np.eye(3))
    )
    policy = SimpleNamespace(
        controller=SimpleNamespace(kin=kin, phase=SimpleNamespace(value="LOWER")),
        task=SimpleNamespace(estimator=Estimator()),
    )
    state = np.zeros(50)
    width = inputs(np.zeros(368), state, np.zeros(12), np.zeros(11), np.zeros(3), np.zeros(3)).size
    weights = dict(groups="normal,contact")
    for group, bias in (("normal", np.array([0.2, 0.0, -0.1])), ("contact", np.array([1.0]))):
        weights.update(
            {
                group + "_mean": np.zeros(width),
                group + "_scale": np.ones(width),
                group + "_coef": np.zeros((width, len(bias))),
                group + "_bias": bias,
            }
        )
    path = tmp_path / "weights.npz"
    np.savez(path, **weights)
    task = CorrectedTaskState(policy, path)
    obs = dict(state=state, rgb=np.zeros((3, 3, 84, 84), dtype=np.uint8))
    sample = task.update(obs)
    assert sample["contacts"]["target_rack"]
    assert sample["tilt_deg"] == pytest.approx(np.rad2deg(np.arccos(-sample["plate_rot"][1, 2])))
    np.testing.assert_allclose(sample["plate_rot"], sample["plate_rot_rel"])
    assert task.snapshot(state) is sample
    task.reset()
    assert task.first_site is None
