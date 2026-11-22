import numpy as np

from method.modular.loaded_wrist import FOCAL, _outline, register


def test_front_facing_plate_subtends_its_metric_lip_diameter():
    bounds = _outline(np.array([[0, 0, -0.50]]), np.array([[0.0, 0.0, 1.0]]), np.array([41.5]))
    radius_px = FOCAL * 0.11415 / (0.50 - 0.0174)
    np.testing.assert_allclose(bounds[0, 0], [41.5 - radius_px, 41.5 + radius_px], atol=0.02)


def test_blank_wrist_cannot_supply_a_two_view_pose():
    head = np.zeros((84, 84), dtype=bool)
    head[12:34, 20:37] = True
    assert register(head, np.zeros_like(head), np.zeros(3), np.eye(3), np.zeros(3)) is None
