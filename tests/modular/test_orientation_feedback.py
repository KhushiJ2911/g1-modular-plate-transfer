import numpy as np

from method.modular.estimated import SITE_ROTATION
from method.modular.orientation_feedback import PlacementOrientation
from method.modular.phases import Phase


def test_disable_signed_feedback_preserves_the_tilt_gate():
    task = {"site_rot": SITE_ROTATION, "plate_rot": np.eye(3), "tilt_deg": 13.0}
    correction = PlacementOrientation(disable_signed=True)
    result = correction.update(np.zeros(50), task, Phase.LOWER)
    np.testing.assert_array_equal(result["plate_rot"], SITE_ROTATION)
    assert result["tilt_deg"] == 13.0
    assert correction.update(np.zeros(50), task, Phase.GRASP) is task


def test_disagreement_cannot_drive_signed_correction():
    task = {"site_rot": SITE_ROTATION, "plate_rot": np.eye(3), "tilt_deg": 5.0}
    correction = PlacementOrientation()
    result = correction.update(np.zeros(50), task, Phase.LOWER)
    np.testing.assert_allclose(result["plate_rot"], SITE_ROTATION, atol=1e-10)
    assert result["tilt_deg"] == 5.0


def test_calibrated_grasp_normal_uses_measured_hand_after_closure():
    from method.modular.orientation_feedback import LEFT_GRASP_NORMAL, CalibratedGraspOrientation

    class Kin:
        def set_state(self, state):
            pass

        def hand_pose(self, side):
            return np.zeros(3), np.eye(3)

    state = np.zeros(50)
    task = {"site_rot": SITE_ROTATION, "plate_rot": np.eye(3), "tilt_deg": 8.0}
    model = CalibratedGraspOrientation(Kin())
    assert model.update(state, task, Phase.GRASP) is task
    state[44] = 0.9
    result = model.update(state, task, Phase.GRASP)
    np.testing.assert_allclose(result["plate_rot"][:, 2], LEFT_GRASP_NORMAL)
    np.testing.assert_allclose(result["plate_rot"].T @ result["plate_rot"], np.eye(3), atol=1e-12)
    assert 8.0 < result["tilt_deg"] < 20.0
