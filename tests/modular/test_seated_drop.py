import numpy as np
import pytest

from method.modular.phases import Phase
from submission.policy_modular_round4 import RoundFourController


def controller():
    c = RoundFourController(None)
    c.machine.phase = Phase.LOWER
    c.machine.elapsed = 140
    c.target_xyz = np.array([0.5, -0.1, 0.85])
    c.measured_hand_z = c.lower_start_z = 0.9
    c.commanded_hand_z = 0.88
    return c


def observe(c, xyz, fk_depth, touching=True, held=True):
    c.fk_plate_z = c.target_xyz[2] + fk_depth
    task = {
        "plate_xyz": c.target_xyz + xyz,
        "held": held,
        "tilt_deg": 5,
        "contacts": {"target_rack": touching, "table": False, "floor": False},
    }
    c.machine.tick()
    return c._release_confirmed(task)


def test_seated_drop_requires_sustained_loaded_contact_and_resets():
    c = controller()
    assert not observe(c, [0.005, 0, 0.022], 0.020, touching=False)
    for _ in range(7):
        assert not observe(c, [0.0, 0.045, -0.032], -0.030)
    assert observe(c, [0.0, 0.045, -0.032], -0.030)
    c._enter({}, [0.5, 0, 0.9], np.eye(3))
    assert c.drop_step is None and not c.descent_heights and not c.over_pocket
    c.reset()
    assert c.drop_step is None and c.phase == Phase.APPROACH


@pytest.mark.parametrize(
    "condition", ["vision_only", "no_contact", "off_center", "lost", "no_load"]
)
def test_seated_drop_rejects_uncorroborated_motion(condition):
    c = controller()
    initial_x = 0.07 if condition == "off_center" else 0.0
    observe(c, [initial_x, 0, 0.020], 0.020, touching=False)
    if condition == "no_load":
        c.commanded_hand_z = c.measured_hand_z
    for _ in range(16):
        assert not observe(
            c,
            [0.0, 0.045, -0.032],
            0.020 if condition == "vision_only" else -0.030,
            touching=condition != "no_contact",
            held=condition != "lost",
        )


def test_combined_uses_stationary_pickup_estimate_even_with_wrist_rgb():
    c = RoundFourController(None)
    c.machine.phase = Phase.GRASP
    c.initial_samples = [np.array([0.40, 0.1, 0.8])]
    state = np.zeros(50)
    task = {"plate_xyz": np.array([0.44, 0.1, 0.82]), "left_wrist_rgb": np.zeros((3, 84, 84))}
    observed = c._pickup_task(state, task)
    np.testing.assert_allclose(observed["plate_xyz"], [0.42, 0.1, 0.8])
    assert not observed["held"]
    assert not c.view.faces and not c.view.rims
