import numpy as np
import pytest

from method.modular import layout
from method.modular.controller import ModularController
from method.modular.phases import SETTLE_STEPS, TIMEOUTS, Phase, PhaseMachine
from method.modular.run_privileged import development_seeds, failure_bucket


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
    controller.act(state, task)  # The pads can still touch a plate when fully open.
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


def test_rollout_seeds_enforce_full_range_and_episode_cap():
    assert list(development_seeds(3000000, 2)) == [3000000, 3000001]
    assert list(development_seeds(1000000, 2)) == [1000000, 1000001]
    for start, count in ((2600000, 1), (7310000, 1), (7309999, 2), (3000000, 61)):
        with pytest.raises(ValueError):
            development_seeds(start, count)


def test_stationary_pickup_refreshes_once_per_attempt():
    from method.modular.robust_controller import SOURCE_FRAMES, StationaryPickup

    pickup = StationaryPickup()
    state = np.zeros(50)
    task = placement()
    for step in range(SOURCE_FRAMES):
        task["plate_xyz"] = np.array([0.5, 0.2, 0.8]) if step else np.ones(3)
        initial = pickup.update(state, task, Phase.APPROACH, step, False)
    np.testing.assert_allclose(initial["plate_xyz"], [0.5, 0.2, 0.8])
    for step in range(1, SOURCE_FRAMES + 1):
        task["plate_xyz"] = np.array([0.52, 0.19, 0.805])
        near = pickup.update(state, task, Phase.PREGRASP, step, False)
    task["plate_xyz"] = np.ones(3)
    grasp = pickup.update(state, task, Phase.GRASP, 5, True)
    np.testing.assert_allclose(grasp["plate_xyz"], near["plate_xyz"])
    retry = pickup.update(state, task, Phase.PREGRASP, 1, False)
    np.testing.assert_allclose(retry["plate_xyz"], task["plate_xyz"])


def test_rigid_grasp_tracks_the_hand_without_blending_visual_position():
    from method.modular.robust_controller import RigidGrasp

    kin = FixedKinematics()
    grasp = RigidGrasp(kin)
    pose = np.zeros(50)
    pose[layout.GRIPPERS] = 0.9
    task = {**placement(), "held": True, "plate_rot": np.eye(3)}
    hand, rotation = kin.hand_pose("left")
    grasp.capture(task, hand, rotation)
    pose[layout.PELVIS_XYZ] = [0.04, 0.03, 0.07]
    vision = {**task, "plate_xyz": np.ones(3)}
    held = grasp.update(pose, vision, vision, Phase.LIFT)
    np.testing.assert_allclose(held["plate_xyz"], task["plate_xyz"] + pose[layout.PELVIS_XYZ])
    assert held["held"]
    vision["held"] = False
    for _ in range(6):
        held = grasp.update(pose, vision, vision, Phase.LIFT)
    assert not held["held"]
    assert grasp.update(pose, vision, vision, Phase.SETTLE) is vision


def test_standoff_site_replaces_provisional_navigation_geometry():
    from method.modular.robust_controller import STANDOFF_FRAMES, StandoffSite

    site = StandoffSite()
    pose = np.zeros(50)
    task = {**placement(), "plate_rot": np.eye(3)}
    site.update_at_phase(pose, task, Phase.APPROACH)
    initial = task["site_xyz"].copy()
    final = initial + [0.03, 0.01, 0.0]
    task["site_xyz"] = final
    for _ in range(STANDOFF_FRAMES - 1):
        estimate = site.update_at_phase(pose, task, Phase.ALIGN)
        np.testing.assert_allclose(estimate["site_xyz"], initial)
    estimate = site.update_at_phase(pose, task, Phase.ALIGN)
    assert site.locked
    np.testing.assert_allclose(estimate["site_xyz"], final)
    task["site_xyz"] = initial
    estimate = site.update_at_phase(pose, task, Phase.LOWER)
    np.testing.assert_allclose(estimate["site_xyz"], final)
    np.testing.assert_allclose(estimate["target_site_rel"], final)
    assert estimate["contacts"]["target_rack"]


def test_seating_feedback_requires_depth_or_near_rack_stall():
    from method.modular.robust_controller import SeatingFeedback

    seating = SeatingFeedback()
    task = {**placement(), "held": True}
    task["plate_xyz"][2] = task["site_xyz"][2] - 0.021
    for elapsed in range(9):
        assert not seating.ready(task, task["site_xyz"], elapsed)
    assert seating.ready(task, task["site_xyz"], 9)
    task["plate_xyz"][1] += 0.04
    assert not seating.ready(task, task["site_xyz"], 209)
    task["plate_xyz"] = task["site_xyz"] - [0.0, 0.0, 0.039]
    seating.depth_history.clear()
    assert seating.ready(task, task["site_xyz"], 1)


def test_pickup_slip_requires_fresh_pregrasp_and_fresh_measurement():
    from method.modular.pickup import WINDOW, PickupController

    ctl = PickupController(FixedKinematics())
    task = {**placement(), "plate_rot": np.eye(3), "held": True}
    state = np.zeros(50)
    for _ in range(WINDOW):
        ctl.act(state, task)
    ctl.machine.phase = Phase.GRASP
    for _ in range(WINDOW):
        ctl._pickup_task(state, task)
    ctl.machine.phase = Phase.LIFT
    hand, rot = ctl.kin.hand_pose("left")
    ctl._enter(task, hand, rot)
    # The visual and FK centers still agree within the slip tolerance, but
    # the hand has only risen 4 cm when vision would already allow CARRY.
    state[44] = 0.9
    state[7:9] = -0.016
    state[20] = 0.04
    rising = {**task, "plate_xyz": task["plate_xyz"] + [0, 0, 0.09]}
    measured = ctl._pickup_task(state, rising)
    assert measured["held"] and not ctl._lift_ready(measured)
    # Empty fingers trigger a bounded retry while vision still claims held.
    state[44] = 1.0
    state[7:9] = -0.020
    for _ in range(10):
        ctl.act(state, task)
    assert ctl.phase == Phase.PREGRASP and ctl.machine.grasp_retries == 1
    assert not ctl.near_samples and ctl.hand_to_center is None
    displaced = {**task, "plate_xyz": task["plate_xyz"] + [0.02, -0.03, 0.004]}
    ctl.machine.phase = Phase.GRASP
    for _ in range(WINDOW):
        target = ctl._pickup_task(state, displaced)
    np.testing.assert_allclose(target["plate_xyz"], displaced["plate_xyz"])
    ctl.reset()
    assert ctl.phase == Phase.APPROACH and not ctl.initial_samples
