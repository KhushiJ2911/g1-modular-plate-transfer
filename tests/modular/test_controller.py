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
    pose[layout.PELVIS_YAW] = np.pi / 2
    vision = {**task, "plate_xyz": np.ones(3)}
    held = grasp.update(pose, vision, vision, Phase.LIFT)
    x, y, z = task["plate_xyz"]
    np.testing.assert_allclose(held["plate_xyz"], [-y + 0.04, x + 0.03, z + 0.07])
    np.testing.assert_allclose(held["plate_rot"][:, 0], [0.0, 1.0, 0.0], atol=1e-12)
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


def test_alignment_cannot_end_while_site_is_provisional():
    from method.modular.robust_controller import RobustPlacementController

    controller = RobustPlacementController(FixedKinematics(), None, late_site=True)
    task = placement()
    assert not controller._ready_to_lower(task, task["plate_xyz"])
    controller.site_contacts.locked = True
    assert controller._ready_to_lower(task, task["plate_xyz"])


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


def test_seated_release_requires_support_and_measured_leveling():
    from method.modular.controller import _rotation_from_vector
    from method.modular.orientation_feedback import LEFT_GRASP_NORMAL
    from method.modular.placement_release import SeatedReleaseController

    kin = FixedKinematics()
    controller = SeatedReleaseController(kin, None, None)
    task = {**placement(), "held": True}
    task["contacts"]["target_rack"] = False
    controller.machine.phase = Phase.LOWER
    controller.machine.elapsed = 98
    controller.machine.step = 410
    controller.target_xyz = task["site_xyz"]
    controller.target_rot = np.eye(3)
    reference = controller.target_rot[:, 2]
    tilt_axis = np.cross(LEFT_GRASP_NORMAL, reference)
    aligned = _rotation_from_vector(
        tilt_axis / np.linalg.norm(tilt_axis) * np.arccos(LEFT_GRASP_NORMAL @ reference)
    )
    tilted = _rotation_from_vector([0.17, 0, 0]) @ aligned
    hand = np.array([0.38, 0.20, 0.78])
    controller.hand = hand, tilted

    assert not controller._ready_to_release(task)
    assert controller.level_pose is None
    task["contacts"]["target_rack"] = True
    assert not controller._ready_to_release(task)
    np.testing.assert_allclose(controller.level_pose[0], hand)
    assert np.linalg.norm(controller.level_pose[1] @ LEFT_GRASP_NORMAL - reference) < 0.01

    controller.machine.step += 16
    assert not controller._ready_to_release(task)
    controller.hand = hand, aligned
    assert controller._ready_to_release(task)
    task["contacts"]["table"] = True
    assert not controller._ready_to_release(task)


def test_post_open_clearance_requires_recontact_and_stops_on_plate_motion():
    from method.modular.placement_release import GuardedClearance

    class Evidence:
        contact = False

        def reset(self):
            self.contact = False

        def update(self, state):
            return self.contact

    detector = Evidence()
    clearance = GuardedClearance(detector)
    pose = np.zeros(50)
    task = placement()
    hand = task["plate_xyz"] - [0.085, 0.0, 0.0]
    rotation = np.eye(3)
    assert clearance.update(pose, task, 10, hand, rotation) is None
    detector.contact = True
    task["contacts"]["target_rack"] = False
    assert clearance.update(pose, task, 11, hand, rotation) is None
    task["contacts"]["target_rack"] = True
    target, orient = clearance.update(pose, task, 12, hand, rotation)
    np.testing.assert_allclose(target, hand - [0.0008, 0.0, 0.0])
    np.testing.assert_array_equal(orient, rotation)
    displaced = {**task, "plate_xyz": task["plate_xyz"] + [0.0, 0.016, 0.0]}
    for step in range(13, 16):
        clearance.update(pose, displaced, step, hand, rotation)
    assert clearance.finished
    assert clearance.update(pose, task, 16, hand, rotation) is None


def test_loaded_grasp_normal_begins_at_carry_and_persists_across_reset():
    from method.modular.orientation_feedback import LEFT_GRASP_NORMAL
    from method.modular.placement_release import SeatedReleaseController

    kin = FixedKinematics()
    controller = SeatedReleaseController(kin, None, None)
    baseline = SeatedReleaseController(kin, None, None)
    state = np.zeros(50)
    state[layout.GRIPPERS] = 0.9
    task = {**placement(), "plate_rot": np.eye(3)}
    measured = np.array([0.0, -1.0, 0.0])
    controller.set_loaded_grasp_normal(measured)
    controller.reset()
    for phase in (Phase.GRASP, Phase.LIFT):
        pre_carry = controller.orientation.update(state, task, phase)
        np.testing.assert_allclose(pre_carry["plate_rot"][:, 2], LEFT_GRASP_NORMAL)
    carried = controller.orientation.update(state, task, Phase.CARRY)
    np.testing.assert_array_equal(carried["plate_rot"][:, 2], measured)
    baseline_pose = baseline.orientation.update(state, task, Phase.CARRY)
    np.testing.assert_allclose(baseline_pose["plate_rot"][:, 2], LEFT_GRASP_NORMAL)
    controller.hand = kin.hand_pose("left")
    controller._start_leveling(measured)
    np.testing.assert_array_equal(controller.level_normal, measured)
    with pytest.raises(ValueError, match="hand-frame normal"):
        controller.set_loaded_grasp_normal(np.zeros(3))


def test_seating_stall_requires_a_descending_command_not_static_sag():
    from method.modular.fk_seating import DescentMonitor

    monitor = DescentMonitor()
    for _ in range(12):
        monitor.update(0.8, 0.775)
    assert not monitor.seated()
    for step in range(12):
        monitor.update(0.8 - 0.001 * step, 0.775 - 0.001 * step)
    assert not monitor.seated()
    for step in range(12):
        monitor.update(0.8, 0.775 - 0.001 * step)
    assert monitor.seated()
    for step in range(12):
        monitor.update(0.8, 0.845 - 0.0002 * step)
    assert monitor.seated()


def test_fk_release_tolerates_low_site_only_with_seating_evidence():
    from method.modular.fk_seating import FKSeatingController

    controller = FKSeatingController(FixedKinematics())
    task = {**placement(), "held": True}
    task["plate_xyz"][2] = task["site_xyz"][2] - 0.009
    controller.hand_to_center = np.zeros(3)
    controller.lower_start_z = 0.91
    controller.measured_hand_z = 0.8
    controller.machine.elapsed = 110
    controller.target_xyz = task["site_xyz"]
    assert not controller._ready_to_release(task)

    task["contacts"]["target_rack"] = False
    for step in range(12):
        controller.descent.update(0.8, 0.775 - 0.001 * step)
    assert controller._ready_to_release(task)

    task["plate_xyz"][1] += 0.05
    assert not controller._ready_to_release(task)
    task["plate_xyz"][1] -= 0.05
    task["tilt_deg"] = 13
    assert not controller._ready_to_release(task)


def test_low_site_uses_stationary_source_support_height():
    from method.modular.site_seating import SupportReferencedController

    controller = SupportReferencedController(FixedKinematics())
    controller.machine.phase = Phase.CARRY
    controller.initial_samples = [np.array([0.55, 0.24, 0.818])]
    controller.source = np.array([0.55, 0.24, 0.78])
    controller.target_xyz = np.array([0.55, -0.24, 0.83])
    pos, rot = controller.kin.hand_pose("left")
    controller._enter({"plate_rel": np.zeros(3)}, pos, rot)
    np.testing.assert_allclose(controller.target_xyz, [0.55, -0.24, 0.85])
    assert controller.low_site_adjustment > 0


def test_settle_height_fallback_keeps_freeze_and_drift_guards():
    from method.modular.site_seating import SupportReferencedController

    controller = SupportReferencedController(FixedKinematics())
    task = placement()
    controller.target_xyz = task["site_xyz"].copy()
    controller.target_xyz[2] += 0.029
    controller.release_z = task["plate_xyz"][2]
    for _ in range(controller.settle_samples.maxlen):
        controller.settle_samples.append(np.r_[task["plate_xyz"], task["tilt_deg"]])
    controller.open_steps = SETTLE_STEPS - 1
    assert not controller._ready_to_retreat(task)
    controller.open_steps = SETTLE_STEPS
    assert controller._ready_to_retreat(task)

    task["tilt_deg"] = 20
    assert not controller._ready_to_retreat(task)
    task["tilt_deg"] = 6
    shifted = np.r_[task["plate_xyz"] + [0, 0, 0.012], 6]
    controller.settle_samples.append(shifted)
    assert not controller._ready_to_retreat(task)
