"""CPU-only contract checks using fake timesteps/environments, never rollouts."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from method.teacher.diagnostics import PrivilegedSensors, failure_category, good_placement, plate_up
from method.teacher.run_teacher import append_record, parse_seeds, read_records, run
from method.teacher.teacher import METHOD_DIR, TARGET_RACK_XY, Teacher


def reset_geometry(teacher, demo=0, grasp_shift=None, place_shift=None):
    """A fake geometry reader; this never constructs or changes simulator state."""
    g = np.zeros(2) if grasp_shift is None else np.asarray(grasp_shift)
    h = np.zeros(2) if place_shift is None else np.asarray(place_shift)
    row = teacher.meta[demo]
    plate = np.asarray(row["plate_pose_after_warmup"]).copy()
    plate[:2] += g
    target = np.zeros(3)
    target[:2] = TARGET_RACK_XY + np.asarray(row["target_rack_offset"])[:2] + h
    target_rack = SimpleNamespace(get_position=lambda: target.copy())
    target_rack.sites = [np.asarray(x) + np.r_[h, 0.0] for x in row["target_sites"]]
    prop = SimpleNamespace(pose=plate, held=True, contacts=set())
    prop.get_pose = lambda: prop.pose.copy()
    prop.is_colliding = lambda other: id(other) in prop.contacts
    robot = SimpleNamespace(grippers={"left": object(), "right": object()})
    robot.is_gripper_holding_object = lambda plate, side: plate.held if side == "left" else False
    return SimpleNamespace(
        inner_env=SimpleNamespace(
            plates=[prop],
            rack_start=SimpleNamespace(
                get_position=lambda: np.r_[
                    np.array([0.55, 0.24]) + np.asarray(row["start_rack_offset"])[:2], 0.70
                ]
            ),
            rack_target=target_rack,
            data=SimpleNamespace(bind=lambda site: SimpleNamespace(xpos=site)),
            robot=robot,
            table=object(),
            floor=object(),
            _advaitic_initial_slot=row["slot"],
        )
    )


class FakeEpisode:
    def __init__(self, teacher, success=True, interrupt=False):
        self.inner_env = reset_geometry(teacher).inner_env
        self.state0 = teacher.states[0][0].copy()
        self.success = success
        self.interrupt = interrupt
        self.closed = False

    def timestep(self):
        state = self.state0.copy()
        state[0] = self.index
        return SimpleNamespace(
            low_dim_obs=state,
            rgb_obs=np.full((3, 3, 84, 84), self.index, dtype=np.uint8),
            last=lambda: self.index >= 5,
        )

    def reset(self, *, seed):
        self.index = 0
        self.executed = []
        return self.timestep()

    def step(self, action):
        if self.interrupt and self.index == 1:
            raise KeyboardInterrupt
        self.executed.append(action.copy())
        self.index += 1
        return self.timestep()

    def episode_fell(self):
        return False

    def episode_succeeded(self):
        return self.success

    def episode_termination(self):
        return "success" if self.success else "timeout"

    def close(self):
        self.closed = True


class TeacherChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.teacher = Teacher(mode="v1")

    def test_all_usable_demo_trajectories_at_zero_error(self):
        # If observations match the reference trajectory exactly, feedback and
        # gates must preserve every recorded action for all 59 usable demos.
        teacher = self.teacher
        for k in teacher.demo_ids.tolist():
            teacher.reset(reset_geometry(teacher, demo=k))
            candidates = teacher.nearest_demos(
                teacher.plate_xy, teacher.states[k][0, 18:20], top_n=60
            )
            self.assertEqual(candidates[0], k)
            self.assertEqual(len(candidates), 59)
            self.assertNotIn(45, candidates)
            for t, expected in enumerate(teacher.actions[k]):
                ts = SimpleNamespace(low_dim_obs=teacher.states[k][t])
                action = teacher.act(ts)
                self.assertEqual(action.dtype, np.float32)
                np.testing.assert_allclose(action, expected, atol=1e-6)
            self.assertEqual(teacher.wait_steps, {"grasp": 0, "release": 0})
            final_ts = SimpleNamespace(low_dim_obs=teacher.states[k][-1])
            for _ in range(3):
                np.testing.assert_array_equal(teacher.act(final_ts), teacher.actions[k][-1])

    def test_shifts_blend_and_rotated_feedback(self):
        teacher = self.teacher
        g, h = np.array([0.001, -0.001]), np.array([0.04, -0.03])
        teacher.reset(reset_geometry(teacher, grasp_shift=g, place_shift=h))
        first = teacher.states[0][0].copy()
        teacher.act(SimpleNamespace(low_dim_obs=first))
        self.assertEqual(teacher.chosen_demo, 0)
        np.testing.assert_allclose(teacher.grasp_shift, g, atol=1e-7)
        np.testing.assert_allclose(teacher.place_shift, h, atol=1e-7)
        start = teacher.meta[0]["grasp_step"] + 30
        end = teacher.meta[0]["release_step"] - 40
        np.testing.assert_allclose(teacher.shift_at(start), g)
        np.testing.assert_allclose(teacher.shift_at(end), h)
        np.testing.assert_allclose(teacher.shift_at((start + end) / 2), (g + h) / 2)
        teacher.demo_step = 1
        state = teacher.states[0][1].copy()
        state[18:20] += g - np.array([0.02, 0.0])
        state[21] = np.pi / 2
        action = teacher.act(SimpleNamespace(low_dim_obs=state))
        expected_xy = teacher.actions[0][1, :2] + np.array([0.0, -0.04])
        np.testing.assert_allclose(action[:2], expected_xy, atol=1e-6)
        np.testing.assert_array_equal(action[2:], teacher.actions[0][1, 2:])
        teacher.demo_step = 1
        state[18:20] -= 100
        saturated = teacher.act(SimpleNamespace(low_dim_obs=state))
        self.assertTrue(np.isfinite(saturated).all())
        self.assertTrue((np.abs(saturated) <= 1).all())

    def test_grasp_release_gates_and_wait_limit(self):
        teacher = self.teacher
        for event in ("grasp", "release"):
            teacher.reset(reset_geometry(teacher))
            teacher.act(SimpleNamespace(low_dim_obs=teacher.states[0][0]))
            t = teacher.meta[0][f"{event}_step"]
            teacher.demo_step = t
            state = teacher.states[0][t].copy()
            state[18:20] += teacher.shift_at(t)
            state[18] -= 0.03
            ts = SimpleNamespace(low_dim_obs=state)
            for _ in range(100):
                action = teacher.act(ts)
                self.assertEqual(teacher.demo_step, t)
                np.testing.assert_array_equal(action[2:], teacher.actions[0][t - 1, 2:])
            action = teacher.act(ts)
            self.assertEqual(teacher.demo_step, t + 1)
            self.assertEqual(teacher.wait_steps[event], 100)
            np.testing.assert_array_equal(action[2:], teacher.actions[0][t, 2:])
            # With zero error there is no wait before the gripper event.
            teacher.reset(reset_geometry(teacher))
            teacher.act(SimpleNamespace(low_dim_obs=teacher.states[0][0]))
            teacher.demo_step = t
            state = teacher.states[0][t].copy()
            state[18:20] += teacher.shift_at(t)
            teacher.act(SimpleNamespace(low_dim_obs=state))
            self.assertEqual(teacher.wait_steps[event], 0)
            self.assertEqual(teacher.demo_step, t + 1)

    def test_seed_parser_and_interrupted_jsonl(self):
        self.assertEqual(parse_seeds(["10:16:2", "12,20", "22"]), [10, 12, 14, 20, 22])
        with self.assertRaises(ValueError):
            parse_seeds(["1:2:0"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.jsonl"
            with path.open("a+") as stream:
                append_record(stream, {"seed": 10, "completed": True})
                stream.write('{"seed":11')
                stream.flush()
                append_record(stream, {"seed": 12, "completed": True})
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(set(read_records(stream)), {10, 12})
            self.assertIn('{"seed":11\n', path.read_text())

    def test_runner_alignment_failures_and_resume(self):
        # Exercise real runner/logging/NPZ code, but never construct a simulator.
        import advaitic_eval.environment

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            args = SimpleNamespace(
                seeds=["100000:100002"],
                out=directory / "episodes.jsonl",
                save_demos=directory,
                demos=METHOD_DIR / "data/demos_lowdim.npz",
                meta=METHOD_DIR / "data/demo_meta.json",
                mode="v1",
                kp=2.0,
                tolerance=0.01,
                max_wait=100,
                dart_noise=0.0,
            )
            fake = FakeEpisode(self.teacher)
            with patch.object(advaitic_eval.environment, "make_environment", return_value=fake):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(run(args), 0)
            self.assertTrue(fake.closed)
            with np.load(directory / "ep_100000.npz") as data:
                self.assertEqual(data["state"].shape, (5, 50))
                self.assertEqual(data["action"].shape, (5, 20))
                self.assertEqual(data["rgb"].shape, (3, 3, 3, 84, 84))
                self.assertEqual(data["rgb"].dtype, np.uint8)
                self.assertEqual(data["state"].dtype, np.float32)
                self.assertEqual(data["action"].dtype, np.float32)
                np.testing.assert_array_equal(data["state"][:, 0], np.arange(5))
                np.testing.assert_array_equal(data["rgb"][:, 0, 0, 0, 0], [0, 2, 4])
                self.assertEqual(int(data["rgb_stride"]), 2)
                self.assertTrue(data["success"])
                np.testing.assert_array_equal(data["action"], data["action_executed"])
                self.assertEqual(float(data["dart_noise"]), 0.0)
                self.assertEqual(data["priv_plate_pos"].shape, (5, 3))
                self.assertEqual(data["priv_held"].shape, (5,))
            before = args.out.read_bytes()
            with patch.object(advaitic_eval.environment, "make_environment") as factory:
                with contextlib.redirect_stdout(io.StringIO()):
                    run(args)
                factory.assert_not_called()
            self.assertEqual(args.out.read_bytes(), before)
            args.dart_noise = 1.0
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                run(args)
            args.dart_noise = 0.0
            args.mode = "v2"
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                run(args)
            args.mode = "v1"
            args.seeds = ["100002"]
            fake = FakeEpisode(self.teacher, success=False)
            with patch.object(advaitic_eval.environment, "make_environment", return_value=fake):
                with contextlib.redirect_stdout(io.StringIO()):
                    run(args)
            self.assertFalse((directory / "ep_100002.npz").exists())
            with args.out.open("a+") as stream:
                rows = read_records(stream)
                self.assertEqual(len(rows), 3)
                self.assertFalse(rows[100002]["success"])
            before = args.out.read_bytes()
            args.seeds = ["100003"]
            fake = FakeEpisode(self.teacher, interrupt=True)
            with patch.object(advaitic_eval.environment, "make_environment", return_value=fake):
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    self.assertRaises(KeyboardInterrupt),
                ):
                    run(args)
            self.assertTrue(fake.closed)
            self.assertEqual(args.out.read_bytes(), before)
            self.assertFalse((directory / "ep_100003.npz").exists())


class V2Checks(unittest.TestCase):
    def setUp(self):
        self.teacher = Teacher()
        self.env = reset_geometry(self.teacher)
        self.plate = self.env.inner_env.plates[0]
        self.teacher.reset(self.env)
        self.teacher.act(SimpleNamespace(low_dim_obs=self.teacher.states[0][0]))
        self.assertEqual(self.teacher.chosen_demo, 0)
        self.plate.pose[3:] = [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]

    def state_at(self, t):
        state = self.teacher.states[0][t].copy()
        state[18:20] += self.teacher.shift_at(t)
        return SimpleNamespace(low_dim_obs=state)

    def place_at_site(self, offset=(0.0, 0.0, 0.0)):
        site = self.env.inner_env.rack_target.sites[2]
        self.plate.pose[:3] = site + np.asarray(offset)
        self.plate.contacts.add(id(self.env.inner_env.rack_target))

    def open_at_release(self):
        teacher = self.teacher
        t = teacher.meta[0]["release_step"]
        teacher.grasp_physical_step = 0
        teacher.previous_gripper = 1.0
        teacher.demo_step = t
        self.place_at_site()
        action = teacher.act(self.state_at(t))
        self.assertLess(action[18], 0)
        self.plate.held = False
        return action

    def test_object_gate_ignores_pelvis_and_uses_nearest_of_six(self):
        teacher = self.teacher
        t = teacher.meta[0]["release_step"] - 40
        teacher.demo_step = t
        self.place_at_site((0.025, 0.0, 0.02))
        ts = self.state_at(t)  # pelvis exactly on its nominal trajectory
        action = teacher.act(ts)
        self.assertEqual(teacher.demo_step, t)
        self.assertLess(action[0], 0)
        self.assertAlmostEqual(float(np.linalg.norm(action[:2])), 0.06, places=5)
        self.assertEqual(teacher.current_snapshot["nearest_site"], 2)
        self.assertGreater(teacher.current_snapshot["site_distance"], 0.025)
        self.place_at_site((0.009, 0.0, 0.02))
        action = teacher.act(ts)
        self.assertEqual(teacher.demo_step, t + 1)
        np.testing.assert_allclose(action[:2], 0, atol=1e-6)
        # At release, position wait is bounded even when the object never moves.
        t = teacher.meta[0]["release_step"]
        teacher.demo_step = t
        self.place_at_site((0.03, 0.0, 0.0))
        ts = self.state_at(t)
        for _ in range(150):
            teacher.act(ts)
            self.assertEqual(teacher.demo_step, t)
        teacher.act(ts)
        self.assertEqual(teacher.demo_step, t + 1)
        self.assertEqual(teacher.wait_steps["release"], 150)

    def test_deadzone_and_rotated_pelvis_gate(self):
        teacher = self.teacher
        t = teacher.meta[0]["grasp_step"]
        teacher.demo_step = t
        ts = self.state_at(t)
        ts.low_dim_obs[18] -= 0.016
        ts.low_dim_obs[21] = np.pi / 2
        action = teacher.act(ts)
        self.assertEqual(teacher.demo_step, t)
        np.testing.assert_allclose(action[:2], [0.0, -0.06], atol=1e-6)
        ts.low_dim_obs[18] += 0.002
        action = teacher.act(ts)
        self.assertEqual(teacher.demo_step, t + 1)
        np.testing.assert_allclose(action[:2], 0, atol=1e-6)

    def test_tilt_correction_sign_bounds_timeout_and_successful_exit(self):
        teacher = self.teacher
        self.place_at_site()
        angle = np.radians(30)
        self.plate.pose[3:] = [
            np.cos((np.pi / 2 + angle) / 2),
            np.sin((np.pi / 2 + angle) / 2),
            0.0,
            0.0,
        ]
        teacher.sensors.joint_geometry = lambda dims: (
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            np.zeros((2, 3)),
            np.zeros(3),
        )
        t = teacher.meta[0]["release_step"]
        teacher.demo_step = t
        ts = self.state_at(t)
        for _ in range(60):
            action = teacher.act(ts)
            self.assertEqual(teacher.demo_step, t)
            self.assertGreater(action[18], 0)
        self.assertLess(teacher.wrist_offset[0], 0)  # moves +120 degrees toward +90
        self.assertLessEqual(np.max(np.abs(teacher.wrist_offset)), 0.18)
        teacher.act(ts)
        self.assertEqual(teacher.demo_step, t + 1)
        self.assertTrue(
            teacher.diagnostics(success=False, fell=False)["tilt_correction"][
                "unresolved_at_release"
            ]
        )
        # A live observed improvement releases immediately after one correction.
        self.setUp()
        teacher = self.teacher
        self.place_at_site()
        self.plate.pose[3:] = [np.cos(np.pi / 3), np.sin(np.pi / 3), 0.0, 0.0]
        teacher.sensors.joint_geometry = lambda dims: (
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            np.zeros((2, 3)),
            np.zeros(3),
        )
        teacher.demo_step = t
        teacher.act(self.state_at(t))
        self.assertEqual(teacher.demo_step, t)
        self.plate.pose[3:] = [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]
        teacher.act(self.state_at(t))
        self.assertEqual(teacher.demo_step, t + 1)
        self.assertEqual(teacher.tilt_correction_steps, 1)

    def test_safe_retreat_stop_lift_and_no_reclosing(self):
        teacher = self.teacher
        opened = self.open_at_release()
        release_t = teacher.demo_step
        ts = self.state_at(release_t)
        for _ in range(25):
            np.testing.assert_array_equal(teacher.act(ts), opened)
            self.assertEqual(teacher.demo_step, release_t)
        replay = teacher.act(ts)
        self.assertEqual(teacher.demo_step, release_t + 1)
        self.plate.pose[0] += 0.016
        teacher.sensors.joint_geometry = lambda dims: (
            np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
            np.array([[-0.2, 0.0, -0.1], [-0.1, 0.0, 0.1]]),
            np.zeros(3),
        )
        for _ in range(50):
            # Full arm stop with open fingers, before lifting.
            action = teacher.act(ts)
            np.testing.assert_array_equal(action[4:18], replay[4:18])
            self.assertTrue(np.all(action[18:20] == -1))
            if teacher.recovery_phase == "lift":
                break
        self.assertEqual(teacher.recovery_phase, "lift")
        frozen_clock = teacher.demo_step
        # Two hinges whose local positional Jacobian spans x/z.
        teacher.sensors.joint_geometry = lambda dims: (
            np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
            np.array([[-0.2, 0.0, -0.1], [-0.1, 0.0, 0.1]]),
            np.zeros(3),
        )
        for _ in range(50):
            action = teacher.act(ts)
            self.assertEqual(teacher.demo_step, frozen_clock)
            np.testing.assert_array_equal(action[[5, 6, 8, 9, 10]], replay[[5, 6, 8, 9, 10]])
            self.assertTrue(np.all(action[18:20] == -1))
        self.assertIsNone(teacher.recovery_phase)
        self.assertFalse(np.allclose(action[[4, 7]], replay[[4, 7]]))
        teacher.act(ts)
        self.assertEqual(teacher.demo_step, frozen_clock + 1)
        self.assertEqual(teacher.recovery_events[0]["reason"], "plate_displacement")

    def test_held_at_50_triggers_recovery_and_physical_diagnostics(self):
        teacher = self.teacher
        self.open_at_release()
        self.plate.held = True
        t = teacher.demo_step
        # Before the deadline, a stable touching plate does not trigger.
        while teacher.physical_step - teacher.release_physical_step < 50:
            teacher.act(self.state_at(t))
        teacher.act(self.state_at(t))
        self.assertEqual(teacher.recovery_events[0]["reason"], "held_50_steps_after_opening")
        self.assertEqual(teacher.recovery_phase, "hold")
        # The grasp+60 observation uses physical time, including paused demo time.
        teacher.reset(self.env)
        teacher.grasp_physical_step = 10
        teacher.physical_step = 69
        teacher.observe()
        self.assertIsNone(teacher.lifted_snapshot)
        self.plate.pose[2] += 0.04
        teacher.physical_step = 70
        teacher.observe()
        self.assertEqual(teacher.lifted_snapshot["step"], 70)
        self.plate.held = False
        teacher.physical_step = 71
        teacher.observe()
        self.assertTrue(teacher.dropped_in_carry)

    def test_live_joint_adapter_resolves_arm_names_with_legs_first(self):
        inner = self.env.inner_env
        names = [
            "left_hip_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
        ]
        joints = [SimpleNamespace(name=f"g1/{name}", id=i) for i, name in enumerate(names)]
        inner.robot.limb_joints = joints
        inner.model = SimpleNamespace(bind=lambda joint: joint)
        inner.data.xaxis = np.arange(15).reshape(5, 3)
        inner.data.xanchor = np.arange(15, 30).reshape(5, 3)
        sides = Enum("Sides", ["RIGHT", "LEFT"])
        inner.robot.grippers = {
            sides.RIGHT: SimpleNamespace(pinch_position=np.ones(3)),
            sides.LEFT: SimpleNamespace(pinch_position=np.zeros(3)),
        }
        inner.robot.get_hand_pos = lambda side: np.zeros(3) if side == sides.LEFT else np.ones(3)
        axes, anchors, hand = PrivilegedSensors(inner).joint_geometry([8, 9, 4, 7])
        np.testing.assert_array_equal(axes, inner.data.xaxis[[3, 4, 1, 2]])
        np.testing.assert_array_equal(anchors, inner.data.xanchor[[3, 4, 1, 2]])
        np.testing.assert_array_equal(hand, np.zeros(3))

    def test_failure_categories_and_success_geometry(self):
        teacher = self.teacher
        self.place_at_site()
        end = teacher.observe()
        np.testing.assert_allclose(plate_up(end["pose"][3:]), [0.0, -1.0, 0.0], atol=1e-12)
        self.assertTrue(good_placement(end))
        initial = dict(end, pose=[*end["pose"][:2], end["pose"][2] - 0.04, *end["pose"][3:]])
        lifted = end

        def classify(final=end, release=None, drop=False, fell=False, success=False, lift=lifted):
            return failure_category(success, fell, initial, lift, release, final, drop)

        self.assertEqual(classify(success=True), "success")
        self.assertEqual(classify(fell=True), "fell")
        self.assertEqual(classify(lift=initial), "grasp_miss")
        self.assertEqual(classify(drop=True), "drop_in_carry")
        self.assertEqual(classify(final=dict(end, site_distance=0.06)), "place_position")
        self.assertEqual(classify(final=dict(end, tilt_deg=25)), "place_tilt")
        self.assertEqual(classify(), "not_released")
        self.assertEqual(classify(final=dict(end, held=False)), "other")
        self.assertEqual(
            classify(final=dict(end, site_distance=0.06), release=end), "knocked_after_release"
        )


if __name__ == "__main__":
    unittest.main()
