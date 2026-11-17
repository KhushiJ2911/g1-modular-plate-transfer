"""CPU-only contract checks using fake timesteps/environments, never rollouts."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

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
    return SimpleNamespace(
        inner_env=SimpleNamespace(
            plates=[SimpleNamespace(get_pose=lambda: plate.copy())],
            rack_target=SimpleNamespace(get_position=lambda: target.copy()),
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
        return self.timestep()

    def step(self, action):
        if self.interrupt and self.index == 1:
            raise KeyboardInterrupt
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
        cls.teacher = Teacher()

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
                kp=2.0,
                tolerance=0.01,
                max_wait=100,
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
            before = args.out.read_bytes()
            with patch.object(advaitic_eval.environment, "make_environment") as factory:
                with contextlib.redirect_stdout(io.StringIO()):
                    run(args)
                factory.assert_not_called()
            self.assertEqual(args.out.read_bytes(), before)
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


if __name__ == "__main__":
    unittest.main()
