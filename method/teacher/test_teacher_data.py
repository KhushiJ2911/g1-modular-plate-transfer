"""DART and label contracts with fake environments; these are not rollouts."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from method.teacher.dart import DartNoise, validate_scale
from method.teacher.label_human import replay_labels
from method.teacher.label_human import run as label_humans
from method.teacher.labels import STEP_KEYS, PrivilegedLabels, pelvis_relative
from method.teacher.run_teacher import run_episode
from method.teacher.teacher import Teacher
from method.teacher.test_teacher import FakeEpisode, reset_geometry


class DartChecks(unittest.TestCase):
    @staticmethod
    def reference(mode="v1"):
        actions = np.zeros((180, 20), dtype=np.float32)
        actions[:, 18:20] = -1
        actions[40:110, 18] = 1
        actions[75:140, 19] = 1
        return SimpleNamespace(
            actions={0: actions},
            chosen_demo=0,
            config={"mode": mode},
            release_physical_step=None,
        )

    def test_reproducibility_guard_boundaries_and_untouched_dimensions(self):
        teacher = self.reference()
        # Pause for 22 physical steps immediately before the first close.
        clock = [*range(40), *([40] * 22), *range(40, 180)]
        clean, actual = [], []
        noise, same, different = DartNoise(18, 1), DartNoise(18, 1), DartNoise(19, 1)
        other = []
        for physical, demo in enumerate(clock):
            action = teacher.actions[0][demo].copy()
            if 40 <= physical < 62:
                action[18] = -1
            clean.append(action)
            actual.append(noise.execute(action, teacher, demo))
            np.testing.assert_array_equal(actual[-1], same.execute(action, teacher, demo))
            other.append(different.execute(action, teacher, demo))
        clean, actual = np.array(clean), np.array(actual)
        transitions = np.flatnonzero(np.any(clean[1:, 18:] != clean[:-1, 18:], axis=1)) + 1
        for event in transitions:
            np.testing.assert_array_equal(
                actual[event - 15 : event + 16], clean[event - 15 : event + 16]
            )
        np.testing.assert_array_equal(actual[25:62], clean[25:62])
        # Immediately outside the protected windows drift can be applied.
        self.assertTrue(np.any(actual[24] != clean[24]))
        self.assertTrue(np.any(actual[-1] != clean[-1]))
        frozen = [2, 3, *range(11, 20)]
        np.testing.assert_array_equal(actual[:, frozen], clean[:, frozen])
        self.assertFalse(np.array_equal(actual, other))
        self.assertTrue(np.isfinite(actual).all())
        self.assertLessEqual(np.abs(actual).max(), 1)

    def test_v2_release_and_retreat_are_protected(self):
        teacher = self.reference("v2")
        noise = DartNoise(10, 1)
        for t, clean in enumerate(teacher.actions[0]):
            if t == 110:
                teacher.release_physical_step = t
            executed = noise.execute(clean, teacher, t)
            if t >= 110:
                np.testing.assert_array_equal(executed, clean)

    def test_ou_spread_correlation_clipping_and_off(self):
        teacher = self.reference()
        teacher.actions[0][:, 18:20] = -1
        clean = np.zeros(20, dtype=np.float32)
        clean[18:20] = -1
        noise = DartNoise(123, 1)
        draws = np.array([noise.execute(clean, teacher, 0) for _ in range(30000)])[200:]
        np.testing.assert_allclose(draws.std(axis=0)[:2], 0.10, rtol=0.05)
        np.testing.assert_allclose(draws.std(axis=0)[4:11], 0.02, rtol=0.05)
        self.assertAlmostEqual(np.corrcoef(draws[:-1, 0], draws[1:, 0])[0, 1], 0.85, delta=0.02)
        high = DartNoise(2, 100)
        saturated = np.array([high.execute(clean, teacher, 0) for _ in range(50)])
        self.assertEqual(np.max(np.abs(saturated[:, :2])), 1)
        off = DartNoise(2, 0)
        np.testing.assert_array_equal(off.execute(clean, teacher, 0), clean)
        for bad in (-1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                validate_scale(bad)


class LabelChecks(unittest.TestCase):
    def setUp(self):
        self.teacher = Teacher(mode="v1")

    def test_observed_pelvis_rotation_nearest_site_and_offsets(self):
        env = reset_geometry(self.teacher)
        state = np.zeros(50)
        state[18:22] = [1, 2, 3, np.pi / 2]
        np.testing.assert_allclose(pelvis_relative([2, 4, 6], state), [2, -1, 3], atol=1e-6)
        labels = PrivilegedLabels(env)
        snap = labels.sensors.snapshot(0)
        sites = np.array(env.inner_env.rack_target.sites)
        nearest = np.argmin(np.linalg.norm(sites - np.array(snap["pose"][:3]), axis=1))
        labels.append(state)
        arrays = labels.arrays()
        np.testing.assert_allclose(
            arrays["priv_target_site_rel"][0], pelvis_relative(sites[nearest], state)
        )
        np.testing.assert_allclose(
            arrays["priv_plate_rel"][0], pelvis_relative(snap["pose"][:3], state)
        )
        for prefix in ("start", "target"):
            np.testing.assert_allclose(
                arrays[f"priv_{prefix}_rack_offset"],
                self.teacher.meta[0][f"{prefix}_rack_offset"][:2],
                atol=1e-7,
            )
        for key, value in arrays.items():
            self.assertEqual(value.dtype, np.bool_ if key == "priv_held" else np.float32)
        self.assertEqual(arrays["priv_held"].shape, (1,))

    def test_runner_keeps_clean_labels_and_records_sent_actions(self):
        class DriftingEpisode(FakeEpisode):
            def step(self, action):
                # Model a base response so subsequent observations depend on
                # executed noisy actions, rather than the clean labels.
                self.state0[18:20] += 0.01 * action[:2]
                return super().step(action)

        env = DriftingEpisode(self.teacher)
        with tempfile.TemporaryDirectory() as directory:
            record = run_episode(env, self.teacher, 123, Path(directory), dart_noise=1.0)
            with np.load(record["demo_file"]) as data:
                np.testing.assert_array_equal(data["action_executed"], env.executed)
                np.testing.assert_array_equal(data["state"][:, 0], np.arange(5))
                self.assertTrue(np.any(data["action"] != data["action_executed"]))
                for key in STEP_KEYS:
                    self.assertEqual(len(data[key]), 5)
                np.testing.assert_array_equal(data["priv_plate_pos"][0], data["priv_plate_pos"][-1])
                # Every saved action must be the teacher's unperturbed response
                # to the real observation, not a perturbed internal trajectory.
                clean_teacher = Teacher(mode="v1")
                clean_teacher.reset(env)
                for state, expected in zip(data["state"], data["action"], strict=True):
                    np.testing.assert_array_equal(
                        clean_teacher.act(SimpleNamespace(low_dim_obs=state)), expected
                    )

    def human_source(self, path):
        env = FakeEpisode(self.teacher)
        env.reset(seed=18)
        states = []
        for _ in range(5):
            states.append(env.timestep().low_dim_obs)
            env.step(np.zeros(20, dtype=np.float32))
        np.savez(path, state=states, action=np.zeros((5, 20), dtype=np.float32), seed=18)

    def test_human_alignment_success_gate_and_pre_action_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "ep_0.npz"
            self.human_source(source)
            arrays, deviation = replay_labels(FakeEpisode(self.teacher), source)
            self.assertEqual(deviation, 0)
            self.assertEqual(arrays["priv_plate_pos"].shape, (5, 3))
            with self.assertRaisesRegex(ValueError, "did not terminate successfully"):
                replay_labels(FakeEpisode(self.teacher, success=False), source)
            env = FakeEpisode(self.teacher)
            env.state0[1] += 0.051
            with self.assertRaisesRegex(AssertionError, "step=0.*max_deviation="):
                replay_labels(env, source)
            with np.load(source) as data:
                states, actions = data["state"].copy(), data["action"].copy()
            states[-1, 1] += 0.051
            np.savez(source, state=states, action=actions, seed=18)
            with self.assertRaisesRegex(AssertionError, "step=4.*max_deviation="):
                replay_labels(FakeEpisode(self.teacher), source)

    def test_human_resume_exclusion_and_no_failed_sidecars(self):
        import advaitic_eval.environment

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            self.human_source(directory / "ep_0.npz")
            self.human_source(directory / "ep_1.npz")
            args = SimpleNamespace(demos_dir=directory, out_dir=None, indices=["0,45"])
            env = FakeEpisode(self.teacher)
            with (
                patch.object(advaitic_eval.environment, "make_environment", return_value=env),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(label_humans(args), 0)
            self.assertTrue(env.closed)
            self.assertFalse((directory / "ep_45.priv.npz").exists())
            before = (directory / "ep_0.priv.npz").read_bytes()
            with (
                patch.object(advaitic_eval.environment, "make_environment") as factory,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(label_humans(args), 0)
                factory.assert_not_called()
            self.assertEqual((directory / "ep_0.priv.npz").read_bytes(), before)
            args.indices = ["1"]
            with (
                patch.object(
                    advaitic_eval.environment,
                    "make_environment",
                    return_value=FakeEpisode(self.teacher, success=False),
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(label_humans(args), 1)
            self.assertFalse((directory / "ep_1.priv.npz").exists())


if __name__ == "__main__":
    unittest.main()
