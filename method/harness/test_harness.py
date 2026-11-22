"""Synthetic stream/unit checks, never task-solving evidence. Run with unittest."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from method.harness.data import audit_files, check_seed, pelvis_relative, snapshot_labels
from method.harness.label_replay import replay_episode
from submission.harness import (
    ARM_HIGH,
    ARM_LOW,
    DEFAULT_CONFIG,
    ReleaseSupervisor,
    config_from_env,
    pose_hold_action,
)


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            **DEFAULT_CONFIG,
            "stable_steps": 4,
            "held_steps": 2,
            "lost_steps": 4,
            "reopen_steps": 4,
            "grasp_timeout_steps": 6,
            "settle_steps": 6,
        }
        self.supervisor = ReleaseSupervisor(self.config, log=False)
        self.state = np.zeros(50, np.float32)
        self.state[20], self.state[44] = 0.75, 0.7
        self.actions = np.zeros((2, 20), np.float32)
        self.actions[:, 18] = 1
        self.estimate = {
            "plate_rel": np.array([0.5, 0, 0.2], np.float32),
            "target_site_rel": np.array([0.5, 0.3, 0.2], np.float32),
            "held": 0.95,
            "tilt_deg": 5.0,
        }

    def step(self):
        return self.supervisor.step(self.estimate, self.state, self.actions)

    def carry(self):
        self.step()
        self.assertEqual(self.supervisor.phase, "GRASP")
        self.step()  # first held observation, zero elapsed evidence
        self.step()
        self.assertEqual(self.supervisor.phase, "CARRY")

    def place(self):
        self.carry()
        self.estimate["target_site_rel"] = self.estimate["plate_rel"] + [0.02, 0, 0]
        self.step()
        self.assertEqual(self.supervisor.phase, "PLACE")

    def test_full_sequence_release_latches_observed_pose(self):
        self.place()
        for _ in range(2):
            self.assertTrue((self.step()[:, 18] == 1).all())
        before = self.state.copy()
        released = self.step()
        self.assertEqual(self.supervisor.phase, "RELEASE")
        self.assertTrue((released[:, 18:] == -1).all())
        np.testing.assert_array_equal(released[0], pose_hold_action(before))
        self.state[:18] = 1.0  # drift cannot move the latched target
        self.estimate["held"] = 0.0
        self.estimate["tilt_deg"] = 80.0
        for _ in range(10):
            np.testing.assert_array_equal(self.step(), released)
        self.assertEqual(self.supervisor.phase, "SETTLE")
        self.assertTrue(self.supervisor.settled)
        self.supervisor.reset()
        self.assertEqual(self.supervisor.phase, "APPROACH")
        self.assertIsNone(self.supervisor.hold)
        self.assertIsNone(self.supervisor.release_time)

    def test_release_gates_and_streak_reset(self):
        for field in ("distance", "tilt", "negative_tilt", "held", "arm_speed", "base_speed"):
            with self.subTest(field=field):
                self.setUp()
                self.place()
                self.step()
                self.step()
                bad = copy.deepcopy(self.estimate)
                state = self.state.copy()
                if field == "distance":
                    bad["target_site_rel"] = bad["plate_rel"] + [0.045, 0, 0]
                elif field == "tilt":
                    bad["tilt_deg"] = 16.0
                elif field == "negative_tilt":
                    bad["tilt_deg"] = -5.0
                elif field == "held":
                    bad["held"] = 0.1
                elif field == "arm_speed":
                    state[22] = 0.9
                else:
                    state[40] = 0.1
                self.supervisor.step(bad, state, self.actions)
                self.assertIsNone(self.supervisor.release_time)
                self.step()
                self.step()
                self.assertIsNone(self.supervisor.release_time)
                self.step()
                self.assertIsNotNone(self.supervisor.release_time)

    def test_grasp_miss_and_carry_drop_reopen_then_retry(self):
        self.estimate["held"] = 0.0
        self.step()
        for _ in range(3):
            self.step()
        self.assertEqual(self.supervisor.phase, "APPROACH")
        self.assertTrue((self.step()[:, 18:] == -1).all())
        self.step()
        self.assertEqual(self.supervisor.phase, "GRASP")
        self.estimate["held"] = 0.95
        self.step()
        self.step()
        self.assertEqual(self.supervisor.phase, "CARRY")
        self.estimate["held"] = 0.0
        self.step()
        self.step()
        self.assertEqual(self.supervisor.phase, "CARRY")
        output = self.step()
        self.assertEqual(self.supervisor.phase, "APPROACH")
        self.assertTrue((output[:, 18:] == -1).all())

    def test_grasp_requires_observed_closure(self):
        self.state[44] = 0
        for _ in range(4):
            self.step()
        self.assertNotEqual(self.supervisor.phase, "CARRY")

    def test_early_bc_open_blocked_and_place_exit(self):
        self.carry()
        self.actions[:, 18] = -1
        self.assertTrue((self.step()[:, 18] == 1).all())
        self.estimate["target_site_rel"] = self.estimate["plate_rel"] + [0.07, 0, 0]
        self.step()
        self.assertEqual(self.supervisor.phase, "PLACE")
        self.step()
        self.assertIsNone(self.supervisor.release_time)
        self.estimate["target_site_rel"] = self.estimate["plate_rel"] + [0.1, 0, 0]
        self.step()
        self.assertEqual(self.supervisor.phase, "CARRY")

    def test_hold_mapping_and_contract(self):
        self.state[:7] = ARM_LOW[:7]
        self.state[9:16] = ARM_HIGH[7:]
        hold = pose_hold_action(self.state)
        np.testing.assert_allclose(hold[4:11], -1, atol=1e-6)
        np.testing.assert_allclose(hold[11:18], 1, atol=1e-6)
        for length in (1, 2, 50):
            out = self.supervisor.step(
                self.estimate, self.state, np.zeros((length, 20), np.float32)
            )
            self.assertEqual(out.shape, (length, 20))
            self.assertEqual(out.dtype, np.float32)
            self.assertLessEqual(np.abs(out).max(), 1)

    def test_env_and_invalid_values(self):
        cfg = config_from_env({"ADV_H_STABLE_STEPS": "12", "ADV_H_TEMPORAL_ENSEMBLE": "1"})
        self.assertEqual(cfg["stable_steps"], 12)
        self.assertTrue(cfg["temporal_ensemble"])
        for key, value in (
            ("STABLE_STEPS", "0"),
            ("HELD_THRESHOLD", "nan"),
            ("RELEASE_DISTANCE_M", "0.2"),
            ("TEMPORAL_ENSEMBLE", "maybe"),
            ("PLACE_KP", "-1"),
            ("PLACE_VMAX", "nan"),
            ("PLACE_SERVO", "maybe"),
        ):
            with self.assertRaises(ValueError):
                config_from_env({"ADV_H_" + key: value})
        self.estimate["held"] = float("nan")
        with self.assertRaises(ValueError):
            self.step()

    def test_place_servo_sign_frame_clipping_and_bc_preservation(self):
        self.config.update(place_servo=True, place_vmax=0.05)
        self.supervisor = ReleaseSupervisor(self.config, log=False)
        self.place()
        self.actions[:, :18] = 0.3
        for yaw in (0.0, np.pi / 2, -np.pi / 3):
            self.state[18:22] = [1, 2, 0.75, yaw]
            plate, site = np.array([1.5, 2.3, 0.95]), np.array([1.56, 2.29, 0.95])
            self.estimate["plate_rel"] = pelvis_relative(plate, self.state)
            self.estimate["target_site_rel"] = pelvis_relative(site, self.state)
            c, s = np.cos(self.state[21]), np.sin(self.state[21])
            expected = np.clip([2 * (c * 0.06 - s * 0.01), 2 * (-s * 0.06 - c * 0.01)], -0.05, 0.05)
            output = self.step()
            np.testing.assert_allclose(output[:, :2], np.tile(expected, (2, 1)), atol=1e-6)
            np.testing.assert_array_equal(output[:, 2:18], self.actions[:, 2:18])
        self.estimate["target_site_rel"] = self.estimate["plate_rel"] - [0.03, 0, 0]
        np.testing.assert_allclose(self.step()[:, 0], -0.05, atol=1e-6)  # plate +x of site
        self.estimate["target_site_rel"] = self.estimate["plate_rel"].copy()
        np.testing.assert_array_equal(self.step()[:, :2], 0)

    def test_place_flags_gating_grasp_approach_and_release(self):
        # Default-off flags preserve brief-13 actions, including PLACE base commands.
        self.actions[:, :18] = 0.3
        self.place()
        np.testing.assert_array_equal(self.step()[:, :18], self.actions[:, :18])
        self.config.update(place_servo=True, tilt_hold=True)
        self.supervisor = ReleaseSupervisor(self.config, log=False)
        for _ in range(3):  # APPROACH -> GRASP -> CARRY; no premature base override
            np.testing.assert_array_equal(self.step()[:, :18], self.actions[:, :18])
        self.step()  # PLACE
        self.estimate["held"] = 0.0
        np.testing.assert_array_equal(self.step()[:, :18], self.actions[:, :18])
        self.estimate["held"] = 0.95
        self.estimate["target_site_rel"] = self.estimate["plate_rel"] + [0.1, 0, 0]
        np.testing.assert_array_equal(self.step()[:, :18], self.actions[:, :18])
        self.assertEqual(self.supervisor.phase, "CARRY")
        self.estimate["target_site_rel"] = self.estimate["plate_rel"] + [0.02, 0, 0]
        for _ in range(4):
            output = self.step()
        self.assertEqual(self.supervisor.phase, "RELEASE")
        np.testing.assert_array_equal(output[0], pose_hold_action(self.state))
        np.testing.assert_array_equal(self.step(), output)  # SETTLE remains latched

    def test_tilt_hold_independent_and_clears_when_level(self):
        self.config["tilt_hold"] = True
        self.supervisor = ReleaseSupervisor(self.config, log=False)
        self.place()
        self.actions[:, :18] = 0.3
        self.estimate["tilt_deg"] = 30.0
        for _ in range(5):
            output = self.step()
            np.testing.assert_array_equal(output[:, [0, 1, 3]], 0)
            np.testing.assert_array_equal(output[:, 4:18], self.actions[:, 4:18])
            self.assertTrue((output[:, 18] == 1).all())
            self.assertIsNone(self.supervisor.release_time)
        self.estimate["tilt_deg"] = 5.0
        np.testing.assert_array_equal(self.step()[:, :18], self.actions[:, :18])
        self.supervisor.config["place_servo"] = True
        self.estimate["tilt_deg"] = 30.0
        np.testing.assert_array_equal(self.step()[:, [0, 1, 3]], 0)  # hold takes priority
        cfg = config_from_env(
            {
                "ADV_H_PLACE_SERVO": "1",
                "ADV_H_TILT_HOLD": "1",
                "ADV_H_PLACE_KP": "3.0",
                "ADV_H_PLACE_VMAX": "0.07",
            }
        )
        self.assertEqual(
            [cfg[k] for k in ("place_servo", "tilt_hold", "place_kp", "place_vmax")],
            [True, True, 3.0, 0.07],
        )


class ReplayTests(unittest.TestCase):
    def test_offline_release_timing_uses_geometry_only_for_reporting(self):
        from method.harness.offline_replay import recorded_releases, replay

        length = 24
        state = np.zeros((length, 50), np.float32)
        state[:, 20], state[:, 44] = 0.75, 0.7
        actions = np.zeros((length, 20), np.float32)
        actions[:, 18] = 1
        actions[20:, 18] = -1
        held = np.ones(length, bool)
        held[20:] = False
        labels = {
            "priv_plate_rel": np.zeros((length, 3), np.float32),
            "priv_target_site_rel": np.tile([0.025, 0, 0], (length, 1)).astype(np.float32),
            "priv_plate_tilt_deg": np.full(length, 7, np.float32),
            "priv_held": held,
        }

        def estimate(obs):
            self.assertEqual(set(obs), {"rgb", "state"})
            return {
                "plate_rel": np.zeros(3),
                "target_site_rel": np.array([0.02, 0, 0]),
                "tilt_deg": 5.0,
                "held": 0.95,
            }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ep_2200000.npz"
            np.savez(
                path,
                state=state,
                rgb=np.zeros((length // 2, 3, 3, 84, 84), np.uint8),
                rgb_stride=2,
                action_executed=actions,
                **labels,
            )
            with patch("sys.stderr"):
                result = replay(path, estimate, DEFAULT_CONFIG)
        self.assertEqual(recorded_releases(actions, held), [20])
        self.assertEqual(result["harness_would_release"]["step"], 16)
        self.assertEqual(result["timing_delta_first_steps"], -4)
        self.assertAlmostEqual(result["harness_would_release"]["true_distance_cm"], 2.5, places=5)
        self.assertEqual(result["harness_would_release"]["true_tilt_deg"], 7)
        self.assertEqual(result["data_releases"][0]["step"], 20)
        self.assertTrue(result["counterfactual_only"])

    def test_reserved_seed_boundaries_and_duplicate_audit(self):
        for seed in (2600000, 2699999, 7310000, 7319999):
            with self.assertRaises(ValueError):
                check_seed(seed)
        for seed in (2599999, 2700000, 7309999, 7320000):
            self.assertEqual(check_seed(seed), seed)
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"ep_{i}.npz" for i in range(2)]
            for path in paths:
                np.savez(path, seed=2200000)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                audit_files(paths)

    def test_pre_action_alignment_teacher_rounding_and_divergence(self):
        state = np.zeros((3, 50), np.float32)
        state[:, 18] = np.arange(3)
        state[:, 21] = 0.5
        executed = np.zeros((3, 20), np.float32)
        snapshot = {
            "pose": [0.1234567890123, 0.2, 0.7, 1, 0, 0, 0],
            "site_xyz": [0.1234567890123, 0.25, 0.7],
            "tilt_deg": 3.123456789,
            "held": True,
        }

        class FakeEnv:
            inner_env = None

            def reset(self, seed):
                self.t = 0
                return SimpleNamespace(low_dim_obs=state[0])

            def step(self, action):
                self.t += 1
                return SimpleNamespace(low_dim_obs=state[min(self.t, 2)], last=lambda: False)

        sensors = SimpleNamespace(snapshot=lambda t: snapshot)
        arrays = dict(seed=2200000, state=state.copy(), action_executed=executed)
        with patch("method.harness.label_replay.PrivilegedSensors", return_value=sensors):
            labels = replay_episode(FakeEnv(), arrays)
            for t in range(3):
                expected = snapshot_labels(snapshot, state[t])
                for key, value in expected.items():
                    np.testing.assert_array_equal(labels[key][t], value)
            replay_episode(FakeEnv(), {**arrays, **labels})
            broken = state.copy()
            broken[1, 1] = 1e-6
            with self.assertRaisesRegex(RuntimeError, "diverged at t=1"):
                replay_episode(FakeEnv(), {**arrays, "state": broken})
            wrong = {**arrays, **labels, "priv_held": np.zeros(3, bool)}
            with self.assertRaisesRegex(RuntimeError, "label mismatch"):
                replay_episode(FakeEnv(), wrong)
        self.assertEqual(labels["priv_held"].dtype, np.bool_)
        self.assertEqual(labels["priv_plate_rel"].dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
