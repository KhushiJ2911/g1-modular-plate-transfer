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
    SEATING_AND_REGRIP_DEFAULTS,
    TILT_AND_STALL_DEFAULTS,
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

    def test_seat_cue_requires_persistent_error_against_last_executed_target(self):
        cfg = config_from_env({"ADV_H_SEAT_CUE": "1"})
        self.supervisor = ReleaseSupervisor({**cfg, "held_steps": 2, "stable_steps": 4}, log=False)
        self.state[5] = 0.3
        self.actions[:, 9] = pose_hold_action(self.state)[9]
        self.actions[0, 9] += 0.2  # only the last row precedes the next observation
        self.place()
        self.state[3] = 0.3  # elbow error alone is insufficient
        for _ in range(5):
            self.assertTrue((self.step()[:, 18] == 1).all())
        self.state[5] -= 0.035
        for _ in range(3):  # 4 elapsed control steps after the first positive observation
            self.assertTrue((self.step()[:, 18] == 1).all())
        self.state[5] += 0.035
        self.step()  # one matching observation clears the seat streak
        self.state[5] -= 0.035
        for _ in range(3):
            self.assertTrue((self.step()[:, 18] == 1).all())
        self.assertTrue((self.step()[:, 18] == -1).all())
        self.assertEqual(self.supervisor.phase, "RELEASE")

    def test_seat_only_bypasses_strict_distance_and_tilt_inside_place_region(self):
        for only in (False, True):
            with self.subTest(only=only):
                self.supervisor = ReleaseSupervisor(
                    {**self.config, "seat_cue": True, "seat_cue_only": only}, log=False
                )
                self.place()
                self.state[5] = 0.07  # zero action commands zero radians on wrist pitch
                self.estimate["target_site_rel"] = self.estimate["plate_rel"] + [0.05, 0, 0]
                self.estimate["tilt_deg"] = 20
                self.state[27] = 0.5  # wrist pitch velocity prevents release in both modes
                for _ in range(5):
                    self.assertTrue((self.step()[:, 18] == 1).all())
                self.state[27] = 0
                for _ in range(4):
                    self.step()
                self.assertEqual(self.supervisor.release_time is not None, only)

    def test_regrip_timing_advance_frame_and_once_per_episode(self):
        # Exercise each of Astra's two independent shallow-pinch tests.
        for fingers in ([0.003, 0.003], [0.005, -0.03]):
            with self.subTest(fingers=fingers):
                self.setUp()
                self.supervisor = ReleaseSupervisor(
                    {**self.config, "regrip": True, "grasp_timeout_steps": 100}, log=False
                )
                self.actions = np.repeat(self.actions[:1], 7, axis=0)
                self.state[7:9] = fingers
                self.state[21] = np.pi / 2
                self.carry()
                chunks = [self.step()]
                self.assertEqual(self.supervisor.phase, "REGRIP")
                start = self.supervisor.phase_since
                latched = pose_hold_action(self.state)
                self.estimate["held"] = 0.0  # intentional opening bypasses lost-plate recovery
                while (age := self.supervisor.time - start) < 100:
                    if age >= 35:
                        self.state[18] = 0.035  # measured base reached its target
                    chunks.append(self.step())
                trace = np.concatenate(chunks)
                grip = 2 * np.clip((np.arange(len(trace)) - 60) / 25, 0, 1) - 1
                np.testing.assert_allclose(trace[:, 18], grip)
                np.testing.assert_allclose(trace[:, 0], 0, atol=1e-7)
                np.testing.assert_allclose(trace[10:35, 1], -0.075)
                np.testing.assert_array_equal(np.r_[trace[:10, 1], trace[35:, 1]], 0)
                np.testing.assert_array_equal(
                    trace[:, 2:18], np.tile(latched[2:18], (len(trace), 1))
                )
                self.step()
                self.assertEqual(self.supervisor.phase, "GRASP")
                self.estimate["held"] = 0.95
                for _ in range(8):
                    self.assertTrue((self.step()[:, 18] == 1).all())
                self.assertEqual(self.supervisor.phase, "CARRY")
                self.supervisor.reset()
                self.assertIsNone(self.supervisor.regrip_pose)

    def test_regrip_skips_good_pinch_and_is_default_off(self):
        for enabled, fingers in ((True, [0.01, -0.01]), (False, [0, 0])):
            with self.subTest(enabled=enabled):
                config = config_from_env({"ADV_H_REGRIP": "1"} if enabled else {})
                self.supervisor = ReleaseSupervisor({**config, "held_steps": 2}, log=False)
                self.state[7:9] = fingers
                self.carry()
                for _ in range(12):
                    self.assertTrue((self.step()[:, 18] == 1).all())
                self.assertEqual(self.supervisor.phase, "CARRY")

    def test_env_and_invalid_values(self):
        legacy = {
            key: value
            for key, value in self.config.items()
            if key not in SEATING_AND_REGRIP_DEFAULTS and key not in TILT_AND_STALL_DEFAULTS
        }
        self.assertEqual(ReleaseSupervisor(legacy, log=False).config, self.config)
        defaults = config_from_env({})
        self.assertFalse(defaults["tilt_fix"])
        self.assertFalse(defaults["stall_release"])
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
            ("SEAT_CUE_ONLY", "1"),
            ("SEAT_ERROR_RAD", "0"),
            ("SEAT_STEPS", "0"),
            ("REGRIP", "maybe"),
            ("TILT_FIX", "maybe"),
            ("TILT_FIX_DEG", "181"),
            ("TILT_PROBE_RAD", "0"),
            ("TILT_PROBE_STEPS", "0"),
            ("TILT_STEP_RAD", "-0.01"),
            ("TILT_LIMIT_RAD", "nan"),
            ("STALL_RELEASE", "maybe"),
            ("STALL_STEPS", "0"),
            ("STALL_DISTANCE_M", "-0.05"),
            ("STALL_TILT_DEG", "181"),
        ):
            with self.assertRaises(ValueError):
                config_from_env({"ADV_H_" + key: value})
        self.estimate["held"] = float("nan")
        with self.assertRaises(ValueError):
            self.step()

    def test_tilt_descent_from_commanded_wrist_offsets(self):
        # The estimator sees only |A @ offset + b|. Feed back the last executed
        # command, with different nominal BC targets for the two rows.
        self.actions[:, :18] = 0.3
        self.actions[0, 8:10] = [-0.2, 0.1]
        rad_per_unit = (ARM_HIGH[4:6] - ARM_LOW[4:6]) / 2
        for bias, noise in ((24.0, 0.0), (-24.0, 1.4), (60.0, 0.0)):
            with self.subTest(bias=bias, noise=noise):
                self.supervisor = ReleaseSupervisor({**self.config, "tilt_fix": True}, log=False)
                self.estimate["target_site_rel"] = self.estimate["plate_rel"] + [0.06, 0, 0]
                self.estimate["tilt_deg"] = abs(bias)
                self.state[20] = 0.75
                self.carry()
                height = pose_hold_action(self.state)[2]
                output = self.step()  # enter PLACE
                self.state[20] += 0.01  # active correction keeps the original base height
                for t in range(320):
                    offset = (output[:, 8:10] - self.actions[:, 8:10]) * rad_per_unit
                    np.testing.assert_allclose(offset[0], offset[-1], atol=1e-7)
                    self.assertLessEqual(np.max(np.abs(offset)), 0.18 + 1e-7)
                    np.testing.assert_array_equal(output[:, 4:8], self.actions[:, 4:8])
                    np.testing.assert_array_equal(output[:, 10:18], self.actions[:, 10:18])
                    tilt = abs(np.dot([65, -55], offset[-1]) + bias)
                    # Deterministic, alternating observation error at the reported MAE.
                    self.estimate["tilt_deg"] = tilt + (-1) ** t * noise
                    if abs(bias) == 24 and t >= 270:
                        self.assertLessEqual(tilt, 10.0)
                    if tilt > 12:
                        np.testing.assert_array_equal(output[:, [0, 1, 3]], 0)
                        np.testing.assert_allclose(output[:, 2], height)
                    output = self.step()
                if bias == 60:  # optimum is still out of reach at the configured limits
                    np.testing.assert_allclose(offset[-1], [-0.18, 0.18], atol=0.021)
                    self.assertLess(tilt, 41)
                else:
                    self.assertTrue(
                        (np.sign(offset[-1]) == np.sign(bias) * np.array([-1, 1])).all()
                    )
                self.assertEqual(self.supervisor.phase, "PLACE")

    def test_tilt_probe_dwell_ema_and_corrected_release_latch(self):
        self.supervisor = ReleaseSupervisor(
            {**self.config, "tilt_fix": True, "place_servo": True}, log=False
        )
        self.estimate["tilt_deg"] = 20.0
        self.actions[:, 8:10] = [[0.1, 0.25], [0.2, -0.1]]
        self.place()
        rad_per_unit = (ARM_HIGH[4:6] - ARM_LOW[4:6]) / 2

        def check_offset(expected):
            output = self.step()
            actual = (output[:, 8:10] - self.actions[:, 8:10]) * rad_per_unit
            np.testing.assert_allclose(actual, np.tile(expected, (2, 1)), atol=1e-7)
            return output

        check_offset([0, 0])  # age 2
        check_offset([0, 0])
        check_offset([0.02, 0])  # age 6: first roll probe
        self.estimate["tilt_deg"] = 17
        for _ in range(2):
            check_offset([0.02, 0])
        self.estimate["tilt_deg"] = 21  # final noisy sample; probe EMA still beats 20
        check_offset([0.01, 0])
        for _ in range(2):
            check_offset([0.01, 0])
        check_offset([0.01, 0.02])  # next axis is pitch
        self.estimate["tilt_deg"] = 5
        check_offset([0.01, 0.02])
        before_release = check_offset([0.01, 0.02])
        self.actions[:, 8:10] = -0.9
        released = self.step()
        self.assertEqual(self.supervisor.phase, "RELEASE")
        hold = pose_hold_action(self.state)
        hold[8:10] = before_release[-1, 8:10]
        np.testing.assert_array_equal(released, np.tile(hold, (2, 1)))
        self.state[:7] = 0.7
        np.testing.assert_array_equal(self.step(), released)
        self.assertEqual(self.supervisor.phase, "SETTLE")

    def test_tilt_probe_interruption_limits_and_place_reentry(self):
        self.supervisor = ReleaseSupervisor(
            {**self.config, "tilt_fix": True, "tilt_probe_steps": 2}, log=False
        )
        self.estimate["tilt_deg"] = 25
        self.actions[:, :18] = 0.4
        self.actions[:, 8] = 0.999
        self.place()
        np.testing.assert_array_equal(self.step()[:, 8], 1)  # clips normalized target
        self.estimate["held"] = 0
        np.testing.assert_array_equal(self.step()[:, :18], self.actions[:, :18])
        self.estimate.update(held=0.95, tilt_deg=10)
        np.testing.assert_array_equal(self.step()[:, :18], self.actions[:, :18])
        self.estimate["tilt_deg"] = 25
        np.testing.assert_array_equal(self.step()[:, 8], self.actions[:, 8])  # fresh baseline
        np.testing.assert_array_equal(self.step()[:, 8], 1)
        self.estimate["target_site_rel"] = self.estimate["plate_rel"] + [0.1, 0, 0]
        np.testing.assert_array_equal(self.step()[:, :18], self.actions[:, :18])
        self.assertEqual(self.supervisor.phase, "CARRY")
        self.estimate["target_site_rel"] = self.estimate["plate_rel"] + [0.04, 0, 0]
        np.testing.assert_array_equal(self.step()[:, 8:10], self.actions[:, 8:10])
        self.assertEqual(self.supervisor.phase, "PLACE")

    def test_stall_release_uses_current_gates_after_300_elapsed_place_steps(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                self.setUp()
                self.supervisor = ReleaseSupervisor(
                    {**self.config, "stall_release": enabled, "seat_cue": True}, log=False
                )
                self.estimate["tilt_deg"] = 18
                self.state[22] = 1  # quiet and seat gates never pass
                self.place()
                start = self.supervisor.phase_since
                self.estimate["plate_rel"] = np.zeros(3)
                self.estimate["target_site_rel"] = [0.05, 0, 0]
                while self.supervisor.time - start <= 300:
                    self.assertTrue((self.step()[:, 18] == 1).all())
                for distance, tilt, held in (
                    (0.051, 18, 1),
                    (0.05, 18.1, 1),
                    (0.05, -1, 1),
                    (0.05, 18, 0),
                ):
                    self.estimate.update(target_site_rel=[distance, 0, 0], tilt_deg=tilt, held=held)
                    self.assertTrue((self.step()[:, 18] == 1).all())
                self.estimate["held"] = 0.95
                released = self.step()
                self.assertEqual(self.supervisor.phase, "RELEASE" if enabled else "PLACE")
                if enabled:
                    hold = np.tile(pose_hold_action(self.state), (2, 1))
                    np.testing.assert_array_equal(released, hold)
                    np.testing.assert_array_equal(self.step(), hold)
                    self.assertEqual(self.supervisor.phase, "SETTLE")

    def test_stall_timer_restarts_on_place_entry(self):
        config = config_from_env(
            {
                "ADV_H_STALL_RELEASE": "1",
                "ADV_H_STALL_STEPS": "4",
                "ADV_H_TILT_FIX": "1",
                "ADV_H_TILT_PROBE_STEPS": "2",
                "ADV_H_HELD_STEPS": "2",
            }
        )
        self.supervisor = ReleaseSupervisor(config, log=False)
        self.estimate["tilt_deg"] = 18
        self.place()
        self.step()
        self.estimate["target_site_rel"] = self.estimate["plate_rel"] + [0.10, 0, 0]
        self.step()
        self.assertEqual(self.supervisor.phase, "CARRY")
        self.estimate["target_site_rel"] = self.estimate["plate_rel"] + [0.04, 0, 0]
        entry = self.supervisor.time
        for _ in range(2):  # new PLACE ages 0, 2
            self.assertTrue((self.step()[:, 18] == 1).all())
        self.estimate["tilt_deg"] = 16
        corrected = self.step()  # age 4: roll probe improved
        self.assertTrue((corrected[:, 18] == 1).all())
        self.assertGreater(corrected[-1, 8], self.actions[-1, 8])
        released = self.step()
        self.assertTrue((released[:, 18:] == -1).all())
        self.assertEqual(self.supervisor.release_time - entry, 6)
        np.testing.assert_array_equal(released[-1, 8:10], corrected[-1, 8:10])
        np.testing.assert_array_equal(self.step(), released)

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
        self.estimate["tilt_deg"] = 25
        for _ in range(10):
            np.testing.assert_array_equal(self.step()[:, :18], self.actions[:, :18])
        self.estimate["tilt_deg"] = 5
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
