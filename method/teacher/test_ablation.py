"""Controller and paired-analysis contracts; fake geometry is not a rollout."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from method.teacher.compare import compare, load_records, mcnemar_exact
from method.teacher.dart import DartNoise
from method.teacher.run_teacher import add_controller_arguments, controller_options, run
from method.teacher.teacher import FLAGS, METHOD_DIR, Teacher
from method.teacher.test_teacher import FakeEpisode, reset_geometry


class AblationChecks(unittest.TestCase):
    def setup_teacher(self, **flags):
        teacher = Teacher(**flags)
        env = reset_geometry(teacher)
        teacher.reset(env)
        teacher.act(SimpleNamespace(low_dim_obs=teacher.states[0][0]))
        return teacher, env

    def state_at(self, teacher, t):
        state = teacher.states[0][t].copy()
        state[18:20] += teacher.shift_at(t)
        return SimpleNamespace(low_dim_obs=state)

    def test_defaults_presets_and_explicit_overrides(self):
        parser = argparse.ArgumentParser()
        add_controller_arguments(parser)
        for args, expected in [([], False), (["--mode", "v2"], True)]:
            teacher = Teacher(**controller_options(parser.parse_args(args)))
            self.assertTrue(all(teacher.config[name] is expected for name in FLAGS))
            self.assertEqual(teacher.config["legacy_v2"], expected)
        for name in FLAGS:
            teacher = Teacher(**{name: True})
            self.assertEqual([n for n in FLAGS if teacher.config[n]], [name])
        opts = parser.parse_args(["--mode", "v2", "--no-tilt-correct"])
        teacher = Teacher(**controller_options(opts))
        self.assertFalse(teacher.config["tilt_correct"])
        self.assertFalse(teacher.config["legacy_v2"])

    def test_alignment_wait_exit_and_frozen_lower_release_retreat(self):
        teacher, env = self.setup_teacher(align_then_lower=True)
        t = teacher.alignment_gate_steps[0]
        teacher.demo_step = t
        # Change the fake geometry reader, never simulator state.
        pose = env.inner_env.plates[0].pose
        site = env.inner_env.rack_target.sites[2]
        pose[:3] = site + [0.02, 0, 0.15]
        ts = self.state_at(teacher, t)
        action = teacher.act(ts)
        self.assertEqual(teacher.demo_step, t)
        np.testing.assert_array_equal(action[4:], teacher.actions[0][t, 4:])
        np.testing.assert_allclose(np.linalg.norm(action[:2]), 0.06, atol=1e-6)
        pose[0] -= 0.011
        action = teacher.act(ts)
        self.assertEqual(teacher.demo_step, t + 1)
        np.testing.assert_allclose(action[:2], 0, atol=1e-6)
        self.assertFalse(teacher.alignment_timed_out)
        for step in range(t + 1, len(teacher.actions[0])):
            ts = self.state_at(teacher, step)
            ts.low_dim_obs[18:20] += 1  # Pelvis mismatch must not re-enable base.
            action = teacher.act(ts)
            np.testing.assert_allclose(action[:2], 0, atol=1e-6)
            np.testing.assert_array_equal(action[4:], teacher.actions[0][step, 4:])
            self.assertEqual(teacher.demo_step, step + 1)
        action = teacher.act(ts)
        np.testing.assert_allclose(action[:2], 0, atol=1e-6)
        self.assertEqual(teacher.wait_steps["release"], 0)
        self.assertEqual(teacher.tilt_correction_steps, 0)
        self.assertEqual(teacher.recovery_events, [])

    def test_alignment_timeout_and_noise_during_lowering(self):
        teacher, env = self.setup_teacher(align_then_lower=True)
        t = teacher.alignment_gate_steps[0]
        teacher.demo_step = t
        for _ in range(150):
            teacher.act(self.state_at(teacher, t))
            self.assertEqual(teacher.demo_step, t)
        action = teacher.act(self.state_at(teacher, t))
        self.assertEqual(teacher.demo_step, t + 1)
        self.assertTrue(teacher.alignment_timed_out)
        self.assertEqual(teacher.wait_steps["pre_release"], 150)
        noise = DartNoise(1, 1)
        for _ in range(20):
            executed = noise.execute(action, teacher, t)
            np.testing.assert_array_equal(executed[:2], action[:2])

    def test_feed_forward_is_preserved_except_grasp_intervention(self):
        reference, _ = self.setup_teacher()
        for flags in (
            {"tilt_correct": True},
            {"guarded_retreat": True},
            {"align_then_lower": True},
            {"deadzone_min_speed": True},
        ):
            teacher, _ = self.setup_teacher(**flags)
            for t in (20, teacher.meta[0]["grasp_step"] + 50):
                teacher.demo_step = reference.demo_step = t
                ts = self.state_at(teacher, t)
                np.testing.assert_allclose(teacher.act(ts), reference.act(ts), atol=1e-6)
        teacher, _ = self.setup_teacher(grasp_feedback_only=True)
        teacher.demo_step = 20
        action = teacher.act(self.state_at(teacher, 20))
        np.testing.assert_allclose(action[:2], 0, atol=1e-6)
        t = teacher.meta[0]["grasp_step"] + 50
        teacher.demo_step = reference.demo_step = t
        ts = self.state_at(teacher, t)
        np.testing.assert_allclose(teacher.act(ts), reference.act(ts), atol=1e-6)

    def test_all_modes_mask_dart_from_actual_release_onward(self):
        for mode in ("v1", "v2"):
            teacher, _ = self.setup_teacher(mode=mode)
            noise = DartNoise(4, 1)
            clean = teacher.actions[0][300].copy()
            teacher.release_physical_step = 100
            for _ in range(50):
                np.testing.assert_array_equal(noise.execute(clean, teacher, 300), clean)

    def test_flags_are_logged_and_change_run_identity(self):
        import advaitic_eval.environment

        teacher = Teacher()
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                seeds=["100"],
                out=Path(directory) / "episodes.jsonl",
                save_demos=None,
                demos=METHOD_DIR / "data/demos_lowdim.npz",
                meta=METHOD_DIR / "data/demo_meta.json",
                mode="v1",
                kp=2.0,
                tolerance=None,
                max_wait=100,
                dart_noise=0.0,
            )
            with (
                patch.object(
                    advaitic_eval.environment, "make_environment", return_value=FakeEpisode(teacher)
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                run(args)
            row = json.loads(args.out.read_text())
            for name in FLAGS:
                self.assertIs(row[name], False)
                self.assertIs(row["teacher_config"][name], False)
                setattr(args, name, True)
                with self.assertRaisesRegex(ValueError, "configuration changed"):
                    run(args)
                setattr(args, name, None)

    def test_audit_binds_data_and_gates_are_clear(self):
        teacher = Teacher(align_then_lower=True)
        audit = json.loads(Path(__file__).with_name("alignment_audit.json").read_text())
        self.assertEqual(len(audit["demos"]), 59)
        for k, t in teacher.alignment_gate_steps.items():
            gate = audit["demos"][str(k)]["gate"]
            self.assertEqual(t, gate["step"])
            self.assertGreaterEqual(gate["clearance_m"], 0.005)
            self.assertFalse(gate["target_contact"])
            self.assertGreater(t, teacher.meta[k]["grasp_step"] + 30)


class ComparisonChecks(unittest.TestCase):
    def test_exact_test_and_pairing(self):
        self.assertEqual(mcnemar_exact(0, 0), 1)
        self.assertEqual(mcnemar_exact(21, 0), 2**-20)
        self.assertEqual(mcnemar_exact(1, 1), 1)
        self.assertAlmostEqual(mcnemar_exact(1, 9), 22 / 1024)
        a = {
            s: {"seed": s, "slot": s % 6, "success": s != 0, "failure_category": "place_position"}
            for s in range(6)
        }
        b = {s: dict(r, success=False) for s, r in a.items()}
        with contextlib.redirect_stdout(io.StringIO()) as out:
            compare(a, b)
        self.assertIn("A=5/6 B=0/6", out.getvalue())
        with self.assertRaisesRegex(ValueError, "Seed sets differ"):
            compare(a, {0: b[0]})
        b[0]["slot"] = 1
        with self.assertRaisesRegex(ValueError, "different source slots"):
            compare(a, b)

    def test_duplicate_incomplete_and_mixed_logs_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.jsonl"
            row = {"completed": True, "seed": 1, "slot": 0, "success": False}
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate seed"):
                load_records(path)
            path.write_text(json.dumps(dict(row, completed=False)))
            with self.assertRaisesRegex(ValueError, "completed seed"):
                load_records(path)
            path.write_text(json.dumps(row) + "\n" + json.dumps(dict(row, seed=2, run_id="x")))
            with self.assertRaisesRegex(ValueError, "mixed run IDs"):
                load_records(path)

    def test_two_worker_split_merge_and_resume(self):
        script = (
            Path(__file__)
            .with_name("ablate.sh")
            .read_text()
            .split("<<'PY'\n", 1)[1]
            .rsplit("\nPY", 1)[0]
        )
        calls = []

        class Worker:
            def __init__(self, cmd, **kwargs):
                calls.append(cmd)
                start, stop, step = map(int, cmd[cmd.index("--seeds") + 1].split(":"))
                output = Path(cmd[cmd.index("--out") + 1])
                output.write_text(
                    "".join(
                        json.dumps(
                            {
                                "seed": s,
                                "slot": s % 6,
                                "completed": True,
                                "success": s % 2 == 0,
                                "run_id": "test",
                            }
                        )
                        + "\n"
                        for s in range(start, stop, step)
                    )
                )

            def wait(self):
                return 0

            def poll(self):
                return 0

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "merged.jsonl"
            argv = ["-", str(output), "100", "107", "--align-then-lower"]
            for _ in range(2):
                with (
                    patch("sys.argv", argv),
                    patch("subprocess.Popen", Worker),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    exec(compile(script, "ablate.sh", "exec"), {})
                self.assertEqual(set(load_records(output)), set(range(100, 107)))
            self.assertEqual(len(calls), 4)
            self.assertEqual(calls[0][calls[0].index("--seeds") + 1], "100:107:2")
            self.assertEqual(calls[1][calls[1].index("--seeds") + 1], "101:107:2")


if __name__ == "__main__":
    unittest.main()
