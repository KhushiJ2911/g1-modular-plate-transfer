"""Privileged demonstration-transfer controller, for training data only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from method.teacher.diagnostics import PrivilegedSensors, failure_category, plate_up

METHOD_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = METHOD_DIR.parent
# Fixed MovePlateG1 nominal rack origin, from the pinned move_plates.py.
TARGET_RACK_XY = np.array([0.55, -0.24])
FLAGS = (
    "align_then_lower",
    "tilt_correct",
    "guarded_retreat",
    "deadzone_min_speed",
    "grasp_feedback_only",
)


class Teacher:
    """Privileged demo transfer: v1-based ablations and the legacy v2 preset.

    Only actions affect physics. reset/observe read live privileged geometry.
    Call reset(env), then act(ts) for each step; observe() also samples the final
    post-action state and is called by the runner for complete diagnostics.
    """

    def __init__(
        self,
        demos_path: str | Path = METHOD_DIR / "data/demos_lowdim.npz",
        meta_path: str | Path = METHOD_DIR / "data/demo_meta.json",
        *,
        mode: str = "v1",
        align_then_lower: bool | None = None,
        tilt_correct: bool | None = None,
        guarded_retreat: bool | None = None,
        deadzone_min_speed: bool | None = None,
        grasp_feedback_only: bool | None = None,
        kp: float = 2.0,
        tolerance: float | None = None,
        max_wait: int = 100,
        lift_steps: int = 30,
        pre_release_steps: int = 40,
    ):
        if mode not in ("v1", "v2"):
            raise ValueError("mode must be v1 or v2")
        if tolerance is None:
            tolerance = 0.01 if mode == "v1" else 0.015
        if not np.isfinite(kp) or kp < 0:
            raise ValueError("kp must be finite and nonnegative")
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        if any(not isinstance(v, int) or v < 0 for v in (max_wait, lift_steps)):
            raise ValueError("max_wait and lift_steps must be nonnegative integers")
        if not isinstance(pre_release_steps, int) or pre_release_steps < 1:
            raise ValueError("pre_release_steps must be a positive integer")
        self.config = {
            "mode": mode,
            "kp": float(kp),
            "tolerance": float(tolerance),
            "max_wait": max_wait,
            "lift_steps": lift_steps,
            "pre_release_steps": pre_release_steps,
        }
        overrides = dict(
            zip(
                FLAGS,
                (
                    align_then_lower,
                    tilt_correct,
                    guarded_retreat,
                    deadzone_min_speed,
                    grasp_feedback_only,
                ),
                strict=True,
            )
        )
        if any(value is not None and type(value) is not bool for value in overrides.values()):
            raise ValueError("Controller flags must be booleans or None")
        self.config.update(
            {name: mode == "v2" if value is None else value for name, value in overrides.items()}
        )
        # The unmodified v2 preset retains the old continuous placement loop.
        # Any explicit flag selects the independent v1-based ablation path.
        self.config["legacy_v2"] = mode == "v2" and all(
            value is None for value in overrides.values()
        )
        self.config.update(
            object_tolerance=0.01,
            object_gate_tolerance=0.015,
            object_max_wait=150,
            min_planar_speed=0.06,
            tilt_limit_deg=14.0,
            tilt_max_wait=60,
            wrist_step_rad=0.01,
            wrist_limit_rad=0.18,
            release_settle_steps=25,
            held_check_steps=50,
            retreat_distance=0.015,
            retreat_tilt_deg=5.0,
            recovery_hold_steps=50,
            recovery_lift_steps=50,
            recovery_lift_m=0.06,
            recovery_joint_limit_rad=0.35,
            recovery_rejoin_steps=50,
        )
        metadata = json.loads(Path(meta_path).read_text())
        self.meta = {int(row["demo"]): row for row in metadata}
        if len(self.meta) != len(metadata):
            raise ValueError("Duplicate demonstration IDs in metadata")
        self.demo_ids = np.array(sorted(k for k in self.meta if k != 45), dtype=int)
        if not len(self.demo_ids):
            raise ValueError("No usable demonstrations")
        self.actions = {}
        self.states = {}
        with np.load(demos_path, allow_pickle=False) as data:
            for k in self.demo_ids:
                a = np.asarray(data[f"a{k}"], dtype=np.float32)
                s = np.asarray(data[f"s{k}"], dtype=np.float32)
                row = self.meta[k]
                if a.ndim != 2 or a.shape[1] != 20 or s.shape != (len(a) + 1, 50):
                    raise ValueError(f"Invalid demonstration shapes: {k}")
                if not np.isfinite(a).all() or not np.isfinite(s).all():
                    raise ValueError(f"Nonfinite demonstration: {k}")
                if np.any(np.abs(a) > 1) or int(data[f"seed{k}"]) != row["seed"]:
                    raise ValueError(f"Invalid actions or mismatched demo seed: {k}")
                grasp, release = row["grasp_step"], row["release_step"]
                if not 0 < grasp < release < len(a):
                    raise ValueError(f"Invalid grasp/release timing: {k}")
                if not (a[grasp - 1, 18] <= 0 < a[grasp, 18]):
                    raise ValueError(f"Grasp metadata does not match gripper transition: {k}")
                if not (a[release - 1, 18] > 0 >= a[release, 18]):
                    raise ValueError(f"Release metadata does not match gripper transition: {k}")
                if grasp + lift_steps >= release - pre_release_steps:
                    raise ValueError(f"No carry interval for shift blending: {k}")
                if not np.allclose(s[0, 18:22], row["pelvis_start"], atol=1e-6):
                    raise ValueError(f"Mismatched initial pelvis metadata: {k}")
                self.actions[k], self.states[k] = a, s
        self.alignment_gate_steps = {
            int(k): self.meta[k]["release_step"] - pre_release_steps for k in self.demo_ids
        }
        if self.config["align_then_lower"] and not self.config["legacy_v2"]:
            audit = json.loads(Path(__file__).with_name("alignment_audit.json").read_text())
            if (
                audit["demos_sha256"] != hashlib.sha256(Path(demos_path).read_bytes()).hexdigest()
                or audit["meta_sha256"] != hashlib.sha256(Path(meta_path).read_bytes()).hexdigest()
                or lift_steps != audit["lift_steps"]
                or pre_release_steps != audit["pre_release_steps"]
            ):
                raise ValueError("Alignment gates require the audited demos and timing settings")
            self.alignment_gate_steps = {
                int(k): audit["demos"][str(k)]["gate"]["step"] for k in self.demo_ids
            }
        self.relative_plate_xy = np.array(
            [
                np.asarray(self.meta[k]["plate_pose_after_warmup"])[:2] - self.states[k][0, 18:20]
                for k in self.demo_ids
            ]
        )
        layout = json.loads((REPO_DIR / "src/advaitic_eval/action_layout.json").read_text())
        self.low = np.asarray(layout["action_low"], dtype=np.float64)
        self.high = np.asarray(layout["action_high"], dtype=np.float64)
        self._ready = False

    def nearest_demos(self, plate_xy, pelvis_xy, top_n: int = 1) -> list[int]:
        """Return up to top_n demo IDs, ranked by relative plate xy distance.

        Ties resolve by demo ID. Neither seed nor sampled source slot is used.
        This function can be called independently of reset/act.
        """
        if not isinstance(top_n, int) or top_n < 1:
            raise ValueError("top_n must be a positive integer")
        relative_xy = np.asarray(plate_xy, dtype=float) - np.asarray(pelvis_xy, dtype=float)
        if relative_xy.shape != (2,) or not np.isfinite(relative_xy).all():
            raise ValueError("plate_xy and pelvis_xy must be finite xy vectors")
        distances = np.linalg.norm(self.relative_plate_xy - relative_xy, axis=1)
        order = np.argsort(distances, kind="stable")[:top_n]
        return self.demo_ids[order].tolist()

    def reset(self, env):
        """Read privileged reset geometry without stepping or changing the sim.

        Selection is finalized on the first act(reset_timestep), because this
        API receives env rather than the reset timestep containing pelvis0.
        The privileged sensor retains a read-only reference for live feedback.
        """
        self._ready = False
        inner = env.inner_env
        self.sensors = PrivilegedSensors(inner)
        self.physical_step = 0
        self.initial_snapshot = self.sensors.snapshot(0)
        self.current_snapshot = self.initial_snapshot
        self.grasp_physical_step = None
        self.release_physical_step = None
        self.lifted_snapshot = None
        self.release_snapshot = None
        self.held_50_snapshot = None
        self.dropped_in_carry = False
        self.previous_gripper = -1.0
        self.wrist_offset = np.zeros(2)
        self.tilt_correction_steps = 0
        self.tilt_before_correction = None
        self.wrist_sensitivity = None
        self.release_action = None
        self.last_action = None
        self.retreat_baseline = None
        self.recovery_phase = None
        self.recovery_step = 0
        self.recovery_events = []
        self.held_check_done = False
        self.rejoin_step = None
        self.plate_xy = np.asarray(inner.plates[0].get_pose(), dtype=float)[:2].copy()
        self.target_offset = (
            np.asarray(inner.rack_target.get_position(), dtype=float)[:2] - TARGET_RACK_XY
        )
        if not np.isfinite(self.plate_xy).all() or not np.isfinite(self.target_offset).all():
            raise ValueError("Nonfinite privileged geometry")
        self.chosen_demo = None
        self.grasp_shift = np.zeros(2)
        self.place_shift = np.zeros(2)
        self.demo_step = 0
        self.wait_steps = {"grasp": 0, "release": 0}
        if self.config["align_then_lower"]:
            self.wait_steps["pre_release"] = 0
        self.alignment_done = False
        self.alignment_snapshot = None
        self.alignment_timed_out = False
        self.max_tracking_error = 0.0
        self.tracking_error = 0.0
        self.object_tracking_error = None
        self._ready = True

    def shift_at(self, demo_step: int) -> np.ndarray:
        """World xy translation at a demonstration-clock index."""
        row = self.meta[self.chosen_demo]
        start = row["grasp_step"] + self.config["lift_steps"]
        end = (
            self.alignment_gate_steps[self.chosen_demo]
            if (self.config["align_then_lower"] and not self.config["legacy_v2"])
            else row["release_step"] - self.config["pre_release_steps"]
        )
        weight = np.clip((demo_step - start) / (end - start), 0.0, 1.0)
        return (1.0 - weight) * self.grasp_shift + weight * self.place_shift

    def _act_replay(self, ts) -> np.ndarray:
        """Return one finite normalized float32 (20,) action."""
        if not self._ready:
            raise RuntimeError("Call reset(env) after the environment reset first")
        state = np.asarray(ts.low_dim_obs, dtype=float)
        if state.shape != (50,) or not np.isfinite(state).all():
            raise ValueError("Expected finite unnormalized (50,) state")
        if self.chosen_demo is None:
            self.chosen_demo = self.nearest_demos(self.plate_xy, state[18:20])[0]
            k = self.chosen_demo
            self.grasp_shift = (self.plate_xy - state[18:20]) - (
                np.asarray(self.meta[k]["plate_pose_after_warmup"])[:2] - self.states[k][0, 18:20]
            )
            self.place_shift = (
                self.target_offset - np.asarray(self.meta[k]["target_rack_offset"])[:2]
            )
        k, t = self.chosen_demo, self.demo_step
        actions, states, row = self.actions[k], self.states[k], self.meta[k]
        desired_xy = states[min(t, len(actions)), 18:20] + self.shift_at(t)
        error_world = desired_xy - state[18:20]
        self.tracking_error = float(np.linalg.norm(error_world))
        self.max_tracking_error = max(self.max_tracking_error, self.tracking_error)
        if self.config["legacy_v2"]:
            return self._act_v2(state, error_world)
        if any(self.config[name] for name in FLAGS):
            return self._act_ablation(state, error_world)
        if t >= len(actions):
            # Exact final-action hold; no continuing base corrections after replay.
            action = actions[-1].copy()
            action[18:20] = -1.0
            return action

        paused = False
        for event in ("grasp", "release"):
            if (
                t == row[f"{event}_step"]
                and self.tracking_error >= self.config["tolerance"]
                and self.wait_steps[event] < self.config["max_wait"]
            ):
                self.wait_steps[event] += 1
                paused = True
                break
        action = actions[t - 1 if paused else t].copy()
        c, s = np.cos(state[21]), np.sin(state[21])
        error_body = np.array(
            [c * error_world[0] + s * error_world[1], -s * error_world[0] + c * error_world[1]]
        )
        feed_forward = self.low[:2] + (actions[t, :2] + 1.0) * (self.high[:2] - self.low[:2]) / 2.0
        raw_xy = (np.zeros(2) if paused else feed_forward) + self.config["kp"] * error_body
        raw_xy = np.clip(raw_xy, self.low[:2], self.high[:2])
        action[:2] = 2.0 * (raw_xy - self.low[:2]) / (self.high[:2] - self.low[:2]) - 1.0
        if not paused:
            self.demo_step += 1
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def observe(self) -> dict:
        """Sample the current pre-action/terminal state without advancing clocks."""
        snap = self.sensors.snapshot(self.physical_step)
        self.current_snapshot = snap
        if (
            self.release_physical_step is not None
            and self.physical_step >= self.release_physical_step + 50
            and self.held_50_snapshot is None
        ):
            self.held_50_snapshot = snap
        if self.grasp_physical_step is not None:
            if self.physical_step >= self.grasp_physical_step + 60:
                if self.lifted_snapshot is None:
                    self.lifted_snapshot = snap
                if self.release_snapshot is None and not snap["held"]:
                    self.dropped_in_carry = True
        return snap

    def act(self, ts) -> np.ndarray:
        """Return an action and record the physical timing of gripper events."""
        if not self._ready:
            raise RuntimeError("Call reset(env) after the environment reset first")
        self.observe()
        action = self._act_replay(ts)
        if self.previous_gripper <= 0 < action[18] and self.grasp_physical_step is None:
            self.grasp_physical_step = self.physical_step
        if (
            self.previous_gripper > 0 >= action[18]
            and self.grasp_physical_step is not None
            and self.release_snapshot is None
        ):
            # Command-time pose, immediately BEFORE the first opening action.
            self.release_physical_step = self.physical_step
            self.release_snapshot = self.current_snapshot
            self.retreat_baseline = self.release_snapshot
            self.release_action = action.copy()
        self.previous_gripper = float(action[18])
        self.last_action = action.copy()
        self.physical_step += 1
        return action

    def diagnostics(self, *, success: bool, fell: bool) -> dict:
        end = self.observe()
        release = self.release_snapshot
        return {
            "mode": self.config["mode"],
            **{name: self.config[name] for name in FLAGS},
            "alignment": {
                "gate_demo_step": None
                if self.chosen_demo is None
                else self.alignment_gate_steps[self.chosen_demo],
                "done": self.alignment_done,
                "timed_out": self.alignment_timed_out,
                "snapshot": self.alignment_snapshot,
            },
            "grasp_physical_step": self.grasp_physical_step,
            "release_physical_step": self.release_physical_step,
            "plate_at_grasp_plus_60": self.lifted_snapshot,
            "plate_at_release": release,
            "plate_at_end": end,
            "plate_50_steps_after_opening": self.held_50_snapshot,
            "nearest_site_distance_at_release": None
            if release is None
            else release["site_distance"],
            "nearest_site_distance_at_end": end["site_distance"],
            "tilt_deg_at_release": None if release is None else release["tilt_deg"],
            "tilt_deg_at_end": end["tilt_deg"],
            "held_at_end": end["held"],
            "contacts_at_end": end["contacts"],
            "drop_in_carry": self.dropped_in_carry,
            "tilt_correction": {
                "steps": self.tilt_correction_steps,
                "before_deg": self.tilt_before_correction,
                "wrist_offset_rad": self.wrist_offset.tolist(),
                "normal_sensitivity_per_rad": self.wrist_sensitivity,
                "unresolved_at_release": release is not None and release["tilt_deg"] > 14,
            },
            "recovery_events": self.recovery_events,
            "failure_category": failure_category(
                success,
                fell,
                self.initial_snapshot,
                self.lifted_snapshot,
                release,
                end,
                self.dropped_in_carry,
            ),
        }

    def _normalize_xy(self, raw_xy) -> np.ndarray:
        raw_xy = np.clip(raw_xy, self.low[:2], self.high[:2])
        return 2 * (raw_xy - self.low[:2]) / (self.high[:2] - self.low[:2]) - 1

    def _base_feedback(self, error_world, yaw, tolerance) -> np.ndarray:
        """No feed-forward: stop in tolerance, otherwise beat the walk deadzone."""
        distance = float(np.linalg.norm(error_world))
        if distance <= tolerance:
            return self._normalize_xy(np.zeros(2))
        speed = max(self.config["min_planar_speed"], self.config["kp"] * distance)
        velocity = speed * error_world / distance
        c, s = np.cos(yaw), np.sin(yaw)
        return self._normalize_xy(
            [c * velocity[0] + s * velocity[1], -s * velocity[0] + c * velocity[1]]
        )

    def _correct_tilt(self):
        """Small bounded wrist roll/pitch correction, assuming a rigid grasp."""
        if self.tilt_before_correction is None:
            self.tilt_before_correction = self.current_snapshot["tilt_deg"]
        axes, _, _ = self.sensors.joint_geometry([8, 9])
        normal = plate_up(self.current_snapshot["pose"][3:])
        jac = np.cross(axes, normal).T
        if self.wrist_sensitivity is None:
            self.wrist_sensitivity = np.linalg.norm(jac, axis=0).tolist()
        # d(normal)/dq = world_axis cross normal. Damping bounds near-singular
        # wrists; only the local tilt-reducing component is requested.
        delta = np.linalg.solve(
            jac.T @ jac + 0.05 * np.eye(2), jac.T @ (np.array([0.0, -1.0, 0.0]) - normal)
        )
        length = np.linalg.norm(delta)
        delta *= min(1.0, self.config["wrist_step_rad"] / max(length, 1e-12))
        self.wrist_offset = np.clip(
            self.wrist_offset + delta,
            -self.config["wrist_limit_rad"],
            self.config["wrist_limit_rad"],
        )
        self.tilt_correction_steps += 1

    def _add_wrist_offset(self, action):
        action[8:10] += 2 * self.wrist_offset / (self.high[8:10] - self.low[8:10])
        return np.clip(action, -1, 1).astype(np.float32)

    def _pelvis_velocity(self, state, error, feed_forward, *, feedback_only=False):
        """V1 feed-forward + P, with independently selected deadzone handling."""
        distance = float(np.linalg.norm(error))
        velocity = self.config["kp"] * error
        if self.config["deadzone_min_speed"]:
            if distance <= self.config["tolerance"]:
                velocity = np.zeros(2)
            elif np.linalg.norm(velocity) < self.config["min_planar_speed"]:
                velocity = self.config["min_planar_speed"] * error / distance
        c, s = np.cos(state[21]), np.sin(state[21])
        body = np.array([c * velocity[0] + s * velocity[1], -s * velocity[0] + c * velocity[1]])
        return self._normalize_xy(body + (0 if feedback_only else feed_forward))

    def _act_ablation(self, state, pelvis_error):
        """Independent interventions on the unchanged v1 replay controller."""
        cfg, t, k = self.config, self.demo_step, self.chosen_demo
        actions, row = self.actions[k], self.meta[k]
        zero = self._normalize_xy(np.zeros(2))
        if cfg["guarded_retreat"] and self.release_snapshot is not None:
            return self._retreat_action()
        if t >= len(actions):
            action = self._add_wrist_offset(actions[-1].copy())
            action[18:20] = -1
            if self.alignment_done:
                action[:2] = zero
            return action
        if (
            cfg["align_then_lower"]
            and t == self.alignment_gate_steps[k]
            and not self.alignment_done
        ):
            snap = self.current_snapshot
            error = np.asarray(snap["site_xyz"])[:2] - np.asarray(snap["pose"])[:2]
            self.object_tracking_error = float(np.linalg.norm(error))
            if (
                self.object_tracking_error > cfg["object_tolerance"]
                and self.wait_steps["pre_release"] < cfg["object_max_wait"]
            ):
                self.wait_steps["pre_release"] += 1
                action = self._add_wrist_offset(actions[t].copy())
                # Always beat the walking deadzone at this above-rack gate.
                action[:2] = self._base_feedback(error, state[21], cfg["object_tolerance"])
                return action
            self.alignment_done = True
            self.alignment_timed_out = self.object_tracking_error > cfg["object_tolerance"]
            self.alignment_snapshot = snap
        paused = False
        for event in ("grasp", "release"):
            # After alignment, release never restarts pelvis locomotion or waits
            # for a pelvis path which the frozen base can no longer track.
            if event == "release" and self.alignment_done:
                continue
            if (
                t == row[f"{event}_step"]
                and self.tracking_error >= cfg["tolerance"]
                and self.wait_steps[event] < cfg["max_wait"]
            ):
                self.wait_steps[event] += 1
                paused = True
                break
        if (
            cfg["tilt_correct"]
            and t == row["release_step"]
            and self.current_snapshot["tilt_deg"] > cfg["tilt_limit_deg"]
            and self.tilt_correction_steps < cfg["tilt_max_wait"]
        ):
            self._correct_tilt()
            paused = True
        action = self._add_wrist_offset(actions[t - 1 if paused else t].copy())
        feed_forward = self.low[:2] + (actions[t, :2] + 1.0) * (self.high[:2] - self.low[:2]) / 2.0
        # The grasp intervention ends once the demo has issued its close action;
        # carry/lowering retain v1 feed-forward unless alignment froze the base.
        feedback_only = paused or (cfg["grasp_feedback_only"] and t <= row["grasp_step"])
        action[:2] = (
            zero
            if self.alignment_done
            else self._pelvis_velocity(
                state, pelvis_error, feed_forward, feedback_only=feedback_only
            )
        )
        if not paused:
            if t == row["release_step"] and cfg["guarded_retreat"]:
                action[:2] = zero
                action[18:20] = -1
            self.demo_step += 1
        return action

    def _act_v2(self, state, pelvis_error) -> np.ndarray:
        k, t, cfg = self.chosen_demo, self.demo_step, self.config
        actions, row = self.actions[k], self.meta[k]
        if self.release_snapshot is not None:
            return self._retreat_action()
        if t >= len(actions):
            action = actions[-1].copy()
            action[:2] = self._normalize_xy(np.zeros(2))
            action[18:20] = -1
            return self._add_wrist_offset(action)
        blend_end = row["release_step"] - cfg["pre_release_steps"]
        object_phase = t >= blend_end
        error = pelvis_error
        tolerance = cfg["tolerance"]
        if object_phase:
            snap = self.current_snapshot
            error = np.asarray(snap["site_xyz"])[:2] - np.asarray(snap["pose"])[:2]
            self.object_tracking_error = float(np.linalg.norm(error))
            tolerance = cfg["object_tolerance"]
        paused = False
        if t == row["grasp_step"]:
            if (
                self.tracking_error > cfg["tolerance"]
                and self.wait_steps["grasp"] < cfg["max_wait"]
            ):
                self.wait_steps["grasp"] += 1
                paused = True
        event = None
        if t == blend_end:
            event = "pre_release"
        elif t == row["release_step"]:
            event = "release"
        if event is not None:
            position_bad = self.object_tracking_error >= cfg["object_gate_tolerance"]
            tilt_bad = (
                event == "release"
                and self.current_snapshot["tilt_deg"] > cfg["tilt_limit_deg"]
                and self.tilt_correction_steps < cfg["tilt_max_wait"]
            )
            if (position_bad or tilt_bad) and self.wait_steps[event] < cfg["object_max_wait"]:
                self.wait_steps[event] += 1
                paused = True
                if tilt_bad:
                    self._correct_tilt()
        # At blend_end hold that pose: t-1 is appropriate for the actual gripper
        # transitions, but would step backwards across the pre-lowering gate.
        index = t - 1 if paused and event != "pre_release" else t
        action = self._add_wrist_offset(actions[index].copy())
        action[:2] = self._base_feedback(error, state[21], tolerance)
        if not paused:
            if t == row["release_step"]:
                action[:2] = self._normalize_xy(np.zeros(2))
                action[18:20] = -1
            self.demo_step += 1
        return action

    def _start_recovery(self, reason):
        self.recovery_phase = "hold"
        self.recovery_step = 0
        self.recovery_action = self.last_action.copy()
        self.recovery_action[:2] = self._normalize_xy(np.zeros(2))
        self.recovery_action[18:20] = -1
        self.rejoin_step = None
        self.recovery_events.append(
            {"step": self.physical_step, "reason": reason, "plate": self.current_snapshot}
        )

    def _recovery_action(self):
        cfg = self.config
        action = self.recovery_action.copy()
        if self.recovery_phase == "hold":
            self.recovery_step += 1
            if self.recovery_step >= cfg["recovery_hold_steps"]:
                self.recovery_phase = "lift"
                self.recovery_step = 0
                _, _, self.lift_start_xyz = self.sensors.joint_geometry([4, 7])
                self.lift_start_targets = action[[4, 7]].astype(float).copy()
            return action
        self.recovery_step += 1
        axes, anchors, hand = self.sensors.joint_geometry([4, 7])
        jac = np.cross(axes, hand - anchors).T
        fraction = min(1.0, self.recovery_step / cfg["recovery_lift_steps"])
        goal = self.lift_start_xyz + np.array([0.0, 0.0, cfg["recovery_lift_m"] * fraction])
        delta = np.linalg.solve(jac.T @ jac + 0.002 * np.eye(2), jac.T @ (goal - hand))
        delta = np.clip(delta, -0.015, 0.015)
        normalized = 2 * delta / (self.high[[4, 7]] - self.low[[4, 7]])
        limit = 2 * cfg["recovery_joint_limit_rad"] / (self.high[[4, 7]] - self.low[[4, 7]])
        action[[4, 7]] = np.clip(
            action[[4, 7]] + normalized,
            self.lift_start_targets - limit,
            self.lift_start_targets + limit,
        )
        action = np.clip(action, -1, 1).astype(np.float32)
        self.recovery_action = action.copy()
        if self.recovery_step >= cfg["recovery_lift_steps"]:
            self.recovery_phase = None
            self.rejoin_step = 0
            self.rejoin_action = action.copy()
            # Re-arm against additional disturbance rather than repeatedly
            # reacting to the same already-recorded displacement.
            self.retreat_baseline = self.current_snapshot
        return action

    def _retreat_action(self) -> np.ndarray:
        cfg, snap = self.config, self.current_snapshot
        elapsed = self.physical_step - self.release_physical_step
        reason = None
        if elapsed >= cfg["held_check_steps"] and not self.held_check_done:
            self.held_check_done = True
            if snap["held"]:
                if self.recovery_phase is None:
                    reason = "held_50_steps_after_opening"
                else:
                    # The requested recovery is already in progress. Retain
                    # the exact deadline observation without restarting it.
                    self.recovery_events[-1]["also_held_after_50_steps"] = True
        if self.recovery_phase is not None:
            return self._recovery_action()
        baseline = self.retreat_baseline
        distance = np.linalg.norm(np.asarray(snap["pose"])[:3] - np.asarray(baseline["pose"])[:3])
        if elapsed > cfg["release_settle_steps"]:
            if distance > cfg["retreat_distance"]:
                reason = reason or "plate_displacement"
            elif snap["tilt_deg"] > baseline["tilt_deg"] + cfg["retreat_tilt_deg"]:
                reason = reason or "plate_tilt_increased"
        if reason is not None:
            self._start_recovery(reason)
            return self._recovery_action()
        if elapsed <= cfg["release_settle_steps"]:
            return self.release_action.copy()
        actions = self.actions[self.chosen_demo]
        action = self._add_wrist_offset(actions[min(self.demo_step, len(actions) - 1)].copy())
        # No locomotion after release: do not drag a plate while withdrawing.
        action[:2] = self._normalize_xy(np.zeros(2))
        action[18:20] = -1
        if self.rejoin_step is not None:
            self.rejoin_step += 1
            weight = min(1.0, self.rejoin_step / cfg["recovery_rejoin_steps"])
            action[4:18] = (1 - weight) * self.rejoin_action[4:18] + weight * action[4:18]
            if weight == 1:
                self.rejoin_step = None
        if self.demo_step < len(actions):
            self.demo_step += 1
        return np.clip(action, -1, 1).astype(np.float32)
