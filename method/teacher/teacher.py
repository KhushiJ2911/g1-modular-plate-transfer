"""Privileged demonstration-transfer controller, for training data only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

METHOD_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = METHOD_DIR.parent
# Fixed MovePlateG1 nominal rack origin, from the pinned move_plates.py.
TARGET_RACK_XY = np.array([0.55, -0.24])


class Teacher:
    """Replay the nearest human arm motion while shifting its base trajectory.

    At reset, read the true plate and destination rack positions. The first
    act() supplies the starting pelvis from the reset observation, selects the
    nearest plate-relative-to-pelvis demonstration (excluding demo 45), and
    computes small world-frame grasp and placement shifts. Replay both arms,
    grippers, height and yaw on a single demonstration clock. Track the shifted
    pelvis with the recorded base velocity plus proportional feedback, rotating
    the world position error into the current body's frame.

    Hold the grasp shift until 30 demo steps after closing, then blend toward
    the destination rack shift until 40 steps before release. Before closing
    or opening, pause the clock and hold the preceding arm/gripper targets for
    at most 100 physical steps if tracking error exceeds 1 cm. Pauses suppress
    velocity feed-forward so feedback can settle at a fixed pelvis target.
    After replay, hold the final recorded action with both grippers open.

    This uses simulator-only information and is NOT an evaluation policy.
    Numerical operations require only NumPy; no simulator imports or writes.
    Call env.reset(seed=...), reset(env), then act(ts) once per env.step().
    """

    def __init__(
        self,
        demos_path: str | Path = METHOD_DIR / "data/demos_lowdim.npz",
        meta_path: str | Path = METHOD_DIR / "data/demo_meta.json",
        *,
        kp: float = 2.0,
        tolerance: float = 0.01,
        max_wait: int = 100,
        lift_steps: int = 30,
        pre_release_steps: int = 40,
    ):
        if not np.isfinite(kp) or kp < 0:
            raise ValueError("kp must be finite and nonnegative")
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        if any(not isinstance(v, int) or v < 0 for v in (max_wait, lift_steps)):
            raise ValueError("max_wait and lift_steps must be nonnegative integers")
        if not isinstance(pre_release_steps, int) or pre_release_steps < 1:
            raise ValueError("pre_release_steps must be a positive integer")
        self.config = {
            "kp": float(kp),
            "tolerance": float(tolerance),
            "max_wait": max_wait,
            "lift_steps": lift_steps,
            "pre_release_steps": pre_release_steps,
        }
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
        No environment reference is retained.
        """
        self._ready = False
        inner = env.inner_env
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
        self.max_tracking_error = 0.0
        self.tracking_error = 0.0
        self._ready = True

    def shift_at(self, demo_step: int) -> np.ndarray:
        """World xy translation at a demonstration-clock index."""
        row = self.meta[self.chosen_demo]
        start = row["grasp_step"] + self.config["lift_steps"]
        end = row["release_step"] - self.config["pre_release_steps"]
        weight = np.clip((demo_step - start) / (end - start), 0.0, 1.0)
        return (1.0 - weight) * self.grasp_shift + weight * self.place_shift

    def act(self, ts) -> np.ndarray:
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
