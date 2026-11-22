"""Build estimator-only datasets from recorded Flywheel2 actions and saved RGB."""

import argparse
import json
import os
from pathlib import Path
from time import monotonic

os.environ["ORT_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import onnxruntime
from bigym.loco import make
from bigym.loco.config import DEFAULT_CAMERA_KEYS

from method.dagger.collect import save_npz_atomic
from method.dagger.diagnostics import PrivilegedSensors
from method.harness.data import (
    ESTIMATOR_PHASES,
    LABEL_KEYS,
    dagger2_challenge_seed,
    flywheel2_split,
    snapshot_labels,
)
from submission.harness import DEFAULT_CONFIG


class OcclusionRenderer:
    """Render geometry IDs twice, hiding grippers only in the diagnostic scene."""

    def __init__(self, inner):
        self.inner = inner
        model = inner.model
        self.cameras = [inner.cameras_map[name][0] for name in DEFAULT_CAMERA_KEYS]
        gripper_bodies = {model.body(g.body.name).id for g in inner.robot.grippers.values()}
        for body, parent in enumerate(model.body_parentid):
            if body and parent in gripper_bodies:
                gripper_bodies.add(body)
        self.gripper = set(np.flatnonzero(np.isin(model.geom_bodyid, list(gripper_bodies))))
        self.plate = [model.geom(geom.name).id for geom in inner.plates[0].geoms]
        if not self.gripper or not len(self.plate):
            raise ValueError("Missing diagnostic geometry IDs")
        self.renderer = mujoco.Renderer(model, 84, 84)
        self.renderer.enable_segmentation_rendering()

    def measure(self):
        counts = []
        geom_type = mujoco.mjtObj.mjOBJ_GEOM
        for camera in self.cameras:
            self.renderer.update_scene(self.inner.data, camera)
            observed = self.renderer.render()
            gripper_pixels = (observed[..., 1] == geom_type) & np.isin(
                observed[..., 0], list(self.gripper)
            )
            if gripper_pixels.any():
                scene = self.renderer.scene
                for geom in scene.geoms[: scene.ngeom]:
                    if geom.objtype == geom_type and geom.objid in self.gripper:
                        geom.pos[:] = (1000, 1000, 1000)
                exposed = self.renderer.render()
            else:
                exposed = observed
            plate_pixels = (exposed[..., 1] == geom_type) & np.isin(exposed[..., 0], self.plate)
            counts.append((np.count_nonzero(gripper_pixels & plate_pixels), plate_pixels.sum()))
        return counts


def replay(env, source, destination, split):
    start = monotonic()
    with np.load(source, allow_pickle=False) as episode:
        seed = int(episode["seed"])
        if split == "challenge":
            dagger2_challenge_seed(seed)
        elif flywheel2_split(seed) != split:
            raise ValueError(f"Unexpected split for {seed}: {split}")
        if seed != int(source.stem[3:]) or int(episode["rgb_stride"]) != 2:
            raise ValueError(f"Unexpected seed or RGB stride in {source}")
        states = episode["state"]
        actions = episode["action_executed"]
        success = bool(episode["success"])
        if actions.shape != (len(states), 20) or states.shape[1:] != (50,):
            raise ValueError(f"Unexpected observation/action dimensions in {source}")
        if not np.isfinite(actions).all() or (np.abs(actions) > 1).any():
            raise ValueError(f"Invalid recorded actions in {source}")
        ts = env.reset(seed=seed)
        sensors = PrivilegedSensors(env.inner_env)
        held_out = split in ("test", "challenge")
        renderer = OcclusionRenderer(env.inner_env) if held_out else None
        rows, labels, phases, contact, occlusion = [], [], [], [], []
        seen_hold = False
        opened = False
        try:
            for t, action in enumerate(actions):
                if not np.array_equal(ts.low_dim_obs, states[t]):
                    difference = float(np.abs(ts.low_dim_obs - states[t]).max())
                    raise ValueError(f"Replay {seed} diverged at step {t} by {difference:g}")
                if t % 2 == 0:
                    snap = sensors.snapshot(t)
                    seen_hold |= snap["held"]
                    if seen_hold and t > 0:
                        opened = actions[t - 1, 18] < 0
                    distance = snap["site_distance"]
                    touching = snap["contacts"]["target_rack"]
                    if t % 4 == 0 and (
                        distance <= 0.10 or (not held_out and (distance <= 0.14 or t % 32 == 0))
                    ):
                        if not seen_hold:
                            phase = "approach"
                        elif opened:
                            phase = "released"
                        elif snap["held"]:
                            phase = (
                                "place"
                                if distance <= DEFAULT_CONFIG["place_distance_m"] or touching
                                else "carry"
                            )
                        else:
                            phase = "lost"
                        row = snapshot_labels(snap, states[t])
                        labels.append(np.concatenate([np.atleast_1d(row[k]) for k in LABEL_KEYS]))
                        rows.append(t)
                        phases.append(ESTIMATOR_PHASES.index(phase))
                        contact.append(touching)
                        occlusion.append(renderer.measure() if renderer else [(-1, -1)] * 3)
                ts = env.step(action)
                if ts.last() and t + 1 < len(actions):
                    raise ValueError(f"Replay {seed} ended before saved trajectory at step {t}")
        finally:
            if renderer:
                renderer.renderer.close()
        rows = np.asarray(rows, dtype=np.int32)
        rgb = episode["rgb"][rows // 2] if len(rows) else np.empty((0, 3, 3, 84, 84), np.uint8)

    save_npz_atomic(
        destination,
        seed=np.int32(seed),
        step=rows,
        state=states[rows],
        rgb=rgb,
        target=np.asarray(labels, np.float32).reshape(-1, 8),
        phase=np.asarray(phases, np.uint8),
        contact=np.asarray(contact, bool),
        occlusion=np.asarray(occlusion, np.int16).reshape(-1, 3, 2),
        success=success,
        trace_steps=len(actions),
    )
    return {
        "seed": seed,
        "split": split,
        "rows": len(rows),
        "steps": len(actions),
        "seconds": round(monotonic() - start, 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=["test", "dev", "train", "challenge"], required=True)
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--limit", type=int)
    opts = parser.parse_args()
    shard, count = map(int, opts.shard.split("/"))
    if not 0 <= shard < count or (opts.limit is not None and opts.limit < 1):
        parser.error("Expected shard i/n with 0 <= i < n and a positive limit")
    files = sorted(opts.source.rglob("ep_*.npz"))
    if opts.split == "challenge":
        files = [p for p in files if is_challenge_episode(p)]
    else:
        files = [p for p in files if flywheel2_split(int(p.stem[3:])) == opts.split]
    if len({p.stem for p in files}) != len(files):
        raise ValueError("Duplicate source seeds")
    dest = opts.out / opts.split
    dest.mkdir(parents=True, exist_ok=True)
    onnxruntime.disable_telemetry_events()
    env = make("move_plate", camera_keys=())
    try:
        for source in files[shard::count][: opts.limit]:
            out = dest / source.name
            if not out.exists():
                print(json.dumps(replay(env, source, out, opts.split)), flush=True)
    finally:
        env.close()


def is_challenge_episode(path):
    seed = dagger2_challenge_seed(int(path.stem[3:]))
    with np.load(path, allow_pickle=False) as episode:
        summary = json.loads(episode["summary_json"].item())
    if summary["seed"] != seed:
        raise ValueError(f"Summary seed mismatch in {path}")
    return (
        summary["outcome_category"] in ("success", "reached_target_then_failed")
        or summary["end"]["site_distance"] <= 0.10
    )


if __name__ == "__main__":
    main()
