"""Offline relabel of a DAgger directory's gripper labels with a different oracle "open" rule.

Analysis/data tool. Never touches the environment code or the source directory.

Usage:
  relabel_dagger.py SRC_DIR --verify                       # original rule must reproduce stored labels
  relabel_dagger.py SRC_DIR --out DST_DIR --open-dist D --open-tilt T

How: replays each episode's saved `action_executed` from its seed (camera-free BiGym env,
deterministic physics), recomputes the privileged snapshot BEFORE each action exactly as
method/dagger/collect.py does, and recomputes method/dagger/diagnostics.gripper_label.
Safety: for every episode the ORIGINAL rule must reproduce the stored gripper labels and
masks bit-for-bit, otherwise nothing is written for that episode and the run fails.
Only action[:, 18] and label_mask[:, 18] change; every other array is copied unchanged.
The "closed" rule is unchanged; only the "open" distance/tilt limits change.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/keyu/advaitic/wt-vla")
import method.dagger.diagnostics as diag  # noqa: E402
from method.dagger.diagnostics import GRIPPER_DIM, PrivilegedSensors  # noqa: E402
import onnxruntime  # noqa: E402

onnxruntime.disable_telemetry_events()
from bigym.loco import make  # noqa: E402


def label(snapshot, start_z, open_dist, open_tilt):
    """diagnostics.gripper_label with configurable open limits (closed rule unchanged)."""
    touching = snapshot["contacts"]["target_rack"]
    if touching and snapshot["site_distance"] <= open_dist and snapshot["tilt_deg"] <= open_tilt:
        return -1.0
    if snapshot["held"] and snapshot["pose"][2] - start_z >= diag.CLOSED_LIFT_M and not touching:
        return 1.0
    return None


def replay_snapshots(env, seed, executed):
    env.reset(seed=seed)
    sensors = PrivilegedSensors(env.inner_env)
    snaps = [sensors.snapshot(0)]
    for t, a in enumerate(executed):
        ts = env.step(a)
        snaps.append(sensors.snapshot(t + 1))
        if ts.last() and t + 1 < len(executed):
            raise RuntimeError(f"seed {seed}: env ended at {t + 1} < {len(executed)} saved steps")
    return snaps  # snaps[i] = state before action i


def labels_for(snaps, n, open_dist, open_tilt):
    start_z = snaps[0]["pose"][2]
    return [label(snaps[i], start_z, open_dist, open_tilt) for i in range(n)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--open-dist", type=float)
    p.add_argument("--open-tilt", type=float)
    p.add_argument("--shard", default="0/1", help="i/n: process every n-th episode from i")
    a = p.parse_args()
    shard_i, shard_n = map(int, a.shard.split("/"))
    if not 0 <= shard_i < shard_n:
        p.error("--shard must be i/n with 0 <= i < n")
    if not a.verify and (a.out is None or a.open_dist is None or a.open_tilt is None):
        p.error("need --verify, or --out with --open-dist and --open-tilt")
    if a.out is not None and a.out.resolve() == a.src.resolve():
        p.error("--out must differ from the source directory")
    all_files = sorted(a.src.glob("ep_*.npz"), key=lambda f: int(f.stem[3:]))
    files = all_files[shard_i::shard_n]
    env = make("move_plate", camera_keys=())
    totals = {"orig_open": 0, "orig_closed": 0, "new_open": 0, "new_closed": 0,
              "open_to_undefined": 0, "undefined_to_open": 0, "other_changes": 0}
    per_ep = []
    if a.out:
        a.out.mkdir(parents=True, exist_ok=shard_n > 1)
        clash = [f.name for f in files if (a.out / f.name).exists()]
        if clash:
            raise RuntimeError(f"refusing to overwrite {len(clash)} files in {a.out}")
    for f in files:
        z = np.load(f, allow_pickle=False)
        arrays = {k: z[k] for k in z.files}
        seed = int(arrays["seed"])
        act, exe, mask = arrays["action"], arrays["action_executed"], arrays["label_mask"]
        n = len(act)
        if bool(arrays["relabel_execute"]):
            raise RuntimeError(f"{f}: relabel_execute episodes not supported")
        if (mask[:, :GRIPPER_DIM].any() or mask[:, GRIPPER_DIM + 1:].any()):
            raise RuntimeError(f"{f}: unexpected non-gripper labels")
        snaps = replay_snapshots(env, seed, exe)
        # 1) the original rule must reproduce the stored labels bit-for-bit
        orig = labels_for(snaps, n, diag.OPEN_DISTANCE_M, diag.OPEN_TILT_DEG)
        o_mask = np.array([x is not None for x in orig])
        o_val = np.array([exe[i, GRIPPER_DIM] if x is None else x for i, x in enumerate(orig)],
                         dtype=np.float32)
        if not (np.array_equal(o_mask, mask[:, GRIPPER_DIM])
                and np.array_equal(o_val, act[:, GRIPPER_DIM])):
            bad = np.flatnonzero((o_mask != mask[:, GRIPPER_DIM]) | (o_val != act[:, GRIPPER_DIM]))
            raise RuntimeError(f"{f}: original labels NOT reproduced at {len(bad)} steps, "
                               f"first {bad[:5].tolist()}; refusing to write")
        # other dims of `action` must equal executed (collector only changes dim 18)
        other = [d for d in range(20) if d != GRIPPER_DIM]
        if not np.array_equal(act[:, other], exe[:, other]):
            raise RuntimeError(f"{f}: non-gripper action differs from executed")
        rec = {"seed": seed, "steps": n, "orig_open": int((o_val[o_mask] == -1).sum()),
               "orig_closed": int((o_val[o_mask] == 1).sum())}
        if a.out:
            new = labels_for(snaps, n, a.open_dist, a.open_tilt)
            n_mask = np.array([x is not None for x in new])
            n_val = np.array([exe[i, GRIPPER_DIM] if x is None else x for i, x in enumerate(new)],
                             dtype=np.float32)
            rec["new_open"] = int((n_val[n_mask] == -1).sum())
            rec["new_closed"] = int((n_val[n_mask] == 1).sum())
            o_lab = [None if x is None else float(x) for x in orig]
            n_lab = [None if x is None else float(x) for x in new]
            for x, y in zip(o_lab, n_lab):
                if x == y:
                    continue
                if x == -1.0 and y is None:
                    totals["open_to_undefined"] += 1
                elif x is None and y == -1.0:
                    totals["undefined_to_open"] += 1
                else:
                    totals["other_changes"] += 1
            out_act = act.copy()
            out_mask = mask.copy()
            out_act[:, GRIPPER_DIM] = n_val
            out_mask[:, GRIPPER_DIM] = n_mask
            arrays["action"] = out_act
            arrays["label_mask"] = out_mask
            np.savez_compressed(a.out / f.name, **arrays)
            totals["new_open"] += rec["new_open"]
            totals["new_closed"] += rec["new_closed"]
        totals["orig_open"] += rec["orig_open"]
        totals["orig_closed"] += rec["orig_closed"]
        per_ep.append(rec)
        print(json.dumps(rec), flush=True)
    summary = {"src": str(a.src.resolve()), "episodes": len(files), "verified": True,
               "shard": a.shard,
               "open_dist": a.open_dist, "open_tilt": a.open_tilt,
               "orig_rule": {"open_dist": diag.OPEN_DISTANCE_M, "open_tilt": diag.OPEN_TILT_DEG,
                             "closed_lift": diag.CLOSED_LIFT_M}, "totals": totals}
    print(json.dumps(summary), flush=True)
    if a.out:
        name = "relabel_config.json" if shard_n == 1 else f"relabel_config.shard{shard_i}.json"
        (a.out / name).write_text(json.dumps(summary, indent=1))
        for extra in ("collector_config.json", "summary.jsonl"):
            if (a.src / extra).exists():
                (a.out / extra).write_bytes((a.src / extra).read_bytes())


if __name__ == "__main__":
    main()
