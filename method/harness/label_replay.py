"""Training-only deterministic action_executed replay; source archives are immutable.

python -m method.harness.label_replay SRC --out NEW_DIR
python -m method.harness.label_replay SRC --verify --limit 1
"""

import argparse
import json
import os
from pathlib import Path

os.environ["ORT_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from method.dagger.collect import save_npz_atomic
from method.dagger.diagnostics import PrivilegedSensors
from method.harness.data import LABEL_KEYS, audit_files, check_seed, snapshot_labels


def replay_episode(env, arrays):
    seed = check_seed(arrays["seed"])
    executed = arrays["action_executed"]
    states = arrays["state"]
    n = len(executed)
    if n < 1 or executed.shape != (n, 20) or states.shape != (n, 50):
        raise ValueError("Expected nonempty action_executed (T,20), state (T,50)")
    if not np.isfinite(executed).all() or np.abs(executed).max() > 1:
        raise ValueError("Invalid executed actions")
    ts = env.reset(seed=seed)
    sensors = PrivilegedSensors(env.inner_env)
    rows = {key: [] for key in LABEL_KEYS}
    for t, action in enumerate(executed):
        actual = np.asarray(ts.low_dim_obs, np.float32)
        if not np.array_equal(actual, states[t]):
            error = float(np.max(np.abs(actual - states[t])))
            raise RuntimeError(f"seed={seed} replay diverged at t={t}: max state error={error}")
        for key, value in snapshot_labels(sensors.snapshot(t), actual).items():
            rows[key].append(value)
        ts = env.step(action)
        if ts.last() and t + 1 < n:
            raise RuntimeError(f"seed={seed}: early termination at {t + 1}/{n}")
    labels = {key: np.asarray(values) for key, values in rows.items()}
    for key in LABEL_KEYS:
        if key in arrays and not np.array_equal(labels[key], arrays[key]):
            raise RuntimeError(f"seed={seed}: teacher label mismatch for {key}")
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out", type=Path)
    mode.add_argument("--verify", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard", default="0/1", help="i/n, disjoint shards may share NEW_DIR")
    args = parser.parse_args()
    i, n = map(int, args.shard.split("/"))
    if not 0 <= i < n or (args.limit is not None and args.limit < 1):
        parser.error("Require 0 <= shard i < n and positive limit")
    files = sorted(
        (p for p in args.src.glob("ep_*.npz") if p.stem[3:].isdigit()),
        key=lambda p: int(p.stem[3:]),
    )[i::n]
    if not files:
        parser.error("No episodes in this source/shard")
    if args.limit:
        files = files[: args.limit]
    audit_files(files)
    if args.out:
        source, destination = args.src.resolve(), args.out.resolve()
        if destination == source or source in destination.parents or destination in source.parents:
            parser.error("--out must be a separate directory, outside source")
        args.out.mkdir(parents=True, exist_ok=n > 1)
        if any((args.out / path.name).exists() for path in files):
            parser.error("Refusing to overwrite output episodes")
    # Same pinned camera-free replay used by relabel_dagger.py. Cameras do not
    # participate in physics; state equality is checked at EVERY saved step.
    import onnxruntime

    onnxruntime.disable_telemetry_events()
    from bigym.loco import make

    env = make("move_plate", camera_keys=())
    try:
        for path in files:
            with np.load(path, allow_pickle=False) as episode:
                arrays = {key: episode[key] for key in episode.files}
            if args.verify and not all(key in arrays for key in LABEL_KEYS):
                raise ValueError(f"{path}: --verify requires all four teacher labels")
            labels = replay_episode(env, arrays)
            if args.out:
                save_npz_atomic(args.out / path.name, **{**arrays, **labels})
            print(
                json.dumps(
                    {
                        "episode": path.name,
                        "steps": len(arrays["state"]),
                        "state_equal": True,
                        "existing_labels_equal": [key for key in LABEL_KEYS if key in arrays],
                    }
                ),
                flush=True,
            )
    finally:
        env.close()
    record = {
        "source": str(args.src.resolve()),
        "episodes": len(files),
        "shard": args.shard,
        "state_check": "bit-for-bit every pre-action state",
        "labels": LABEL_KEYS,
        "training_only_privileged": True,
    }
    if args.out:
        (args.out / f"label_replay.{i}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
