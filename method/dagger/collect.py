"""Collect student-visited observations with privileged gripper-only labels."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Before simulator, controller or model imports; workers inherit these settings.
os.environ["ORT_DISABLE_TELEMETRY"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))

from advaitic_eval.environment import (  # noqa: E402
    make_environment,
    observation,
    observed_source_slot,
    validate_actions,
)
from advaitic_eval.policy_process import PolicyProcess  # noqa: E402
from method.dagger.diagnostics import (  # noqa: E402
    CLOSED_LIFT_M,
    GRIPPER_DIM,
    OPEN_DISTANCE_M,
    OPEN_TILT_DEG,
    PrivilegedSensors,
    gripper_label,
    outcome_category,
)

RGB_STRIDE = 2
# Restrict to the designated training block, also excluding the public/final
# evaluator block (7,310,000+) and any future reviewer seeds outside this block.
TRAIN_SEED_START = 2_000_000
TRAIN_SEED_STOP = 3_000_000


def parse_seeds(spec):
    """Python-style A:B, stop exclusive; refuse anything outside training."""
    try:
        start, stop = map(int, spec.split(":"))
    except ValueError as exc:
        raise ValueError("--seeds must be A:B (stop exclusive)") from exc
    if not TRAIN_SEED_START <= start < stop <= TRAIN_SEED_STOP:
        raise ValueError("Seeds must be in the training block [2000000, 3000000)")
    return range(start, stop)


def save_npz_atomic(path, **arrays):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def read_records(stream):
    stream.seek(0)
    records = {}
    for line in stream:
        try:
            row = json.loads(line)
        except ValueError:
            continue  # Preserve an interrupted trailing write; next append separates it.
        if row.get("completed") is not True or type(row.get("seed")) is not int:
            raise ValueError("Invalid completed episode summary")
        if row["seed"] in records:
            raise ValueError(f"Duplicate seed in summary: {row['seed']}")
        records[row["seed"]] = row
    return records


def append_summary(path, record):
    """Short shared-log lock: two collectors can share a directory and JSONL."""
    with path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        records = read_records(stream)
        if record["seed"] in records:
            if records[record["seed"]] != record:
                raise ValueError(f"Conflicting summary for seed {record['seed']}")
            return
        stream.seek(0, os.SEEK_END)
        if stream.tell():
            stream.seek(stream.tell() - 1)
            if stream.read(1) != "\n":
                stream.write("\n")
        stream.write(json.dumps(record, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def collect_episode(env, process, seed, *, max_steps, stop_after_placed, relabel_execute):
    started = time.monotonic()
    ts = env.reset(seed=seed)
    process.reset()
    sensors = PrivilegedSensors(env.inner_env)
    initial = end = sensors.snapshot(0)
    start_z = initial["pose"][2]
    rgb, states, actions, executed, masks = [], [], [], [], []
    steps, max_lift, held_steps = 0, 0.0, 0
    first_held = first_placed = None
    counts = {"open": 0, "closed": 0, "undefined": 0}
    termination = None
    while not ts.last() and termination is None:
        # Same call cadence and full returned chunk execution as evaluator.runner.
        chunk = validate_actions(process.act(observation(ts)))
        for student_action in chunk:
            label = gripper_label(end, start_z)
            if label == -1 and first_placed is None:
                first_placed = steps
            action = student_action.copy()
            label_mask = np.zeros(20, dtype=bool)
            if label is not None:
                action[GRIPPER_DIM] = label
                label_mask[GRIPPER_DIM] = True
            action_executed = action.copy() if relabel_execute else student_action.copy()
            states.append(observation(ts)["state"].copy())
            actions.append(action)
            executed.append(action_executed)
            masks.append(label_mask)
            if steps % RGB_STRIDE == 0:
                rgb.append(observation(ts)["rgb"].copy())
            counts["undefined" if label is None else "open" if label == -1 else "closed"] += 1
            ts = env.step(action_executed)
            steps += 1
            end = sensors.snapshot(steps)
            max_lift = max(max_lift, end["pose"][2] - start_z)
            if end["held"]:
                held_steps += 1
                if first_held is None:
                    first_held = steps
            if gripper_label(end, start_z) == -1 and first_placed is None:
                first_placed = steps
            if ts.last():
                break
            if first_placed is not None and steps >= first_placed + stop_after_placed:
                termination = "stop_after_placed"
            elif steps >= max_steps:
                termination = "max_steps"
            if termination is not None:
                break
    fell = bool(env.episode_fell())
    success = bool(env.episode_succeeded() and not fell and termination is None)
    record = {
        "seed": seed,
        "slot": int(observed_source_slot(env)),
        "completed": True,
        "success": success,
        "fell": fell,
        "steps": steps,
        "termination": termination or env.episode_termination(),
        "relabel_execute": relabel_execute,
        "outcome_category": outcome_category(success, fell, max_lift, first_placed, end, start_z),
        "max_lift_m": max_lift,
        "first_held": first_held,
        "held_steps": held_steps,
        "first_placed": first_placed,
        "label_counts": counts,
        "label_fractions": {key: value / steps for key, value in counts.items()},
        "initial": initial,
        "end": end,
        "wall_time": time.monotonic() - started,
    }
    arrays = {
        "rgb": np.asarray(rgb, dtype=np.uint8),
        "rgb_stride": np.int64(RGB_STRIDE),
        "state": np.asarray(states, dtype=np.float32),
        "action": np.asarray(actions, dtype=np.float32),
        "action_executed": np.asarray(executed, dtype=np.float32),
        "label_mask": np.asarray(masks, dtype=bool),
        "seed": np.int64(seed),
        "slot": np.int64(record["slot"]),
        "success": np.bool_(success),
        "relabel_execute": np.bool_(relabel_execute),
    }
    return record, arrays


def run_config(args):
    digest = hashlib.sha256()
    with args.checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    config = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": digest.hexdigest(),
        "policy_python": str(Path(args.policy_python).resolve()),
        "max_steps": args.max_steps,
        "stop_after_placed": args.stop_after_placed,
        "relabel_execute": args.relabel_execute,
        "oracle": {
            "distance_m": OPEN_DISTANCE_M,
            "tilt_deg": OPEN_TILT_DEG,
            "lift_m": CLOSED_LIFT_M,
        },
        "source_sha256": {
            str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path(__file__).with_name("diagnostics.py"),
                REPO_ROOT / "submission/policy.py",
                REPO_ROOT / "submission/model.py",
                REPO_ROOT / "src/advaitic_eval/environment.py",
                REPO_ROOT / "src/advaitic_eval/policy_process.py",
                REPO_ROOT / "src/advaitic_eval/worker.py",
            )
        },
    }
    run_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    with (args.out / ".collect.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = args.out / "collector_config.json"
        if path.exists():
            if json.loads(path.read_text()) != config:
                raise ValueError("Collector/checkpoint configuration changed; choose a new --out")
        else:
            with tempfile.NamedTemporaryFile(mode="w", dir=args.out, delete=False) as stream:
                temporary = Path(stream.name)
                try:
                    json.dump(config, stream, indent=2)
                    stream.flush()
                    os.fsync(stream.fileno())
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
    return run_id


def run(args):
    seeds = parse_seeds(args.seeds)
    if args.max_steps < 1 or args.stop_after_placed < 1 or args.policy_timeout <= 0:
        raise ValueError("max-steps, stop-after-placed and policy-timeout must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    run_id = run_config(args)
    os.environ["ADV_CKPT"] = str(args.checkpoint.resolve())
    summary = args.out / "summary.jsonl"
    env = process = None
    try:
        for seed in seeds:
            # Episode lock prevents duplicate work/publication even on overlapping ranges.
            with (args.out / f".ep_{seed}.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                path = args.out / f"ep_{seed}.npz"
                if path.exists():
                    with np.load(path, allow_pickle=False) as episode:
                        if str(episode["run_id"].item()) != run_id or int(episode["seed"]) != seed:
                            raise ValueError(f"Incompatible existing episode: {path}")
                        record = json.loads(str(episode["summary_json"].item()))
                    # Recover a crash after npz publication and before JSONL append.
                    append_summary(summary, record)
                    print(f"seed={seed} skipped (already saved)", flush=True)
                    continue
                if summary.exists():
                    with summary.open("r", encoding="utf-8") as stream:
                        fcntl.flock(stream, fcntl.LOCK_SH)
                        if seed in read_records(stream):
                            raise ValueError(f"Summary exists but episode is missing: {path}")
                if env is None:
                    env = make_environment()
                    process = PolicyProcess(
                        "submission.policy:make_policy",
                        REPO_ROOT,
                        args.out / f"policy.{os.getpid()}.stderr.log",
                        timeout=args.policy_timeout,
                        python=args.policy_python,
                    )
                record, arrays = collect_episode(
                    env,
                    process,
                    seed,
                    max_steps=args.max_steps,
                    stop_after_placed=args.stop_after_placed,
                    relabel_execute=args.relabel_execute,
                )
                record["run_id"] = run_id
                save_npz_atomic(
                    path,
                    **arrays,
                    run_id=np.asarray(run_id),
                    summary_json=np.asarray(json.dumps(record, allow_nan=False)),
                )
                append_summary(summary, record)
                print(json.dumps(record, allow_nan=False), flush=True)
    finally:
        if process is not None:
            process.close()
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--seeds", required=True, help="A:B; stop exclusive; training block only")
    parser.add_argument(
        "--out", required=True, type=Path, help="Episode directory and shared JSONL"
    )
    parser.add_argument("--policy-python", default="/home/keyu/advaitic/mlenv/bin/python")
    parser.add_argument("--policy-timeout", type=float, default=120.0)
    parser.add_argument("--relabel-execute", action="store_true")
    parser.add_argument("--max-steps", type=int, default=1700)
    parser.add_argument("--stop-after-placed", type=int, default=150)
    args = parser.parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted; rerun the same command to resume completed episodes.\n")
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
