"""Run the privileged teacher in one process and append completed episodes."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# Set before any simulator/controller imports. Training generator uses CPU only.
os.environ["ORT_DISABLE_TELEMETRY"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from method.teacher.dart import DartNoise, validate_scale
from method.teacher.labels import PrivilegedLabels, save_npz_atomic
from method.teacher.teacher import METHOD_DIR, REPO_DIR, Teacher

RGB_STRIDE = 2


def parse_seeds(specs: list[str]) -> list[int]:
    """Accept integers, comma lists and Python-style start:stop[:step] ranges."""
    seeds = []
    for spec in specs:
        for item in spec.split(","):
            if ":" in item:
                parts = [int(v) for v in item.split(":")]
                if len(parts) not in (2, 3) or (len(parts) == 3 and parts[2] == 0):
                    raise ValueError(f"Invalid seed range: {item}")
                seeds.extend(range(*parts))
            else:
                seeds.append(int(item))
    if not seeds or any(s < 0 or s > 2**32 - 1 for s in seeds):
        raise ValueError("Provide at least one seed, each in [0, 2**32-1]")
    return list(dict.fromkeys(seeds))


def read_records(stream) -> dict[int, dict]:
    """Ignore interrupted/malformed lines, preserving them in the append-only log."""
    stream.seek(0)
    records = {}
    for number, line in enumerate(stream, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if row.get("completed") is not True or type(row.get("seed")) is not int:
                raise ValueError("Not a completed episode record")
        except (ValueError, AttributeError) as exc:
            print(f"Ignoring JSONL line {number}: {exc}", file=sys.stderr, flush=True)
            continue
        if row["seed"] in records:
            raise ValueError(f"Duplicate completed seed in log: {row['seed']}")
        records[row["seed"]] = row
    return records


def append_record(stream, record: dict):
    """Append only after completion; fsync before reporting durable progress."""
    payload = json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n"
    stream.seek(0, os.SEEK_END)
    # A previous SIGKILL may have left a partial final line. Preserve its bytes,
    # and separate the next record so it remains independently parseable.
    if stream.tell():
        stream.seek(stream.tell() - 1)
        if stream.read(1) != "\n":
            stream.write("\n")
    stream.write(payload)
    stream.flush()
    os.fsync(stream.fileno())


def save_demo(
    directory: Path, seed: int, slot: int, rgb, states, actions, executed, dart_noise, labels
) -> Path:
    """Atomically publish successful pre-action observations and their actions."""
    return save_npz_atomic(
        directory / f"ep_{seed}.npz",
        rgb=np.asarray(rgb, dtype=np.uint8),
        state=np.asarray(states, dtype=np.float32),
        action=np.asarray(actions, dtype=np.float32),
        action_executed=np.asarray(executed, dtype=np.float32),
        dart_noise=np.float32(dart_noise),
        seed=np.int64(seed),
        slot=np.int64(slot),
        success=np.bool_(True),
        rgb_stride=np.int64(RGB_STRIDE),
        **labels.arrays(),
    )


def run_episode(
    env, teacher: Teacher, seed: int, save_directory: Path | None, *, trace=None, dart_noise=0.0
) -> dict:
    started = time.monotonic()
    noise = DartNoise(seed, dart_noise)
    ts = env.reset(seed=seed)
    from advaitic_eval.environment import observed_source_slot

    slot = observed_source_slot(env)
    teacher.reset(env)
    if trace is not None:
        trace(0, teacher, teacher.current_snapshot, False)
    labels = PrivilegedLabels(env) if save_directory is not None else None
    rgb, states, actions, executed = [], [], [], []
    steps = 0
    while not ts.last():
        if labels is not None:
            labels.append(ts.low_dim_obs, teacher.current_snapshot)
        demo_step = teacher.demo_step
        action = teacher.act(ts)
        if action.shape != (20,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise ValueError("Teacher returned an invalid normalized action")
        action_executed = noise.execute(action, teacher, demo_step)
        if save_directory is not None:
            states.append(np.asarray(ts.low_dim_obs, dtype=np.float32).copy())
            actions.append(action.copy())
            executed.append(action_executed.copy())
            if steps % RGB_STRIDE == 0:
                rgb.append(np.asarray(ts.rgb_obs, dtype=np.uint8).copy())
        ts = env.step(action_executed)
        steps += 1
        snapshot = teacher.observe()
        if trace is not None:
            trace(steps, teacher, snapshot, ts.last())
    fell = bool(env.episode_fell())
    success = bool(env.episode_succeeded() and not fell)
    record = {
        "seed": seed,
        "dart_noise": noise.scale,
        "slot": slot,
        "chosen_demo": teacher.chosen_demo,
        "grasp_shift": teacher.grasp_shift.tolist(),
        "place_shift": teacher.place_shift.tolist(),
        "success": success,
        "completed": True,
        "termination": env.episode_termination(),
        "fell": fell,
        "steps": steps,
        "max_pelvis_tracking_error": teacher.max_tracking_error,
        "wait_steps": dict(teacher.wait_steps),
        "demo_step": teacher.demo_step,
        "demo_file": None,
    }
    record.update(teacher.diagnostics(success=success, fell=fell))
    if success and save_directory is not None:
        record["demo_file"] = str(
            save_demo(
                save_directory, seed, slot, rgb, states, actions, executed, noise.scale, labels
            )
        )
    record["wall_time"] = time.monotonic() - started
    return record


def run(args) -> int:
    dart_noise = validate_scale(args.dart_noise)
    seeds = parse_seeds(args.seeds)
    teacher = Teacher(
        args.demos,
        args.meta,
        mode=args.mode,
        kp=args.kp,
        tolerance=args.tolerance,
        max_wait=args.max_wait,
    )
    # Prevent accidentally resuming a changed controller/data configuration.
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).with_name("teacher.py"),
        Path(__file__).with_name("diagnostics.py"),
        Path(__file__).with_name("dart.py"),
        Path(__file__).with_name("labels.py"),
        args.demos,
        args.meta,
    ):
        digest.update(path.read_bytes())
    digest.update((REPO_DIR / "src/advaitic_eval/action_layout.json").read_bytes())
    digest.update(json.dumps(teacher.config, sort_keys=True).encode())
    digest.update(json.dumps({"dart_noise": dart_noise}, sort_keys=True).encode())
    run_id = digest.hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.save_demos is not None:
        args.save_demos = args.save_demos.resolve()
        args.save_demos.mkdir(parents=True, exist_ok=True)
    with args.out.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Output JSONL is already in use; use one log per process") from exc
        records = read_records(stream)
        if any(row.get("run_id") != run_id for row in records.values()):
            raise ValueError("Controller/data configuration changed; choose a new --out JSONL")
        pending = [seed for seed in seeds if seed not in records]
        print(
            f"Requested {len(seeds)}; skipped {len(seeds) - len(pending)}; pending {len(pending)}"
        )
        if pending:
            # Import this worktree's evaluator, even with a sibling editable env.
            sys.path.insert(0, str(REPO_DIR / "src"))
            from advaitic_eval.environment import make_environment

            env = make_environment()
            try:
                for seed in pending:
                    record = run_episode(env, teacher, seed, args.save_demos, dart_noise=dart_noise)
                    record.update(run_id=run_id, teacher_config=teacher.config)
                    append_record(stream, record)
                    records[seed] = record
                    print(
                        f"seed={seed} slot={record['slot']} demo={record['chosen_demo']} "
                        f"success={record['success']} category={record['failure_category']} "
                        f"end={record['termination']} "
                        f"steps={record['steps']} error={record['max_pelvis_tracking_error']:.4f}m",
                        flush=True,
                    )
            finally:
                env.close()
        successes = sum(records[seed]["success"] for seed in seeds)
        print(
            f"Privileged teacher: {successes}/{len(seeds)} successful ({successes / len(seeds):.1%})"
        )
        for slot in range(6):
            rows = [records[s] for s in seeds if records[s]["slot"] == slot]
            print(f"  slot {slot}: {sum(r['success'] for r in rows)}/{len(rows)}")
    return 0


def add_controller_arguments(parser):
    parser.add_argument("--mode", choices=("v1", "v2"), default="v2")
    parser.add_argument("--demos", type=Path, default=METHOD_DIR / "data/demos_lowdim.npz")
    parser.add_argument("--meta", type=Path, default=METHOD_DIR / "data/demo_meta.json")
    parser.add_argument("--kp", type=float, default=2.0, help="Base position gain in 1/s")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="Pelvis gate tolerance in m (v1: .01; v2: .015)",
    )
    parser.add_argument(
        "--max-wait", type=int, default=100, help="Pelvis gate wait limit (v2: grasp only)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", nargs="+", required=True, help="e.g. 100000:100030 or 100000,100003 100008"
    )
    parser.add_argument("--out", type=Path, required=True, help="Append-only episode JSONL")
    parser.add_argument("--save-demos", type=Path, help="Save successful episodes atomically here")
    parser.add_argument(
        "--dart-noise",
        type=float,
        default=0.0,
        help="OU action noise scale (finite, nonnegative; default: 0 = off)",
    )
    add_controller_arguments(parser)
    args = parser.parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        print(
            "Interrupted; unfinished episode was not logged. Resume with the same command.",
            file=sys.stderr,
        )
        return 130
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
