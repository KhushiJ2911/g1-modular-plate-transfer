"""Read-only simulator replay -> full-action tracking labels (analysis/data only).

Run with ADV_THROTTLE=1; this tool never executes its generated labels. See
track_label.md for the phase metric, action indexing, and sanity-check limits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ["ORT_DISABLE_TELEMETRY"] = "1"

import numpy as np

from method.teacher.teacher import METHOD_DIR, Teacher

GRIPPER_DIM = 18
CLOSED_LIFT_M = 0.03  # method/dagger/diagnostics.py; strict relabel rule unchanged.
OPEN_DISTANCE_M = 0.04
OPEN_TILT_DEG = 10.0
FORBIDDEN_SEEDS = ((2600000, 2700000), (7310000, 7320000))


@dataclass(frozen=True)
class TrackConfig:
    window: int = 25
    arm_weight: float = 1.0
    gripper_weight: float = 0.1
    arm_scale_rad: float = 0.2
    gripper_scale: float = 0.6
    # Left arm qpos, NOT action offsets 4:11 or the adjacent finger joints 7:9.
    arm_indices: tuple[int, ...] = tuple(range(7))
    gripper_index: int = 44
    drop_cost: float | None = None

    def __post_init__(self):
        if type(self.window) is not int or self.window < 0:
            raise ValueError("window must be a nonnegative integer")
        for name in ("arm_weight", "gripper_weight", "arm_scale_rad", "gripper_scale"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not self.arm_indices
            or len(set(self.arm_indices)) != len(self.arm_indices)
            or any(type(i) is not int or not 0 <= i < 50 for i in self.arm_indices)
            or type(self.gripper_index) is not int
            or not 0 <= self.gripper_index < 50
        ):
            raise ValueError("arm/gripper indices must be distinct valid state indices")
        if self.gripper_index in self.arm_indices:
            raise ValueError("gripper index overlaps arm indices")
        if self.drop_cost is not None and (not np.isfinite(self.drop_cost) or self.drop_cost < 0):
            raise ValueError("drop_cost must be finite and nonnegative, or null")


def validate_seed(seed: int):
    if not 0 <= seed < 2**32 or any(lo <= seed < hi for lo, hi in FORBIDDEN_SEEDS):
        raise ValueError(f"Refusing prohibited/invalid seed {seed}")


def match_phase(state, demo_states, previous: int, config: TrackConfig) -> tuple[int, float]:
    """Inclusive forward window; earliest phase wins exact ties; no clock prior.

    demo_states contains only phases with an available action (s[:-1]). Costs
    are dimensionless: weighted mean squared left-arm error plus gripper error.
    """
    if not 0 <= previous < len(demo_states):
        raise ValueError("previous phase outside demonstration")
    stop = min(previous + config.window + 1, len(demo_states))
    refs = demo_states[previous:stop].astype(np.float64)
    arm = (refs[:, config.arm_indices] - state[list(config.arm_indices)]) / config.arm_scale_rad
    grip = (refs[:, config.gripper_index] - state[config.gripper_index]) / config.gripper_scale
    costs = config.arm_weight * np.mean(arm**2, axis=1) + config.gripper_weight * grip**2
    if not np.isfinite(costs).all():
        raise ValueError("Nonfinite phase matching cost")
    offset = int(np.argmin(costs))
    return previous + offset, float(costs[offset])


def strict_gripper_label(snapshot: dict, start_z: float) -> float | None:
    """Same strict relabel_dagger.py rule: contact AND 4cm AND 10deg to open."""
    touching = snapshot["contacts"]["target_rack"]
    if (
        touching
        and snapshot["site_distance"] <= OPEN_DISTANCE_M
        and snapshot["tilt_deg"] <= OPEN_TILT_DEG
    ):
        return -1.0
    if snapshot["held"] and snapshot["pose"][2] - start_z >= CLOSED_LIFT_M and not touching:
        return 1.0
    return None


def assert_state_exact(actual, expected, *, seed: int, step: int):
    actual = np.asarray(actual, dtype=np.float32)
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise RuntimeError(f"seed {seed}, step {step}: replay state shape/dtype mismatch")
    # Compare bytes as requested (also catches a difference in signed zero).
    if actual.tobytes() != expected.tobytes():
        bits = actual.view(np.uint32) != expected.view(np.uint32)
        dims = np.flatnonzero(bits).tolist()
        delta = float(np.max(np.abs(actual.astype(float) - expected)))
        raise RuntimeError(
            f"seed {seed}, step {step}: replay state NOT bit-for-bit equal; "
            f"dims={dims}, max_abs={delta}; refusing to publish episode"
        )


def tracking_action(teacher: Teacher, ts, phase: int, snapshot: dict, start_z: float):
    """Reuse v1 base feedback at t*, then use absolute arms/grippers at t*+1."""
    teacher.demo_step = phase
    action = teacher._act_replay(ts)
    actions = teacher.actions[teacher.chosen_demo]
    next_phase = min(phase + 1, len(actions) - 1)
    action[4:] = actions[next_phase, 4:]
    grip = strict_gripper_label(snapshot, start_z)
    if grip is not None:
        action[GRIPPER_DIM] = grip
    return action, grip


def difference_report(labels, executed) -> dict:
    diff = np.abs(labels.astype(np.float64) - executed)
    worst = np.argmax(diff, axis=0)
    top = np.argsort(np.max(diff[:, 4:18], axis=1), kind="stable")[-10:][::-1]
    return {
        "mean_abs_by_dim": diff.mean(axis=0).tolist(),
        "max_abs_by_dim": diff.max(axis=0).tolist(),
        "max_step_by_dim": worst.tolist(),
        "different_steps_by_dim_at_1e-6": (diff > 1e-6).sum(axis=0).tolist(),
        "arm_mean_abs": float(diff[:, 4:18].mean()),
        "arm_max_abs": float(diff[:, 4:18].max()),
        "worst_arm_steps": [
            {
                "step": int(i),
                "max_abs": float(diff[i, 4:18].max()),
                "dim": int(4 + np.argmax(diff[i, 4:18])),
            }
            for i in top
        ],
    }


def replay_episode(env, teacher, arrays, config: TrackConfig) -> tuple[dict, dict]:
    seed = int(arrays["seed"])
    validate_seed(seed)
    states, executed = arrays["state"], arrays["action_executed"]
    n = len(executed)
    if (
        n < 1
        or states.shape != (n, 50)
        or states.dtype != np.float32
        or executed.shape != (n, 20)
        or executed.dtype != np.float32
        or arrays["action"].shape != executed.shape
        or not np.isfinite(states).all()
        or not np.isfinite(executed).all()
        or np.any(np.abs(executed) > 1)
    ):
        raise ValueError(f"seed {seed}: invalid episode arrays")
    if bool(arrays.get("relabel_execute", False)):
        raise ValueError("relabel_execute episodes are not supported")
    if teacher.config["mode"] != "v1" or teacher.config["max_wait"] != 0:
        raise ValueError("Tracking requires a v1 teacher with max_wait=0")
    ts = env.reset(seed=seed)
    assert_state_exact(ts.low_dim_obs, states[0], seed=seed, step=0)
    teacher.reset(env)
    # Finalize demo choice and both shifts using the original Teacher implementation.
    teacher._act_replay(ts)
    demo_states = teacher.states[teacher.chosen_demo][:-1]
    phase = 0
    phases = np.empty(n, dtype=np.int64)
    costs = np.empty(n, dtype=np.float64)
    labels = np.empty((n, 20), dtype=np.float32)
    start_z = teacher.initial_snapshot["pose"][2]
    oracle_counts = {"open": 0, "closed": 0, "demo_fallback": 0}
    for i in range(n):
        assert_state_exact(ts.low_dim_obs, states[i], seed=seed, step=i)
        if ts.last():
            raise RuntimeError(f"seed {seed}: environment ended before saved action {i}")
        snapshot = teacher.sensors.snapshot(i)
        phase, cost = match_phase(states[i], demo_states, phase, config)
        labels[i], grip = tracking_action(teacher, ts, phase, snapshot, start_z)
        oracle_counts["demo_fallback" if grip is None else "open" if grip < 0 else "closed"] += 1
        phases[i], costs[i] = phase, cost
        # Only the ORIGINAL executed action ever reaches physics.
        ts = env.step(executed[i])
    keep = np.ones(n, dtype=bool) if config.drop_cost is None else costs <= config.drop_cost
    output = {
        "action": labels,
        "label_mask": np.repeat(keep[:, None], 20, axis=1),
        "track_phase": phases,
        "track_cost": costs,
    }
    report = {
        "seed": seed,
        "steps": n,
        "verified_states": n,
        "chosen_demo": int(teacher.chosen_demo),
        "grasp_shift": teacher.grasp_shift.tolist(),
        "place_shift": teacher.place_shift.tolist(),
        "phase": phases.tolist(),
        "phase_every_50": [[i, int(phases[i])] for i in range(0, n, 50)],
        "phase_end": int(phases[-1]),
        "demo_last_phase": len(demo_states) - 1,
        "phase_stalls": int(np.count_nonzero(np.diff(phases) == 0)),
        "phase_max_jump": int(np.diff(phases).max(initial=0)),
        "cost_quantiles_0_50_90_99_100": np.quantile(costs, [0, 0.5, 0.9, 0.99, 1]).tolist(),
        "dropped_steps": int((~keep).sum()),
        "dropped_fraction": float((~keep).mean()),
        "fraction_above_cost_4": float((costs > 4).mean()),
        "gripper_rule_counts": oracle_counts,
        **difference_report(labels, executed),
    }
    for row in report["worst_arm_steps"]:
        row["phase"] = int(phases[row["step"]])
        row["cost"] = float(costs[row["step"]])
    return output, report


def publish_npz(destination: Path, arrays: dict):
    """Atomic no-clobber publication, including under concurrent shards."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".npz", delete=False) as f:
            temporary = Path(f.name)
            np.savez_compressed(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.link(temporary, destination)  # fails if destination already exists
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json_new(path: Path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", default="0/1", help="i/n, zero based; at most two workers")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("track_config.json")
    )
    parser.add_argument("--window", type=int, help="Override inclusive forward window")
    parser.add_argument(
        "--drop-cost",
        type=float,
        nargs="?",
        const=4.0,
        help="Mask all 20 dims when cost exceeds threshold (flag alone: 4)",
    )
    parser.add_argument("--demos", type=Path, default=METHOD_DIR / "data/demos_lowdim.npz")
    parser.add_argument("--meta", type=Path, default=METHOD_DIR / "data/demo_meta.json")
    parser.add_argument(
        "-o", "--report", type=Path, help="New JSON report; default: OUT/track_report*.json"
    )
    args = parser.parse_args(argv)
    try:
        shard_i, shard_n = map(int, args.shard.split("/"))
        if not 0 <= shard_i < shard_n <= 2:
            raise ValueError
    except ValueError:
        parser.error("--shard must be i/n with 0 <= i < n <= 2")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if os.environ.get("ADV_THROTTLE") != "1":
        parser.error("Set ADV_THROTTLE=1 before simulator replay")
    src, out = args.src.resolve(), args.out.resolve()
    if not src.is_dir() or out == src or src in out.parents or out in src.parents:
        parser.error("SRC must exist and OUT must be a separate directory, without nesting")
    settings = json.loads(args.config.read_text())
    if args.window is not None:
        settings["window"] = args.window
    if args.drop_cost is not None:
        settings["drop_cost"] = args.drop_cost
    if "arm_indices" in settings:
        settings["arm_indices"] = tuple(settings["arm_indices"])
    config = TrackConfig(**settings)
    files = sorted(src.glob("ep_*.npz"), key=lambda p: int(p.stem[3:]))
    files = files[shard_i::shard_n]
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        parser.error("No episodes selected")
    suffix = "" if shard_n == 1 else f".shard{shard_i}"
    report_path = args.report or out / f"track_report{suffix}.json"
    config_path = out / f"track_config{suffix}.json"
    if report_path.resolve() == src or src in report_path.resolve().parents:
        parser.error("Report must not be inside SRC")
    # Check every selected seed before constructing a simulator or publishing data.
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            validate_seed(int(data["seed"]))
    out.mkdir(parents=True, exist_ok=shard_n > 1)
    for path in [out / f.name for f in files] + [report_path, config_path]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    command = shlex.join(["python", "-m", "method.teacher.track_label", *(argv or sys.argv[1:])])
    provenance = {
        "source": str(src),
        "output": str(out),
        "command": command,
        "shard": args.shard,
        "config": asdict(config),
        "oracle": {
            "open_distance_m": OPEN_DISTANCE_M,
            "open_tilt_deg": OPEN_TILT_DEG,
            "closed_lift_m": CLOSED_LIFT_M,
        },
        "demo_sha256": hashlib.sha256(args.demos.read_bytes()).hexdigest(),
        "meta_sha256": hashlib.sha256(args.meta.read_bytes()).hexdigest(),
        "labeler_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "phase_metric": "left arm qpos MSE / .2rad^2 + .1 * (gripper delta / .6)^2; "
        "scales/weights configurable; inclusive forward window, earliest ties",
        "plate_progress": "Unavailable in aligned demo archive; not synthesized from a mismatched replay",
        "action_indexing": "base v1(t*), arms and grippers demo[min(t*+1,last)], strict gripper override",
        "waits": False,
        "replay_state_check": "float32 bytes at every saved pre-action state",
    }
    write_json_new(config_path, provenance)
    import onnxruntime

    onnxruntime.disable_telemetry_events()
    from bigym.loco import make

    teacher = Teacher(args.demos, args.meta, mode="v1", max_wait=0)
    env = make("move_plate", camera_keys=())
    records = []
    try:
        for path in files:
            # Delay loading RGB until replay has passed; never mutate the source archive.
            with np.load(path, allow_pickle=False) as data:
                arrays = {key: data[key] for key in ("seed", "state", "action", "action_executed")}
                if "relabel_execute" in data:
                    arrays["relabel_execute"] = data["relabel_execute"]
                replacements, record = replay_episode(env, teacher, arrays, config)
                arrays = {key: data[key] for key in data.files if key not in replacements}
                arrays.update(replacements)
                publish_npz(out / path.name, arrays)
            record["file"] = path.name
            records.append(record)
            print(
                json.dumps(
                    {
                        key: record[key]
                        for key in (
                            "file",
                            "seed",
                            "steps",
                            "chosen_demo",
                            "phase_end",
                            "phase_stalls",
                            "arm_mean_abs",
                            "arm_max_abs",
                            "dropped_fraction",
                        )
                    }
                ),
                flush=True,
            )
    finally:
        env.close()
    total = sum(r["steps"] for r in records)
    summary = {
        "episodes": len(records),
        "steps": total,
        "verified_states": total,
        "dropped_fraction": sum(r["dropped_steps"] for r in records) / total,
        "fraction_above_cost_4": sum(r["fraction_above_cost_4"] * r["steps"] for r in records)
        / total,
    }
    write_json_new(report_path, {**provenance, "summary": summary, "episodes": records})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
