"""Camera-free development rollouts scored with BiGym's official episode latch."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TEACHER_ROOT = REPO.with_name("wt-teacher")
POLICIES = ("teacher", "teacher-freeze", "modular")
_env = None
_policy = None
_kind = None


def make_policy(kind: str):
    if kind == "modular":
        from .controller import ModularController
        from .kinematics import G1Kinematics

        return ModularController(G1Kinematics(), None)

    # The replay teacher is a sibling-worktree diagnostic dependency only.
    if str(TEACHER_ROOT) not in sys.path:
        sys.path.append(str(TEACHER_ROOT))
    from method.teacher.teacher import Teacher

    teacher = Teacher(mode="v1")
    if kind == "teacher-freeze":
        from .teacher_freeze import TeacherFreeze

        return TeacherFreeze(teacher)
    return teacher


def init_worker(kind: str):
    global _env, _policy, _kind
    import onnxruntime
    from bigym.loco import make

    onnxruntime.disable_telemetry_events()
    _env = make("move_plate", camera_keys=())
    _policy, _kind = make_policy(kind), kind


def run_episode(seed: int) -> dict:
    from advaitic_eval.environment import validate_actions

    from .privileged import PrivilegedTaskState

    env, policy, kind = _env, _policy, _kind
    started = time.monotonic()
    ts = env.reset(seed=seed)
    source = PrivilegedTaskState(env) if kind == "modular" else None
    teacher = getattr(policy, "teacher", policy) if source is None else None
    if source is None:
        policy.reset(env)
    else:
        policy.reset()
    steps, held_steps, max_lift, max_held_lift = 0, 0, 0.0, 0.0
    first_held, min_xy, policy_error, initial_z = None, None, None, None
    phases = Counter()
    phase_entries = []

    while True:
        state = ts.low_dim_obs
        if source is None:
            snap = teacher.observe() if ts.last() else teacher.sensors.snapshot(steps)
            z = snap["pose"][2]
        else:
            snap = source.snapshot(state)
            z = snap["plate_xyz"][2]
        if initial_z is None:
            initial_z = z
        lift = z - initial_z
        max_lift = max(max_lift, lift)
        if snap["held"]:
            held_steps += 1
            max_held_lift = max(max_held_lift, lift)
            if first_held is None:
                first_held = steps
        if lift > 0.03:
            site_xy = snap["site_xy_distance"]
            min_xy = site_xy if min_xy is None else min(min_xy, site_xy)
        if ts.last():
            break

        try:
            action = policy.act(ts) if source is None else policy.act(state, snap)
            action = validate_actions(action)
            if action.shape != (1, 20):
                raise ValueError("Expected one action per privileged rollout step")
            action = action[0]
        except (ValueError, RuntimeError, KeyError, np.linalg.LinAlgError) as exc:
            policy_error = f"{type(exc).__name__}: {exc}"
            break
        if source is None:
            if hasattr(policy, "phase") and policy.phase != "TEACHER":
                phase = policy.phase
            elif teacher.release_snapshot is not None:
                phase = "POST_RELEASE"
            elif teacher.grasp_physical_step is not None:
                phase = "GRASP_THROUGH_PLACE"
            else:
                phase = "APPROACH"
        else:
            phase = policy.phase.value
        phases[phase] += 1
        if not phase_entries or phase_entries[-1]["phase"] != phase:
            phase_entries.append(
                {
                    "phase": phase,
                    "step": steps,
                    "held": bool(snap["held"]),
                    "lift_m": float(lift),
                    "site_distance_m": snap["site_distance"],
                    "tilt_deg": snap["tilt_deg"],
                    "plate_xyz": list(snap["pose"][:3] if source is None else snap["plate_xyz"]),
                    "pelvis": state[18:22].tolist(),
                }
            )
        ts = env.step(action)
        steps += 1

    fell = bool(env.episode_fell())
    success = bool(env.episode_succeeded() and not fell and policy_error is None)
    result = {
        "seed": seed,
        "policy": kind,
        "success": success,
        "fell": fell,
        "termination": "policy_error" if policy_error else env.episode_termination(),
        "policy_error": policy_error,
        "steps": steps,
        "phase_steps": dict(phases),
        "phase_entries": phase_entries,
        "first_held": first_held,
        "held_steps": held_steps,
        "max_lift_m": float(max_lift),
        "max_held_lift_m": float(max_held_lift),
        "m2": bool(max_held_lift >= 0.05),
        "min_target_xy_while_lifted": min_xy,
        "end_held": bool(snap["held"]),
        "end_contacts": snap["contacts"],
        "end_site_dist": snap["site_distance"],
        "end_tilt": snap["tilt_deg"],
        "wall_seconds": time.monotonic() - started,
    }
    result["failure_bucket"] = failure_bucket(result)
    if teacher is not None:
        diag = teacher.diagnostics(success=success, fell=fell)
        result.update(
            teacher_failure_category="policy_error" if policy_error else diag["failure_category"],
            release_step=diag["release_physical_step"],
            release_site_distance=diag["nearest_site_distance_at_release"],
            chosen_demo=teacher.chosen_demo,
        )
    if kind == "teacher-freeze":
        result.update(
            frozen_steps=policy.frozen_steps,
            fully_open_freeze_steps=policy.open_steps,
            retreat_guard_tripped=policy.guard_tripped,
        )
    if source is not None:
        result["transitions"] = policy.transitions
    return result


def failure_bucket(row: dict) -> str:
    """Replay-analysis stages using log evidence together with official outcome."""
    if row["success"]:
        return "success"
    if row["termination"] in {"fell", "physics_error", "policy_error"}:
        return row["termination"]
    if row["end_contacts"].get("floor"):
        return "plate_on_floor"
    if row["first_held"] is None:
        return "A never grasped"
    if not row["m2"]:
        return "B grasped, never lifted"
    nearest = row["min_target_xy_while_lifted"]
    if nearest is None or nearest > 0.05:
        return "C lifted, never reached rack"
    if row["end_contacts"]["target_rack"]:
        return "D1 on rack, never released" if row["end_held"] else "D2 on rack, not settled"
    if row["end_held"]:
        return "E held off rack"
    return "E reached rack, then dropped"


def summarize(records: list[dict]) -> dict:
    total = len(records)
    successes = sum(row["success"] for row in records)
    return {
        "episodes": total,
        "successes": successes,
        "success_rate": successes / total,
        "m2_episodes": sum(row["m2"] for row in records),
        "failure_buckets": dict(Counter(row["failure_bucket"] for row in records)),
        "teacher_categories": dict(
            Counter(
                row["teacher_failure_category"]
                for row in records
                if "teacher_failure_category" in row
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--seed-start", required=True, type=int)
    parser.add_argument("--episodes", required=True, type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    stop = args.seed_start + args.episodes
    if not 1 <= args.episodes <= 60 or not 1 <= args.workers <= args.episodes:
        parser.error("Use 1–60 episodes and 1–episodes workers")
    if not any(
        first <= args.seed_start < stop <= last
        for first, last in ((1_000_000, 1_100_000), (3_000_000, 3_100_000))
    ):
        parser.error("Use an approved 1000000+ dev or 3000000+ privileged seed block")
    os.environ["ORT_DISABLE_TELEMETRY"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("MUJOCO_GL", "egl")
    seeds = range(args.seed_start, stop)
    started = time.monotonic()
    records = []
    with ExitStack() as stack:
        stream = stack.enter_context(args.out.open("x", encoding="utf-8"))
        if args.workers == 1:
            init_worker(args.policy)
            stack.callback(_env.close)
            results = map(run_episode, seeds)
        else:
            pool = stack.enter_context(
                ProcessPoolExecutor(
                    max_workers=args.workers,
                    mp_context=mp.get_context("spawn"),
                    initializer=init_worker,
                    initargs=(args.policy,),
                )
            )
            futures = [pool.submit(run_episode, seed) for seed in seeds]
            results = (future.result() for future in as_completed(futures))
        for row in results:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            records.append(row)
            print(row["seed"], row["success"], row["failure_bucket"], row["steps"], flush=True)
    summary = summarize(records)
    summary.update(
        policy=args.policy,
        seed_start=args.seed_start,
        workers=args.workers,
        wall_seconds=time.monotonic() - started,
    )
    data = json.dumps(summary, indent=2) + "\n"
    args.out.with_suffix(".summary.json").write_text(data, encoding="utf-8")
    print(data, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
