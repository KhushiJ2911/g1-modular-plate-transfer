"""Run real teacher actions and print compact privileged traces, without saving demos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Import runner first: it disables ORT telemetry and CUDA before sim imports.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from method.teacher.run_teacher import add_controller_arguments, parse_seeds, run_episode
from method.teacher.teacher import REPO_DIR, Teacher


def print_trace(step, teacher, snapshot, terminal):
    if step % 25 and not terminal:
        return
    xyz = snapshot["pose"][:3]
    contacts = snapshot["contacts"]
    print(
        f"{step:4d} demo={teacher.demo_step:4d} "
        f"xyz=({xyz[0]:+.3f},{xyz[1]:+.3f},{xyz[2]:.3f}) "
        f"d={snapshot['site_distance']:.3f} xy={snapshot['site_xy_distance']:.3f} "
        f"tilt={snapshot['tilt_deg']:5.1f} "
        f"rack={int(contacts['target_rack'])} table={int(contacts['table'])} "
        f"floor={int(contacts['floor'])} held={int(snapshot['held'])} "
        f"phase={teacher.recovery_phase or 'replay'}{' END' if terminal else ''}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", required=True)
    add_controller_arguments(parser)
    args = parser.parse_args()
    try:
        seeds = parse_seeds(args.seeds)
        teacher = Teacher(
            args.demos,
            args.meta,
            mode=args.mode,
            kp=args.kp,
            tolerance=args.tolerance,
            max_wait=args.max_wait,
        )
        sys.path.insert(0, str(REPO_DIR / "src"))
        from advaitic_eval.environment import make_environment

        env = make_environment()
        try:
            for seed in seeds:
                print(f"seed={seed} mode={args.mode}", flush=True)
                record = run_episode(env, teacher, seed, None, trace=print_trace)
                print(json.dumps(record, allow_nan=False), flush=True)
        finally:
            env.close()
    except KeyboardInterrupt:
        return 130
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
