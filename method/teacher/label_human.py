"""Replay human actions to produce aligned privileged-label sidecars, CPU only."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["ORT_DISABLE_TELEMETRY"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from method.teacher.labels import PrivilegedLabels, save_npz_atomic
from method.teacher.run_teacher import parse_seeds
from method.teacher.teacher import REPO_DIR

HUMAN_DIR = Path("/home/keyu/advaitic/data/human60")
ALIGNMENT_TOLERANCE = 0.05


def replay_labels(env, source: Path) -> tuple[dict, float]:
    """Return labels only for a fully aligned, successful real action replay."""
    with np.load(source, allow_pickle=False) as data:
        actions = np.asarray(data["action"], dtype=np.float32)
        states = np.asarray(data["state"], dtype=np.float32)
        seed = int(data["seed"])
    if actions.ndim != 2 or actions.shape[1] != 20 or not len(actions):
        raise ValueError("Expected nonempty (T,20) human actions")
    if states.shape != (len(actions), 50):
        raise ValueError("Expected human state rows aligned with (T,20) actions")
    if not np.isfinite(actions).all() or np.any(np.abs(actions) > 1):
        raise ValueError("Human actions must be finite and normalized to [-1,1]")
    if not np.isfinite(states).all() or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Nonfinite human states or invalid seed")
    ts = env.reset(seed=seed)
    labels = PrivilegedLabels(env)
    max_deviation = 0.0
    for step, action in enumerate(actions):
        if ts.last():
            raise ValueError(
                f"Replay terminated early at step={step}/{len(actions)}; "
                f"max_deviation={max_deviation:.9g}"
            )
        observed = np.asarray(ts.low_dim_obs, dtype=np.float32)
        if observed.shape != (50,) or not np.isfinite(observed).all():
            raise AssertionError(f"Invalid replay state at step={step}")
        deviation = float(np.max(np.abs(observed - states[step])))
        max_deviation = max(max_deviation, deviation)
        if deviation > ALIGNMENT_TOLERANCE:
            raise AssertionError(
                f"State alignment failed at step={step}: "
                f"max_deviation={max_deviation:.9g} > {ALIGNMENT_TOLERANCE}"
            )
        # Use the observed replay pelvis, just as in the teacher runner.
        labels.append(observed)
        ts = env.step(action)
    if not ts.last() or not env.episode_succeeded() or env.episode_fell():
        raise ValueError(
            f"Replay did not terminate successfully after {len(actions)} actions; "
            f"termination={env.episode_termination()}; max_deviation={max_deviation:.9g}"
        )
    return {
        **labels.arrays(),
        "seed": np.int64(seed),
        "success": np.bool_(True),
        "max_state_deviation": np.float32(max_deviation),
    }, max_deviation


def run(args) -> int:
    indices = parse_seeds(args.indices)
    output = args.out_dir if args.out_dir is not None else args.demos_dir
    output.mkdir(parents=True, exist_ok=True)
    pending, skipped, flagged = [], 0, 0
    for index in indices:
        if index == 45:
            print("demo=45 FLAG known non-reproducing demo; no sidecar written", flush=True)
            continue
        destination = output / f"ep_{index}.priv.npz"
        if destination.exists():
            skipped += 1
            print(f"demo={index} SKIP existing={destination}", flush=True)
        else:
            pending.append((index, destination))
    written, max_deviation = 0, 0.0
    if pending:
        sys.path.insert(0, str(REPO_DIR / "src"))
        from advaitic_eval.environment import make_environment

        env = make_environment()
        try:
            for index, destination in pending:
                try:
                    arrays, deviation = replay_labels(env, args.demos_dir / f"ep_{index}.npz")
                    save_npz_atomic(destination, **arrays)
                except (AssertionError, ValueError, RuntimeError, OSError, KeyError) as exc:
                    flagged += 1
                    print(f"demo={index} FLAG {type(exc).__name__}: {exc}", flush=True)
                    continue
                written += 1
                max_deviation = max(max_deviation, deviation)
                print(
                    f"demo={index} OK steps={len(arrays['priv_held'])} "
                    f"max_deviation={deviation:.9g} sidecar={destination}",
                    flush=True,
                )
        finally:
            env.close()
    print(
        f"Human labels: written={written} skipped_existing={skipped} "
        f"known_excluded={int(45 in indices)} flagged={flagged} "
        f"max_deviation_written={max_deviation:.9g}",
        flush=True,
    )
    return 1 if flagged else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demos-dir", type=Path, default=HUMAN_DIR)
    parser.add_argument(
        "--indices", nargs="+", default=["0:60"], help="Demo indices, e.g. 0:60 (45 excluded)"
    )
    parser.add_argument(
        "--out-dir", type=Path, help="Sidecar directory (default: next to the human NPZs)"
    )
    args = parser.parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        print("Interrupted; unfinished sidecar not published. Repeat to resume.", file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
