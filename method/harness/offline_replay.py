"""Counterfactual release timing on fixed recordings, NOT a simulated policy rollout."""

import argparse
import json
from pathlib import Path

import numpy as np

from method.bc.dataset import episode_files, read_priv
from method.harness.data import LABEL_KEYS, audit_files
from method.harness.estimator import Estimator
from submission.harness import ReleaseSupervisor, config_from_env
from submission.policy import BCPolicy


def recorded_releases(executed, held):
    """Close-to-open edges after actual contact; exclude the initial open approach."""
    seen_held = np.maximum.accumulate(held.astype(bool))
    edges = (executed[1:, 18] < 0) & (executed[:-1, 18] > 0) & seen_held[1:]
    return (np.flatnonzero(edges) + 1).tolist()


def geometry(labels, step):
    if step is None:
        return None
    return {
        "step": int(step),
        "true_distance_cm": float(
            np.linalg.norm(labels["priv_plate_rel"][step] - labels["priv_target_site_rel"][step])
            * 100
        ),
        "true_tilt_deg": float(labels["priv_plate_tilt_deg"][step]),
        "true_held": bool(labels["priv_held"][step]),
    }


def replay(path, estimator, config, bc=None, max_steps=None):
    with np.load(path, allow_pickle=False) as episode:
        state, rgb = episode["state"], episode["rgb"]
        if int(episode["rgb_stride"]) != 2:
            raise ValueError(f"{path}: expected rgb_stride=2")
        executed = episode["action_executed"]
        labels, available = read_priv(path, episode, len(state))
    if not all(available[key] for key in LABEL_KEYS):
        raise ValueError(f"{path}: need true labels for diagnostics (label first)")
    supervisor = ReleaseSupervisor(config)
    if bc:
        bc.reset()
    n = min(len(state), max_steps or len(state))
    decision = None
    estimated_at_release = None
    for t in range(0, n, config["exec_steps"]):
        obs = {"rgb": rgb[t // 2], "state": state[t]}
        estimate = estimator(obs)
        # Recorded actions are merely proposals: the supervisor owns both
        # grippers. True geometry below never enters estimator/supervisor.
        proposed = bc(obs) if bc else executed[t : min(t + config["exec_steps"], n)]
        supervisor.step(estimate, obs["state"], proposed)
        if supervisor.release_time is not None:
            decision = t
            estimated_at_release = {
                "distance_cm": float(
                    np.linalg.norm(estimate["plate_rel"] - estimate["target_site_rel"]) * 100
                ),
                "tilt_deg": estimate["tilt_deg"],
                "held_probability": estimate["held"],
            }
            break  # release is latched forever; later recording cannot change it
    data_steps = [t for t in recorded_releases(executed, labels["priv_held"]) if t < n]
    return {
        "episode": str(path),
        "observed_prefix_steps": n,
        "proposal_source": "recomputed_bce1" if bc else "recorded_action_executed",
        "harness_would_release": geometry(labels, decision),
        "estimate_at_release": estimated_at_release,
        "data_releases": [geometry(labels, t) for t in data_steps],
        "timing_delta_first_steps": decision - data_steps[0]
        if decision is not None and data_steps
        else None,
        "counterfactual_only": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dirs", nargs="+", required=True)
    parser.add_argument("--estimator", required=True)
    parser.add_argument(
        "--bc-checkpoint", help="Optional: recompute bce1 proposals from observations"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-steps", type=int, help="Smoke only: inspect this prefix")
    parser.add_argument("-o", "--out", type=Path, help="New JSONL output file")
    args = parser.parse_args()
    if any(x is not None and x < 1 for x in (args.limit, args.max_steps)):
        parser.error("limit and max-steps must be positive")
    files = episode_files(args.data_dirs)
    if args.limit:
        files = files[: args.limit]
    audit_files(files)
    config = config_from_env()
    estimator = Estimator(args.estimator, args.device)
    bc = BCPolicy(args.bc_checkpoint) if args.bc_checkpoint else None
    if bc:
        bc.device = estimator.device
        bc.model.to(bc.device)
        if config["exec_steps"] > bc.horizon:
            raise ValueError("exec_steps exceeds BC horizon")
        bc.exec_steps = config["exec_steps"]
        bc.temporal_ensemble = config["temporal_ensemble"]
        bc.ensemble_decay = config["ensemble_decay"]
    stream = args.out.open("x") if args.out else None
    try:
        for path in files:
            record = {**replay(path, estimator, config, bc, args.max_steps), "config": config}
            line = json.dumps(record, allow_nan=False)
            print(line, flush=True)
            if stream:
                stream.write(line + "\n")
                stream.flush()
    finally:
        if stream:
            stream.close()


if __name__ == "__main__":
    main()
