"""Rendered, privileged input-substitution diagnostics for the frozen J controller."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .estimated import EstimatedTaskState
from .observed_controller import LockedSiteContacts, StableSourcePlate
from .orientation_feedback import CalibratedGraspOrientation, PlacementOrientation
from .phases import Phase
from .run_privileged import development_seeds

SOURCE_PHASES = {Phase.APPROACH, Phase.PREGRASP, Phase.GRASP, Phase.LIFT}
PLACE_PHASES = {Phase.ALIGN, Phase.LOWER, Phase.RELEASE, Phase.SETTLE, Phase.RETREAT}
PLATE = ("plate_xyz", "plate_rel")
SITE = ("site_xyz", "site_rot", "target_site_rel", "target_site_rot_rel")
ORIENTATION = ("plate_rot", "plate_rot_rel", "tilt_deg")
GROUPS = {
    "truth": (),
    "source_xyz": PLATE,
    "source_xyz_robust": PLATE,
    "placement_xyz": PLATE,
    "destination": SITE,
    "orientation": ORIENTATION,
    "orientation_no_feedback": ORIENTATION,
    "orientation_robust": ORIENTATION,
    "orientation_grasp": ORIENTATION,
    "tilt_only": ORIENTATION,
    "held": ("held",),
    "contacts": ("contacts",),
    "destination_contacts": (*SITE, "contacts"),
    "destination_contacts_locked": (*SITE, "contacts"),
    "all": (),
    "all_site_fixed": (),
    "all_source_fixed": (),
    "all_source_site_fixed": (),
    "all_robust": (),
    "contacts_robust": ("contacts",),
}


def substitute(variant, phase, truth, estimate, locked_site, basis):
    if variant in {
        "all",
        "all_site_fixed",
        "all_source_fixed",
        "all_source_site_fixed",
        "all_robust",
    }:
        return estimate
    keys = GROUPS[variant]
    if variant in {"source_xyz", "source_xyz_robust"} and phase not in SOURCE_PHASES:
        keys = ()
    if variant == "placement_xyz" and phase in SOURCE_PHASES:
        keys = ()
    task = {**truth, **{key: estimate[key] for key in keys}}
    if variant == "destination_contacts_locked" and locked_site is not None:
        # Change only the destination reference used by the adapter's
        # target-contact predicate. Keep its visual plate position unchanged.
        delta = estimate["plate_xyz"] - locked_site
        task["contacts"] = {
            **estimate["contacts"],
            "target_rack": bool(np.linalg.norm(delta) < 0.052 and delta[2] <= -0.012),
        }
    if variant == "tilt_only":
        # Same unsigned tilt gate, with the adapter's pre-hold convention
        # throughout the rollout. This removes the hand-propagated rotation.
        angle = np.deg2rad(90 - estimate["tilt_deg"])
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
        task.update(plate_rot=rotation, plate_rot_rel=basis.T @ rotation)
    return task


def _first_failure(transitions, final_phase, success):
    if success:
        return None
    for edge in transitions:
        if edge["reason"] != "exit_check":
            return edge["from"]
    if final_phase == "HOLD":
        return "SETTLE/RETREAT"
    return final_phase


def _run_episode(env, policy, observer, seed, variant):
    from advaitic_eval.environment import observation, observed_source_slot, validate_actions

    from .kinematics import pelvis_rotation
    from .privileged import PrivilegedTaskState

    start = time.monotonic()
    ts = env.reset(seed=seed)
    source = PrivilegedTaskState(env)
    slot = observed_source_slot(env)
    policy.reset()
    observer.reset()
    contacts_filter = LockedSiteContacts() if variant == "contacts_robust" else None
    source_filter = (
        StableSourcePlate()
        if variant
        in {"source_xyz_robust", "all_source_fixed", "all_source_site_fixed", "all_robust"}
        else None
    )
    orientation_filter = (
        PlacementOrientation(disable_signed=variant == "orientation_no_feedback")
        if variant in {"orientation_no_feedback", "orientation_robust", "all_robust"}
        else None
    )
    if variant == "orientation_grasp":
        orientation_filter = CalibratedGraspOrientation(observer.kin)
    initial = source(ts.low_dim_obs)
    z0 = initial["plate_xyz"][2]
    lift = 0.0
    phase_steps = Counter()
    sample = {}
    errors = {
        "pickup_xyz_cm": [],
        "target_switch_cm": [],
        "normal_deg": [],
        "signed_roll_error_deg": [],
    }
    contact_disagreements = 0
    placement_samples = 0
    while True:
        obs = observation(ts)
        truth = source(obs["state"])
        phase = policy.phase
        if truth["held"]:
            lift = max(lift, float(truth["plate_xyz"][2] - z0))
        if ts.last():
            break
        estimate = observer.update(obs)
        if not phase_steps:
            sample["target_error_frame1_cm"] = (
                100 * (estimate["site_xyz"] - initial["site_xyz"])
            ).tolist()
            sample["target_estimate_frame1"] = estimate["site_xyz"].tolist()
            sample["target_truth_frame1"] = initial["site_xyz"].tolist()
        site_xyz = getattr(policy, "target_xyz", None) if policy.last_raw is not None else None
        basis = pelvis_rotation(obs["state"])
        task_estimate = estimate
        if source_filter is not None:
            task_estimate = source_filter.update(
                obs["state"], task_estimate, phase, policy.machine.elapsed, policy.closing
            )
        if orientation_filter is not None:
            task_estimate = orientation_filter.update(obs["state"], task_estimate, phase)
        if contacts_filter is not None:
            task_estimate = contacts_filter.update(obs["state"], estimate)
        task = substitute(variant, phase, truth, task_estimate, site_xyz, basis)
        phase_steps[phase.value] += 1
        if phase in SOURCE_PHASES and phase_steps[phase.value] % 5 == 1:
            errors["pickup_xyz_cm"].append(100 * (estimate["plate_xyz"] - truth["plate_xyz"]))
        if phase in PLACE_PHASES and phase_steps[phase.value] % 5 == 1:
            n_est, n_true = estimate["plate_rot"][:, 2], truth["plate_rot"][:, 2]
            errors["normal_deg"].append(np.rad2deg(np.arccos(np.clip(n_est @ n_true, -1, 1))))
            errors["signed_roll_error_deg"].append(
                np.rad2deg(np.arctan2(n_est[2], -n_est[1]) - np.arctan2(n_true[2], -n_true[1]))
            )
            locked_estimate = np.array(sample["target_estimate_frame1"])
            errors["target_switch_cm"].append(100 * (estimate["site_xyz"] - locked_estimate))
            dist = estimate["plate_xyz"] - locked_estimate
            locked_contact = np.linalg.norm(dist) < 0.052 and dist[2] <= -0.012
            contact_disagreements += int(locked_contact != estimate["contacts"]["target_rack"])
            placement_samples += 1
        previous = policy.phase
        action = validate_actions(policy.act(obs["state"], task))[0]
        if (
            previous == Phase.GRASP
            and policy.phase == Phase.LIFT
            and "prelift_xyz_error_cm" not in sample
        ):
            sample["prelift_xyz_error_cm"] = (
                100 * (estimate["plate_xyz"] - truth["plate_xyz"])
            ).tolist()
        ts = env.step(action)

    fell = bool(env.episode_fell())
    success = bool(env.episode_succeeded() and not fell)
    for key, series in errors.items():
        if series:
            values = np.asarray(series)
            sample[key] = {
                "mean": np.mean(values, axis=0).tolist(),
                "abs_p95": np.percentile(np.abs(values), 95, axis=0).tolist(),
            }
    sample.update(
        placement_contact_disagreements=contact_disagreements,
        placement_frames_sampled=placement_samples,
    )
    return {
        "seed": seed,
        "slot": slot,
        "variant": variant,
        "success": success,
        "m2": lift >= 0.05,
        "max_held_lift_m": lift,
        "steps": sum(phase_steps.values()),
        "fell": fell,
        "termination": env.episode_termination(),
        "first_failed_phase": _first_failure(policy.transitions, policy.phase.value, success),
        "phase_steps": dict(phase_steps),
        "transitions": policy.transitions,
        "prediction_diagnostics": sample,
        "seconds": time.monotonic() - start,
    }


def run(start, count, variants, checkpoint, out):
    import torch

    from advaitic_eval.environment import make_environment
    from submission.estimator import Estimator

    from .controller import ModularController
    from .kinematics import G1Kinematics
    from .observed_controller import LockedSiteController

    seeds = development_seeds(start, count)
    torch.set_num_threads(1)
    estimator = EstimatedTaskState(Estimator(checkpoint), G1Kinematics())
    controller = ModularController(G1Kinematics())
    fixed_controller = LockedSiteController(G1Kinematics())
    env = make_environment()
    try:
        with out.open("x", encoding="utf-8") as stream:
            for seed in seeds:
                for variant in variants:
                    selected = (
                        fixed_controller
                        if variant in {"all_site_fixed", "all_source_site_fixed", "all_robust"}
                        else controller
                    )
                    result = _run_episode(env, selected, estimator, seed, variant)
                    stream.write(json.dumps(result, allow_nan=False) + "\n")
                    stream.flush()
                    print(
                        seed, variant, result["success"], result["first_failed_phase"], flush=True
                    )
    finally:
        env.close()


def summarize(rows):
    baseline = {row["seed"]: row["success"] for row in rows if row["variant"] == "truth"}
    result = {}
    for variant in dict.fromkeys(row["variant"] for row in rows):
        group = [row for row in rows if row["variant"] == variant]
        lost = [row for row in group if baseline.get(row["seed"]) is True and not row["success"]]
        gained = [row for row in group if baseline.get(row["seed"]) is False and row["success"]]
        result[variant] = {
            "episodes": len(group),
            "successes": sum(row["success"] for row in group),
            "m2": sum(row["m2"] for row in group),
            "paired_baseline_successes_lost": len(lost),
            "paired_baseline_failures_rescued": len(gained),
            "lost_first_failed_phases": dict(Counter(row["first_failed_phase"] for row in lost)),
            "first_failed_phases": dict(
                Counter(row["first_failed_phase"] for row in group if not row["success"])
            ),
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=tuple(GROUPS), required=True)
    args = parser.parse_args()
    run(args.start, args.count, args.variants, args.checkpoint, args.out)
