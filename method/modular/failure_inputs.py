"""Attribute observations at the first failed guard on a recorded trajectory.

This is a local counterfactual on the recorded states. Only a separate rollout
with an input substituted can establish whether correcting it changes success.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .controller import GRASP_OFFSET, PREGRASP_OFFSET, SOURCE_STANDOFF_XY


def grasp_target(task, retry, approach):
    normal = task[:, 6:9]
    x = np.broadcast_to([1.0, 0.0, 0.0], normal.shape) - normal[:, :1] * normal
    x /= np.linalg.norm(x, axis=1)[:, None]
    offset = GRASP_OFFSET.copy()
    offset[0] += (0, 0.009, -0.009)[min(2, retry)]
    offset += PREGRASP_OFFSET if approach else 0
    return task[:, :3] + offset[0] * x + offset[1] * np.cross(normal, x) + offset[2] * normal


def release_gates(task, anchor, elapsed):
    error = task[:, :3] - anchor
    return {
        "held": task[:, 10].astype(bool),
        "target contact": task[:, 12].astype(bool),
        "table/floor contact": ~np.any(task[:, 13:15], axis=1),
        "plate z": (error[:, 2] >= -0.047) & (error[:, 2] <= -0.018),
        "tilt": task[:, 9] <= 12,
        "plate xy": ((np.abs(error[:, 0]) < 0.027) & (np.abs(error[:, 1]) < 0.022))
        | ((elapsed >= 120) & (np.linalg.norm(error, axis=1) < 0.039)),
    }


def diagnose(f):
    info = json.loads(str(f["info"]))
    events = info["transitions"]
    abnormal = [e for e in events if e["reason"] != "exit_check"]
    first = (
        abnormal[0]
        if abnormal
        else {
            "from": str(f["phase"][-1]),
            "step": info["shadow_steps"] - 1,
            "reason": "termination",
        }
    )
    phase, end = first["from"], first["step"]
    starts = [e["step"] + 1 for e in events if e["step"] < end]
    mask = (f["phase"] == phase) & (f["step"] <= end) & (f["step"] >= max(starts, default=0))
    if not np.any(mask):
        raise ValueError(f"No frames for {first}")
    p, t, hand, state, steps = (f[k][mask] for k in ("pred", "truth", "hand", "state", "step"))
    anchor = np.asarray(info["first_site"])

    def distance(value):
        return np.linalg.norm(value, axis=1)

    above = float(np.percentile(distance(p[:, :3] - t[:, :3]), 95) * 100)
    inputs = {}
    if phase == "LOWER":
        ep = release_gates(p, anchor, steps - max(starts, default=0) + 1)
        gt = release_gates(t, anchor, steps - max(starts, default=0) + 1)
        for k in ep:
            # Number of frames at which that one condition changes the gate.
            others = np.logical_and.reduce([v for name, v in ep.items() if name != k])
            inputs[k] = int(np.sum(others & (ep[k] != gt[k])))
        evidence = ", ".join(f"{k} {n}" for k, n in inputs.items() if n)
        label = (
            max(inputs, key=inputs.get)
            if max(inputs.values())
            else "no single condition flips release"
        )
        disagreement = {k: int(np.sum(ep[k] != gt[k])) for k in ep}
        if not max(inputs.values()) and max(disagreement.values()):
            label = max(disagreement, key=disagreement.get) + " (coupled gates)"
        details = (
            f"Single-input release-gate flips (frames): {evidence or '0'}. "
            f"Total condition disagreements {disagreement}; estimated/true rack frames "
            f"{int(p[:, 12].sum())}/{int(t[:, 12].sum())}."
        )
    elif phase in ("GRASP", "PREGRASP"):
        pp = grasp_target(p, 0, phase == "PREGRASP")
        tp = grasp_target(t, 0, phase == "PREGRASP")
        d1, d2 = distance(pp - hand[:, :3]), distance(tp - hand[:, :3])
        held = int(np.sum(p[:, 10] != t[:, 10]))
        closest = int(np.argmin(d1))
        target_error = 100 * distance(pp - tp)[closest]
        label = (
            "held"
            if held >= 4 and t[:, 10].any()
            else (
                "grasp target geometry" if target_error >= 1.0 else "unresolved grasp/arm tracking"
            )
        )
        details = f"At nearest predicted reach, hand error estimated/true {100 * d1[closest]:.2f}/{100 * d2[closest]:.2f} cm; target displacement {target_error:.2f} cm; held mismatch {held}/{len(p)}"
    elif phase == "APPROACH":
        pp = distance(p[:, :2] - SOURCE_STANDOFF_XY - state[:, 18:20])
        tt = distance(t[:, :2] - SOURCE_STANDOFF_XY - state[:, 18:20])
        flips = (pp < 0.023) != (tt < 0.023)
        label = "source plate xy" if np.any(flips) else "unresolved approach/gait"
        details = f"Standoff distance gate disagreed {flips.sum()}/{len(p)} frames"
    elif phase == "ALIGN":
        target = anchor + [0, 0, 0.09]
        pos_p = distance(p[:, :3] - target) < 0.023
        pos_t = distance(t[:, :3] - target) < 0.023
        tilt_p = p[:, 9] < 14
        tilt_t = t[:, 9] < 14
        inputs = {
            "plate xyz": int(np.sum((pos_p != pos_t) & tilt_p)),
            "tilt": int(np.sum((tilt_p != tilt_t) & pos_p)),
        }
        label = max(inputs, key=inputs.get) if max(inputs.values()) else "unresolved alignment"
        details = f"Single-input alignment gate flips {inputs}"
    elif phase in ("SETTLE", "RETREAT"):
        enter = int(np.searchsorted(f["step"], max(starts, default=0), side="left"))
        reference = max(enter - 1, 0)

        def conditions(q, origin):
            if phase == "SETTLE":
                return {
                    "target contact": q[:, 12] > 0,
                    "plate xyz": distance(q[:, :3] - anchor) < 0.047,
                    "tilt": q[:, 9] < 17,
                }
            return {
                "target contact": q[:, 12] > 0,
                "plate displacement": distance(q[:, :3] - origin[:3]) <= 0.012,
                "tilt": q[:, 9] <= min(19, origin[9] + 4),
            }

        ep = conditions(p, f["pred"][reference])
        gt = conditions(t, f["truth"][reference])
        disagreement = {k: int(np.sum(ep[k] != gt[k])) for k in ep}
        label = (
            max(disagreement, key=disagreement.get)
            if max(disagreement.values())
            else "no seating-gate disagreement"
        )
        details = f"Seating condition disagreements {disagreement} over {len(p)} frames"

    else:
        held = int((p[:, 10] != t[:, 10]).sum())
        source = int((p[:, 11] != t[:, 11]).sum())
        label = "held" if held else "source contact" if phase == "LIFT" and source else "unresolved"
        details = f"Held disagreed {held}/{len(p)}, source contact {source}/{len(p)}"
    # A nominal phase exit can be premature before the first timeout or retry.
    # Restrict these checks to handoff gates directly observable in the trace.
    handoffs = []
    for event in events:
        if event["reason"] != "exit_check":
            continue
        i = int(np.searchsorted(f["step"], event["step"], side="right") - 1)
        if i < 0 or f["phase"][i] != event["from"]:
            continue
        if event["from"] == "GRASP" and f["pred"][i, 10] and not f["truth"][i, 10]:
            handoffs.append(("GRASP", event["step"], "false held"))
        elif event["from"] == "LOWER" and f["gate"][i, 0] and not f["gate"][i, 1]:
            checks = release_gates(
                f["truth"][i : i + 1],
                anchor,
                np.array(
                    [
                        event["step"]
                        - max([e["step"] for e in events if e["step"] < event["step"]], default=0)
                    ]
                ),
            )
            handoffs.append(
                (
                    "LOWER",
                    event["step"],
                    "release with incorrect " + "/".join(k for k, v in checks.items() if not v[0]),
                )
            )
        elif event["from"] == "SETTLE" and f["truth"][i, 10] and not f["pred"][i, 10]:
            handoffs.append(("SETTLE", event["step"], "plate still held with open gripper"))
    if (
        phase in ("SETTLE", "RETREAT")
        and (t[:, 10].astype(bool) & ~p[:, 10].astype(bool)).mean() > 0.5
    ):
        # Retained contact with the gripper can also prevent official task success.
        details += f"; open/held disagreement {int(np.sum(t[:, 10] != p[:, 10]))}/{len(p)}"
    if info["max_action_delta"] > 1e-4 or info["max_state_delta"] > 1e-5:
        label = "shadow policy mismatch; " + label
    closure = np.flatnonzero((f["phase"] == "GRASP") & (f["state"][:, 44] > 0.05))
    preclose = max(0, closure[0] - 1) if len(closure) else None
    lower = f["phase"] == "LOWER"
    lean = np.degrees(
        np.arctan2(f["pred"][lower, 8], -f["pred"][lower, 7])
        - np.arctan2(f["truth"][lower, 8], -f["truth"][lower, 7])
    )
    return dict(
        first_false_handoff=handoffs[0] if handoffs else None,
        preclose_step=int(f["step"][preclose]) if preclose is not None else None,
        preclose_plate_error_xyz_cm=np.round(
            100 * (f["pred"][preclose, :3] - f["truth"][preclose, :3]), 2
        ).tolist()
        if preclose is not None
        else None,
        lower_signed_lean_bias_deg=round(float(np.mean((lean + 180) % 360 - 180)), 2)
        if lean.size
        else None,
        seed=info["seed"],
        phase=phase,
        step=end,
        reason=first["reason"],
        input=label,
        evidence=details,
        phase_plate_p95_cm=round(above, 3),
        initial_site_cm=round(100 * np.linalg.norm(anchor - f["truth"][0, 3:6]), 3),
        normal_mae_deg=round(
            np.degrees(np.arccos(np.clip((p[:, 6:9] * t[:, 6:9]).sum(1), -1, 1))).mean(), 3
        ),
        max_action_delta=info["max_action_delta"],
        max_state_delta=info["max_state_delta"],
        last_event=events[-1] if events else None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--glob", default="**/*.npz")
    parser.add_argument("--physics", type=Path)
    args = parser.parse_args()
    reports = []
    for file in sorted(args.directory.glob(args.glob)):
        with np.load(file) as record:
            if "info" in record and not json.loads(str(record["info"]))["outcome"]["success"]:
                reports.append(diagnose(record))
    if args.physics:
        archived = {r["seed"]: r for r in map(json.loads, args.physics.read_text().splitlines())}
        if len(reports) != len(archived) or {r["seed"] for r in reports} != set(archived):
            raise ValueError("A failure is missing or was processed twice")
        for row in reports:
            physical = archived[row["seed"]]
            row.update(
                block=physical["block"],
                episode=physical["episode"],
                slot=physical["source_slot"] + 1,
                physical_peak_lift_cm=physical["peak_lift_cm"],
                rack_contact_frames=physical["target_contact_frames"],
            )
    reports.sort(key=lambda r: r["seed"])
    if reports:
        with args.csv.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(reports[0]))
            writer.writeheader()
            writer.writerows(reports)
