"""Camera-free diagnostic of source pose, loaded grasp and the first destination opening."""

import json
import runpy
import sys
import tempfile
from pathlib import Path

import numpy as np


def analyze(folder):
    import bigym.loco

    sys.path.insert(0, "/home/keyu/advaitic/wt-teacher")
    from method.teacher.diagnostics import PrivilegedSensors, plate_up

    folder = Path(folder)
    report = json.loads((folder / "report.json").read_text())
    ep = report["episodes"][0]
    if not report["integrity_ok"] or ep["policy_error"] or not ep["completed"]:
        raise ValueError("Invalid official episode")
    with np.load(folder / "episode_000.actions.npz") as f:
        commands = f["actions"][:, 18]
    make = bigym.loco.make
    snapshot = PrivilegedSensors.snapshot
    envs, events = [], {}
    last_held = False
    opened = None
    previous = -1

    def capture(*args, **kwargs):
        env = make(*args, **kwargs)
        envs.append(env)
        return env

    def sample(sensors, step):
        nonlocal last_held, opened, previous
        s = snapshot(sensors, step)
        if step == previous:
            return s
        previous = step
        xyz = np.array(s["pose"][:3])
        names = []
        if step == 0:
            names.append("source")
        if step == 40:
            names.append("settled_source")
        if s["held"] and "lift" not in events and xyz[2] > events["source"]["xyz"][2] + 0.075:
            names.append("lift")
        if "lift" in events and s["held"] and s["site_xy_distance"] < 0.06:
            if "at_target" not in events:
                names.append("at_target")
            if s["contacts"]["target_rack"] and "support" not in events:
                names.append("support")
        if "at_target" in events and opened is None and step < len(commands) and commands[step] < 0:
            opened = step
            names.append("open")
        if opened is not None:
            if step > opened and last_held and not s["held"] and "released" not in events:
                names.append("released")
            if not last_held and s["held"] and "released" in events and "recontact" not in events:
                names.append("recontact")
            if step - opened in (5, 15, 30):
                names.append(f"open_plus_{step - opened}")
        if step == len(commands):
            names.append("end")
        last_held = s["held"]
        if names:
            inner = sensors.inner
            side = next(x for x in inner.robot.grippers if x.name == "LEFT")
            hand = inner.robot.get_hand_pos(side)
            wrist_body = next(
                b
                for b in range(inner.model.nbody)
                if inner.model.body(b).name.endswith("left_wrist_yaw_link")
            )
            rot = inner.data.xmat[wrist_body].reshape(3, 3)
            normal = plate_up(s["pose"][3:])
            record = dict(
                step=step,
                xyz=xyz.tolist(),
                site=s["site_xyz"],
                target_index=s["nearest_site"],
                normal=normal.tolist(),
                tilt=s["tilt_deg"],
                held=s["held"],
                rack=s["contacts"]["target_rack"],
                hand=hand.tolist(),
                hand_rotation=rot.tolist(),
                center_in_hand=(rot.T @ (xyz - hand)).tolist(),
                normal_in_hand=(rot.T @ normal).tolist(),
            )
            for name in names:
                events[name] = record
        return s

    bigym.loco.make = capture
    PrivilegedSensors.snapshot = sample
    try:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "replay.jsonl"
            sys.argv = ["/root/analysis/replay_eval.py", str(folder), str(output)]
            runpy.run_path(sys.argv[0], run_name="__main__")
            base = json.loads(output.read_text())
        env = envs[0]
        if (
            previous,
            bool(env.episode_succeeded()),
            bool(env.episode_fell()),
            str(env.episode_termination()),
        ) != (ep["steps"], ep["success"], ep["fell"], ep["termination"]):
            raise ValueError("Terminal replay mismatch")
        return dict(
            folder=folder.parent.name,
            seed=ep["seed"],
            slot=ep["source_slot"],
            success=ep["success"],
            steps=ep["steps"],
            events=events,
            base=base,
        )
    finally:
        PrivilegedSensors.snapshot = snapshot
        bigym.loco.make = make
        for env in envs:
            env.close()


def compare_pair(base, clearance):
    """Validate the common approach and measure separation on both public action logs."""
    rows = [analyze(folder) for folder in (base, clearance)]
    if rows[0]["seed"] != rows[1]["seed"] or rows[0]["slot"] != rows[1]["slot"]:
        raise ValueError("Unpaired reports")
    actions = []
    for folder in (base, clearance):
        with np.load(Path(folder) / "episode_000.actions.npz") as f:
            actions.append(f["actions"])
    n = min(map(len, actions))
    difference = np.max(np.abs(actions[0][:n] - actions[1][:n]), axis=1)
    changed = np.flatnonzero(difference > 1e-5)
    first = int(changed[0]) if len(changed) else None
    opened = [r["events"].get("open", {}).get("step") for r in rows]
    end = min([n, *(s + 1 for s in opened if s is not None)])
    reports = [
        json.loads((Path(folder) / "report.json").read_text()) for folder in (base, clearance)
    ]
    return dict(
        seed=rows[0]["seed"],
        first_changed=first,
        max_action_delta_through_open=float(np.max(difference[:end])),
        openings=opened,
        prefix_match=opened[0] == opened[1] and not np.any(difference[:end] > 1e-5),
        action_delta_first=(actions[1][first] - actions[0][first]).tolist()
        if first is not None
        else None,
        provenance=[
            {
                k: v
                for k, v in r.items()
                if k in ("submission_sha256", "substrate", "python", "policy")
            }
            for r in reports
        ],
        base=rows[0],
        clearance=rows[1],
    )


if __name__ == "__main__":
    paths = sys.argv[1:]
    print(json.dumps(compare_pair(*paths) if len(paths) == 2 else analyze(*paths)))
