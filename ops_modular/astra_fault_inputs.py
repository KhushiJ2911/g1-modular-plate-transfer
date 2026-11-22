"""Measure stateless public inputs around camera-free verified dev failure events."""

import inspect
import json
import os
import re
import subprocess
import time
from pathlib import Path

import modal

from ops_modular.modular_sim import image, volume

app = modal.App("modular-astra-fault-inputs")
records = modal.Volume.from_name("modular-runs")


def diagnose(folder, event, checkpoint):
    import mujoco
    import numpy as np
    from bigym.loco import make
    from bigym.loco.config import DEFAULT_CAMERA_KEYS

    from method.modular.kinematics import G1Kinematics, pelvis_rotation
    from method.modular.orientation_feedback import LEFT_GRASP_NORMAL
    from method.modular.privileged import PrivilegedTaskState
    from submission.estimator import Estimator

    seed = event["seed"]
    with np.load(Path(folder) / "episode_000.actions.npz") as f:
        actions = f["actions"]
    with np.load(checkpoint, allow_pickle=False) as w:
        n = w["loaded_normal"] if "loaded_normal" in w else LEFT_GRASP_NORMAL
    est = Estimator("/runs/est_v1b/last.pt", device="cpu")
    env = make("move_plate", camera_keys=())
    kin = G1Kinematics()
    renderer = None
    initial = []
    points = []
    steps = {i: "initial" for i in range(6)}
    near = event.get("target_approach_step")
    release = event.get("open_at_target")
    if near is not None:
        steps.update({near + delta: "arriving" for delta in (0, 30, 60)})
    if release:
        opening = release["step"]
        steps.update({opening + d: f"open{d:+d}" for d in (-6, -2, 0, 4, 12, 32)})
    held = event.get("first_held")
    if held is not None:
        steps.update({held + d: "pickup" for d in (0, 15, 45)})
    last_seat = -100
    first_seat = None
    try:
        ts = env.reset(seed=seed)
        inner = env.inner_env
        src = PrivilegedTaskState(env)
        cams = [
            next(
                i for i in range(inner.model.ncam) if inner.model.camera(i).name.split("/")[-1] == k
            )
            for k in DEFAULT_CAMERA_KEYS
        ]
        renderer = mujoco.Renderer(inner.model, height=84, width=84)
        for i in range(len(actions) + 1):
            state = ts.low_dim_obs
            true = src(state)
            if (
                near is not None
                and i >= near
                and true["contacts"]["target_rack"]
                and true["held"]
                and true["site_xy_distance"] < 0.05
                and (not release or i < release["step"])
            ):
                if first_seat is None:
                    first_seat = i
                if i - last_seat >= 16 and i - first_seat <= 144:
                    steps.setdefault(i, "supported")
                    last_seat = i
            if i in steps:
                views = []
                for cam in cams:
                    renderer.update_scene(inner.data, camera=cam)
                    views.append(renderer.render().transpose(2, 0, 1))
                raw = est({"rgb": np.stack(views), "state": state})
                basis = pelvis_rotation(state)
                origin = state[18:21]
                site = origin + basis @ raw["target_site_rel"]
                plate = origin + basis @ raw["plate_rel"]
                if i < 6:
                    initial.append(site.copy())
                lock = np.median(initial, axis=0)
                kin.set_state(state)
                hand_normal = basis @ kin.hand_pose("left")[1] @ n
                fk_tilt = np.rad2deg(
                    np.arccos(np.clip(hand_normal @ true["site_rot"][:, 2], -1, 1))
                )
                angle = np.rad2deg(np.arccos(np.clip(hand_normal @ true["plate_rot"][:, 2], -1, 1)))
                points.append(
                    dict(
                        step=i,
                        event=steps[i],
                        held=true["held"],
                        visual_held=float(raw["held"]),
                        target_contact=true["contacts"]["target_rack"],
                        true_tilt=true["tilt_deg"],
                        fk_tilt=float(fk_tilt),
                        raw_visual_tilt=float(raw["tilt_deg"]),
                        fk_normal_error_deg=float(angle),
                        plate_error_cm=(100 * (plate - true["plate_xyz"])).tolist(),
                        site_lock_error_cm=(100 * (lock - true["site_xyz"])).tolist(),
                        observed_gap_cm=(100 * (plate - lock)).tolist(),
                        true_gap_cm=(100 * (true["plate_xyz"] - true["site_xyz"])).tolist(),
                    )
                )
            if i < len(actions):
                ts = env.step(actions[i])
        ep = json.loads((Path(folder) / "report.json").read_text())["episodes"][0]
        if (bool(env.episode_succeeded()), bool(env.episode_fell()), env.episode_termination()) != (
            ep["success"],
            ep["fell"],
            ep["termination"],
        ):
            raise ValueError("Camera-free physics did not reproduce the official terminal result")
    finally:
        if renderer is not None:
            renderer.close()
        env.close()
    return dict(seed=seed, bucket=event["bucket"], points=points)


@app.function(
    image=image,
    cpu=1,
    memory=2304,
    timeout=1200,
    max_containers=2,
    single_use_containers=True,
    volumes={"/runs": volume, "/records": records},
)
def audit(name, folder, event, checkpoint):
    import hashlib

    if (
        not re.fullmatch(r"astra_[a-z0-9_]+", name)
        or not re.fullmatch(re.escape(name) + r"-10[0-9]{5}(-r[0-9]+)?", folder)
        or not 1000000 <= event["seed"] < 1030000
        or not (
            folder == f"{name}-{event['seed']}" or folder.startswith(f"{name}-{event['seed']}-r")
        )
    ):
        raise ValueError("Expected a completed sharded dev failure")
    if Path(checkpoint).parent != Path("/runs/heads"):
        raise ValueError("Expected a development checkpoint")
    from ops_modular.modular_sim import WEIGHTS_SHA

    if hashlib.sha256(Path("/runs/est_v1b/last.pt").read_bytes()).hexdigest() != WEIGHTS_SHA:
        raise ValueError("Unexpected base estimator")
    dest = (
        Path("/records/astra-integration") / name / "fault_inputs" / (str(event["seed"]) + ".json")
    )
    if dest.exists():
        return json.loads(dest.read_text())
    dest.parent.mkdir(exist_ok=True, parents=True)
    t = time.monotonic()
    args = repr((f"/runs/modular/{folder}/eval", event, checkpoint))
    src = "import json\nfrom pathlib import Path\n" + inspect.getsource(diagnose)
    p = subprocess.run(
        ["/root/policy/bin/python", "-c", src + f"\nprint(json.dumps(diagnose(*{args})))"],
        cwd="/root/code",
        env={
            **os.environ,
            "MUJOCO_GL": "osmesa",
            "PYOPENGL_PLATFORM": "osmesa",
            "LP_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        },
        capture_output=True,
        text=True,
    )
    if p.returncode:
        raise RuntimeError(p.stderr[-1900:])
    result = json.loads(p.stdout.splitlines()[-1])
    result["seconds"] = time.monotonic() - t
    dest.write_text(json.dumps(result))
    records.commit()
    return result


@app.function(cpu=0.125, memory=256, timeout=86400, volumes={"/records": records})
def monitor(name, checkpoint, smoke_name, smoke_id):
    first = modal.FunctionCall.from_id(smoke_id).get()
    if not first["points"] or first["points"][0]["event"] != "initial":
        raise ValueError("Camera or policy-input smoke produced no usable first frame")
    if not re.fullmatch(r"astra_[a-z0-9_]+", name):
        raise ValueError("Expected a dev round")
    folder = Path("/records/astra-integration") / name
    ids = {first["seed"]: smoke_id} if name == smoke_name else {}
    while True:
        records.reload()
        summary_file = folder / "summary.json"
        event_file = folder / "replays.json"
        if summary_file.exists() and event_file.exists():
            summary = json.loads(summary_file.read_text())
            events = json.loads(event_file.read_text())
            journal = folder / "fault_calls.json"
            if journal.exists():
                ids.update({int(k): v for k, v in json.loads(journal.read_text()).items()})
            failures = [r for r in summary["rows"] if not r["success"] and not r.get("policy_error")]
            for row in failures:
                seed = row["seed"]
                event = events.get(str(seed))
                if seed not in ids and event and "bucket" in event:
                    ids[seed] = audit.spawn(name, row["folder"], event, checkpoint).object_id
                    journal.write_text(json.dumps(ids))
                    records.commit()
            if summary["decision"] != "pending" and all(row["seed"] in ids for row in failures):
                break
        time.sleep(40)
    for cid in ids.values():
        modal.FunctionCall.from_id(cid).get()
    return len(ids)
