"""Matched generation-only placement continuations from public pickup action prefixes."""

import json
import os
import time
from pathlib import Path

import modal

from ops_modular.modular_sim import image

app = modal.App("modular-astra-grasp-transfer")
volume = modal.Volume.from_name("modular-runs")


def continuation(seed, archive, calibration, fixed_site=None, observe_plate=None):
    import numpy as np
    from bigym.loco import make

    from method.modular import layout
    from method.modular.controller import ModularController
    from method.modular.kinematics import G1Kinematics, pelvis_rotation
    from method.modular.orientation_feedback import LEFT_GRASP_NORMAL
    from method.modular.phases import Phase
    from method.modular.privileged import PrivilegedTaskState
    from method.modular.refined import align_normal

    with np.load(Path(archive) / "episode_000.actions.npz") as f:
        actions = f["actions"]
    env = make("move_plate", camera_keys=())
    perception = None
    try:
        ts = env.reset(seed=seed)
        truth = PrivilegedTaskState(env)

        def observe(state):
            task = truth(state)
            if fixed_site is not None:
                site = np.asarray(fixed_site)
                delta = task["plate_xyz"] - site
                task.update(
                    site_xyz=site,
                    target_site_rel=pelvis_rotation(state).T @ (site - state[18:21]),
                    site_distance=float(np.linalg.norm(delta)),
                    site_xy_distance=float(np.linalg.norm(delta[:2])),
                )
            return task

        kin = G1Kinematics()
        ctl = ModularController(kin)
        source = observe(ts.low_dim_obs)
        ctl._initialize(ts.low_dim_obs, source)
        z0 = source["plate_xyz"][2]
        consecutive = 0
        for i, action in enumerate(actions):
            ts = env.step(action)
            state = ts.low_dim_obs
            task = observe(state)
            consecutive = (
                consecutive + 1 if task["held"] and task["plate_xyz"][2] > z0 + 0.075 else 0
            )
            if consecutive >= 12:
                break
            if ts.last():
                return dict(
                    secure=False, success=False, fell=bool(env.episode_fell()), prefix=i + 1
                )
        if consecutive < 12:
            raise ValueError("Action archive stopped before terminal or secure lift")
        # All three-dimensional translation, site and contact fields now come
        # from labels. Both arms start from precisely the same physical grasp
        # and inherited action, with the nominal rigid transform for transit.
        ctl.last_raw = layout.denormalize_action(action)
        kin.set_state(state)
        p, r = kin.hand_pose("left")
        basis = pelvis_rotation(state)
        hand = state[18:21] + basis @ p
        hand_rot = basis @ r
        nominal = align_normal(task["plate_rot"], hand_rot @ LEFT_GRASP_NORMAL)
        ctl.machine.step = i + 1
        ctl.machine.phase = Phase.LIFT
        ctl._enter({**task, "plate_rot": nominal}, hand, hand_rot)
        ctl.machine.phase = Phase.CARRY
        if observe_plate is not None:
            perception = observe_plate(env)
            task = perception(state, task, Phase.CARRY)
        ctl._enter(task, hand, hand_rot)
        n = np.asarray(calibration) if calibration is not None else LEFT_GRASP_NORMAL
        n = n / np.linalg.norm(n)
        error = float(
            np.rad2deg(np.arccos(np.clip((hand_rot @ n) @ task["plate_rot"][:, 2], -1, 1)))
        )
        while not ts.last():
            state = ts.low_dim_obs
            task = observe(state)
            if perception is not None and ctl.phase != Phase.HOLD:
                task = perception(state, task, ctl.phase)
            if ctl.phase in (Phase.CARRY, Phase.ALIGN, Phase.LOWER):
                kin.set_state(state)
                basis = pelvis_rotation(state)
                normal = basis @ kin.hand_pose("left")[1] @ n
                rot = align_normal(task["plate_rot"], normal)
                tilt = float(np.rad2deg(np.arccos(np.clip(normal @ ctl.target_rot[:, 2], -1, 1))))
                task = {**task, "plate_rot": rot, "plate_rot_rel": basis.T @ rot, "tilt_deg": tilt}
            ts = env.step(ctl.act(state, task))
        return dict(
            secure=True,
            prefix=i + 1,
            normal_error_deg=error,
            success=bool(env.episode_succeeded() and not env.episode_fell()),
            fell=bool(env.episode_fell()),
            termination=env.episode_termination(),
            transitions=ctl.transitions,
            **({"position_errors": perception.stats()} if perception is not None else {}),
        )
    finally:
        if perception is not None:
            perception.close()
        env.close()


@app.function(
    image=image,
    cpu=1,
    memory=1536,
    volumes={"/runs": volume},
    timeout=1200,
    max_containers=2,
    single_use_containers=True,
)
def compare(seed, archive, vector, name):
    import inspect
    import subprocess

    if not (
        3000700 <= seed < 3000850
        and Path(archive).parent.parent == Path("/runs/grasp-validation")
        and Path(archive).name == "eval"
    ):
        raise ValueError("Use a fresh full public-pickup archive")
    if name not in ("v1", "signed"):
        raise ValueError(name)
    volume.reload()
    report_path = Path(archive) / "report.json"
    for _ in range(5):
        if report_path.is_file():
            break
        time.sleep(8)
        volume.reload()
    report = json.loads(report_path.read_text())
    ep = report["episodes"][0]
    if (
        not report["integrity_ok"]
        or ep["seed"] != seed
        or ep["policy_error"]
        or not ep["completed"]
    ):
        raise ValueError("Invalid source report")
    folder = Path("/runs/astra-integration/grasp-transfer") / name
    folder.mkdir(exist_ok=True, parents=True)
    f = folder / f"{seed}.json"
    if f.exists():
        r = json.loads(f.read_text())
        if r["vector"] != vector or r["archive"] != archive:
            raise ValueError("Conflicting diagnostic")
        return r
    src = (
        "import json\nfrom pathlib import Path\n"
        + inspect.getsource(continuation)
        + f"\nprint(json.dumps([continuation({seed},{archive!r},n) for n in (None,{vector!r})]))"
    )
    t = time.monotonic()
    p = subprocess.run(
        ["/root/policy/bin/python", "-c", src],
        cwd="/root/code",
        env={**os.environ, "MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"},
        capture_output=True,
        text=True,
    )
    if p.returncode:
        raise ValueError(p.stderr[-1600:])
    a, b = json.loads(p.stdout.splitlines()[-1])
    if (a["prefix"], a["secure"]) != (b["prefix"], b["secure"]):
        raise ValueError("Mismatched physical handoff")
    r = dict(
        seed=seed,
        archive=archive,
        vector=vector,
        prior=a,
        candidate=b,
        seconds=time.monotonic() - t,
    )
    f.write_text(json.dumps(r))
    volume.commit()
    return r


@app.function(cpu=0.125, memory=256, volumes={"/runs": volume}, timeout=86400)
def schedule(name, inputs, vector, smoke_id):
    smoke = modal.FunctionCall.from_id(smoke_id).get()
    if not smoke["prior"]["secure"] or smoke["vector"] != vector:
        raise ValueError("Unusable paired handoff smoke")
    if len(inputs) != 30 or len(set(s for s, _ in inputs)) != 30:
        raise ValueError("Expected full paired generation plan")
    folder = Path("/runs/astra-integration/grasp-transfer") / name
    ids = {smoke["seed"]: smoke_id}
    volume.reload()
    if (folder / "calls.json").exists():
        raise ValueError("Existing study")
    while len(ids) < len(inputs):
        for s, c in inputs:
            if s in ids:
                continue
            try:
                rec = modal.FunctionCall.from_id(c).get(timeout=0)
            except (TimeoutError, modal.exception.ConnectionError):
                continue
            if rec["seed"] != s or rec["variant"] != "obs_pickup":
                raise ValueError("Unexpected source")
            ids[s] = compare.spawn(s, rec["archive"], vector, name).object_id
        (folder / "calls.json").write_text(json.dumps(ids))
        volume.commit()
        if len(ids) < len(inputs):
            time.sleep(45)
    rows = [modal.FunctionCall.from_id(i).get() for i in ids.values()]
    summary = dict(
        prior=sum(z["prior"]["success"] for z in rows),
        candidate=sum(z["candidate"]["success"] for z in rows),
        gains=sum(z["candidate"]["success"] and not z["prior"]["success"] for z in rows),
        losses=sum(z["prior"]["success"] and not z["candidate"]["success"] for z in rows),
        secure=sum(z["prior"]["secure"] for z in rows),
        complete=len(rows),
    )
    (folder / "summary.json").write_text(json.dumps(summary))
    volume.commit()
    return summary
