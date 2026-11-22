"""Change the leveling pivot on matched, physically reached generation prefixes."""

import json
import os
import time
from pathlib import Path

import modal

from ops_modular.modular_sim import image

app = modal.App("modular-astra-pivot")
data = modal.Volume.from_name("modular-data")
if modal.is_local():
    image = image.add_local_file(Path(__file__), "/root/code/pivot_audit.py")


def trial(seed, mode):
    import numpy as np
    from bigym.loco import make

    from method.modular import layout
    from method.modular.controller import _rotation_from_vector
    from method.modular.kinematics import G1Kinematics, pelvis_rotation
    from method.modular.privileged import PrivilegedTaskState

    with np.load(f"/data/astra-dagger/sol/{seed}.npz") as f:
        actions, steps, phase = f["actions"], f["step"], f["phase"]
        meta = json.loads(str(f["meta"]))
        lower = steps[phase == "LOWER"]
        record = dict(seed=seed, mode=mode, baseline=meta["success"], opened=False)
        if not len(lower):
            return record
        post = np.flatnonzero(actions[lower[0] :, 18] < 0)
        if not len(post):
            return record
        opening = lower[0] + post[0]
        idx = np.flatnonzero(steps <= opening)[-1]
        check_step, check_state = steps[idx], f["state"][idx]
    env = make("move_plate", camera_keys=())
    kin = G1Kinematics()
    try:
        ts = env.reset(seed=seed)
        labels = PrivilegedTaskState(env)
        for t, a in enumerate(actions[:opening]):
            if t == check_step:
                np.testing.assert_array_equal(ts.low_dim_obs, check_state)
            ts = env.step(a)
        state = ts.low_dim_obs
        if opening == check_step:
            np.testing.assert_array_equal(state, check_state)
        task = labels(state)
        if ts.last():
            raise ValueError("Replay ended before first release")
        record.update(
            opened=True,
            step=int(opening),
            held=bool(task["held"]),
            support=bool(task["contacts"]["target_rack"]),
            initial_tilt=task["tilt_deg"],
        )
        kin.set_state(state)
        p, r = kin.hand_pose("left")
        basis = pelvis_rotation(state)
        raw = layout.denormalize_action(actions[opening]).copy()
        normal = task["plate_rot_rel"][:, 2]
        target = task["target_site_rot_rel"][:, 2]
        center = task["plate_rel"].copy()
        pivot = p.copy()
        if mode in ("center", "bottom"):
            pivot = center.copy()
        if mode == "bottom":
            down = basis.T @ np.array([0.0, 0.0, -1.0])
            radial = down - np.dot(normal, down) * normal
            pivot = pivot + 0.114 * radial / max(np.linalg.norm(radial), 1e-8)
        corr = np.cross(normal, target)
        adjust = _rotation_from_vector(corr * min(1.0, 0.22 / max(np.linalg.norm(corr), 1e-8)))
        desired = pivot + adjust @ (p - pivot)
        goal = kin.ik(
            "left",
            desired,
            adjust @ r,
            q_init=raw[layout.LEFT_ARM_ACTION],
            rot_weight=0.22,
            iters=30,
        )
        raw[layout.GRIPPER_ACTION] = [1.0, 0.0]
        for _ in range(22):
            raw[layout.LEFT_ARM_ACTION] += np.clip(goal - raw[layout.LEFT_ARM_ACTION], -0.02, 0.02)
            ts = env.step(layout.normalize_action(raw))
            if ts.last():
                break
        op = labels(ts.low_dim_obs)
        record.update(
            release_tilt=op["tilt_deg"],
            shift_cm=float(100 * np.linalg.norm(task["plate_xyz"] - op["plate_xyz"])),
            command_shift_cm=float(100 * np.linalg.norm(desired - p)),
        )
        raw[layout.GRIPPER_ACTION] = [0.0, 0.0]
        for _ in range(180):
            if ts.last():
                break
            ts = env.step(layout.normalize_action(raw))
        final = labels(ts.low_dim_obs)
        return dict(
            **record,
            success=bool(env.episode_succeeded() and not env.episode_fell()),
            fell=bool(env.episode_fell()),
            final_tilt=final["tilt_deg"],
            final_held=bool(final["held"]),
            final_support=bool(final["contacts"]["target_rack"]),
        )
    finally:
        env.close()


@app.function(
    image=image,
    cpu=1,
    memory=1536,
    timeout=1800,
    max_containers=2,
    single_use_containers=True,
    volumes={"/data": data},
)
def compare(seed):
    import inspect
    import subprocess

    if not 3110200 <= seed < 3110260:
        raise ValueError("Reserved generation diagnostic range")
    folder = Path("/data/astra-integration/pivot")
    folder.mkdir(exist_ok=True, parents=True)
    dest = folder / f"{seed}.json"
    if dest.exists():
        return json.loads(dest.read_text())
    t = time.monotonic()
    src = (
        "import json\n"
        + inspect.getsource(trial)
        + f"\nprint(json.dumps([trial({seed},s) for s in ('hand','center','bottom')]))"
    )
    p = subprocess.run(
        ["/root/policy/bin/python", "-c", src],
        cwd="/root/code",
        env={**os.environ, "MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"},
        capture_output=True,
        text=True,
    )
    if p.returncode:
        raise RuntimeError(p.stderr[-2000:])
    record = dict(arms=json.loads(p.stdout.splitlines()[-1]), seconds=time.monotonic() - t)
    dest.write_text(json.dumps(record))
    data.commit()
    return record


@app.function(cpu=0.125, memory=256, timeout=86400, volumes={"/data": data})
def batch(smoke_id):
    smoke = modal.FunctionCall.from_id(smoke_id).get()
    if not all(r["opened"] and r["held"] and not r["fell"] for r in smoke["arms"]):
        raise ValueError("Matched pivot smoke did not reach a held opening")
    folder = Path("/data/astra-integration/pivot")
    calls = folder / "calls.json"
    if calls.exists():
        raise ValueError("Active pivot study")
    ids = {
        s: compare.spawn(s).object_id
        for s in range(3110200, 3110220)
        if s != smoke["arms"][0]["seed"]
    }
    calls.write_text(json.dumps(ids))
    data.commit()
    # At most 20 generation prefixes (60 intervention episodes).
    rows = {smoke["arms"][0]["seed"]: smoke}
    for s, c in ids.items():
        rows[s] = modal.FunctionCall.from_id(c).get()
    (folder / "result.json").write_text(json.dumps(rows))
    data.commit()
    return {
        k: sum(
            r["success"]
            for row in rows.values()
            for r in row["arms"]
            if r["mode"] == k and r["opened"]
        )
        for k in ("hand", "center", "bottom")
    }
