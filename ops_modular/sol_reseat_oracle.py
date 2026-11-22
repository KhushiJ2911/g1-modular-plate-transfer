"""Camera-free diagnostic of one supported plate reseat on generation actions."""

import inspect
import json
import os
import subprocess

import modal

from ops_modular.modular_sim import image

APP = "modular-sol-reseat-oracle-v2"
GENERATION = range(3_110_220, 3_110_232)
app = modal.App(APP)
data = modal.Volume.from_name("modular-data")


def trigger_reseat(task, step, lower_step, open_step, released):
    x, y, depth = task["plate_xyz"] - task["site_xyz"]
    on_rack = (
        task["held"]
        and task["contacts"]["target_rack"]
        and not (task["contacts"]["table"] or task["contacts"]["floor"])
        and abs(x) < 0.032
        and abs(y) < 0.032
        and -0.057 < depth < -0.015
    )
    if not on_rack:
        return None
    if open_step is None:
        return "seated_closed" if step >= lower_step + 140 else None
    if released and step >= open_step + 12:
        return "recaptured"
    return None


def trial(seed, refine):
    import numpy as np
    from bigym.loco import make

    from method.modular import layout
    from method.modular.controller import _rotation_from_vector
    from method.modular.kinematics import G1Kinematics, pelvis_rotation
    from method.modular.privileged import PrivilegedTaskState

    with np.load(f"/data/astra-dagger/sol/{seed}.npz") as trace:
        actions = trace["actions"]
        metadata = json.loads(str(trace["meta"]))
        lower = trace["step"][trace["phase"] == "LOWER"]
    result = {"seed": seed, "baseline": metadata["success"], "trigger": None}
    if not len(lower) or metadata["success"]:
        return result

    env = make("move_plate", camera_keys=())
    kin = G1Kinematics()
    try:
        ts = env.reset(seed=seed)
        labels = PrivilegedTaskState(env)
        open_step = None
        released = False
        for step, action in enumerate(actions):
            task = labels(ts.low_dim_obs)
            if step >= lower[0]:
                if open_step is None and action[18] < 0:
                    open_step = step
                released |= open_step is not None and not task["held"]
                reason = trigger_reseat(task, step, lower[0], open_step, released)
                if reason:
                    result.update(trigger=reason, step=step, before_tilt=task["tilt_deg"])
                    break
            ts = env.step(action)
            if ts.last():
                return result
        if not result["trigger"]:
            return result

        site = task["site_xyz"].copy()
        normal = task["site_rot"][:, 2].copy()
        raw = layout.denormalize_action(action)
        raw[layout.BASE_VELOCITY_ACTION] = 0
        raw[layout.GRIPPER_ACTION] = [1, 0]
        raw[layout.LEFT_ARM_ACTION] = ts.low_dim_obs[layout.LEFT_ARM]

        def hand_world():
            state = ts.low_dim_obs
            kin.set_state(state)
            pos, rot = kin.hand_pose("left")
            basis = pelvis_rotation(state)
            return state[layout.PELVIS_XYZ] + basis @ pos, basis @ rot

        def move(position, rotation, frames, *, must_hold=False):
            nonlocal ts
            for _ in range(frames):
                if ts.last():
                    return False
                state = ts.low_dim_obs
                if must_hold and not labels(state)["held"]:
                    return False
                if position is not None:
                    basis = pelvis_rotation(state).T
                    arm = raw[layout.LEFT_ARM_ACTION]
                    q = kin.ik(
                        "left",
                        basis @ (position - state[layout.PELVIS_XYZ]),
                        basis @ rotation,
                        q_init=arm,
                        rot_weight=0.22,
                        iters=14,
                    )
                    arm += np.clip(q - arm, -0.014, 0.014)
                ts = env.step(layout.normalize_action(raw))
            return True

        hand, rotation = hand_world()
        move(hand, rotation, 14)
        caught = labels(ts.low_dim_obs)
        result["regrasped"] = caught["held"]
        if caught["held"] and not ts.last():
            hand, rotation = hand_world()
            adjust = np.cross(caught["plate_rot"][:, 2], normal)
            correction = _rotation_from_vector(
                adjust * min(1.0, 0.28 / max(1e-8, np.linalg.norm(adjust)))
            )
            level_rotation = correction @ rotation
            offset = correction @ (hand - caught["plate_xyz"])
            seat = site + [0.0, 0.0, -0.033]
            raised = seat + [0.0, 0.0, 0.048]
            completed_lift = move(hand + [0.0, 0.0, 0.045], rotation, 28, must_hold=True)
            aligned = completed_lift and move(raised + offset, level_rotation, 28, must_hold=True)
            if aligned:
                for fraction in np.linspace(0.0, 1.0, 42):
                    alive = move(raised + fraction * (seat - raised) + offset, level_rotation, 1)
                    if not alive:
                        break
        if refine and not ts.last():
            for _ in range(2):
                current = labels(ts.low_dim_obs)
                if not (current["held"] and current["contacts"]["target_rack"]):
                    break
                if current["tilt_deg"] <= 5.5:
                    break
                hand, rotation = hand_world()
                axis = np.cross(current["plate_rot"][:, 2], normal)
                adjust = axis * min(1.0, 0.15 / max(1e-8, np.linalg.norm(axis)))
                if not move(hand, _rotation_from_vector(adjust) @ rotation, 20, must_hold=True):
                    break
        raw[layout.GRIPPER_ACTION] = [0, 0]
        release = labels(ts.low_dim_obs)
        result.update(
            release_tilt=release["tilt_deg"], release_support=release["contacts"]["target_rack"]
        )
        move(None, None, 180)
        final = labels(ts.low_dim_obs)
        return dict(
            **result,
            success=bool(env.episode_succeeded() and not env.episode_fell()),
            final_tilt=final["tilt_deg"],
            final_held=bool(final["held"]),
            fell=bool(env.episode_fell()),
        )
    finally:
        env.close()


@app.function(
    image=image, cpu=1, memory=1536, timeout=240, max_containers=2, volumes={"/data": data}
)
def episode(seed, refine=False):
    if seed not in GENERATION:
        raise ValueError("Use the fixed generation diagnostic range")
    source = (
        "import json, time\n"
        + inspect.getsource(trigger_reseat)
        + inspect.getsource(trial)
        + "\nstart = time.monotonic()\n"
        + f"print(json.dumps({{**trial({seed}, {refine!r}), 'seconds':time.monotonic() - start}}))"
    )
    result = subprocess.run(
        ["/root/policy/bin/python", "-c", source],
        cwd="/root/code",
        capture_output=True,
        env={**os.environ, "MUJOCO_GL": "egl"},
        text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr[-1800:])
    return json.loads(result.stdout.splitlines()[-1])
