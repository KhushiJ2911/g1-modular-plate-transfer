"""Paired camera-free placement continuations with controlled height bias."""

import hashlib
import inspect
import json
import os
import subprocess
import time
from pathlib import Path

import modal

from ops_modular.modular_sim import image

APP = "modular-sol-height-hybrid"
CHECKPOINT = "64dfe105e7f716c0ab5b7a9c487b7a8a1cac06565070b390e75992dfbee4deff"
app = modal.App(APP)
volume = modal.Volume.from_name("advaitic-runs")


def scenario(seed, archive, normal, mode, plate_bias, site_bias):
    import numpy as np
    from bigym.loco import make

    from advaitic_eval.environment import validate_actions
    from method.modular import layout
    from method.modular.contact_perception import PerceivedRackContact
    from method.modular.fk_seating import FKSeatingController
    from method.modular.kinematics import G1Kinematics, pelvis_rotation
    from method.modular.orientation_feedback import LEFT_GRASP_NORMAL
    from method.modular.phases import Phase
    from method.modular.pickup import PickupController
    from method.modular.privileged import PrivilegedTaskState
    from method.modular.refined import align_normal

    with np.load(Path(archive) / "episode_000.actions.npz", allow_pickle=False) as f:
        actions = f["actions"]
    env = make("move_plate", camera_keys=())
    kin = G1Kinematics()
    policy = (PickupController if mode == "base" else FKSeatingController)(kin)
    policy.site_contacts = PerceivedRackContact(policy.site_contacts)
    policy.set_loaded_grasp_normal(normal)
    try:
        ts = env.reset(seed=seed)
        truth = PrivilegedTaskState(env)
        initial = truth(ts.low_dim_obs)
        z0 = initial["plate_xyz"][2]
        target = initial["site_xyz"] + [0.0, 0.0, site_bias]
        policy._initialize(ts.low_dim_obs, initial)
        task = {**initial, "site_xyz": target}
        for _ in range(6):
            policy.site_contacts.update(ts.low_dim_obs, task)
        policy.target_xyz = target

        def hand_world(state):
            kin.set_state(state)
            pos, rot = kin.hand_pose("left")
            basis = pelvis_rotation(state)
            return state[layout.PELVIS_XYZ] + basis @ pos, basis @ rot

        secure_frames, offset = 0, None
        for prefix, a in enumerate(actions, start=1):
            ts = env.step(a)
            state = ts.low_dim_obs
            physical = truth(state)
            if (
                offset is None
                and a[18] > 0
                and physical["held"]
                and state[layout.GRIPPERS][0] >= 0.8
            ):
                hand, rotation = hand_world(state)
                offset = rotation.T @ (initial["plate_xyz"] - hand)
            secure_frames = (
                secure_frames + 1
                if physical["held"] and physical["plate_xyz"][2] > z0 + 0.075
                else 0
            )
            if secure_frames >= 12 or ts.last():
                policy.machine.step = prefix
                break
        if offset is None or secure_frames < 12 or ts.last():
            raise ValueError(f"Archive for {seed} did not reach a secure public lift by step {prefix}")

        hand, rotation = hand_world(state)
        nominal = align_normal(physical["plate_rot"], rotation @ LEFT_GRASP_NORMAL)
        policy.last_raw = layout.denormalize_action(a)
        policy.machine.phase = Phase.LIFT
        policy._enter({**physical, "plate_rot": nominal}, hand, rotation)
        policy.hand_to_center = offset
        policy.machine.phase = Phase.CARRY
        policy._enter(physical, hand, rotation)
        opening, supported_steps, proprio_steps = None, 0, 0
        heights = []
        while not ts.last():
            state = ts.low_dim_obs
            physical = truth(state)
            plate = physical["plate_xyz"].copy()
            phase = policy.phase
            if phase in (Phase.ALIGN, Phase.LOWER):
                plate[2] += plate_bias
            basis = pelvis_rotation(state)
            task = {
                **physical,
                "plate_xyz": plate,
                "plate_rel": basis.T @ (plate - state[layout.PELVIS_XYZ]),
                "site_xyz": target,
            }
            if phase == Phase.LOWER:
                supported_steps += bool(physical["contacts"]["target_rack"] and physical["held"])
                if mode == "fk":
                    hand, rotation = hand_world(state)
                    heights.append(float((hand + rotation @ offset)[2] - physical["plate_xyz"][2]))
            a = validate_actions(policy.act(state, task))[0]
            if phase == Phase.LOWER:
                if mode == "fk":
                    proprio_steps += bool(policy.descent.seated())
                if a[18] < 0:
                    opening = dict(
                        step=policy.machine.step,
                        support=bool(physical["contacts"]["target_rack"]),
                        held=bool(physical["held"]),
                        tilt_deg=physical["tilt_deg"],
                        site_depth=float(physical["plate_xyz"][2] - physical["site_xyz"][2]),
                    )
            ts = env.step(a)
        return dict(
            seed=seed,
            arm=mode,
            plate_bias=plate_bias,
            site_bias=site_bias,
            success=bool(env.episode_succeeded() and not env.episode_fell()),
            fell=bool(env.episode_fell()),
            transitions=policy.transitions,
            opening=opening,
            supported_lower_steps=supported_steps,
            proprio_seating_steps=proprio_steps,
            fk_error_median_cm=float(100 * np.median(heights)) if heights else None,
        )
    finally:
        env.close()


@app.function(
    image=image,
    cpu=1,
    memory=1792,
    timeout=600,
    max_containers=12,
    volumes={"/runs": volume},
)
def episode(seed, archive, include_site_bias):
    if not 3_000_700 <= seed < 3_000_750:
        raise ValueError("Use the predeclared public-prefix development range")
    if not Path(archive).is_relative_to("/runs/grasp-validation") or Path(archive).name != "eval":
        raise ValueError("Expected existing public pickup archive")
    volume.reload()
    directory = Path(archive)
    report = json.loads((directory / "report.json").read_text())
    if (
        not report["integrity_ok"]
        or report["infrastructure_error"]
        or report["episodes"][0]["seed"] != seed
    ):
        raise ValueError("Archive integrity or seed mismatch")
    checkpoint = Path("/runs/heads") / f"{CHECKPOINT}.npz"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != CHECKPOINT:
        raise ValueError("Wrong loaded grasp calibration")
    normal = [-0.25436843862225744, -0.967107356788533, 0.00024057924249259576]
    biases = [(0.0, 0.0), (0.020, 0.0)]
    if include_site_bias:
        biases.append((0.0, -0.020))
    grid = [(mode, p, s) for p, s in biases for mode in ("base", "fk")]
    source = (
        "from pathlib import Path\nimport json\n"
        + inspect.getsource(scenario)
        + f"\nprint(json.dumps([scenario({seed}, {archive!r}, {normal!r}, *case) for case in {grid!r}]))"
    )
    started = time.monotonic()
    result = subprocess.run(
        ["/root/policy/bin/python", "-c", source],
        cwd="/root/code",
        env={**os.environ, "MUJOCO_GL": "egl"},
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr[-3000:])
    return {
        "seed": seed,
        "slot": report["episodes"][0]["source_slot"],
        "rows": json.loads(result.stdout.splitlines()[-1]),
        "seconds": time.monotonic() - started,
    }
