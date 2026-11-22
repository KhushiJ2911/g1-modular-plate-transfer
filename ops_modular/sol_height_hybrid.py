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

APP = "modular-sol-height-hybrid-v4"
CHECKPOINT = "64dfe105e7f716c0ab5b7a9c487b7a8a1cac06565070b390e75992dfbee4deff"
app = modal.App(APP)
volume = modal.Volume.from_name("modular-runs")
weights = modal.Volume.from_name("advaitic-runs")


def scenario(seed, archive, normal, mode, plate_bias, site_bias, hide_rack_contact=False):
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
            if a[18] < 0:
                # Mirror the pickup controller discard on a missed grasp.
                offset = None
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
            raise ValueError(
                f"Archive for {seed} did not reach a secure public lift by step {prefix}"
            )

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
        windows = []
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
            if hide_rack_contact and phase == Phase.LOWER:
                task["contacts"] = {**physical["contacts"], "target_rack": False}
            if phase == Phase.LOWER:
                supported_steps += bool(physical["contacts"]["target_rack"] and physical["held"])
                if mode == "fk":
                    hand, rotation = hand_world(state)
                    heights.append(float((hand + rotation @ offset)[2] - physical["plate_xyz"][2]))
            a = validate_actions(policy.act(state, task))[0]
            if phase == Phase.LOWER:
                if mode == "fk":
                    proprio_steps += bool(policy.descent.seated())
                    if hide_rack_contact and policy.machine.step % 6 == 0:
                        samples = np.array(policy.descent.samples)
                        if len(samples) == policy.descent.samples.maxlen:
                            before = np.median(samples[:3], axis=0)
                            after = np.median(samples[-3:], axis=0)
                            windows.append(
                                {
                                    "step": policy.machine.step,
                                    "support": physical["contacts"]["target_rack"],
                                    "measured_mm": float(1000 * (before[0] - after[0])),
                                    "commanded_mm": float(1000 * (before[1] - after[1])),
                                    "deflection_mm": float(1000 * (after[0] - after[1])),
                                }
                            )
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
            hide_rack_contact=hide_rack_contact,
            success=bool(env.episode_succeeded() and not env.episode_fell()),
            fell=bool(env.episode_fell()),
            transitions=policy.transitions,
            opening=opening,
            supported_lower_steps=supported_steps,
            proprio_seating_steps=proprio_steps,
            fk_error_median_cm=float(100 * np.median(heights)) if heights else None,
            **({"windows": windows} if hide_rack_contact else {}),
        )
    finally:
        env.close()


@app.function(
    image=image,
    cpu=1,
    memory=1792,
    timeout=600,
    max_containers=12,
    volumes={"/runs": volume, "/weights": weights},
)
def episode(seed, archive, include_site_bias, proprio_only=False):
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
    checkpoint = Path("/weights/heads") / f"{CHECKPOINT}.npz"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != CHECKPOINT:
        raise ValueError("Wrong loaded grasp calibration")
    biases = [(0.0, 0.0), (0.020, 0.0)]
    if include_site_bias:
        biases.append((0.0, -0.020))
    grid = [(mode, p, s) for p, s in biases for mode in ("base", "fk")]
    if proprio_only:
        grid = [("base", 0.020, 0.0, True), ("fk", 0.020, 0.0, True)]
    source = (
        "from pathlib import Path\nimport json\nimport numpy as np\n"
        + inspect.getsource(scenario)
        + f"\nwith np.load({str(checkpoint)!r}, allow_pickle=False) as weights:\n"
        + '    normal = weights["loaded_normal"]\n'
        + f"print(json.dumps([scenario({seed}, {archive!r}, normal, *case) for case in {grid!r}]))"
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
