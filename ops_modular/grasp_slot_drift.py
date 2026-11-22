"""Measure how much of the true post-open wrist displacement is visible to public FK."""

import json
import sys
from pathlib import Path

import numpy as np


def rack_support(inner, plate_ids, rack_ids):
    """Resolve the rack's net force on the plate and its vertical load centroid."""
    import mujoco

    force = np.zeros(3)
    centroid = np.zeros(3)
    weight = 0.0
    for index, contact in enumerate(inner.data.contact):
        a, b = int(contact.geom1), int(contact.geom2)
        if a in rack_ids and b in plate_ids:
            sign = 1
        elif b in rack_ids and a in plate_ids:
            sign = -1
        else:
            continue
        local = np.zeros(6)
        mujoco.mj_contactForce(inner.model, inner.data, index, local)
        world = sign * contact.frame.reshape(3, 3).T @ local[:3]
        force += world
        load = max(0.0, float(world[2]))
        weight += load
        centroid += load * contact.pos
    return force.tolist(), (centroid / weight).tolist() if weight else None


def batch(jobs):
    from bigym.loco import make

    from method.modular import layout
    from method.modular.kinematics import G1Kinematics, pelvis_rotation

    env = make("move_plate", camera_keys=())
    kin = G1Kinematics()
    out = []
    try:
        for job in jobs:
            path = Path("/runs/modular", job["folder"], "eval")
            ep = json.loads((path / "report.json").read_text())["episodes"][0]
            with np.load(path / "episode_000.actions.npz") as f:
                actions = f["actions"]
            t0 = job["open"]
            ts = env.reset(seed=ep["seed"])
            inner = env.inner_env
            side = next(s for s in inner.robot.grippers if s.name == "LEFT")
            plate = inner.plates[0]
            plate_ids = {inner.model.bind(g).id for g in plate.colliders}
            rack_ids = {inner.model.bind(g).id for g in inner.rack_target.colliders}
            frames = []
            for t in range(min(t0 + 16, len(actions))):
                if t >= t0 - 15:
                    state = ts.low_dim_obs
                    kin.set_state(state)
                    basis = pelvis_rotation(state)
                    world = state[18:21] + basis @ kin.hand_pose("left")[0]
                    J = basis @ kin.jacobian("left")[:3]
                    raw = layout.denormalize_action(actions[t])
                    tracking = raw[4:11] - state[:7]
                    true = inner.robot.get_hand_pos(side)
                    frames.append(
                        dict(
                            step=t,
                            fk=world.tolist(),
                            hand=true.tolist(),
                            tracking_xyz=(J @ tracking).tolist(),
                            qerr=float(np.linalg.norm(tracking)),
                            base=state[18:22].tolist(),
                            base_velocity=state[40:44].tolist(),
                            fingers=state[7:9].tolist(),
                            norm_cmd=raw[4:11].tolist(),
                            plate=plate.get_position().tolist(),
                            rack_support=rack_support(inner, plate_ids, rack_ids),
                        )
                    )
                ts = env.step(actions[t])
            if np.linalg.norm(np.array(frames[15]["hand"]) - job["hand"]) > 1e-6:
                raise ValueError("Did not reproduce initial opening pose")
            out.append(
                dict(
                    folder=job["folder"],
                    seed=ep["seed"],
                    slot=ep["source_slot"],
                    success=ep["success"],
                    frames=frames,
                )
            )
    finally:
        env.close()
    return out


def dwell_forks(jobs, intervention):
    """Paired mechanical intervention on the observed opening pose, through the 75-step dwell."""
    from bigym.loco import make

    from method.modular import layout
    from method.modular.estimated import SITE_ROTATION
    from method.modular.kinematics import G1Kinematics, pelvis_rotation
    from method.modular.phases import SETTLE_STEPS, Phase
    from submission.policy_slot_clearance import OpeningPlaneFeedback

    class OpenDwell:
        """The incumbent holds its open action throughout the mandatory dwell."""

        def __init__(self, raw):
            self.last_raw = raw
            self.controller = self
            self.kin = kin
            self.target_rot = SITE_ROTATION
            self.phase = Phase.SETTLE

        def reset(self):
            pass

        def __call__(self, obs):
            return layout.normalize_action(self.last_raw)

    env = make("move_plate", camera_keys=())
    kin = G1Kinematics()
    rows = []
    try:
        for job in jobs:
            path = Path(job["path"])
            ep = json.loads((path / "report.json").read_text())["episodes"][0]
            with np.load(path / "episode_000.actions.npz") as f:
                actions = f["actions"]
            t0 = job["open"]
            open_command = actions[t0]
            original = actions[t0 : t0 + SETTLE_STEPS]
            if np.max(abs(original - open_command)) > 1e-5 or open_command[18] >= 0:
                raise ValueError("Expected the controller's constant open dwell")
            arms = {}
            for mode in ("original", intervention):
                ts = env.reset(seed=ep["seed"])
                for cmd in actions[:t0]:
                    ts = env.step(cmd)
                inner = env.inner_env
                side = next(s for s in inner.robot.grippers if s.name == "LEFT")
                start = inner.robot.get_hand_pos(side).copy()
                if np.linalg.norm(start - job["hand"]) > 1e-6:
                    raise ValueError("Fork did not reproduce opening pose")
                state = ts.low_dim_obs
                kin.set_state(state)
                row = SITE_ROTATION[:, 2] @ pelvis_rotation(state) @ kin.jacobian("left")[:3]
                raw = layout.denormalize_action(open_command)
                target = raw[layout.LEFT_ARM_ACTION]
                preload = float(row @ (target - state[layout.LEFT_ARM]))
                command = open_command
                feedback = OpeningPlaneFeedback(OpenDwell(raw)) if mode == "feedback" else None
                if mode == "unload" and row @ row > 1e-12:
                    target -= row * preload / (row @ row)
                    command = layout.normalize_action(raw)
                free = touch = drift = None
                for step in range(1, SETTLE_STEPS + 1):
                    if feedback is not None:
                        command = feedback({"state": ts.low_dim_obs})
                    ts = env.step(command)
                    held = inner.robot.is_gripper_holding_object(inner.plates[0], side)
                    if not held and free is None:
                        free = step
                    if held and free is not None and touch is None:
                        touch = step
                    if step == 15:
                        drift = float(
                            SITE_ROTATION[:, 2] @ (inner.robot.get_hand_pos(side) - start)
                        )
                    if ts.last():
                        break
                arms[mode] = dict(
                    first_free=free,
                    recontact=touch,
                    drift15=drift,
                    preload=preload,
                    steps=step,
                    success=bool(env.episode_succeeded()),
                    fell=bool(env.episode_fell()),
                )
            rows.append(dict(path=str(path), seed=ep["seed"], slot=ep["source_slot"], **arms))
    finally:
        env.close()
    return rows


if __name__ == "__main__":
    jobs = json.loads(sys.argv[1])
    if len(sys.argv) == 3:
        if sys.argv[2] not in ("unload", "feedback"):
            raise ValueError("Expected an opening intervention")
        print(json.dumps(dwell_forks(jobs, sys.argv[2])))
    else:
        print(json.dumps(batch(jobs)))
