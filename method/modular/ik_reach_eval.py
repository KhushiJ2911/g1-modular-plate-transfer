"""Training-only, camera-free diagnostics for M0 and M1. All live reads are here.

M0 samples one endpoint per seed after bounded random commands. M1 follows a
geometric pre-grasp with the left hand open. Only the plate pose and state50
are passed to the controller. Actual site poses are read for scoring.
"""

import argparse
import json
import os
import sys
from contextlib import redirect_stdout

os.environ["ORT_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import onnxruntime
from bigym.loco import make

from . import layout
from .kinematics import G1Kinematics, pelvis_rotation, rotation_vector, world_to_pelvis


def validate_seeds(start, count):
    if not (1 <= count <= 60 and 3_000_000 <= start < start + count <= 2**32):
        raise ValueError("Use 1 to 60 privileged development seeds starting at 3000000 or higher")
    for lo, hi in ((1_000_000, 1_100_000), (2_600_000, 2_700_000), (7_310_000, 7_320_000)):
        if start < hi and start + count > lo:
            raise ValueError("Protected seed block")


def live_hand(env, side):
    inner = env.inner_env
    key = next(s for s in inner.robot.grippers if s.name.lower() == side)
    site = inner.data.bind(inner.robot.grippers[key].wrist_site)
    return inner.robot.get_hand_pos(key), site.xmat.reshape(3, 3).copy()


def pose_error(p, r, target_p, target_r):
    return float(np.linalg.norm(p - target_p)), float(
        np.rad2deg(np.linalg.norm(rotation_vector(target_r @ r.T)))
    )


def fk_episode(env, kin, seed):
    ts = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    base = env.raw_hold_action()
    steps = int(rng.integers(10, 101))
    for t in range(steps):
        if t % 12 == 0:
            raw = base.copy()
            raw[:2] = rng.uniform(-0.24, 0.24, 2)
            raw[2] += rng.uniform(-0.06, 0.04)
            raw[3] = rng.uniform(-0.25, 0.25)
            raw[4:18] += rng.uniform(-0.48, 0.48, 14)
            raw[18:] = rng.random(2)
            action = layout.normalize_action(raw)
            np.testing.assert_allclose(
                layout.denormalize_action(action), env.denormalize_action(action), atol=5e-7
            )
        ts = env.step(action)
        if ts.last():
            break
    state = ts.low_dim_obs
    inner = env.inner_env
    grippers = list(inner.robot.grippers.values())
    suffixes = [
        *layout.JOINT_NAMES[:7],
        "left_dex1_finger_joint_1",
        "left_dex1_finger_joint_2",
        *layout.JOINT_NAMES[7:],
        "right_dex1_finger_joint_1",
        "right_dex1_finger_joint_2",
        "pelvis_x",
        "pelvis_y",
        "pelvis_z",
        "pelvis_rz",
    ]
    ids = [inner.model.joint(inner.robot.namespace + n).id for n in suffixes]
    expected = np.r_[
        inner.data.qpos[inner.model.jnt_qposadr[ids]],
        inner.data.qvel[inner.model.jnt_dofadr[ids]],
        [g.qpos for g in grippers],
        state[18:22],
    ].astype(np.float32)
    np.testing.assert_allclose(state, expected, atol=1e-6, rtol=0)
    kin.set_state(state)
    record = {"seed": seed, "steps": t + 1, "fell": env.episode_fell()}
    # Independent diagnostic isolates geometry below the unobserved waist.
    torso = inner.data.body(inner.robot.namespace + "torso_link")
    pelvis = inner.data.body(inner.robot.namespace + "pelvis")
    local_torso = kin.data.body("torso_link")
    for side in ("left", "right"):
        hp, hr = live_hand(env, side)
        p, r = kin.hand_pose(side)
        record[side + "_m"], record[side + "_deg"] = pose_error(
            p, r, world_to_pelvis(state, hp), pelvis_rotation(state).T @ hr
        )
        record[side + "_pelvis_body_m"], _ = pose_error(
            p, r, pelvis.xmat.reshape(3, 3).T @ (hp - pelvis.xpos),
            pelvis.xmat.reshape(3, 3).T @ hr
        )
        record[side + "_torso_m"], record[side + "_torso_deg"] = pose_error(
            local_torso.xmat.reshape(3, 3).T @ (p - local_torso.xpos),
            local_torso.xmat.reshape(3, 3).T @ r,
            torso.xmat.reshape(3, 3).T @ (hp - torso.xpos),
            torso.xmat.reshape(3, 3).T @ hr,
        )
    return record


def pregrasp(plate_pose):
    """Left pad center just in front of the near-upper rim, jaws across the plate.

    Project the world forward axis into the plate plane for the approach axis.
    This is invariant to rotation of the circular plate about its own normal.
    """
    plate_rot = np.empty(9)
    mujoco.mju_quat2Mat(plate_rot, plate_pose[3:])
    y = -plate_rot.reshape(3, 3)[:, 2]
    forward = np.array([1.0, 0.0, 0.0])
    x = forward - y * y[0]
    x /= np.linalg.norm(x)
    z = np.cross(x, y)
    return plate_pose[:3] - 0.14 * x + 0.07 * z, np.column_stack((x, y, z))


class Reach:
    def __init__(self, kin, state):
        self.kin = kin
        self.command = state[layout.LEFT_ARM].copy()
        self.right_hold = state[layout.RIGHT_ARM].copy()
        self.solution = self.command.copy()
        self.bias = np.zeros(7)
        self.stopped = False

    def act(self, state, plate_pose):
        p, r = pregrasp(plate_pose)
        rot = pelvis_rotation(state)
        raw = np.zeros(20, dtype=np.float32)
        raw[2] = 0.74
        raw[layout.RIGHT_ARM_ACTION] = self.right_hold
        stand_error = (plate_pose[:2] - [0.47, 0.16]) - state[layout.PELVIS_XY]
        distance = np.linalg.norm(stand_error)
        if distance <= 0.035:
            self.stopped = True
        if not self.stopped:
            v = stand_error * (np.clip(2 * distance, 0.09, 0.42) / distance)
            raw[:2] = rot[:2, :2].T @ v
        if self.stopped:
            self.kin.set_state(state)
            self.solution = self.kin.ik(
                "left", world_to_pelvis(state, p), rot.T @ r, q_init=self.solution, iters=12
            )
            # Integral action removes steady arm-servo tracking error under gravity.
            err = self.solution - state[layout.LEFT_ARM]
            self.bias = np.clip(self.bias + 0.07 * np.clip(err, -0.08, 0.08), -0.12, 0.12)
            self.command += np.clip(self.solution + self.bias - self.command, -0.055, 0.055)
        raw[layout.LEFT_ARM_ACTION] = self.command
        return layout.normalize_action(raw)


def reach_episode(env, kin, seed, trace=False):
    ts = env.reset(seed=seed)
    reach = Reach(kin, ts.low_dim_obs)
    plate = env.inner_env.plates[0]
    settled = 0
    tail = []
    for step in range(1, 401):
        ts = env.step(reach.act(ts.low_dim_obs, plate.get_pose()))
        p, r = live_hand(env, "left")
        target_p, target_r = pregrasp(plate.get_pose())
        meters, deg = pose_error(p, r, target_p, target_r)
        tail = (tail + [(meters, deg)])[-10:]
        settled += int(reach.stopped)
        if trace and (step % 25 == 0 or settled == 120):
            state = ts.low_dim_obs
            kin.set_state(state)
            print(
                json.dumps(
                    dict(
                        step=step,
                        stopped=reach.stopped,
                        pelvis=state[18:22].tolist(),
                        error=(p - target_p).tolist(),
                        deg=deg,
                        fk_bias=(world_to_pelvis(state, p) - kin.hand_pose("left")[0]).tolist(),
                        q=state[layout.LEFT_ARM].tolist(),
                        solution=reach.solution.tolist(),
                    )
                ),
                flush=True,
            )
        if ts.last() or env.episode_fell() or settled == 120:
            break
    # Use the worst of the ten final samples. Holding for a fixed interval
    # prevents a single pass through the tolerance during motion from counting.
    meters, deg = np.max(tail, axis=0).tolist()
    fell = env.episode_fell()
    return dict(
        seed=seed,
        steps=step,
        stopped=reach.stopped,
        settled_steps=settled,
        final10_max_m=meters,
        final10_max_deg=deg,
        fell=fell,
        reached=bool(settled == 120 and not fell and meters <= 0.01 and deg <= 10),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["fk", "reach"])
    parser.add_argument("--start", type=int, default=3_000_000)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    validate_seeds(args.start, args.count)
    onnxruntime.disable_telemetry_events()
    with redirect_stdout(sys.stderr):
        env = make("move_plate", camera_keys=())
    kin = G1Kinematics()
    try:
        for seed in range(args.start, args.start + args.count):
            if args.mode == "fk":
                record = fk_episode(env, kin, seed)
            else:
                record = reach_episode(env, kin, seed, args.trace)
            print(json.dumps(record), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
