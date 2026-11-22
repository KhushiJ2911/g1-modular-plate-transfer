"""Collect public observations under frozen v1b, then label saved actions in a separate replay."""

import argparse
import json
from pathlib import Path

import numpy as np

CONTACTS = ("source_rack", "target_rack", "table", "floor")


def collect(seed, out):
    from bigym.loco import make

    from advaitic_eval.environment import make_environment, observation
    from submission.policy_modular import make_policy

    from .privileged import PrivilegedTaskState, pelvis_rotation

    if not 3_100_000 <= seed < 3_200_000:
        raise ValueError("Generation requires the reserved v2 generation range")
    policy = make_policy()
    env = make_environment()
    actions, phases, steps, images, states, baseline = [], [], [], [], [], []
    try:
        ts = env.reset(seed=seed)
        policy.reset()
        prior = None
        while not ts.last():
            obs = observation(ts)
            phase = policy.controller.phase.value
            step = len(actions)
            action = policy(obs)
            if step == 0:
                target = policy.controller.target_xyz.copy()
            if step == 0 or phase != prior or step % (90 if phase == "HOLD" else 6) == 0:
                task = policy.task.snapshot(obs["state"])
                images.append(obs["rgb"].copy())
                states.append(obs["state"].copy())
                steps.append(step)
                phases.append(phase)
                baseline.append(
                    np.r_[
                        task["plate_rel"],
                        task["target_site_rel"],
                        task["tilt_deg"],
                        policy.task.held_probability,
                        task["plate_rot_rel"][:, :2].T.ravel(),
                        [task["contacts"][k] for k in CONTACTS],
                    ]
                )
            actions.append(action.copy())
            prior = phase
            ts = env.step(action)
        outcome = (
            bool(env.episode_succeeded()),
            bool(env.episode_fell()),
            str(env.episode_termination()),
        )
    finally:
        env.close()
    replay = make("move_plate", camera_keys=())
    labels = []
    try:
        ts = replay.reset(seed=seed)
        truth = PrivilegedTaskState(replay)
        rack = truth.inner.rack_target.sites
        xyz = np.array([truth.inner.data.bind(s).xpos.copy() for s in rack])
        # J commands a continuous world-space target. Assign its nearest physical
        # target site once for annotation. No slot ID enters the policy.
        site = int(np.argmin(np.linalg.norm(xyz - target, axis=1)))
        row = 0
        for step, action in enumerate(actions):
            if ts.last():
                raise ValueError("Replay terminated early")
            if row < len(steps) and step == steps[row]:
                state = np.asarray(ts.low_dim_obs, np.float32)
                np.testing.assert_array_equal(state, states[row])
                task = truth(state)
                site_xyz = truth.inner.data.bind(rack[site]).xpos
                locked = pelvis_rotation(state[21]).T @ (site_xyz - state[18:21])
                labels.append(
                    np.r_[
                        task["plate_rel"],
                        locked,
                        task["tilt_deg"],
                        task["held"],
                        task["plate_rot_rel"][:, :2].T.ravel(),
                        [task["contacts"][k] for k in CONTACTS],
                    ]
                )
                row += 1
            ts = replay.step(action)
        if (
            not ts.last()
            or row != len(steps)
            or outcome
            != (
                bool(replay.episode_succeeded()),
                bool(replay.episode_fell()),
                str(replay.episode_termination()),
            )
        ):
            raise ValueError("Replay terminal mismatch")
    finally:
        replay.close()
    meta = dict(
        seed=seed,
        steps=len(actions),
        frames=len(steps),
        success=outcome[0],
        fell=outcome[1],
        termination=outcome[2],
        locked_site=site,
        lock_error_m=float(np.linalg.norm(xyz[site] - target)),
    )
    np.savez_compressed(
        out,
        rgb=np.stack(images),
        state=np.stack(states),
        step=steps,
        phase=phases,
        actions=actions,
        truth=np.array(labels, np.float32),
        baseline=np.array(baseline, np.float32),
        meta=json.dumps(meta),
    )
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=int)
    parser.add_argument("output", type=Path)
    a = parser.parse_args()
    print(json.dumps(collect(a.seed, a.output)))
