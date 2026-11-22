"""Fresh development labels and shadow reconstruction of recorded official failures."""

import argparse
import json
from pathlib import Path

import numpy as np

from .kinematics import pelvis_rotation

CONTACTS = ("source_rack", "target_rack", "table", "floor")


def physical(task):
    return np.r_[
        task["plate_xyz"],
        task["site_xyz"],
        task["plate_rot"][:, 2],
        task["tilt_deg"],
        task["held"],
        [task["contacts"][k] for k in CONTACTS],
    ]


def replay_truth(seed, actions):
    from bigym.loco import make

    from .privileged import PrivilegedTaskState

    env = make("move_plate", camera_keys=())
    try:
        ts = env.reset(seed=seed)
        adapter = PrivilegedTaskState(env)
        records, states = [], []
        for action in actions:
            if ts.last():
                raise RuntimeError("Action log continues beyond terminal frame")
            state = np.asarray(ts.low_dim_obs).copy()
            records.append(physical(adapter(state)))
            states.append(state)
            ts = env.step(action)
        return (
            np.array(records),
            np.array(states),
            {
                "steps": len(actions),
                "success": bool(env.episode_succeeded()),
                "fell": bool(env.episode_fell()),
                "termination": env.episode_termination(),
                "terminal": bool(ts.last()),
            },
        )
    finally:
        env.close()


def audit(seed, driver, output, actions_file=None, refined=None):
    from advaitic_eval.environment import make_environment, observation
    from submission.policy_modular import make_policy

    from .privileged import PrivilegedTaskState

    if actions_file is None and not 3_000_000 <= seed < 3_100_000:
        raise ValueError("Fresh development seed required")
    if actions_file:
        if not 1_000_000 <= seed < 1_100_000:
            raise ValueError("Only original development traces accepted")
        with np.load(actions_file) as f:
            actions = f["actions"]
        truth_log, state_log, replay = replay_truth(seed, actions)
    else:
        actions = replay = None
    policy = make_policy()
    if refined:
        from .refined import attach

        attach(policy, refined)
    embedding = []

    def hook(mod, args):
        embedding[:] = [args[0].detach().cpu().numpy()[0]]

    policy.task.estimator.model.head[0].register_forward_pre_hook(hook)
    controller = policy.controller
    env = make_environment()
    rows = {k: [] for k in ("step", "phase", "features", "state", "hand", "pred", "truth", "gate")}
    max_action = max_state = 0.0
    steps = 0
    pred = None
    try:
        ts = env.reset(seed=seed)
        truth_source = PrivilegedTaskState(env)
        while not ts.last():
            if actions is not None and steps == len(actions):
                break
            phase = controller.phase.value
            if phase == "HOLD" and replay:
                # All future commands in HOLD are constant, including after the first HOLD action.
                frozen = controller.act(np.asarray(ts.low_dim_obs), pred)
                max_action = max(max_action, float(np.max(np.abs(actions[steps:] - frozen))))
                break
            if phase == "HOLD":
                # HOLD has constant normalized action and ignores perception.
                frozen = controller.act(np.asarray(ts.low_dim_obs), pred)
                while not ts.last():
                    ts = env.step(frozen)
                    steps += 1
                break
            obs = observation(ts)
            state = obs["state"]
            pred = policy.task.update(obs)
            truth = truth_source(state)
            if replay:
                max_state = max(max_state, float(np.max(np.abs(state_log[steps] - state))))
                if np.max(np.abs(physical(truth) - truth_log[steps])) > 1e-5:
                    raise RuntimeError("Rendered and camera-free privileged labels disagree")
            controller.kin.set_state(state)
            pos, rot = controller.kin.hand_pose("left")
            basis = pelvis_rotation(state)
            hand = np.r_[state[18:21] + basis @ pos, (basis @ rot).ravel()]
            if not steps:
                first_plate = pred["plate_xyz"].copy()
                first_site = pred["site_xyz"].copy()
            gates = [False, False]
            if phase == "LOWER":
                gates = [controller._ready_to_release(task) for task in (pred, truth)]
            if phase != "HOLD" and (actions_file or steps % 3 == 0):
                for k, v in dict(
                    step=steps,
                    phase=phase,
                    state=state,
                    hand=hand,
                    features=embedding[0],
                    pred=physical(pred),
                    truth=truth_log[steps] if replay else physical(truth),
                    gate=gates,
                ).items():
                    rows[k].append(v)
            action = controller.act(state, truth if driver == "truth" else pred)
            if replay:
                max_action = max(max_action, float(np.max(np.abs(action - actions[steps]))))
                action = actions[steps]
            ts = env.step(action)
            steps += 1
        outcome = replay or {
            "steps": steps,
            "success": bool(env.episode_succeeded()),
            "fell": bool(env.episode_fell()),
            "termination": env.episode_termination(),
        }
        info = dict(
            seed=seed,
            driver=driver,
            outcome=outcome,
            transitions=controller.transitions,
            max_action_delta=max_action,
            max_state_delta=max_state,
            shadow_steps=steps,
            first_plate=first_plate.tolist(),
            first_site=first_site.tolist(),
        )
        arrays = {k: np.asarray(v) for k, v in rows.items()}
        arrays["features"] = arrays["features"].astype(np.float16)
        if replay:
            del arrays["features"]  # Development replay files are never training examples.
        np.savez_compressed(output, **arrays, info=json.dumps(info))
        return info
    finally:
        env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("seed", type=int)
    p.add_argument("driver", choices=["truth", "estimate"])
    p.add_argument("output", type=Path)
    p.add_argument("--actions", type=Path)
    p.add_argument("--refined", type=Path)
    a = p.parse_args()
    print(json.dumps(audit(a.seed, a.driver, a.output, a.actions, a.refined)))
