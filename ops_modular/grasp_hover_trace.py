"""Replay archived actions to distinguish placement gates, reach, and pose errors."""

import json
import sys
from pathlib import Path

import numpy as np


def replay(path, rendered):
    from bigym.loco import make

    from advaitic_eval.environment import make_environment, observation
    from method.modular import layout
    from method.modular.kinematics import G1Kinematics, pelvis_rotation
    from method.modular.privileged import PrivilegedTaskState

    report = json.loads((path / "report.json").read_text())
    episode = report["episodes"][0]
    assert report["integrity_ok"] and episode["completed"] and not episode["policy_error"]
    with np.load(path / "episode_000.actions.npz") as data:
        actions = data["actions"]
    env = make_environment() if rendered else make("move_plate", camera_keys=())
    kin = G1Kinematics()
    frames = []
    err_max = 0
    if rendered:
        from submission.policy_modular_calibrated import make_policy

        policy = make_policy()
        policy.reset()
        ctl = policy.controller
        captured = {}
        original = ctl.site_contacts.update

        def observe(state, task):
            captured["task"] = original(state, task)
            return captured["task"]

        ctl.site_contacts.update = observe
        action = ctl._action

        def capture_action(state, position=None, rotation=None, base=None, close=False):
            captured["position"] = position
            return action(state, position, rotation, base, close)

        ctl._action = capture_action
    try:
        ts = env.reset(seed=episode["seed"])
        sensor = PrivilegedTaskState(env)
        side = next(x for x in env.inner_env.robot.grippers if x.name == "LEFT")
        for t, cmd in enumerate(actions):
            state = ts.low_dim_obs
            truth = sensor(state)
            item = {}
            if rendered:
                phase = ctl.phase.value
                issued = policy(observation(ts))
                task = captured["task"]
                delta = float(np.max(np.abs(issued - cmd)))
                err_max = max(err_max, delta)
                item.update(
                    phase=phase,
                    next_phase=ctl.phase.value,
                    elapsed=ctl.machine.elapsed,
                    error=delta,
                    grip=bool(task["held"]),
                    observed_rack=bool(task["contacts"]["target_rack"]),
                    gap=(task["plate_xyz"] - ctl.target_xyz).tolist(),
                    target=ctl.target_xyz.tolist(),
                    plate=task["plate_xyz"].tolist(),
                    tilt=task["tilt_deg"],
                    correction=ctl.place_correction.tolist(),
                    goal=None
                    if captured.get("position") is None
                    else captured["position"].tolist(),
                )
            if not rendered or phase != ctl.phase.value or t % 5 == 0:
                kin.set_state(state)
                hand = env.inner_env.robot.get_hand_pos(side)
                basis = pelvis_rotation(state)
                measured = state[18:21] + basis @ kin.hand_pose("left")[0]
                raw = layout.denormalize_action(cmd)
                item.update(
                    t=t,
                    true_plate=truth["plate_xyz"].tolist(),
                    site=truth["site_xyz"].tolist(),
                    true_gap=(truth["plate_xyz"] - truth["site_xyz"]).tolist(),
                    true_tilt=truth["tilt_deg"],
                    held=bool(truth["held"]),
                    rack=bool(truth["contacts"]["target_rack"]),
                    hand=hand.tolist(),
                    fk=measured.tolist(),
                    base=state[18:22].tolist(),
                    q=state[:7].tolist(),
                    arm_cmd=raw[4:11].tolist(),
                    open=bool(cmd[18] < 0),
                )
                frames.append(item)
            ts = env.step(cmd)
            if rendered and ctl.phase.value == "HOLD":
                break
        if not rendered:
            assert (
                t + 1,
                env.episode_succeeded(),
                env.episode_fell(),
                env.episode_termination(),
            ) == (episode["steps"], episode["success"], episode["fell"], episode["termination"])
        return dict(
            seed=episode["seed"],
            rendered=rendered,
            frames=frames,
            max_error=err_max,
            transitions=ctl.transitions if rendered else [],
            steps=t + 1,
        )
    finally:
        env.close()


if __name__ == "__main__":
    print(json.dumps(replay(Path(sys.argv[1]), sys.argv[2] == "public")))
