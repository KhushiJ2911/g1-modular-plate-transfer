"""Read-only diagnostics along observation-only pickup rollouts."""

import argparse
import json
from pathlib import Path

import numpy as np

from .kinematics import pelvis_rotation
from .phases import Phase
from .pickup_validation import LiftMetrics
from .privileged import PrivilegedTaskState


def trace(seed, out, policy_factory=None):
    from advaitic_eval.environment import make_environment, observation
    from submission.policy_modular_heads import make_policy

    policy = (policy_factory or make_policy)()
    ctl = policy.controller
    env = make_environment()
    rows, wrist, image_steps, pinches = [], [], [], []
    carry = None
    try:
        ts = env.reset(seed=seed)
        labels = PrivilegedTaskState(env)
        metrics = LiftMetrics(labels(ts.low_dim_obs)["plate_xyz"][2])
        while True:
            obs = observation(ts)
            state = obs["state"]
            task = labels(state)
            metrics.observe(task)
            step = ctl.machine.step
            if ts.last() or ctl.phase == Phase.HOLD or (carry is not None and step - carry >= 60):
                break
            phase, elapsed = ctl.phase, ctl.machine.elapsed
            action = policy(obs)
            if phase == Phase.GRASP and ctl.phase == Phase.LIFT:
                view = ctl.view
                pinches.append(
                    dict(
                        step=step,
                        pregrasp_normal_world=(
                            None if view.rotation is None else view.rotation[:, 2].tolist()
                        ),
                        face_iou=view.face_iou,
                        face_frames=len(view.faces),
                        hand_to_plate=ctl.confirmed_hand_to_plate.tolist(),
                    )
                )
            estimate = policy.task.snapshot(state)
            ctl.kin.set_state(state)
            pos, rot = ctl.kin.hand_pose("left")
            basis = pelvis_rotation(state)
            hand, rot = state[18:21] + basis @ pos, basis @ rot
            if phase in (Phase.PREGRASP, Phase.GRASP, Phase.LIFT) or step < 6:
                rows.append(
                    dict(
                        step=step,
                        phase=phase.value,
                        elapsed=elapsed,
                        closing=ctl.closing,
                        hand=hand.tolist(),
                        rotation=rot.tolist(),
                        state=state.tolist(),
                        used=None if ctl.source is None else ctl.source.tolist(),
                        visual=estimate["plate_xyz"].tolist(),
                        truth=task["plate_xyz"].tolist(),
                        normal=task["plate_rot"][:, 2].tolist(),
                        est_normal=estimate["plate_rot"][:, 2].tolist(),
                        held=task["held"],
                        est_held=bool(estimate["held"]),
                    )
                )
                if step < 6 or phase == Phase.GRASP and (elapsed <= 6 or elapsed % 8 == 0):
                    wrist.append(obs["rgb"][2])
                    image_steps.append(step)
            if carry is None and ctl.phase == Phase.CARRY:
                carry = step
            ts = env.step(action)
        summary = dict(
            seed=seed,
            **metrics.result(),
            transitions=ctl.transitions,
            pinches=pinches,
            carry=carry,
            stopped=step,
        )
        np.savez_compressed(
            out,
            image_steps=image_steps,
            wrist=wrist,
            rows=json.dumps(rows),
            summary=json.dumps(summary),
        )
        return summary
    finally:
        env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("seed", type=int)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    print(json.dumps(trace(a.seed, a.out)))
