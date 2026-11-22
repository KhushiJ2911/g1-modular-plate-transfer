"""Development-only paired pickup substitution and camera-free official-action audit."""

import argparse
import json
import os

import numpy as np

from .controller import ModularController
from .observed_controller import CalibratedPlacementController
from .phases import Phase
from .privileged import PrivilegedTaskState


class LiftMetrics:
    def __init__(self, origin):
        self.origin = origin
        self.max_rise = 0
        self.streak = self.longest = self.empty = 0
        self.acquired = False
        self.early_losses = 0

    def observe(self, task):
        rise = task["plate_xyz"][2] - self.origin
        if task["held"]:
            self.acquired = True
            self.empty = 0
            self.max_rise = max(self.max_rise, float(rise))
        else:
            self.empty += 1
            if self.acquired and self.longest < 30 and self.empty == 5:
                self.early_losses += 1
                self.acquired = False
        self.streak = self.streak + 1 if task["held"] and rise >= 0.05 else 0
        self.longest = max(self.longest, self.streak)

    def result(self):
        return dict(
            lift=self.max_rise >= 0.05, secure=self.longest >= 30, early_losses=self.early_losses
        )


def baseline():
    from submission.policy_modular import make_policy
    from submission.policy_modular_heads import PerceptionPolicy

    return PerceptionPolicy(make_policy(), os.environ["ADV_HEADS"], CalibratedPlacementController)


def hybrid(seed, variant):
    from bigym.loco import make

    from advaitic_eval.environment import (
        make_environment,
        observation,
        observed_source_slot,
        validate_actions,
    )
    from submission.policy_modular_heads import make_policy

    policy = (baseline if variant == "base" else make_policy)()
    ctl = policy.controller
    env = make_environment()
    try:
        ts = env.reset(seed=seed)
        policy.reset()
        truth = PrivilegedTaskState(env)
        metrics = LiftMetrics(truth(ts.low_dim_obs)["plate_xyz"][2])
        slot = observed_source_slot(env)
        handoffs = []
        prefix = []
        while True:
            task = truth(ts.low_dim_obs)
            metrics.observe(task)
            if ts.last():
                break
            phase = ctl.phase
            if phase in (Phase.APPROACH, Phase.PREGRASP, Phase.GRASP, Phase.LIFT):
                obs = observation(ts)
                est = policy.task.update(obs)
                # All source pose, held and source-contact inputs are estimated.
                # The initial destination used by CARRY comes from truth.
                combined = {
                    **task,
                    **{
                        k: est[k]
                        for k in (
                            "plate_xyz",
                            "plate_rel",
                            "plate_rot",
                            "plate_rot_rel",
                            "tilt_deg",
                            "held",
                        )
                    },
                    "contacts": {**task["contacts"], "source_rack": est["contacts"]["source_rack"]},
                }
                if "left_wrist_rgb" in est:
                    combined["left_wrist_rgb"] = est["left_wrist_rgb"]
                action = ctl.act(obs["state"], combined)
            else:
                action = ModularController.act(ctl, ts.low_dim_obs, task)
            if phase != ctl.phase and ctl.phase in (Phase.LIFT, Phase.CARRY):
                handoffs.append(
                    dict(
                        phase=ctl.phase.value,
                        held=task["held"],
                        rise=float(task["plate_xyz"][2] - metrics.origin),
                    )
                )
            action = validate_actions(action)[0]
            ts = env.step(action)
            if prefix is not None:
                prefix.append(action.copy())
                if ctl.phase in (Phase.CARRY, Phase.HOLD) and not ts.last():
                    # All subsequent observations in this hybrid are privileged.
                    # Replay the actual prefix, verify the handoff, then omit cameras.
                    state = ts.low_dim_obs.copy()
                    xyz = truth(state)["plate_xyz"]
                    env.close()
                    env = make("move_plate", camera_keys=())
                    ts = env.reset(seed=seed)
                    for recorded in prefix:
                        ts = env.step(recorded)
                    truth = PrivilegedTaskState(env)
                    np.testing.assert_array_equal(state, ts.low_dim_obs)
                    np.testing.assert_array_equal(xyz, truth(ts.low_dim_obs)["plate_xyz"])
                    prefix = None
        return dict(
            seed=seed,
            variant=variant,
            slot=slot,
            steps=ctl.machine.step,
            success=bool(env.episode_succeeded() and not env.episode_fell()),
            fell=bool(env.episode_fell()),
            **metrics.result(),
            handoffs=handoffs,
            transitions=ctl.transitions,
        )
    finally:
        env.close()


def audit(seed, archive):
    from bigym.loco import make

    env = make("move_plate", camera_keys=())
    try:
        ts = env.reset(seed=seed)
        truth = PrivilegedTaskState(env)
        metrics = LiftMetrics(truth(ts.low_dim_obs)["plate_xyz"][2])
        with np.load(archive) as f:
            for action in f["actions"]:
                if ts.last():
                    raise ValueError("Official replay terminated early")
                metrics.observe(truth(ts.low_dim_obs))
                ts = env.step(action)
        metrics.observe(truth(ts.low_dim_obs))
        if not ts.last():
            raise ValueError("Official replay did not terminate")
        return dict(
            **metrics.result(),
            success=bool(env.episode_succeeded() and not env.episode_fell()),
            fell=bool(env.episode_fell()),
        )
    finally:
        env.close()


def handoff_geometry(seed, archive):
    """Measure the grasp frame after pickup by replaying a completed official run."""
    from .kinematics import G1Kinematics, pelvis_rotation
    from .orientation_feedback import LEFT_GRASP_NORMAL
    from .perception_audit import replay_truth

    with np.load(archive) as file:
        actions = file["actions"]
    truth, states, outcome = replay_truth(seed, actions)
    if not outcome["terminal"]:
        raise ValueError("Incomplete action archive")
    held = truth[:, 10].astype(bool)
    lifted = np.flatnonzero(held & (truth[:, 2] > truth[0, 2] + 0.075))
    if not len(lifted):
        return dict(seed=seed, **outcome, grasp_angle=None)
    kin = G1Kinematics()
    offsets, normals, errors = [], [], []
    # One half-second of loaded geometry immediately after the first lift.
    for step in lifted[lifted < lifted[0] + 25][::3]:
        state = states[step]
        basis = pelvis_rotation(state)
        kin.set_state(state)
        pos, rot = kin.hand_pose("left")
        rot = basis @ rot
        hand = state[18:21] + basis @ pos
        offsets.append(rot.T @ (truth[step, :3] - hand))
        normal = rot.T @ truth[step, 6:9]
        normals.append(normal)
        errors.append(np.rad2deg(np.arccos(np.clip(normal @ LEFT_GRASP_NORMAL, -1, 1))))
    opened = np.flatnonzero((np.arange(len(actions)) > lifted[0]) & (actions[:, 18] < 0))
    event = opened[0] if len(opened) else -1
    local_normal = np.mean(normals, axis=0)
    released = np.flatnonzero(~held[event:]) if len(opened) else []
    return dict(
        seed=seed,
        **outcome,
        grasp_angle=float(np.median(errors)),
        normal_in_hand=(local_normal / np.linalg.norm(local_normal)).tolist(),
        center_in_hand=np.median(offsets, axis=0).tolist(),
        opened=bool(len(opened)),
        frames_to_physical_release=int(released[0]) if len(released) else None,
        reacquired_after_open=bool(len(released) and held[event + released[0] + 1 :].any()),
        held_at_open_or_end=bool(held[event]),
        on_rack_at_open_or_end=bool(truth[event, 12]),
        tilt_at_open_or_end=float(truth[event, 9]),
        final_distance=float(np.linalg.norm(truth[-1, :3] - truth[-1, 3:6])),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("seed", type=int)
    p.add_argument("variant", choices=("base", "pickup", "audit", "handoff"))
    p.add_argument("--archive")
    a = p.parse_args()
    if a.variant in ("audit", "handoff"):
        result = (audit if a.variant == "audit" else handoff_geometry)(a.seed, a.archive)
    else:
        result = hybrid(a.seed, a.variant)
    print(json.dumps(result))
