"""Recover the original destination standoff after a loaded insertion pushes the base back."""

import os

import numpy as np

from method.modular import layout
from method.modular.controller import TARGET_STANDOFF_XY
from method.modular.phases import Phase
from method.modular.pickup import PickupController


class RetryStandoff:
    """Compose before a placement controller in the MRO. Its release gate is untouched."""

    def reset(self):
        super().reset()
        self.retry_hand_z = None
        self.stance_goal = None

    def _enter(self, task, hand_world, hand_rot):
        super()._enter(task, hand_world, hand_rot)
        self.retry_hand_z = None
        if self.phase == Phase.ALIGN and self.machine.transitions[-1]["from"] == Phase.LOWER.value:
            self.retry_hand_z = hand_world[2]

    def act(self, obs_state, task_state=None):
        previous_goal = self.stance_goal
        self.stance_goal = None
        if self.phase == Phase.ALIGN and self.retry_hand_z is not None:
            state = np.asarray(obs_state)
            goal = self.target_xyz[:2] - TARGET_STANDOFF_XY
            displaced = np.linalg.norm(goal - state[layout.PELVIS_XY]) >= 0.025
            braking = (
                previous_goal is not None
                and np.linalg.norm(state[layout.PELVIS_LINEAR_VEL][:2]) >= 0.075
            )
            if displaced or braking:
                self.kin.set_state(state)
                hand, _ = self.kin.hand_pose("left")
                height = state[layout.PELVIS_HEIGHT] + hand[2]
                # Let the normal retry raise the hand by the controller's CARRY clearance.
                if height >= self.retry_hand_z + 0.065:
                    self.stance_goal = goal if displaced else state[layout.PELVIS_XY]
        return super().act(obs_state, task_state)

    def _action(self, state, position=None, rotation=None, base=None, close=False):
        if self.phase == Phase.ALIGN and self.stance_goal is not None:
            base = self.stance_goal
        return super()._action(state, position, rotation, base, close)

    def _ready_to_lower(self, task, desired):
        if not task["held"] or task["contacts"]["target_rack"]:
            self.stance_goal = None
        return self.stance_goal is None and super()._ready_to_lower(task, desired)


class HoverController(RetryStandoff, PickupController):
    pass


def make_policy():
    from submission.policy_modular import make_policy as make_base
    from submission.policy_modular_heads import PerceptionPolicy

    checkpoint = os.environ["ADV_HEADS"]
    policy = PerceptionPolicy(make_base(), checkpoint, HoverController)
    with np.load(checkpoint, allow_pickle=False) as data:
        policy.controller.set_loaded_grasp_normal(data["loaded_normal"])
    return policy
