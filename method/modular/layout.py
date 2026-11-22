"""Release layout. All slices are half-open and use the evaluator's order.

Qpos[0:22] and qvel[22:44] have the same joint order, with radians for
hinges, meters for slides, and velocities per second. Base x/y are world
axes. The last four entries repeat measured base coordinates. Gripper
closure is a scalar per hand (0 open, 1 closed), distinct from finger slides.
"""

import json
from importlib.resources import files

import numpy as np

STATE_DIM = 50
ACTION_DIM = 20
LEFT_ARM = slice(0, 7)
LEFT_FINGERS = slice(7, 9)
RIGHT_ARM = slice(9, 16)
RIGHT_FINGERS = slice(16, 18)
PELVIS_XY = slice(18, 20)
PELVIS_XYZ = slice(18, 21)
PELVIS_HEIGHT = 20
PELVIS_YAW = 21
LEFT_ARM_VEL = slice(22, 29)
LEFT_FINGERS_VEL = slice(29, 31)
RIGHT_ARM_VEL = slice(31, 38)
RIGHT_FINGERS_VEL = slice(38, 40)
PELVIS_LINEAR_VEL = slice(40, 43)
PELVIS_YAW_RATE = 43
GRIPPERS = slice(44, 46)
PELVIS_POSE_DUPLICATE = slice(46, 50)

BASE_VELOCITY_ACTION = slice(0, 2)  # forward, left, m/s in the yaw-aligned pelvis frame
HEIGHT_ACTION = 2
YAW_RATE_ACTION = 3
LEFT_ARM_ACTION = slice(4, 11)
RIGHT_ARM_ACTION = slice(11, 18)
GRIPPER_ACTION = slice(18, 20)

_release = json.loads(files("advaitic_eval").joinpath("action_layout.json").read_text())
JOINT_NAMES = tuple(_release["limb_names"])
ACTION_LOW = np.asarray(_release["action_low"], dtype=np.float32)
ACTION_HIGH = np.asarray(_release["action_high"], dtype=np.float32)


def _action(value):
    value = np.asarray(value, dtype=np.float32)
    if value.shape[-1:] != (ACTION_DIM,) or not np.isfinite(value).all():
        raise ValueError("Expected finite actions with 20 entries on the last axis")
    return value


def normalize_action(raw):
    """Clip raw targets to the release limits and map to [-1, 1]."""
    return np.clip(2 * (_action(raw) - ACTION_LOW) / (ACTION_HIGH - ACTION_LOW) - 1, -1, 1)


def denormalize_action(norm):
    return ACTION_LOW + (np.clip(_action(norm), -1, 1) + 1) * 0.5 * (ACTION_HIGH - ACTION_LOW)
