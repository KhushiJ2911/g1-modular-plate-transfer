"""Feedback controller for privileged development, with explicit pose inputs."""

from __future__ import annotations

import mujoco
import numpy as np

from . import layout
from .kinematics import pelvis_rotation, rotation_vector
from .phases import SETTLE_STEPS, Phase, PhaseMachine

# One pooled pre-close calibration, in the plate's local frame. At each
# episode, apply the same transform to the current estimated plate pose.
# Sources and extraction command are recorded in the sol development report.
GRASP_OFFSET = np.array([-0.090, 0.024, 0.009])
GRASP_ROTATION_VECTOR = np.array([-1.485, 0.140, 0.055])
PREGRASP_OFFSET = np.array([-0.080, 0.035, 0.0])
SOURCE_STANDOFF_XY = np.array([0.50, 0.205])
TARGET_STANDOFF_XY = np.array([0.46, 0.205])
TUCKED_HAND = np.array([0.27, 0.14, 0.19])
# Seed the local IK solve with the elbow bent forward during the reach.
GRASP_IK_SEED = np.array([-0.55, 0.22, 0.30, 0.80, -0.15, 0.10, 0.00])
LIFT_M = 0.16
LOWER_RAMP_STEPS = 125


def _rotation_from_vector(vector):
    quaternion = np.empty(4)
    rotation = np.empty(9)
    angle = np.linalg.norm(vector)
    if angle < 1e-9:
        return np.eye(3)
    mujoco.mju_axisAngle2Quat(quaternion, vector / angle, angle)
    mujoco.mju_quat2Mat(rotation, quaternion)
    return rotation.reshape(3, 3)


GRASP_ROTATION = _rotation_from_vector(GRASP_ROTATION_VECTOR)


def _base_velocity(state, goal_xy):
    error = goal_xy - state[layout.PELVIS_XY]
    distance = np.linalg.norm(error)
    if distance < 0.017:
        return np.zeros(2)
    speed = np.clip(2.0 * distance, 0.07, 0.24)
    return pelvis_rotation(state)[:2, :2].T @ (error * speed / distance)


class ModularController:
    """Arm targets are generated from current proprioception and task geometry."""

    def __init__(self, kin, task_state_source=None):
        self.kin = kin
        self.task_state_source = task_state_source
        self.reset()

    def reset(self):
        self.machine = PhaseMachine()
        self.last_raw = None
        self.target_xyz = None
        self.target_rot = None
        self.closing = False
        self.grasp_path_start = None
        self.grasp_bias = np.zeros(7)
        self.grasp_streak = 0
        self.lost_streak = 0
        self.lift_origin = None
        self.lift_rot = None
        self.plate_to_hand = None
        self.carry_rel = None
        self.retreat_start = None
        self.retreat_tilt = None
        self.retreat_pose = None
        self.open_steps = 0
        self.placement_streak = 0
        self.place_correction = np.zeros(3)
        self.place_rotation_bias = np.zeros(3)

    @property
    def phase(self):
        return self.machine.phase

    @property
    def transitions(self):
        return self.machine.transitions

    def _initialize(self, state, task):
        self.last_raw = np.zeros(layout.ACTION_DIM)
        self.last_raw[layout.HEIGHT_ACTION] = state[layout.PELVIS_HEIGHT]
        self.last_raw[layout.LEFT_ARM_ACTION] = state[layout.LEFT_ARM]
        self.last_raw[layout.RIGHT_ARM_ACTION] = state[layout.RIGHT_ARM]
        self.target_xyz = np.array(task["site_xyz"], dtype=float)
        self.target_rot = np.array(task["site_rot"], dtype=float)

    def _grasp_target(self, task, phase):
        normal = task["plate_rot"][:, 2]
        x = np.array([1.0, 0.0, 0.0]) - normal[0] * normal
        if np.linalg.norm(x) < 1e-5:
            x = np.array([0.0, 1.0, 0.0]) - normal[1] * normal
        x /= np.linalg.norm(x)
        rot = np.column_stack((x, np.cross(normal, x), normal))
        offset = GRASP_OFFSET.copy()
        offset[0] += (0, 0.009, -0.009)[self.machine.grasp_retries]
        if phase == Phase.PREGRASP:
            offset += PREGRASP_OFFSET
        return task["plate_xyz"] + rot @ offset, rot @ GRASP_ROTATION

    def _hand_from_plate(self, pos, rot):
        offset, rotation = self.plate_to_hand
        return pos + rot @ offset, rot @ rotation

    def _action(self, state, position=None, rotation=None, base=None, close=False):
        raw = self.last_raw.copy()
        raw[layout.BASE_VELOCITY_ACTION] = 0 if base is None else _base_velocity(state, base)
        raw[layout.YAW_RATE_ACTION] = 0
        raw[layout.GRIPPER_ACTION] = (float(close), 0.0)
        if position is not None:
            basis = pelvis_rotation(state).T
            reaching = self.phase in (Phase.APPROACH, Phase.PREGRASP, Phase.GRASP)
            q = self.kin.ik(
                "left",
                basis @ (position - state[layout.PELVIS_XYZ]),
                basis @ rotation,
                q_init=GRASP_IK_SEED if reaching else raw[layout.LEFT_ARM_ACTION],
                rot_weight=0.14,
                iters=32 if reaching else 10,
            )
            if self.phase in (Phase.PREGRASP, Phase.GRASP):
                error = np.clip(q - state[layout.LEFT_ARM], -0.08, 0.08)
                self.grasp_bias = np.clip(self.grasp_bias + 0.07 * error, -0.12, 0.12)
                q += self.grasp_bias
            arms = raw[layout.LEFT_ARM_ACTION]
            rate = 0.012 if self.phase == Phase.RETREAT else 0.035
            arms += np.clip(q - arms, -rate, rate)
        self.last_raw = raw
        return layout.normalize_action(raw)

    def _ready_to_lower(self, task, desired):
        return np.linalg.norm(task["plate_xyz"] - desired) < 0.023 and task["tilt_deg"] < 14

    def _lift_ready(self, task):
        return (
            task["held"] >= 0.5
            and task["plate_xyz"][2] - self.lift_origin[2] >= 0.078
            and not task["contacts"]["source_rack"]
        )

    def _ready_to_release(self, task):
        contact = task["contacts"]
        error = task["plate_xyz"] - self.target_xyz
        retrying = self.machine.elapsed >= 120
        seated = (
            task["held"] >= 0.5
            and contact["target_rack"]
            and not contact["table"]
            and not contact["floor"]
            and -0.047 <= error[2] <= -0.018
            and task["tilt_deg"] <= 12
        )
        centered = abs(error[0]) < 0.027 and abs(error[1]) < 0.022
        return seated and (centered or (retrying and np.linalg.norm(error) < 0.039))

    def _release_confirmed(self, task):
        self.placement_streak = self.placement_streak + 1 if self._ready_to_release(task) else 0
        return self.placement_streak >= 8

    def _ready_to_retreat(self, task):
        return (
            self.open_steps >= SETTLE_STEPS
            and task["contacts"]["target_rack"]
            and np.linalg.norm(task["plate_xyz"] - self.target_xyz) < 0.047
            and task["tilt_deg"] < 17
        )

    def _unsafe_retreat(self, state, task):
        return (
            np.any(state[layout.GRIPPERS] > 0.08)
            or not task["contacts"]["target_rack"]
            or np.linalg.norm(task["plate_xyz"] - self.retreat_start) > 0.012
            or task["tilt_deg"] > min(19, self.retreat_tilt + 4)
        )

    def _enter(self, task, hand_world, hand_rot):
        phase = self.phase
        if phase in (Phase.PREGRASP, Phase.GRASP):
            self.grasp_path_start = hand_world
            if phase == Phase.PREGRASP:
                self.grasp_bias.fill(0)
            self.closing = False
            self.grasp_streak = 0
        elif phase == Phase.LIFT:
            plate_rot = task["plate_rot"]
            self.lift_origin = np.array(task["plate_xyz"])
            self.lift_rot = np.array(plate_rot)
            self.plate_to_hand = (
                plate_rot.T @ (hand_world - self.lift_origin),
                plate_rot.T @ hand_rot,
            )
            self.lost_streak = 0
        elif phase == Phase.CARRY:
            self.carry_rel = np.array(task["plate_rel"])
            self.carry_rel[2] += 0.065
        elif phase == Phase.ALIGN:
            self.place_correction.fill(0)
            self.place_rotation_bias.fill(0)
        elif phase == Phase.LOWER:
            self.placement_streak = 0
        elif phase == Phase.SETTLE:
            self.open_steps = 0
        elif phase == Phase.RETREAT:
            self.retreat_start = np.array(task["plate_xyz"])
            self.retreat_tilt = task["tilt_deg"]
            self.retreat_pose = hand_world, hand_rot

    def act(self, obs_state, task_state=None):
        state = np.asarray(obs_state)
        if task_state is None:
            if self.task_state_source is None:
                raise ValueError("Task state or a task state source is required")
            task_state = self.task_state_source(state)
        task = task_state
        if self.last_raw is None:
            self._initialize(state, task)
        self.kin.set_state(state)
        hand, orient = self.kin.hand_pose("left")
        basis = pelvis_rotation(state)
        hand_world = state[layout.PELVIS_XYZ] + basis @ hand
        hand_rot = basis @ orient
        plate = task["plate_xyz"]
        held = task["held"] >= 0.5
        gripper = state[layout.GRIPPERS][0]
        base_goal = None
        position, rotation = None, None
        close = False
        ready = False
        unsafe = False
        phase = self.phase

        if phase == Phase.APPROACH:
            if self.machine.elapsed >= 24:
                base_goal = plate[:2] - SOURCE_STANDOFF_XY
            position = state[layout.PELVIS_XYZ] + basis @ TUCKED_HAND
            rotation = basis
            ready = (
                base_goal is not None
                and np.linalg.norm(base_goal - state[layout.PELVIS_XY]) < 0.023
                and np.linalg.norm(state[layout.PELVIS_LINEAR_VEL][:2]) < 0.075
                and self.machine.elapsed >= 15
            )
        elif phase in (Phase.PREGRASP, Phase.GRASP):
            position, rotation = self._grasp_target(task, phase)
            distance = np.linalg.norm(position - hand_world)
            blend = min((self.machine.elapsed + 1) / (28 if phase == Phase.GRASP else 22), 1)
            position = self.grasp_path_start + blend * (position - self.grasp_path_start)
            angle = np.linalg.norm(rotation_vector(rotation @ hand_rot.T))
            reached = distance < 0.018 and angle < 0.18 and blend >= 1
            if phase == Phase.PREGRASP:
                ready = reached and self.machine.elapsed >= 12
            else:
                self.closing |= bool(reached)
                if self.closing:
                    position = None
                    close = True
                self.grasp_streak = self.grasp_streak + 1 if held and gripper >= 0.65 else 0
                ready = self.grasp_streak >= 4
        elif phase == Phase.LIFT:
            ramp = min((self.machine.elapsed + 1) / 65, 1)
            desired = self.lift_origin + ramp * np.array([-0.035, 0, LIFT_M])
            position, rotation = self._hand_from_plate(desired, self.lift_rot)
            close = True
            ready = self._lift_ready(task)
        elif phase == Phase.CARRY:
            base_goal = self.target_xyz[:2] - TARGET_STANDOFF_XY
            desired = state[layout.PELVIS_XYZ] + basis @ self.carry_rel
            position, rotation = self._hand_from_plate(desired, self.target_rot)
            close = True
            ready = (
                np.linalg.norm(base_goal - state[layout.PELVIS_XY]) < 0.025
                and np.linalg.norm(state[layout.PELVIS_LINEAR_VEL][:2]) < 0.075
            )
        elif phase in (Phase.ALIGN, Phase.LOWER):
            height = 0.090
            if phase == Phase.LOWER:
                height -= min((self.machine.elapsed + 1) / LOWER_RAMP_STEPS, 1) * 0.135
            desired = self.target_xyz + [0, 0, height]
            # Loaded arm joints sag under constant targets. Integrate the
            # measured plate error slowly, within a bounded IK offset.
            self.place_correction = np.clip(
                self.place_correction + 0.035 * (desired - plate),
                [-0.08, -0.08, -0.075],
                [0.08, 0.08, 0.095],
            )
            position, rotation = self._hand_from_plate(
                desired + self.place_correction, self.target_rot
            )
            normal_error = np.cross(task["plate_rot"][:, 2], self.target_rot[:, 2])
            self.place_rotation_bias = np.clip(
                self.place_rotation_bias + 0.026 * normal_error, -0.22, 0.22
            )
            rotation = _rotation_from_vector(self.place_rotation_bias) @ rotation
            close = True
            if phase == Phase.ALIGN:
                ready = self._ready_to_lower(task, desired)
            else:
                ready = self._release_confirmed(task)
        elif phase == Phase.RELEASE:
            ready = bool(np.all(state[layout.GRIPPERS] <= 0.08))
        elif phase == Phase.SETTLE:
            both_open = np.all(state[layout.GRIPPERS] <= 0.08)
            self.open_steps = self.open_steps + 1 if both_open else 0
            ready = self._ready_to_retreat(task)
        elif phase == Phase.RETREAT:
            start, rotation = self.retreat_pose
            away = -self.target_rot[:, 0]
            travel = min(0.075, (self.machine.elapsed + 1) * 0.0015)
            position = start + travel * away
            ready = travel >= 0.075 and np.linalg.norm(hand_world - position) < 0.014
            unsafe = self._unsafe_retreat(state, task)
        elif phase == Phase.HOLD:
            close = bool(self.last_raw[layout.GRIPPER_ACTION][0])

        loss_sensitive = phase in (Phase.LIFT, Phase.CARRY, Phase.ALIGN, Phase.LOWER)
        self.lost_streak = self.lost_streak + 1 if loss_sensitive and not held else 0
        self.machine.advance(bool(ready), lost=self.lost_streak >= 10, unsafe_retreat=unsafe)
        if self.phase != phase or self.machine.elapsed == 0:
            self._enter(task, hand_world, hand_rot)
        if self.machine.elapsed == 0:
            base_goal = None
            position = None
            if self.phase in (Phase.PREGRASP, Phase.RELEASE, Phase.SETTLE, Phase.RETREAT):
                close = False
        self.machine.tick()
        return self._action(state, position, rotation, base_goal, close)
