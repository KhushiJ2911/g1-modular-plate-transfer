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
PREGRASP_OFFSET = np.array([-0.060, 0.016, 0.0])
STANDOFF_XY = np.array([0.50, 0.235])
LIFT_M = 0.16


def _rotation_from_vector(vector):
    quaternion = np.empty(4)
    rotation = np.empty(9)
    mujoco.mju_axisAngle2Quat(quaternion, vector / np.linalg.norm(vector), np.linalg.norm(vector))
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
        self.grasp_q = None
        self.grasp_streak = 0
        self.lost_streak = 0
        self.lift_origin = None
        self.plate_to_hand = None
        self.carry_rel = None
        self.release_xyz = None
        self.retreat_pose = None
        self.stable_steps = 0
        self.place_correction = np.zeros(3)

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
        rot = task["plate_rot"]
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
            q = self.kin.ik(
                "left",
                basis @ (position - state[layout.PELVIS_XYZ]),
                basis @ rotation,
                q_init=raw[layout.LEFT_ARM_ACTION],
                rot_weight=0.14,
                iters=10,
            )
            arms = raw[layout.LEFT_ARM_ACTION]
            arms += np.clip(q - arms, -0.035, 0.035)
        self.last_raw = raw
        return layout.normalize_action(raw)

    def _ready_to_release(self, task):
        contact = task["contacts"]
        return (
            task["held"] >= 0.5
            and contact["target_rack"]
            and not contact["table"]
            and not contact["floor"]
            and np.linalg.norm(task["plate_xyz"] - self.target_xyz) <= 0.043
            and task["tilt_deg"] <= 16
        )

    def _enter(self, task, hand_world, hand_rot):
        phase = self.phase
        if phase in (Phase.PREGRASP, Phase.GRASP):
            self.grasp_q = None
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
        elif phase == Phase.ALIGN:
            self.place_correction.fill(0)
        elif phase == Phase.RELEASE:
            self.release_xyz = np.array(task["plate_xyz"])
        elif phase == Phase.SETTLE:
            self.stable_steps = 0
        elif phase == Phase.RETREAT:
            self.release_xyz = np.array(task["plate_xyz"])
            away = -self.target_rot[:, 0]
            self.retreat_pose = hand_world + 0.080 * away + [0, 0, 0.025], hand_rot

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
            base_goal = plate[:2] - STANDOFF_XY
            ready = (
                np.linalg.norm(base_goal - state[layout.PELVIS_XY]) < 0.023
                and np.linalg.norm(state[layout.PELVIS_LINEAR_VEL][:2]) < 0.075
                and self.machine.elapsed >= 15
            )
        elif phase in (Phase.PREGRASP, Phase.GRASP):
            position, rotation = self._grasp_target(task, phase)
            distance = np.linalg.norm(position - hand_world)
            angle = np.linalg.norm(rotation_vector(rotation @ hand_rot.T))
            reached = distance < 0.012 and angle < 0.17
            if phase == Phase.PREGRASP:
                ready = reached and self.machine.elapsed >= 12
            else:
                if self.grasp_q is None and reached:
                    self.grasp_q = self.last_raw[layout.LEFT_ARM_ACTION].copy()
                if self.grasp_q is not None:
                    position = None
                    close = True
                self.grasp_streak = self.grasp_streak + 1 if held and gripper >= 0.65 else 0
                ready = self.grasp_streak >= 4
        elif phase == Phase.LIFT:
            ramp = min((self.machine.elapsed + 1) / 65, 1)
            desired = self.lift_origin + ramp * np.array([-0.035, 0, LIFT_M])
            position, rotation = self._hand_from_plate(desired, self.lift_rot)
            close = True
            ready = (
                held
                and plate[2] - self.lift_origin[2] >= 0.078
                and not task["contacts"]["source_rack"]
            )
        elif phase == Phase.CARRY:
            base_goal = self.target_xyz[:2] - STANDOFF_XY
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
                height -= min((self.machine.elapsed + 1) / 100, 1) * 0.135
            desired = self.target_xyz + [0, 0, height]
            # Loaded arm joints sag under constant targets. Integrate the
            # measured plate error slowly, within a bounded IK offset.
            self.place_correction = np.clip(
                self.place_correction + 0.035 * (desired - plate),
                [-0.045, -0.045, -0.075],
                [0.045, 0.045, 0.095],
            )
            position, rotation = self._hand_from_plate(
                desired + self.place_correction, self.target_rot
            )
            close = True
            ready = (
                np.linalg.norm(plate - desired) < 0.023 and task["tilt_deg"] < 14
                if phase == Phase.ALIGN
                else self._ready_to_release(task)
            )
        elif phase == Phase.RELEASE:
            ready = bool(np.all(state[layout.GRIPPERS] <= 0.08) and not held)
        elif phase == Phase.SETTLE:
            open_and_clear = np.all(state[layout.GRIPPERS] <= 0.08) and not held
            self.stable_steps = self.stable_steps + 1 if open_and_clear else 0
            ready = (
                self.stable_steps >= SETTLE_STEPS
                and task["contacts"]["target_rack"]
                and np.linalg.norm(plate - self.release_xyz) < 0.025
            )
        elif phase == Phase.RETREAT:
            position, rotation = self.retreat_pose
            ready = np.linalg.norm(hand_world - position) < 0.014
            unsafe = (
                held
                or not task["contacts"]["target_rack"]
                or np.linalg.norm(plate - self.release_xyz) > 0.015
                or task["tilt_deg"] > 20
            )
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
