"""Pure observation-driven supervisor. No simulator imports or privileged inputs."""

import json
import math
import os
import sys

import numpy as np

# Durations are control steps (50 Hz), measured between actual observations.
# These are unvalidated dev starting points, not a claimed performance result.
SEATING_AND_REGRIP_DEFAULTS = {
    "seat_cue": False,
    "seat_cue_only": False,
    "seat_steps": 5,
    "seat_error_rad": 0.024,
    "regrip": False,
}
DEFAULT_CONFIG = {
    "exec_steps": 2,
    "release_distance_m": 0.035,
    "release_tilt_deg": 12.0,
    "stable_steps": 6,
    **SEATING_AND_REGRIP_DEFAULTS,
    "held_threshold": 0.7,
    "held_steps": 4,
    "lost_steps": 10,
    "place_distance_m": 0.08,
    "place_servo": False,
    "place_kp": 2.0,  # 1/s
    "place_vmax": 0.10,  # m/s, per planar axis
    "tilt_hold": False,
    "arm_speed_rad_s": 0.4,
    "base_speed_m_s": 0.04,
    "gripper_closed_min": 0.2,
    "grasp_timeout_steps": 100,
    "reopen_steps": 10,
    "settle_steps": 75,
    "temporal_ensemble": False,
    # Ablation control: False = return bce1 actions unchanged (same exec cadence, no supervisor).
    "supervise": True,
    "ensemble_decay": 0.01,
}


def config_from_env(environ=None):
    environ = os.environ if environ is None else environ
    config = DEFAULT_CONFIG.copy()
    for key, default in config.items():
        value = environ.get("ADV_H_" + key.upper())
        if value is None:
            continue
        if isinstance(default, bool):
            if value.lower() not in ("true", "false", "1", "0"):
                raise ValueError(f"ADV_H_{key.upper()} must be true/false/1/0")
            config[key] = value.lower() in ("true", "1")
        else:
            config[key] = type(default)(value)
    return validate_config(config)


def validate_config(config):
    config = SEATING_AND_REGRIP_DEFAULTS | dict(config)
    if set(config) != set(DEFAULT_CONFIG):
        raise ValueError("Harness config keys must match DEFAULT_CONFIG")
    for key, value in config.items():
        default = DEFAULT_CONFIG[key]
        if type(value) is not type(default) or not math.isfinite(value):
            raise ValueError(f"Invalid harness {key} type/value")
        if type(default) is not bool and value < 0:
            raise ValueError(f"Harness {key} must be nonnegative")
        if type(default) is int and value < 1:
            raise ValueError(f"Harness {key} must be positive")
    if config["exec_steps"] > 50:
        raise ValueError("exec_steps must be <= 50")
    if not 0 < config["held_threshold"] <= 1 or not 0 <= config["gripper_closed_min"] <= 1:
        raise ValueError("Invalid probability/gripper thresholds")
    if not 0 < config["release_distance_m"] <= config["place_distance_m"]:
        raise ValueError("Require 0 < release_distance_m <= place_distance_m")
    if not 0 < config["release_tilt_deg"] <= 180:
        raise ValueError("release_tilt_deg must be in (0,180]")
    if config["seat_error_rad"] <= 0:
        raise ValueError("Require a positive seat_error_rad")
    if config["seat_cue_only"] and not config["seat_cue"]:
        raise ValueError("seat_cue_only requires seat_cue")
    return config


# Fixed robot calibration from the committed action_layout.json; absolute targets.
ARM_LOW = np.array(
    [-3.0892, -1.5882, -2.618, -1.0472, -1.972219944, -1.614429951, -1.614429951] * 2, np.float32
)
ARM_HIGH = np.array(
    [2.6704, 2.2515, 2.618, 2.0944, 1.972219944, 1.614429951, 1.614429951] * 2, np.float32
)
LEFT_WRIST_PITCH = 5  # qpos and arm offset; absolute target is action[4 + 5]

# Adapted from SWIRL-Lab's BiGym2 Move Plate Astra s1 and Opus s3 (CC BY 4.0).
# Source links, line references and changes are in method/harness/BRIEF18.md.
REGRIP_CHECK_STEPS = 15  # from the first full-close command
REGRIP_STEPS = 100


def pose_hold_action(state):
    action = np.zeros(20, np.float32)
    action[2] = 2 * (state[20] - 0.4) / 0.6 - 1
    joints = np.r_[state[:7], state[9:16]]
    action[4:18] = 2 * (joints - ARM_LOW) / (ARM_HIGH - ARM_LOW) - 1
    action[18:] = -1  # open both grippers, zero base/yaw velocities
    return np.clip(action, -1, 1)


class ReleaseSupervisor:
    def __init__(self, config=None, log=True):
        self.config = validate_config(DEFAULT_CONFIG if config is None else config)
        self.log = log
        self.reset()

    def reset(self):
        self.phase = "APPROACH"
        self.time = 0
        self.last_time = None
        self.phase_since = 0
        self.streaks = {}
        self.reopen_until = 0
        self.release_time = None
        self.hold = None
        self.settled = False
        self.place_wrist_target = None
        self.regrip_check_after = None
        self.regrip_pose = None  # retained after the retry to enforce once per episode
        self.regrip_origin = None

    def transition(self, phase, reason):
        previous = self.phase
        self.phase, self.phase_since = phase, self.time
        self.streaks.clear()
        self.regrip_check_after = None
        if self.log:
            print(
                f"[harness t={self.time}] {previous}>{phase} {reason}", file=sys.stderr, flush=True
            )

    def streak(self, key, condition, elapsed):
        previous, duration = self.streaks.get(key, (False, 0))
        duration = duration + elapsed if condition and previous else 0
        self.streaks[key] = (bool(condition), duration)
        return duration

    def regrip_actions(self, state, actions):
        # Astra holds the arm pose, opens, advances toward world +x, then ramps closed.
        steps = self.time - self.phase_since + np.arange(len(actions))
        actions[:] = self.regrip_pose
        actions[:, 18] = 2 * np.clip((steps - 60) / 25, 0, 1) - 1
        advance = (10 <= steps) & (steps < 60)
        x, y, yaw = self.regrip_origin
        delta = np.array([x + 0.035, y]) - state[18:20]
        velocity = np.clip(2 * delta, -0.09, 0.09)
        velocity[np.abs(delta) < 0.012] = 0
        speed = np.linalg.norm(velocity)
        if 0 < speed < 0.075:
            velocity *= 0.075 / speed
        heading = state[21]
        cos, sin = np.cos(heading), np.sin(heading)
        actions[advance, :2] = (
            cos * velocity[0] + sin * velocity[1],
            -sin * velocity[0] + cos * velocity[1],
        )
        yaw_error = (yaw - heading + np.pi) % (2 * np.pi) - np.pi
        actions[advance, 3] = np.clip(3 * yaw_error, -0.4, 0.4)

    def step(self, estimate, state, bc_actions):
        """One decision per fresh observation; never count a chunk's future as evidence."""
        state = np.asarray(state, np.float32)
        actions = np.asarray(bc_actions, np.float32).copy()
        if state.shape != (50,) or not np.isfinite(state).all():
            raise ValueError("Expected finite (50,) proprio state")
        if (
            actions.ndim != 2
            or actions.shape[1] != 20
            or not 1 <= len(actions) <= 50
            or not np.isfinite(actions).all()
        ):
            raise ValueError("Expected finite (H,20) BC actions, H in [1,50]")
        plate = np.asarray(estimate["plate_rel"], np.float32)
        site = np.asarray(estimate["target_site_rel"], np.float32)
        tilt, probability = float(estimate["tilt_deg"]), float(estimate["held"])
        if (
            plate.shape != (3,)
            or site.shape != (3,)
            or not np.isfinite(np.r_[plate, site, tilt, probability]).all()
            or not 0 <= probability <= 1
        ):
            raise ValueError("Invalid observation-derived estimate")
        elapsed = 0 if self.last_time is None else self.time - self.last_time
        self.last_time = self.time
        c = self.config
        held = probability >= c["held_threshold"]
        distance = float(np.linalg.norm(plate - site))
        reason = f"d={distance:.3f} tilt={tilt:.1f} held={probability:.2f}"

        if self.phase == "RELEASE":
            self.transition("SETTLE", "open latched; hold observed release pose")
        elif self.phase == "REGRIP":
            if self.time - self.phase_since >= REGRIP_STEPS:
                self.transition("GRASP", "regrip complete; confirm hold")
        elif self.phase == "APPROACH":
            # Only grasp initiation uses bce1's proposal. Recovery first opens for
            # a short observation-triggered interval, then permits a new grasp.
            if self.time >= self.reopen_until and actions[0, 18] > 0:
                self.transition("GRASP", "bce1 close proposal")
        elif self.phase == "GRASP":
            confirmed = self.streak("held", held and state[44] >= c["gripper_closed_min"], elapsed)
            if confirmed >= c["held_steps"]:
                check_after = self.phase_since + REGRIP_CHECK_STEPS
                self.transition("CARRY", reason)
                if c["regrip"] and self.regrip_pose is None:
                    self.regrip_check_after = check_after
            elif self.time - self.phase_since >= c["grasp_timeout_steps"]:
                self.reopen_until = self.time + c["reopen_steps"]
                self.transition("APPROACH", "grasp timeout; reopen")
        elif self.phase in ("CARRY", "PLACE"):
            missing = self.streak("lost", not held, elapsed)
            if missing >= c["lost_steps"]:
                self.reopen_until = self.time + c["reopen_steps"]
                self.transition("APPROACH", "lost plate; reopen")
            elif self.phase == "CARRY" and distance <= c["place_distance_m"]:
                self.transition("PLACE", reason)
            elif self.regrip_check_after is not None and self.time >= self.regrip_check_after:
                # A single check near grasp confirmation avoids reopening later in transport.
                self.regrip_check_after = None
                fingers = state[7:9]
                shallow = np.max(fingers) < 0.004 or np.sum(fingers) < -0.023
                if held and state[44] >= c["gripper_closed_min"] and shallow:
                    self.regrip_pose = pose_hold_action(state)
                    self.regrip_origin = state[[18, 19, 21]]
                    self.transition("REGRIP", "shallow left pinch; reopen and advance")
            elif self.phase == "PLACE":
                quiet = (
                    np.max(np.abs(state[22:29])) <= c["arm_speed_rad_s"]
                    and np.linalg.norm(state[40:42]) <= c["base_speed_m_s"]
                )
                safe = (
                    held
                    and distance <= c["release_distance_m"]
                    and 0 <= tilt <= c["release_tilt_deg"]
                    and quiet
                )
                if c["seat_cue_only"]:
                    safe = held and quiet and distance <= c["place_distance_m"]
                stable = self.streak("safe", safe, elapsed)
                seated = True
                if c["seat_cue"]:
                    # Unlike Astra's nominal pose check, require a measurable tracking lag.
                    target = self.place_wrist_target
                    tracking_error = target is not None and (
                        abs(target - state[LEFT_WRIST_PITCH]) >= c["seat_error_rad"]
                    )
                    seated = (
                        self.streak("seat", safe and tracking_error, elapsed) >= c["seat_steps"]
                    )
                if stable >= c["stable_steps"] and seated:
                    self.release_time = self.time
                    self.hold = pose_hold_action(state)
                    self.transition("RELEASE", reason + f" stable={stable}")
                elif distance > c["place_distance_m"]:
                    self.transition("CARRY", "left placement region")

        if self.phase in ("RELEASE", "SETTLE"):
            actions[:] = self.hold
            if self.time - self.release_time >= c["settle_steps"] and not self.settled:
                self.settled = True
                if self.log:
                    print(
                        f"[harness t={self.time}] settle period elapsed; open/hold latched",
                        file=sys.stderr,
                        flush=True,
                    )
            # Continue holding even after the minimum settle period: no success
            # flag is available, so never guess that a re-grasp is appropriate.
        elif self.phase == "REGRIP":
            self.regrip_actions(state, actions)
        else:
            if self.phase == "PLACE" and held and distance <= c["place_distance_m"]:
                if c["place_servo"]:
                    # Labels already use R(-yaw) * (world_xyz - pelvis_xyz).
                    # Move the held plate toward the site, not away from it.
                    velocity = np.clip(
                        c["place_kp"] * (site[:2] - plate[:2]), -c["place_vmax"], c["place_vmax"]
                    )
                    # Fixed planar raw limits [-1,1] m/s: teacher's
                    # 2*(velocity-low)/(high-low)-1 is identity here.
                    actions[:, :2] = velocity
                if c["tilt_hold"] and tilt > c["release_tilt_deg"]:
                    actions[:, [0, 1, 3]] = 0  # recheck next observation; retain BC arms/height
            actions[:, 18] = -1 if self.phase == "APPROACH" else 1
            actions[:, 19] = -1  # bce1 is a left-hand policy; keep unused hand open
        self.time += len(actions)
        actions = np.clip(actions, -1, 1).astype(np.float32)
        if c["seat_cue"] and self.phase == "PLACE":
            i = LEFT_WRIST_PITCH
            self.place_wrist_target = ARM_LOW[i] + (actions[-1, 4 + i] + 1) / 2 * (
                ARM_HIGH[i] - ARM_LOW[i]
            )
        return actions


def log_config(config):
    print("[harness config] " + json.dumps(config, sort_keys=True), file=sys.stderr, flush=True)
