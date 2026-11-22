"""Conservative late release recovery for the calibrated plate controller."""

from collections import deque

import numpy as np

from . import layout
from .controller import ModularController
from .kinematics import pelvis_rotation
from .phases import SETTLE_STEPS, Phase
from .recontact import OpenFingerContact
from .site_seating import SupportReferencedController

NEAR_FRAMES = 24
# The nominal insertion ramps for 125 steps. Continued integral correction
# under load can move the hand relative to an already seated plate.
LATE_LOWER = 125
MAX_STALL_AGE = 24
LATE_SETTLE = SETTLE_STEPS + 16


class FallbackSeatingController(SupportReferencedController):
    """Leave nominal actions intact until a late, well-supported height disagreement."""

    fk_height_for_motion = False
    correct_site_on_carry = False
    hold_measured_release = False

    def reset(self):
        super().reset()
        self.near_frames = 0
        self.stall_step = None
        self.proprio_arming = False
        self.fallback_releases = []
        self.fallback_settles = []
        self.low_site_at_release = False
        self.supported_depths = deque(maxlen=NEAR_FRAMES)
        self.seating_state = None
        self.release_contact = OpenFingerContact() if self.hold_measured_release else None
        self.open_finger_contact = False

    def act(self, obs_state, task_state=None):
        self.seating_state = np.asarray(obs_state) if self.phase == Phase.LOWER else None
        if self.release_contact is not None and self.phase in (
            Phase.RELEASE,
            Phase.SETTLE,
            Phase.RETREAT,
        ):
            self.open_finger_contact = self.release_contact.update(np.asarray(obs_state))
        return super().act(obs_state, task_state)

    def _action(self, state, position=None, rotation=None, base=None, close=False):
        if self.hold_measured_release and self.phase == Phase.RELEASE and self.machine.elapsed == 1:
            # Unload the accumulated IK error before opening. Subsequent
            # RELEASE/SETTLE actions retain this single measured target.
            arm = layout.LEFT_ARM_ACTION
            self.last_raw[arm] = np.clip(
                state[layout.LEFT_ARM], layout.ACTION_LOW[arm], layout.ACTION_HIGH[arm]
            )
        return super()._action(state, position, rotation, base, close)

    def _use_supported_handoff(self):
        return bool(self.fallback_releases or self.fallback_settles)

    def _collect_handoff_samples(self):
        return self._use_supported_handoff() or self.low_site_at_release

    def _enter(self, task, hand_world, hand_rot):
        if self.phase == Phase.LOWER:
            self.near_frames = 0
            self.stall_step = None
            self.proprio_arming = False
            self.supported_depths.clear()
        elif self.phase == Phase.RELEASE and self.proprio_arming:
            self._correct_low_site()
            plate = task["plate_xyz"].copy()
            plate[2] = self.fk_plate_z
            task = {**task, "plate_xyz": plate}
            self.fallback_releases.append(
                {"step": self.machine.step, "site_adjustment_m": self.low_site_adjustment}
            )
        elif self.phase == Phase.RELEASE:
            reference = self._source_site_height()
            if reference is not None and self.fk_plate_z is not None:
                self.low_site_at_release = -0.052 <= self.fk_plate_z - reference <= -0.017
                if self.low_site_at_release:
                    self.release_z = self.fk_plate_z
                    self.settle_samples.clear()
        super()._enter(task, hand_world, hand_rot)

    def _ready_to_retreat(self, task):
        if self.open_finger_contact:
            return False
        if super()._ready_to_retreat(task):
            return True
        if (
            self.low_site_at_release
            and self.open_steps >= LATE_SETTLE
            and self._stable_on_support(task)
        ):
            self.fallback_settles.append({"step": self.machine.step})
            return True
        return False

    def _unsafe_retreat(self, state, task):
        return self.open_finger_contact or super()._unsafe_retreat(state, task)

    def _ready_to_release(self, task):
        contacts = task["contacts"]
        delta = task["plate_xyz"] - self.target_xyz
        near = (
            task["held"]
            and not contacts["table"]
            and not contacts["floor"]
            and np.linalg.norm(delta[:2]) < 0.032
            and -0.048 <= delta[2] <= 0.022
        )
        self.near_frames = self.near_frames + 1 if near else 0
        if self.descent.seated():
            self.stall_step = self.machine.step
        supported = self._supported_seat(task, delta)

        if ModularController._ready_to_release(self, task):
            if self.proprio_arming:
                # The independent fallback streak must not advance the
                # nominal eight-frame confirmation.
                self.placement_streak = 0
            self.proprio_arming = False
            return True

        regular_gate_in_progress = self.placement_streak and not self.proprio_arming
        self.proprio_arming = not regular_gate_in_progress and (
            self._ready_for_fallback(task, delta) or supported
        )
        return self.proprio_arming

    def _supported_seat(self, task, delta):
        """Corroborate continuous rack contact with rigid-grasp or load geometry."""
        state = self.seating_state
        contacts = task["contacts"]
        if (
            state is None
            or self.fk_plate_z is None
            or self.lower_start_z is None
            or not task["held"]
            or not contacts["target_rack"]
            or contacts["table"]
            or contacts["floor"]
            or state[layout.GRIPPERS][0] < 0.8
            or np.mean(state[layout.LEFT_FINGERS]) < -0.0185
            or task["tilt_deg"] > 12
            or not -0.060 <= delta[2] <= 0.016
            or self.lower_start_z - self.measured_hand_z < 0.065
        ):
            self.supported_depths.clear()
            return False

        near = np.linalg.norm(delta[:2]) <= 0.037
        if not near:
            # Wrist occlusion can move the CNN center laterally after seating.
            hand, rot = self.kin.hand_pose("left")
            offset = self.hand_to_center
            if self.loaded_center_bias is not None:
                offset = offset - self.loaded_center_bias
            rigid = state[layout.PELVIS_XYZ] + pelvis_rotation(state) @ (hand + rot @ offset)
            near = (
                np.linalg.norm(rigid[:2] - self.target_xyz[:2]) <= 0.040
                and np.linalg.norm(rigid[:2] - task["plate_xyz"][:2]) <= 0.060
            )
        reference = self._source_site_height()
        if reference is None:
            reference = self.target_xyz[2]
        depth = self.fk_plate_z - reference
        if not (near and -0.065 <= depth <= 0.010):
            self.supported_depths.clear()
            return False
        self.supported_depths.append(float(delta[2]))
        if self.machine.elapsed < LATE_LOWER or len(self.supported_depths) < NEAR_FRAMES:
            return False

        # If the rigid center rides high, require a stationary visible plate
        # and the measured hand to be resisting the commanded descent.
        return depth <= -0.016 or (
            self.commanded_hand_z is not None
            and self.measured_hand_z - self.commanded_hand_z >= 0.012
            and np.ptp(self.supported_depths) <= 0.012
        )

    def _ready_for_fallback(self, task, delta):
        if (
            self.machine.elapsed < LATE_LOWER
            or self.near_frames < NEAR_FRAMES
            or self.fk_plate_z is None
            or self.lower_start_z is None
            or self.measured_hand_z is None
            or self.lower_start_z - self.measured_hand_z < 0.065
            or task["tilt_deg"] > 12
            or delta[2] <= -0.018
        ):
            return False

        source_reference = self._source_site_height()
        reference = self.target_xyz[2] if source_reference is None else source_reference
        fk_depth = self.fk_plate_z - reference
        disagreement = task["plate_xyz"][2] - self.fk_plate_z >= 0.012
        stalled = (
            self.stall_step is not None and self.machine.step - self.stall_step <= MAX_STALL_AGE
        )
        return (
            -0.052 <= fk_depth <= -0.017
            and (disagreement or source_reference is not None)
            and (task["contacts"]["target_rack"] or stalled)
        )


def make_policy():
    """Build bd951bc's calibrated policy with an opt-in late release path."""
    from .fk_seating import make_policy as make_fk_policy

    return make_fk_policy(FallbackSeatingController)
