"""Use the common support plane and settled plate pose during rack handoff."""

from collections import deque

import numpy as np

from . import layout
from .fk_seating import FKSeatingController
from .phases import SETTLE_STEPS, Phase

# The released fixtures use identical dish racks mounted at the same height.
# A seated plate center is about 3 cm below the rack's slot reference
# (pooled generation trajectories 3110000–3110299, not a per-slot value).
SITE_ABOVE_SEATED_CENTER = 0.032


class SupportReferencedController(FKSeatingController):
    """Check a low destination height against the stationary source plate."""

    def reset(self):
        super().reset()
        self.release_z = None
        self.settle_samples = deque(maxlen=9)
        self.low_site_adjustment = 0.0

    def _enter(self, task, hand_world, hand_rot):
        super()._enter(task, hand_world, hand_rot)
        if self.phase == Phase.CARRY and self.initial_samples:
            # A retry may have moved the plate. The first stationary frames
            # preserve the original seated height on the source rack.
            source_z = float(np.median(np.asarray(self.initial_samples)[:, 2]))
            source_reference = source_z + SITE_ABOVE_SEATED_CENTER
            low_by = float(source_reference - self.target_xyz[2])
            # Trust small disagreements between the two camera measurements.
            if 0.012 < low_by < 0.045:
                self.target_xyz = self.target_xyz.copy()
                self.target_xyz[2] = source_reference
                self.low_site_adjustment = low_by
        elif self.phase == Phase.RELEASE:
            self.release_z = float(task["plate_xyz"][2])
            self.settle_samples.clear()
        elif self.phase == Phase.RETREAT and len(self.settle_samples) == self.settle_samples.maxlen:
            settled = np.median(self.settle_samples, axis=0)
            self.retreat_start = settled[:3]
            self.retreat_tilt = float(settled[3])

    def act(self, obs_state, task_state=None):
        state = np.asarray(obs_state)
        task = self._read_task(state, task_state)
        if self.phase == Phase.SETTLE:
            contacts = task["contacts"]
            if (
                np.all(state[layout.GRIPPERS] <= 0.08)
                and contacts["target_rack"]
                and not (contacts["table"] or contacts["floor"])
            ):
                self.settle_samples.append(np.r_[task["plate_xyz"], task["tilt_deg"]])
            else:
                self.settle_samples.clear()
        return super().act(state, task)

    def _ready_to_retreat(self, task):
        contacts = task["contacts"]
        if contacts["table"] or contacts["floor"]:
            return False
        if super()._ready_to_retreat(task):
            return True
        if (
            self.open_steps < SETTLE_STEPS
            or self.release_z is None
            or len(self.settle_samples) != self.settle_samples.maxlen
        ):
            return False
        samples = np.asarray(self.settle_samples)
        settled = np.median(samples, axis=0)
        return (
            contacts["target_rack"]
            and np.linalg.norm(settled[:2] - self.target_xyz[:2]) < 0.032
            and abs(settled[2] - self.release_z) < 0.025
            and settled[3] < 17
            and task["tilt_deg"] < 19
            and np.linalg.norm(np.ptp(samples[:, :3], axis=0)) < 0.008
        )

    def _unsafe_retreat(self, state, task):
        contacts = task["contacts"]
        return super()._unsafe_retreat(state, task) or contacts["table"] or contacts["floor"]


def make_policy():
    """Build the calibrated FK policy with the shared-plane check and guarded handoff."""
    from .fk_seating import make_policy as make_fk_policy

    return make_fk_policy(SupportReferencedController)
