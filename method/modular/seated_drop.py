"""Detect a held plate dropping into the rack after aligning above its rim."""

from collections import deque

import numpy as np

from .fallback_seating import LATE_LOWER
from .phases import Phase
from .site_seating import SITE_ABOVE_SEATED_CENTER


class SeatedDropMixin:
    """Corroborate a rim-to-seat transition with both vision and wrist motion."""

    def reset(self):
        super().reset()
        self.descent_heights = deque(maxlen=12)
        self.over_pocket = False
        self.drop_step = None

    def _enter(self, task, hand_world, hand_rot):
        super()._enter(task, hand_world, hand_rot)
        if self.phase == Phase.LOWER:
            self.descent_heights.clear()
            self.over_pocket = False
            self.drop_step = None

    def _ready_to_release(self, task):
        gap = task["plate_xyz"] - self.target_xyz
        if self.fk_plate_z is None or not task["held"]:
            self.descent_heights.clear()
            self.over_pocket = False
            self.drop_step = None
        else:
            if gap[2] > 0 and np.linalg.norm(gap[:2]) < 0.023:
                self.over_pocket = True
            fk_depth = self.fk_plate_z - self.target_xyz[2]
            heights = np.array([gap[2], fk_depth])
            # The nominal ramp travels only 13 mm in twelve frames. A common
            # drop of a seat depth over this window indicates the rim gave way.
            if (
                self.over_pocket
                and max(heights) < -0.018
                and any(
                    np.all(previous - heights >= SITE_ABOVE_SEATED_CENTER)
                    for previous in self.descent_heights
                )
            ):
                self.drop_step = self.machine.step
            self.descent_heights.append(heights)
        return super()._ready_to_release(task)

    def _ready_for_fallback(self, task, delta):
        if super()._ready_for_fallback(task, delta):
            return True
        if self.drop_step is None or self.machine.step - self.drop_step > LATE_LOWER:
            return False
        contacts = task["contacts"]
        # Keep the ordinary eight-frame confirmation. Contact high on a rim,
        # a visual jump alone, or an unsupported low wrist cannot open the hand.
        return bool(
            self.machine.elapsed >= LATE_LOWER
            and task["held"]
            and contacts["target_rack"]
            and not (contacts["table"] or contacts["floor"])
            and task["tilt_deg"] <= 12
            and np.linalg.norm(delta[:2]) < 0.065
            and -0.060 <= delta[2] <= -0.018
            and self.fk_plate_z is not None
            and -0.065 <= self.fk_plate_z - self.target_xyz[2] <= -0.018
            and self.measured_hand_z is not None
            and self.commanded_hand_z is not None
            and self.measured_hand_z - self.commanded_hand_z >= 0.012
        )
