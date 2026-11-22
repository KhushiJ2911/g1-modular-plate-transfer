"""Keep a corroborated rim pinch across a rejected grasp or lift observation."""

import numpy as np

from method.modular import layout
from method.modular.base_clearance import ClearancePolicy
from method.modular.fallback_seating import FallbackSeatingController
from method.modular.fk_seating import make_policy as make_seated_policy
from method.modular.phases import TIMEOUTS, Phase


class RetainedPinchController(FallbackSeatingController):
    def _pickup_task(self, state, task):
        phase = self.phase
        measured = super()._pickup_task(state, task)
        if measured["held"] or phase not in (Phase.GRASP, Phase.LIFT):
            return measured
        # An empty slide bottoms out at -20 mm. Require both slides to be
        # obstructed and stationary before overriding a visual disagreement.
        bilateral = (
            state[layout.GRIPPERS][0] >= 0.80
            and np.all(state[layout.LEFT_FINGERS] >= -0.0185)
            and np.max(np.abs(state[layout.LEFT_FINGERS_VEL])) < 0.012
        )
        if not bilateral:
            return measured

        if phase == Phase.GRASP:
            # The reach check already passed to issue closure. At the retry
            # deadline use the normal four-frame confirmation and test the
            # load in LIFT instead of opening a stationary pinch.
            if (
                self.closing
                and self.machine.elapsed >= TIMEOUTS[Phase.GRASP] - 4
                and np.linalg.norm(task["plate_xyz"] - self.source) < 0.05
            ):
                return {**measured, "held": True}
        elif (
            self.lost_streak >= 9
            and self.fk_rise >= 0.085
            and not task["contacts"]["source_rack"]
            and task["plate_xyz"][2] > self.lift_origin[2] + 0.065
        ):
            # Continue the normal lift until its original carry gate passes.
            return {**measured, "held": True}
        return measured


def make_policy():
    return ClearancePolicy(make_seated_policy(RetainedPinchController))
