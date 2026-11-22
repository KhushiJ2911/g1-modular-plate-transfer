"""Observation-only pickup, held and seating estimates for the grasp controller."""

import numpy as np

from submission.head_features import head_inputs

from .kinematics import pelvis_rotation
from .refined import CorrectedTaskState, GROUP_PHASES

PICKUP = ("APPROACH", "PREGRASP", "GRASP")


class PerceptionHeads(CorrectedTaskState):
    def __init__(self, policy, checkpoint):
        super().__init__(policy, checkpoint)
        if self.groups != ("contact",):
            raise ValueError("Expected contact-only initialization")
        self.network = "hidden_weight" in self.weights

    def update(self, obs):
        task = super().update(obs)
        state = obs["state"]
        phase = self.controller.phase.value
        if self.network:
            w = self.weights
            x = head_inputs(self.visual, state, task["plate_xyz"], task["site_xyz"],
                            self.held_probability, self.first_plate)
            x = np.clip((x-w["x_mean"])/w["x_std"], -12, 12)
            y = np.maximum(0, x @ w["hidden_weight"] + w["hidden_bias"]) @ w["out_weight"] + w["out_bias"]
            if phase in PICKUP:
                delta = 0.025 * y[:3]
                task["plate_xyz"] = task["plate_xyz"] + delta
                task["plate_rel"] = task["plate_rel"] + pelvis_rotation(state).T @ delta
            if phase in GROUP_PHASES["contact"]:
                task["contacts"]["target_rack"] = bool(y[4] >= 0)
            self.held_probability = float(1 / (1 + np.exp(-np.clip(y[3], -60, 60))))
        elif phase in ("GRASP", "LIFT"):
            # Empty fingers bottom out at -20 mm. A rim generally stops the
            # slides around -16 mm; closure alone can report an empty pinch.
            if state[44] < 0.8 or np.mean(state[7:9]) < -0.0185:
                self.held_probability = 0.0
        task["held"] = self.held_probability >= 0.5
        return task


class PerceivedRackContact:
    """Carry the learned contact through Sol's existing site-lock transformation."""

    def __init__(self, site_filter):
        self.site_filter = site_filter

    def update(self, state, task):
        locked = self.site_filter.update(state, task)
        locked["contacts"]["target_rack"] = task["contacts"]["target_rack"]
        return locked
