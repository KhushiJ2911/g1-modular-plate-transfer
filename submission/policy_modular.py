"""Observation-only G1 feedback policy. ADV_ESTIMATOR selects est_v1b/last.pt."""

import os
from pathlib import Path

from method.modular.controller import ModularController
from method.modular.estimated import EstimatedTaskState
from method.modular.kinematics import G1Kinematics

from .estimator import Estimator


class ModularPolicy:
    def __init__(self, estimator):
        kin = G1Kinematics()
        self.task = EstimatedTaskState(estimator, kin)
        self.controller = ModularController(kin, self.task)

    def reset(self):
        self.controller.kin.reset()
        self.task.reset()
        self.controller.reset()

    def __call__(self, obs):
        self.task.update(obs)
        return self.controller.act(obs["state"])


def make_policy():
    default = Path(__file__).resolve().parents[1] / "checkpoints/est_v1b/last.pt"
    return ModularPolicy(Estimator(os.environ.get("ADV_ESTIMATOR") or default))
