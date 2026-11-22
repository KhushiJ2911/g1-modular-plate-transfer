"""Calibrated grasp orientation with a replaceable perception checkpoint."""

import os
from pathlib import Path

from method.modular.contact_perception import PerceivedRackContact, PerceptionHeads
from method.modular.observed_controller import CalibratedPlacementController

from .policy_modular import ModularPolicy
from .policy_modular import make_policy as make_baseline


class PerceptionPolicy(ModularPolicy):
    def __init__(self, base, checkpoint):
        self.task = base.task
        self.controller = CalibratedPlacementController(base.controller.kin, self.task)
        self.task = PerceptionHeads(self, checkpoint)
        self.controller.task_state_source = self.task
        self.reset()

    def reset(self):
        super().reset()
        self.controller.site_contacts = PerceivedRackContact(self.controller.site_contacts)


def make_policy():
    default = Path(__file__).resolve().parents[1] / "checkpoints/astra/contact_v1.npz"
    return PerceptionPolicy(make_baseline(), os.environ.get("ADV_HEADS") or default)
