"""bce1 with observation-only plate estimation and a release supervisor."""

import os
from pathlib import Path

from .estimator import Estimator
from .harness import ReleaseSupervisor, config_from_env, log_config
from .policy import BCPolicy


class HarnessPolicy:
    def __init__(self, checkpoint, estimator, config=None):
        self.config = config_from_env() if config is None else config
        self.bc = BCPolicy(checkpoint)
        if self.config["exec_steps"] > self.bc.horizon:
            raise ValueError("ADV_H_EXEC_STEPS exceeds BC horizon")
        self.bc.exec_steps = self.config["exec_steps"]
        self.bc.temporal_ensemble = self.config["temporal_ensemble"]
        self.bc.ensemble_decay = self.config["ensemble_decay"]
        self.estimator = Estimator(estimator, self.bc.device)
        self.supervisor = ReleaseSupervisor(self.config)
        log_config(self.config)
        self.reset()

    def reset(self):
        self.bc.reset()
        self.supervisor.reset()

    def __call__(self, obs):
        # Deliberately pass only the two public observations through inference.
        visible = {"rgb": obs["rgb"], "state": obs["state"]}
        estimate = self.estimator(visible)
        actions = self.bc(visible)
        if not self.config["supervise"]:
            return actions
        return self.supervisor.step(estimate, visible["state"], actions)


def make_policy():
    checkpoint = os.environ.get("ADV_CKPT") or Path(__file__).parent / "weights" / "policy.pt"
    estimator = os.environ.get("ADV_ESTIMATOR")
    if not estimator:
        raise ValueError("Set ADV_ESTIMATOR to the trained estimator last.pt")
    return HarnessPolicy(checkpoint, estimator)
