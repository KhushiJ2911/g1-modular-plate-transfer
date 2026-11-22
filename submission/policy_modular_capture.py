"""Measure a pickup-dependent hand-frame normal from head and left-wrist RGB."""

import os

import numpy as np
import torch

from method.modular.kinematics import pelvis_rotation
from method.modular.loaded_wrist import register
from method.modular.plate_mask import PlateMask

from .policy_modular_calibrated import make_policy as make_calibrated


class LiftObservation:
    """Observe an upward-moving, closed hand before using the rigid grasp for placement."""

    def __init__(self, kin, weights):
        self.kin = kin
        self.prior = weights["loaded_normal"].copy()
        self.mask = PlateMask().eval()
        self.mask.load_state_dict(
            {k: torch.from_numpy(weights["mask/" + k]) for k in self.mask.state_dict()}
        )
        self.reset()

    def reset(self):
        self.low = None
        self.age = 0
        self.measurements = []

    def due(self, state):
        if state[44] < 0.8:
            self.reset()
            return False
        if self.age >= 13:
            return False
        self.kin.set_state(state)
        hand, _ = self.kin.hand_pose("left")
        height = (pelvis_rotation(state) @ hand)[2] + state[20]
        self.low = height if self.low is None else min(self.low, height)
        if height < self.low + 0.05:
            return False
        self.age += 1
        return (self.age - 1) % 3 == 0

    @torch.inference_mode()
    def observe(self, rgb, plate_rel):
        masks = (self.mask(torch.from_numpy(rgb[[0, 2]]), torch.tensor([0, 1])) > 0).numpy()
        p, r = self.kin.hand_pose("left")
        torso = self.kin.data.body("torso_link")
        result = register(
            *masks,
            r.T @ (torso.xpos - p),
            r.T @ torso.xmat.reshape(3, 3),
            r.T @ (plate_rel - p),
        )
        if result is not None:
            self.measurements.append(result)
        if self.age < 13 or len(self.measurements) < 3:
            return self.prior
        best = sorted(self.measurements, key=lambda r: r[2])[:3]
        normals = np.array([r[0] for r in best])
        n = np.median(normals, axis=0)
        if np.linalg.norm(normals - n, axis=1).max() >= 0.12:
            return self.prior
        tangent = n - (n @ self.prior) * self.prior
        n = self.prior + tangent * min(0.9, 0.25 / max(1e-8, np.linalg.norm(tangent)))
        return n / np.linalg.norm(n)


class CapturePolicy:
    def __init__(self, base, weights):
        self.base = base
        self.capture = LiftObservation(base.controller.kin, weights)

    def reset(self):
        self.capture.reset()
        self.base.reset()

    def __call__(self, obs):
        state = obs["state"]
        task = self.base.task.update(obs)
        ctl = self.base.controller
        if self.capture.due(state):
            ctl.orientation.set_loaded_normal(self.capture.observe(obs["rgb"], task["plate_rel"]))
        elif state[44] < 0.8:
            ctl.orientation.set_loaded_normal(self.capture.prior)
        return ctl.act(state, task)


def make_policy():
    policy = make_calibrated()
    with np.load(os.environ["ADV_HEADS"], allow_pickle=False) as weights:
        return CapturePolicy(policy, weights)
