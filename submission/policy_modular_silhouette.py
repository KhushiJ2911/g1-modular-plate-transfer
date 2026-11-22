"""The merged pickup and release controller with measured plate-lip geometry."""

import os
from collections import deque

import numpy as np
import torch

from method.modular.loaded_wrist import register
from method.modular.phases import Phase
from method.modular.plate_mask import PlateMask
from submission.policy_modular_integrated import make_visual_release


class LoadedViews:
    """Measure only while the left fingers hold the plate during lowering."""

    def __init__(self, source, controller, weights):
        self.source = source
        self.controller = controller
        self.mask = PlateMask().eval()
        self.mask.load_state_dict(
            {name: torch.from_numpy(weights["mask/" + name]) for name in self.mask.state_dict()}
        )
        self.recent = deque(maxlen=5)
        self.current = None

    def reset(self):
        self.source.reset()
        self.recent.clear()
        self.current = None

    def update(self, obs):
        task = self.source.update(obs)
        state = obs["state"]
        if self.controller.phase != Phase.LOWER or not task["held"] or state[44] < 0.65:
            self.recent.clear()
            self.current = None
            return task
        if self.controller.machine.elapsed % 4 == 0:
            with torch.inference_mode():
                rgb = torch.from_numpy(obs["rgb"][[0, 2]])
                masks = (self.mask(rgb, torch.tensor([0, 1])) > 0).numpy()
            kin = self.controller.kin
            kin.set_state(state)
            hand, rot = kin.hand_pose("left")
            torso = kin.data.body("torso_link")
            result = register(
                *masks,
                rot.T @ (torso.xpos - hand),
                rot.T @ torso.xmat.reshape(3, 3),
                rot.T @ (task["plate_rel"] - hand),
            )
            self.recent.append(result)
            self.current = None
            # Require three independent recent silhouettes. A weighted subset
            # suppresses occluded frames without using task labels at runtime.
            usable = sorted((r for r in self.recent if r is not None), key=lambda r: r[2])
            if len(usable) >= 3:
                values = np.array([r[0] for r in usable[:3]])
                center = np.median(values, axis=0)
                if np.max(np.linalg.norm(values - center, axis=1)) < 0.12:
                    self.current = center / np.linalg.norm(center)
        return task

    def snapshot(self, state):
        return self.source.snapshot(state)

    __call__ = snapshot


class MeasuredNormal:
    def __init__(self, views, fallback):
        self.views = views
        self.fallback = fallback

    def measure(self, state, task, hand, rotation):
        if self.views.current is None:
            return self.fallback.measure(state, task, hand, rotation)
        return self.views.current


def make_policy():
    policy = make_visual_release()
    with np.load(os.environ["ADV_HEADS"], allow_pickle=False) as weights:
        views = LoadedViews(policy.task, policy.controller, weights)
    policy.task = views
    policy.controller.task_state_source = views
    policy.controller.normal_source = MeasuredNormal(views, policy.controller.normal_source)
    return policy
