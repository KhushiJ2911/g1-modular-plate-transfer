"""Continuous corrections for the episode's destination lock and late plate position."""

import numpy as np

PLACE = ("CARRY", "ALIGN", "LOWER", "RELEASE", "SETTLE", "RETREAT")


def site_features(vision, rel):
    return np.concatenate([vision, rel], axis=-1)


def plate_features(vision, state, plate, site):
    return np.concatenate([vision, state[..., :9], state[..., 44:45], plate, plate - site], axis=-1)


def predict(weights, group, x):
    z = np.clip((x - weights[group + "_mean"]) / weights[group + "_scale"], -8, 8)
    return z @ weights[group + "_coef"] + weights[group + "_offset"]


class PlacementEstimate:
    def __init__(self, source, controller, checkpoint, groups):
        self.source = source
        self.controller = controller
        with np.load(checkpoint, allow_pickle=False) as f:
            self.weights = dict(f)
        self.groups = groups
        self.reset()

    def reset(self):
        self.source.reset()
        self.samples = []
        self.site_delta = None
        self.task = None

    def update(self, obs):
        from .kinematics import pelvis_rotation

        task = self.source.update(obs).copy()
        basis = pelvis_rotation(obs["state"])
        if "site" in self.groups:
            if len(self.samples) < 6:
                # J holds the initial viewpoint for the first six observations.
                x = site_features(self.source.visual, task["target_site_rel"])
                self.samples.append(basis @ predict(self.weights, "site", x))
                self.site_delta = np.median(self.samples, axis=0)
            task["site_xyz"] = task["site_xyz"] + self.site_delta
            task["target_site_rel"] = task["target_site_rel"] + basis.T @ self.site_delta
        if "plate" in self.groups and self.controller.phase.value in PLACE:
            # Use the same raw v1b site field as the training target's features.
            raw = self.source.snapshot(obs["state"])
            x = plate_features(
                self.source.visual, obs["state"], raw["plate_rel"], raw["target_site_rel"]
            )
            d = predict(self.weights, "plate", x)
            task["plate_rel"] = task["plate_rel"] + d
            task["plate_xyz"] = task["plate_xyz"] + basis @ d
        delta = task["plate_xyz"] - task["site_xyz"]
        task["site_distance"] = float(np.linalg.norm(delta))
        task["site_xy_distance"] = float(np.linalg.norm(delta[:2]))
        self.task = task
        return task

    def snapshot(self, state):
        self.source.snapshot(state)
        return self.task

    __call__ = snapshot
