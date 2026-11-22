"""Observation-only signed plate-normal and target-rack contact heads.

Both heads use the same frozen multi-camera features as v1b, plus measured arm
geometry and continuous distances from the visually located plate and rack.
Tilt is measured from the corrected normal, consistently with the release gate.
"""

import numpy as np

from .estimated import EstimatedTaskState
from .kinematics import pelvis_rotation

ALL_PHASES = (
    "APPROACH",
    "PREGRASP",
    "GRASP",
    "LIFT",
    "CARRY",
    "ALIGN",
    "LOWER",
    "RELEASE",
    "SETTLE",
    "RETREAT",
)
GROUP_PHASES = {
    "normal": ALL_PHASES,
    "contact": ("CARRY", "ALIGN", "LOWER", "RELEASE", "SETTLE", "RETREAT"),
}


def inputs(visual, state, hand, pred, first_plate, first_site):
    plate, site = pred[..., :3], pred[..., 3:6]
    first_plate = np.broadcast_to(first_plate, plate.shape)
    first_site = np.broadcast_to(first_site, plate.shape)
    return np.concatenate(
        (
            visual,
            state[..., :9],
            state[..., 18:22],
            state[..., 44:45],
            hand[..., :3] - first_plate,
            hand[..., 3:],
            plate - first_plate,
            site - first_site,
            np.abs(plate - first_site),
            np.linalg.norm(plate - first_site, axis=-1, keepdims=True),
            hand[..., :3] - first_site,
            np.abs(hand[..., :3] - first_site),
            first_plate - first_site,
            pred[..., 6:11],
        ),
        axis=-1,
    )


def align_normal(rotation, normal):
    """Update the normal while preserving the observed plate's in-plane heading."""
    length = np.linalg.norm(normal)
    if length < 1e-8:
        return rotation
    z = normal / length
    x = rotation[:, 0] - z * (z @ rotation[:, 0])
    if np.linalg.norm(x) < 1e-8:
        x = np.cross(rotation[:, 1], z)
    x /= np.linalg.norm(x)
    return np.column_stack((x, np.cross(z, x), z))


class CorrectedTaskState(EstimatedTaskState):
    def __init__(self, policy, weights):
        self.controller = policy.controller
        with np.load(weights, allow_pickle=False) as file:
            self.weights = dict(file)
        self.groups = tuple(str(self.weights["groups"]).split(","))
        if not set(self.groups) <= GROUP_PHASES.keys():
            raise ValueError(f"Unknown estimator heads {self.groups}")
        estimator = policy.task.estimator

        def record_visual(module, args):
            self.visual = args[0][0].detach().cpu().numpy()

        estimator.model.head[0].register_forward_pre_hook(record_visual)
        super().__init__(estimator, self.controller.kin)

    def reset(self):
        super().reset()
        self.visual = None
        self.first_plate = None
        self.first_site = None

    def update(self, obs):
        task = super().update(obs)
        if self.first_site is None:
            self.first_site = task["site_xyz"].copy()
            self.first_plate = task["plate_xyz"].copy()
        phase = self.controller.phase.value
        active = [g for g in self.groups if phase in GROUP_PHASES[g]]
        if not active:
            return task
        state = np.asarray(obs["state"])
        self.kin.set_state(state)
        pos, rot = self.kin.hand_pose("left")
        basis = pelvis_rotation(state)
        hand = np.r_[state[18:21] + basis @ pos, (basis @ rot).ravel()]
        pred = np.r_[
            task["plate_xyz"],
            task["site_xyz"],
            task["plate_rot"][:, 2],
            task["tilt_deg"],
            task["held"],
        ]
        x = inputs(self.visual, state, hand, pred, self.first_plate, self.first_site)
        for group in active:
            weights = self.weights
            value = (x - weights[group + "_mean"]) / weights[group + "_scale"] @ weights[
                group + "_coef"
            ] + weights[group + "_bias"]
            if group == "normal":
                rot = align_normal(task["plate_rot"], pred[6:9] + value)
                task["plate_rot"] = rot
                task["plate_rot_rel"] = basis.T @ rot
                task["tilt_deg"] = float(np.rad2deg(np.arccos(np.clip(-rot[1, 2], -1, 1))))
            else:
                task["contacts"]["target_rack"] = bool(value.item() >= 0.5)
        return task


def attach(policy, weights):
    policy.task = CorrectedTaskState(policy, weights)
    policy.controller.task_state_source = policy.task
    return policy
