"""Read-only task labels for camera-free controller development."""

from __future__ import annotations

import mujoco
import numpy as np


def pelvis_rotation(yaw: float) -> np.ndarray:
    """Rotation from the estimator's yaw-aligned pelvis frame to world."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class PrivilegedTaskState:
    """Sample live geometry in the same frame as the estimator labels."""

    def __init__(self, env):
        self.inner = env.inner_env
        self.plate = self.inner.plates[0]

    def snapshot(self, state: np.ndarray) -> dict:
        inner, plate = self.inner, self.plate
        pose = np.asarray(plate.get_pose(), dtype=float)
        sites = inner.rack_target.sites
        positions = np.array([inner.data.bind(site).xpos for site in sites])
        distances = np.linalg.norm(positions - pose[:3], axis=1)
        nearest = int(np.argmin(distances))
        site_rot = inner.data.bind(sites[nearest]).xmat.reshape(3, 3).copy()
        plate_rot = np.empty((3, 3))
        mujoco.mju_quat2Mat(plate_rot.ravel(), pose[3:])
        world_to_pelvis = pelvis_rotation(state[21]).T
        origin = state[18:21]
        return {
            "plate_rel": world_to_pelvis @ (pose[:3] - origin),
            "target_site_rel": world_to_pelvis @ (positions[nearest] - origin),
            "plate_rot_rel": world_to_pelvis @ plate_rot,
            "target_site_rot_rel": world_to_pelvis @ site_rot,
            "plate_xyz": pose[:3].copy(),
            "plate_rot": plate_rot,
            "site_xyz": positions[nearest],
            "site_rot": site_rot,
            "nearest_site": nearest,
            "site_distance": float(distances[nearest]),
            "site_xy_distance": float(np.linalg.norm(positions[nearest, :2] - pose[:2])),
            "tilt_deg": float(np.degrees(np.arccos(np.clip(-plate_rot[1, 2], -1, 1)))),
            "held": any(
                inner.robot.is_gripper_holding_object(plate, side) for side in inner.robot.grippers
            ),
            "contacts": {
                "source_rack": bool(plate.is_colliding(inner.rack_start)),
                "target_rack": bool(plate.is_colliding(inner.rack_target)),
                "table": bool(plate.is_colliding(inner.table)),
                "floor": bool(plate.is_colliding(inner.floor)),
            },
        }

    __call__ = snapshot
