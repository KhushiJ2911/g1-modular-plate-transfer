"""Read-only privileged geometry, using the fixed task's success conventions."""

from __future__ import annotations

import numpy as np


def plate_up(quaternion) -> np.ndarray:
    """Rotate local +z by a wxyz quaternion (as pyquaternion does)."""
    q = np.asarray(quaternion, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


class PrivilegedSensors:
    """Read live poses, contact predicates and joint axes; never write mjData."""

    def __init__(self, inner):
        self.inner = inner
        self.plate = inner.plates[0]

    def snapshot(self, step: int) -> dict:
        inner, plate = self.inner, self.plate
        pose = np.asarray(plate.get_pose(), dtype=float)
        sites = np.array([inner.data.bind(s).xpos.copy() for s in inner.rack_target.sites])
        if pose.shape != (7,) or sites.shape != (6, 3):
            raise ValueError("Expected one xyz/wxyz plate pose and six xyz destination sites")
        if not np.isfinite(pose).all() or not np.isfinite(sites).all():
            raise ValueError("Nonfinite privileged pose/sites")
        nearest = int(np.argmin(np.linalg.norm(sites - pose[:3], axis=1)))
        up = plate_up(pose[3:])
        return {
            "step": step,
            "pose": pose.tolist(),
            "nearest_site": nearest,
            "site_xyz": sites[nearest].tolist(),
            "site_distance": float(np.linalg.norm(sites[nearest] - pose[:3])),
            "site_xy_distance": float(np.linalg.norm(sites[nearest, :2] - pose[:2])),
            "tilt_deg": float(np.degrees(np.arccos(np.clip(-up[1], -1, 1)))),
            "held": any(
                inner.robot.is_gripper_holding_object(plate, side) for side in inner.robot.grippers
            ),
            "contacts": {
                "target_rack": bool(plate.is_colliding(inner.rack_target)),
                "table": bool(plate.is_colliding(inner.table)),
                "floor": bool(plate.is_colliding(inner.floor)),
            },
        }

    def joint_geometry(self, action_dims) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """World hinge axes/anchors and left pinch position, in action order.

        For a hinge, dn/dq = axis cross n and dp/dq = axis cross (p-anchor).
        Reading xaxis/xanchor avoids perturbing a held plate or simulator state.
        """
        inner = self.inner
        # The WBC robot's limb_joints also includes legs/waist. Resolve names,
        # never action-offset indices into that larger list.
        left_names = (
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
        )
        by_name = {j.name.rsplit("/", 1)[-1]: j for j in inner.robot.limb_joints}
        joints = [by_name[left_names[d - 4]] for d in action_dims]
        ids = [inner.model.bind(j).id for j in joints]
        left_side = next(side for side in inner.robot.grippers if side.name == "LEFT")
        return (
            inner.data.xaxis[ids].copy(),
            inner.data.xanchor[ids].copy(),
            # In this G1, get_hand_pos is the end-effector site at the pad
            # center (0.148m along wrist_yaw), unlike the default pinch site
            # created at the wrist body origin by the generic Gripper class.
            inner.robot.get_hand_pos(left_side).copy(),
        )


def good_placement(snapshot: dict) -> bool:
    """Geometry/contact part of success, allowing fingers to touch at opening."""
    return (
        snapshot["site_distance"] <= 0.05
        and snapshot["tilt_deg"] <= 20
        and snapshot["contacts"]["target_rack"]
        and not snapshot["contacts"]["table"]
        and not snapshot["contacts"]["floor"]
    )


def failure_category(success, fell, initial, lifted, release, end, dropped) -> str:
    """Single label; precedence is documented in README, raw evidence is retained."""
    if success:
        return "success"
    if fell or end["contacts"]["floor"]:
        return "fell"
    if lifted is not None and lifted["pose"][2] - initial["pose"][2] < 0.03:
        return "grasp_miss"
    if dropped:
        return "drop_in_carry"
    if release is not None and good_placement(release) and not good_placement(end):
        return "knocked_after_release"
    if end["site_distance"] > 0.05:
        return "place_position"
    if end["tilt_deg"] > 20:
        return "place_tilt"
    if end["held"]:
        return "not_released"
    return "other"
