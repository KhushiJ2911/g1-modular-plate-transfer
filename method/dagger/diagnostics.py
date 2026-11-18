"""Training-only, read-only subset of the teacher's privileged sensors.

Copied from method/teacher/diagnostics.py in the teacher worktree. No cross-worktree
imports or simulator writes. Tilt follows the fixed task's plate convention.
"""

import numpy as np

OPEN_DISTANCE_M = 0.05
OPEN_TILT_DEG = 20.0
CLOSED_LIFT_M = 0.03
GRIPPER_DIM = 18


def plate_up(quaternion):
    q = np.asarray(quaternion, dtype=float)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


class PrivilegedSensors:
    def __init__(self, inner):
        self.inner = inner
        self.plate = inner.plates[0]

    def snapshot(self, step):
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


def gripper_label(snapshot, start_z):
    """Return -1/open, +1/closed, or None/undefined at this pre-action state."""
    touching = snapshot["contacts"]["target_rack"]
    if (
        touching
        and snapshot["site_distance"] <= OPEN_DISTANCE_M
        and snapshot["tilt_deg"] <= OPEN_TILT_DEG
    ):
        return -1.0
    if snapshot["held"] and snapshot["pose"][2] - start_z >= CLOSED_LIFT_M and not touching:
        return 1.0
    return None


def outcome_category(success, fell, max_lift, first_placed, end, start_z):
    """Replay-style categories; raw geometry remains in the episode summary."""
    if success:
        return "success"
    if fell or end["contacts"]["floor"]:
        return "fell"
    if max_lift < CLOSED_LIFT_M:
        return "no_grasp"
    if gripper_label(end, start_z) == -1 and end["held"]:
        return "placed_not_released"
    if first_placed is not None:
        return "reached_target_then_failed"
    return "other"
