"""Conservative placement feedback when the visual tilt lacks a sign."""

from collections import deque

import numpy as np

from .kinematics import pelvis_rotation
from .phases import Phase

PLACEMENT_PHASES = (Phase.ALIGN, Phase.LOWER)


class PlacementOrientation:
    """Filter the unsigned gate and suppress small, uncertain normal corrections."""

    def __init__(self, *, disable_signed=False):
        self.disable_signed = disable_signed
        self._normals = deque(maxlen=5)
        self._tilts = deque(maxlen=5)

    def update(self, state, task, phase):
        if phase not in PLACEMENT_PHASES:
            self._normals.clear()
            self._tilts.clear()
            return task
        site = task["site_rot"]
        reference = site[:, 2]
        if self.disable_signed:
            return {**task, "plate_rot": site, "plate_rot_rel": pelvis_rotation(state).T @ site}

        self._normals.append(task["plate_rot"][:, 2])
        self._tilts.append(task["tilt_deg"])
        normal = np.median(self._normals, axis=0)
        normal /= max(1e-8, np.linalg.norm(normal))
        dot = np.clip(normal @ reference, -1.0, 1.0)
        kin_tilt = np.rad2deg(np.arccos(dot))
        visual_tilt = float(np.median(self._tilts))
        # The wrist orientation carries a sign, but it includes grasp-angle
        # and slip error. Require both signals to support a substantial tilt
        # before feeding a small signed correction into J's integral loop.
        angle = np.deg2rad(np.clip((min(kin_tilt, visual_tilt) - 7.0) * 0.35, 0.0, 2.5))
        tangent = normal - dot * reference
        tangent /= max(1e-8, np.linalg.norm(tangent))
        corrected = np.cos(angle) * reference + np.sin(angle) * tangent
        x = site[:, 0] - corrected * (site[:, 0] @ corrected)
        x /= max(1e-8, np.linalg.norm(x))
        rotation = np.column_stack((x, np.cross(corrected, x), corrected))
        # The signed kinematic proxy helps ALIGN's permissive gate. Once the
        # plate is held against the rack, use the filtered visual magnitude.
        gate = 0.65 * visual_tilt + 0.35 * kin_tilt if phase == Phase.ALIGN else visual_tilt
        return {
            **task,
            "plate_rot": rotation,
            "plate_rot_rel": pelvis_rotation(state).T @ rotation,
            "tilt_deg": float(gate),
        }


# Mean unit plate normal in the left grasp frame, calibrated on the
# privileged-driven ALIGN/LOWER samples in the Sol development report.
# The constant is shared across every rack position and episode.
LEFT_GRASP_NORMAL = np.array([-0.21602, -0.96411, 0.15170])
LEFT_GRASP_NORMAL /= np.linalg.norm(LEFT_GRASP_NORMAL)


class CalibratedGraspOrientation:
    """Use measured arm orientation with one calibrated grasp geometry."""

    def __init__(self, kin):
        self.kin = kin

    def update(self, state, task, phase):
        active = (Phase.GRASP, Phase.LIFT, Phase.CARRY, Phase.ALIGN, Phase.LOWER)
        if phase not in active or state[44] < 0.65:
            return task
        basis = pelvis_rotation(state)
        self.kin.set_state(state)
        hand = basis @ self.kin.hand_pose("left")[1]
        normal = hand @ LEFT_GRASP_NORMAL
        x = task["plate_rot"][:, 0] - normal * (normal @ task["plate_rot"][:, 0])
        if np.linalg.norm(x) < 1e-8:
            x = np.cross(task["plate_rot"][:, 1], normal)
        rotation = np.column_stack(
            (x / np.linalg.norm(x), np.cross(normal, x) / np.linalg.norm(x), normal)
        )
        tilt = float(np.rad2deg(np.arccos(np.clip(normal @ task["site_rot"][:, 2], -1, 1))))
        # A large visual tilt can indicate slip, breaking the rigid transform.
        if task["tilt_deg"] > 27:
            tilt = max(tilt, task["tilt_deg"])
        return {
            **task,
            "plate_rot": rotation,
            "plate_rot_rel": basis.T @ rotation,
            "tilt_deg": tilt,
        }
