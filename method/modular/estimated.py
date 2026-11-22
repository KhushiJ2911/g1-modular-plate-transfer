"""Public-observation adapter for ModularController.

Call update(obs) once per observation. snapshot(state) and __call__(state)
then implement the PrivilegedTaskState interface for that same frame.
"""

import numpy as np

from .kinematics import pelvis_rotation

# The rack sites' common upright frame (x forward, z normal toward -world y).
# All sites have this orientation in the fixed dish-drainer geometry.
SITE_ROTATION = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
PLATE_RADIUS = 0.114  # meters, collision mesh radial extent


class EstimatedTaskState:
    def __init__(self, estimator, kin):
        self.estimator = estimator
        self.kin = kin
        self.reset()

    def reset(self):
        if hasattr(self.estimator, "reset"):
            self.estimator.reset()
        self._state = None
        self._task = None
        self._source = None
        self.held_probability = 0.0
        self._plate_in_hand = None

    def update(self, obs):
        state = np.asarray(obs["state"])
        prediction = self.estimator({"rgb": obs["rgb"], "state": state})
        basis = pelvis_rotation(state)
        plate_rel = np.asarray(prediction["plate_rel"], dtype=float)
        site_rel = np.asarray(prediction["target_site_rel"], dtype=float)
        plate, site = (state[18:21] + basis @ p for p in (plate_rel, site_rel))
        tilt = float(np.clip(prediction["tilt_deg"], 0.0, 180.0))
        self.held_probability = float(prediction["held"])
        held = self.held_probability >= 0.5

        # Tilt magnitude has no sign or azimuth. Start with the rack's usual
        # slight upward lean and zero in-plane spin. While held, propagate the
        # estimated grasp orientation from the observed left arm's rotation.
        radians = np.deg2rad(90.0 - tilt)
        c, s = np.cos(radians), np.sin(radians)
        plate_rot = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
        if "plate_rot_rel" in prediction:
            plate_rot = basis @ prediction["plate_rot_rel"]
        elif held:
            self.kin.set_state(state)
            hand_rot = basis @ self.kin.hand_pose("left")[1]
            if self._plate_in_hand is None:
                self._plate_in_hand = hand_rot.T @ plate_rot
            plate_rot = hand_rot @ self._plate_in_hand
        else:
            self._plate_in_hand = None

        if self._source is None:
            self._source = plate.copy()
        delta = plate - site
        distance = float(np.linalg.norm(delta))
        planar = float(np.linalg.norm(delta[:2]))
        # These are geometric proximity proxies for unseen contact sensors.
        # Reference the observed site height; table top is 0.17 m below it.
        lower_edge = plate[2] - PLATE_RADIUS * np.sqrt(max(0.0, 1.0 - plate_rot[2, 2] ** 2))
        contacts = {
            "source_rack": bool(
                np.linalg.norm(plate[:2] - self._source[:2]) < 0.10
                and plate[2] < self._source[2] + 0.065
            ),
            "target_rack": bool(distance < 0.052 and delta[2] <= -0.012),
            "table": bool(lower_edge <= site[2] - 0.17 + 0.006),
            "floor": bool(lower_edge <= 0.006),
        }
        if "contacts" in prediction:
            contacts = prediction["contacts"].copy()
        self._state = state.copy()
        self._task = {
            "plate_rel": plate_rel,
            "target_site_rel": site_rel,
            "plate_rot_rel": basis.T @ plate_rot,
            "target_site_rot_rel": basis.T @ SITE_ROTATION,
            "plate_xyz": plate,
            "plate_rot": plate_rot,
            "site_xyz": site,
            "site_rot": SITE_ROTATION.copy(),
            # The network estimates the nearest site's coordinates without an ID.
            "nearest_site": -1,
            "site_distance": distance,
            "site_xy_distance": planar,
            "tilt_deg": tilt,
            "held": held,
            "contacts": contacts,
        }
        return self._task

    def snapshot(self, state):
        if self._task is None or not np.array_equal(self._state, state):
            raise ValueError("Call update(obs) with RGB and state for the current frame")
        return self._task

    __call__ = snapshot
