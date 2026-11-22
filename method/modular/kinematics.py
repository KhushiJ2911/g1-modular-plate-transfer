"""Independent G1 arm kinematics in the estimator's yaw-aligned pelvis frame.

The state supplies no waist joints or pelvis tilt. A fixed standing nominal
torso is used for the arm chain. Legs are siblings of that chain in the MJCF.
The original free joint remains at the origin in this local coordinate system.
"""

from importlib.resources import files

import mujoco
import numpy as np

from . import layout

# G1Dex1._EE_SITE_POS in pinned bigym/robots/configs/g1.py. This is the
# local grasp-point offset from wrist_yaw_link used by robot.get_hand_pos.
GRASP_POINT_OFFSET = (0.148, 0.0, 0.0)
SIDES = ("left", "right")


def pelvis_rotation(state50):
    yaw = float(state50[layout.PELVIS_YAW])
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def pelvis_to_world(state50, p):
    return pelvis_rotation(state50) @ p + state50[layout.PELVIS_XYZ]


def world_to_pelvis(state50, p):
    return pelvis_rotation(state50).T @ (np.asarray(p) - state50[layout.PELVIS_XYZ])


def rotation_vector(matrix):
    """Shortest spatial rotation vector of a 3x3 rotation matrix."""
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(matrix, dtype=float).reshape(9))
    vector = np.empty(3)
    mujoco.mju_quat2Vel(vector, quat, 1.0)
    return vector


class G1Kinematics:
    """Queries depend only on the last set_state call. IK preserves that state."""

    pelvis_to_world = staticmethod(pelvis_to_world)
    world_to_pelvis = staticmethod(world_to_pelvis)

    def __init__(self):
        xml = files("bigym").joinpath("envs/xmls/g1/g1_29dof_with_dex1_1.xml")
        spec = mujoco.MjSpec.from_file(str(xml))
        # Match BiGym's G1 loader. The bundled XML otherwise inherits a force
        # limit for joints without a force range, which MuJoCo cannot compile.
        spec.default.joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_AUTO
        for joint in spec.joints:
            joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_AUTO
        for side in SIDES:
            spec.body(f"{side}_wrist_yaw_link").add_site(
                name=f"{side}_end_effector", pos=GRASP_POINT_OFFSET, size=[0.001] * 3
            )
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        # Effective standing torso pitch, calibrated once at standoff in the
        # seed 3000100 open-hand rollout (15 mm vertical error at 0.34 m).
        self.data.joint("waist_pitch_joint").qpos[0] = 0.045
        self._sites = {s: self.model.site(f"{s}_end_effector").id for s in SIDES}
        self._qpos = {}
        self._dofs = {}
        self._limits = {}
        self._fingers = {}
        for side, names, action in zip(
            SIDES,
            (layout.JOINT_NAMES[:7], layout.JOINT_NAMES[7:]),
            (layout.LEFT_ARM_ACTION, layout.RIGHT_ARM_ACTION),
            strict=True,
        ):
            ids = [self.model.joint(n).id for n in names]
            self._qpos[side] = self.model.jnt_qposadr[ids]
            self._dofs[side] = self.model.jnt_dofadr[ids]
            lo, hi = self.model.jnt_range[ids].T
            self._limits[side] = (
                np.maximum(lo, layout.ACTION_LOW[action]),
                np.minimum(hi, layout.ACTION_HIGH[action]),
            )
            fingers = [self.model.joint(f"{side}_dex1_finger_joint_{j}").id for j in (1, 2)]
            self._fingers[side] = self.model.jnt_qposadr[fingers]
        self._forward()

    def _forward(self):
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def set_state(self, state50):
        state = np.asarray(state50)
        if state.shape != (50,) or not np.isfinite(state).all():
            raise ValueError("Expected 50 finite proprioceptive values")
        self.data.qpos[self._qpos["left"]] = state[layout.LEFT_ARM]
        self.data.qpos[self._qpos["right"]] = state[layout.RIGHT_ARM]
        self.data.qpos[self._fingers["left"]] = state[layout.LEFT_FINGERS]
        self.data.qpos[self._fingers["right"]] = state[layout.RIGHT_FINGERS]
        self._forward()

    def hand_pose(self, side):
        i = self._sites[side]
        return self.data.site_xpos[i].copy(), self.data.site_xmat[i].reshape(3, 3).copy()

    def jacobian(self, side):
        linear = np.empty((3, self.model.nv))
        angular = np.empty_like(linear)
        mujoco.mj_jacSite(self.model, self.data, linear, angular, self._sites[side])
        return np.vstack((linear, angular))[:, self._dofs[side]]

    def ik(
        self, side, target_pos, target_rot=None, q_init=None, pos_weight=1, rot_weight=0.3, iters=50
    ):
        """Damped least squares, spatial rotation-log error, hard joint limits.

        Weights multiply residuals (meters and radians) and Jacobian rows.
        For unreachable targets return the lowest-residual iterate visited.
        """
        idx = self._qpos[side]
        lo, hi = self._limits[side]
        original = self.data.qpos[idx].copy()
        q = np.clip(original if q_init is None else q_init, lo, hi)
        best, best_loss = q.copy(), np.inf
        try:
            for _ in range(iters + 1):
                self.data.qpos[idx] = q
                self._forward()
                p, r = self.hand_pose(side)
                error = pos_weight * (np.asarray(target_pos) - p)
                jac = self.jacobian(side)
                if target_rot is None:
                    jac = jac[:3] * pos_weight
                else:
                    angular = rotation_vector(target_rot @ r.T)
                    error = np.r_[error, rot_weight * angular]
                    jac[:3] *= pos_weight
                    jac[3:] *= rot_weight
                loss = error @ error
                if loss < best_loss:
                    best, best_loss = q.copy(), loss
                if loss < 1e-10:
                    break
                normal = jac @ jac.T + 0.008**2 * np.eye(len(error))
                dq = jac.T @ np.linalg.solve(normal, error)
                q_next = np.clip(q + dq / max(1, np.max(np.abs(dq)) / 0.18), lo, hi)
                if np.max(np.abs(q_next - q)) < 1e-8:
                    break
                q = q_next
        finally:
            self.data.qpos[idx] = original
            self._forward()
        return best
