"""Plate position and signed lean from the stationary pregrasp wrist view."""

import mujoco
import numpy as np

# Wrist camera mount and 60 degree vertical FOV from the fixed G1 geometry.
_camera_q, _camera_mat = np.empty(4), np.empty(9)
mujoco.mju_euler2Quat(_camera_q, np.array([0.0, -0.87, -1.57]), "xyz")
mujoco.mju_quat2Mat(_camera_mat, _camera_q)
WRIST_CAMERA = _camera_mat.reshape(3, 3)
FOCAL_PIXELS = 42 / np.tan(np.deg2rad(30))


def rim_lateral(rgb, hand, rotation, height):
    """World y of an upright plate from the bright front lip at PREGRASP."""
    # A continuous vertical highlight distinguishes the rim from isolated
    # highlights on rack pins. Discard cropped and ambiguous wrist views.
    strip = rgb[:, 16:49, 5:79].min(axis=0)
    u = strip.argmax(axis=1)
    v = np.arange(len(u))
    bright = strip[v, u] > 195
    if bright.sum() < 9:
        return None
    u, v = u[bright] + 5, v[bright] + 16
    center = np.median(u)
    if np.median(np.abs(u - center)) > 3:
        return None
    ray = np.array([(center - 41.5) / FOCAL_PIXELS, (41.5 - np.median(v)) / FOCAL_PIXELS, -1])
    # The reached standoff fixes the observed lip at about 10.5 cm depth.
    # Account for the site-to-camera mount, nominal source lean, and the
    # single lateral calibration shared by all continuous source positions.
    point = hand + rotation @ ([-0.028, 0, 0.040] + 0.105 * (WRIST_CAMERA @ ray))
    return point[1] - 0.12 * (point[2] - height) + 0.020


# Test the fixed plate's projected circular lip at a grid of nearby source poses.
# Negative and positive signed lean compete on the same subsampled wrist pixels.
V, U = np.mgrid[3:82:3, 2:83:3]
RAYS = np.stack(
    ((U - 41.5) / FOCAL_PIXELS, (41.5 - V) / FOCAL_PIXELS, -np.ones_like(U)), axis=-1
).reshape(-1, 3)
_dx, _dy, LEAN = (
    g.ravel()
    for g in np.meshgrid(
        [-0.012, 0.0, 0.012],
        np.arange(-0.064, 0.065, 0.004),
        np.deg2rad(np.arange(-42, 25, 6)),
        indexing="ij",
    )
)
NORMALS = np.column_stack((np.zeros_like(LEAN), -np.cos(LEAN), np.sin(LEAN)))
OFFSETS = np.column_stack((_dx, _dy, np.zeros_like(LEAN)))


def _tilted_face(rgb, hand, rotation, source):
    """Fit a disk to the broad white face exposed by a backwards lean."""
    pixels = rgb[:, V, U].reshape(3, -1)
    white = (pixels.min(axis=0) >= 120) & (np.ptp(pixels, axis=0) <= 3)
    if white.mean() < 0.055:
        return None
    rays = RAYS @ (rotation @ WRIST_CAMERA).T
    camera = hand + rotation @ [-0.028, 0.0, 0.040]
    # The visual lip is 18 mm along the plate normal, with radius 114 mm.
    centers = source + OFFSETS
    relative = centers + 0.018 * NORMALS - camera
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.sum(relative * NORMALS, axis=1)[:, None] / (NORMALS @ rays.T)
        radius_sq = (
            np.sum(relative**2, axis=1)[:, None]
            - 2 * depth * (relative @ rays.T)
            + depth**2 * np.sum(rays**2, axis=1)
        )
    disk = (depth > 0) & (radius_sq < 0.114**2)
    iou = (disk & white).sum(axis=1) / (disk | white).sum(axis=1)
    i = np.argmax(iou)
    if iou[i] < 0.75 or LEAN[i] >= np.deg2rad(-12):
        return None
    return np.r_[centers[i, :2], LEAN[i], iou[i]]


class SourceView:
    def __init__(self):
        self.rims = []
        self.faces = []
        self.rotation = None
        self.face_iou = None

    def update(self, rgb, hand, rotation, source, visual_tilt):
        face = _tilted_face(rgb, hand, rotation, source)
        if face is not None:
            self.faces.append(face)
        elif visual_tilt < 18:
            y = rim_lateral(rgb, hand, rotation, source[2])
            if y is not None:
                self.rims.append(y)
        source = source.copy()
        if len(self.faces) >= 3:
            x, y, angle, iou = np.median(self.faces, axis=0)
            self.face_iou = float(iou)
            source[:2] = x, y
            c, s = np.cos(angle), np.sin(angle)
            self.rotation = np.array([[1.0, 0.0, 0.0], [0.0, s, -c], [0.0, c, s]])
        elif len(self.rims) >= 3:
            offset = np.median(self.rims) - source[1]
            source[1] += 0.90 * np.clip(offset, -0.035, 0.035)
        return source
