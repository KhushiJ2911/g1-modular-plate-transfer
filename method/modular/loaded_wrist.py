"""Register a held plate's metric silhouette in calibrated head and wrist views."""

from __future__ import annotations

import numpy as np

from .orientation_feedback import LEFT_GRASP_NORMAL

FOCAL = 42 / np.tan(np.pi / 6)
WRIST_POS = np.array([-0.028, 0.0, 0.040])
HEAD_POS = np.array([0.12, 0.0, 0.37])


def _mount(pitch):
    # G1 camera mount, vertical FOV 60°, torso / wrist Euler order x-y-z.
    c, s = np.cos(pitch), np.sin(pitch)
    z, w = np.cos(-1.57), np.sin(-1.57)
    return np.array([[c * z, -c * w, s], [w, z, 0.0], [-s * z, s * w, c]])


WRIST_ROT = _mount(-0.87)
HEAD_ROT = _mount(-0.52)
# (radius, axial height) in metres, from the concentric rings of the fixed
# plate_01 visual mesh. The front lip, bowl and reverse face have different
# slopes. The wrist camera clips the near rim at 18 mm in this release.
PROFILE = np.array(
    [
        [0.0, 0.0018],
        [0.0380, 0.0018],
        [0.0803, 0.0138],
        [0.1133, 0.0192],
        [0.11415, 0.0174],
        [0.11318, 0.0157],
        [0.08133, 0.0103],
        [0.05769, 0.0032],
        [0.0552, 0],
        [0.0, 0],
    ]
)
RINGS = np.concatenate(
    [
        np.linspace(a, b, max(2, round(np.linalg.norm(b - a) / 0.008)), endpoint=False)
        for a, b in zip(PROFILE[:-1], PROFILE[1:], strict=True)
    ]
    + [PROFILE[-1:]]
)
CLIP = 0.018


def _roots(a, b, c):
    discriminant = np.sqrt(b * b - 4 * a * c)
    return (-b - discriminant) / (2 * a), (-b + discriminant) / (2 * a)


def _outline(center, normal, rows):
    """Return visible horizontal bounds of candidate bowls across image rows.

    Image rays have camera coordinates (u, v, -1). At each fixed v, the
    circular sections of the bowl give quadratic equations in u. Where the
    near plane cuts a sloped section, it supplies a second pair of bounds.
    """
    rays = np.stack([np.zeros(len(rows)), (41.5 - rows) / FOCAL, -np.ones(len(rows))], axis=1)
    n = normal[:, None, None, :]
    circle = (center[:, None, :] + normal[:, None, :] * RINGS[None, :, 1, None])[:, :, None, :]
    radius = RINGS[None, :, 0, None]
    offset = (n * circle).sum(axis=-1, keepdims=True)
    ny = (rays @ normal.T).T[:, None, :]
    nx = normal[:, 0, None, None]
    xdir = np.array([1.0, 0.0, 0.0]) * offset - nx[..., None] * circle
    ydir = offset * rays - ny[..., None] * circle
    a = (xdir * xdir).sum(axis=-1) - (radius * nx) ** 2
    b = 2 * ((ydir * xdir).sum(axis=-1) - radius**2 * ny * nx)
    c = (ydir * ydir).sum(axis=-1) - (radius * ny) ** 2
    endpoints = []
    with np.errstate(invalid="ignore", divide="ignore"):
        for u in _roots(a, b, c):
            depth = offset[..., 0] / (ny + u * nx)
            endpoints.append(np.where(depth >= CLIP, u, np.nan))

    radius, height = PROFILE[:-1].T[:, None, :, None]
    dr, dh = np.diff(PROFILE, axis=0).T[:, None, :, None]
    point = CLIP * rays[None, :, :] - center[:, None, :]
    axial = (point * normal[:, None, :]).sum(axis=2)[:, None, :]
    slope = CLIP * nx
    intercept = radius * dh + dr * (axial - height)
    # For a frustum with radial increment dr and height increment dh,
    # (radius(u)*dh)^2 = (radius0*dh + dr*(height(u)-height0))^2.
    a = (CLIP**2 - slope * slope) * dh**2 - (dr * slope) ** 2
    b = 2 * ((CLIP * point[:, None, :, 0] - axial * slope) * dh**2 - intercept * dr * slope)
    radius_sq = (point * point).sum(axis=2)[:, None, :] - axial**2
    c = radius_sq * dh**2 - intercept**2
    with np.errstate(invalid="ignore", divide="ignore"):
        for u in _roots(a, b, c):
            h = axial + slope * u
            t = (h - height) / np.where(dh != 0, dh, 1)
            r2 = (
                (point * point).sum(axis=2)[:, None, :]
                + 2 * CLIP * point[:, None, :, 0] * u
                + (CLIP * u) ** 2
                - h**2
            )
            planar = (r2 >= np.minimum(radius, radius + dr) ** 2) & (
                r2 <= np.maximum(radius, radius + dr) ** 2
            )
            valid = np.where(dh != 0, (t >= 0) & (t <= 1), planar)
            endpoints.append(np.where(valid, u, np.nan))

    u = np.concatenate(endpoints, axis=1)
    lo = np.min(np.where(np.isfinite(u), u, 1e3), axis=1)
    hi = np.max(np.where(np.isfinite(u), u, -1e3), axis=1)
    return 41.5 + FOCAL * np.stack([lo, hi], axis=2)


def _normal(parameters):
    n = np.column_stack((parameters[:, 3], -np.ones(len(parameters)), parameters[:, 4]))
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def _measurements(head, wrist, torso_pos, torso_rot):
    p = torso_pos + torso_rot @ HEAD_POS
    r = torso_rot @ HEAD_ROT
    # A second perpendicular head scan constrains the ends of the plate.
    quarter_turn = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    views = [
        (head, p, r, 82, 3, 1.5),
        (np.rot90(head), p, r @ quarter_turn, 82, 3, 1.5),
        (wrist, WRIST_POS, WRIST_ROT, 25, 3, 1.0),
    ]
    result = []
    for mask, position, rotation, stop, stride, weight in views:
        rows = np.arange(1, stop, stride)
        columns = np.arange(84)[None, :]
        left = np.where(mask[rows], columns, 1000).min(axis=1)
        right = np.where(mask[rows], columns, -1000).max(axis=1)
        visible = right - left >= 2
        edges = np.stack([left[visible], right[visible]], axis=1)
        observed = (edges > 1) & (edges < 82)
        if observed.sum() >= 3:
            result.append((position, rotation, rows[visible], edges, observed, weight))
        elif mask is wrist:
            return []
    return result


def register(head_mask, wrist_mask, torso_pos, torso_rot, center_guess):
    """Estimate the normal in the left hand from two public-image masks.

    The torso and center guess are expressed in the left grasp frame. Supply
    the torso transform from independent FK and plate center from perception.
    The residual reports how well the two silhouettes fit the rigid geometry.
    """
    views = _measurements(head_mask, wrist_mask, torso_pos, torso_rot)
    if len(views) < 2:
        return None
    scale = np.array([0.02, 0.02, 0.025, 0.16, 0.16])
    x = np.r_[center_guess, LEFT_GRASP_NORMAL[[0, 2]] / -LEFT_GRASP_NORMAL[1]]
    origin = x.copy()

    def residual(choices):
        normal = _normal(choices)
        chunks = []
        for p, r, rows, edges, observed, weight in views:
            curve = _outline((choices[:, :3] - p) @ r, normal @ r, rows)
            difference = np.clip(curve, -4, 87) - edges
            chunks.append(difference[:, observed] * np.sqrt(12 / observed.sum()) * weight)
        chunks.append((choices - origin) / scale * 0.7)
        return np.concatenate(chunks, axis=1)

    perturb = np.r_[np.zeros((1, 5)), np.eye(5)] * 0.04 * scale

    def cost(errors):
        return np.sum(np.log1p((errors / 2.5) ** 2), axis=1)

    # Damped, robust least squares. Coordinates are scaled in metres and
    # normal slopes so the trust region bounds both translation and rotation.
    for _ in range(18):
        samples = residual(x + perturb)
        err = samples[0]
        jac = (samples[1:] - err).T / 0.04
        weight = 1 / (1 + (err / 2.5) ** 2)
        delta = np.linalg.solve(
            jac.T @ (weight[:, None] * jac) + 2 * np.eye(5), -jac.T @ (weight * err)
        )
        delta /= max(1, np.linalg.norm(delta) / 1.3)
        choices = x + np.array([0.0, 0.35, 0.7, 1.0])[:, None] * delta * scale
        losses = cost(residual(choices))
        i = np.argmin(losses)
        x = choices[i]
        if i == 0:
            break
    return _normal(x[None, :])[0], x[:3], float(losses[i])
