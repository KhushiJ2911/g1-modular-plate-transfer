"""Shared, read-only privileged training labels for teacher and human rollouts."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from method.teacher.diagnostics import PrivilegedSensors

# Nominal MovePlateG1 rack origins in the pinned upstream move_plates.py.
START_RACK_XY = np.array([0.55, 0.24])
TARGET_RACK_XY = np.array([0.55, -0.24])
STEP_KEYS = (
    "priv_plate_pos",
    "priv_plate_rel",
    "priv_target_site_rel",
    "priv_plate_tilt_deg",
    "priv_held",
)


def pelvis_relative(world_xyz, state) -> np.ndarray:
    """Translate by observed pelvis xyz, then rotate xy by negative pelvis yaw."""
    offset = np.asarray(world_xyz, dtype=float) - state[18:21]
    c, s = np.cos(state[21]), np.sin(state[21])
    return np.array(
        [c * offset[0] + s * offset[1], -s * offset[0] + c * offset[1], offset[2]],
        dtype=np.float32,
    )


class PrivilegedLabels:
    def __init__(self, env):
        inner = env.inner_env
        self.sensors = PrivilegedSensors(inner)
        self.offsets = {
            "priv_start_rack_offset": np.asarray(
                np.asarray(inner.rack_start.get_position())[:2] - START_RACK_XY, dtype=np.float32
            ),
            "priv_target_rack_offset": np.asarray(
                np.asarray(inner.rack_target.get_position())[:2] - TARGET_RACK_XY, dtype=np.float32
            ),
        }
        if any(v.shape != (2,) or not np.isfinite(v).all() for v in self.offsets.values()):
            raise ValueError("Expected finite xy rack offsets")
        self.rows = {key: [] for key in STEP_KEYS}

    def append(self, state, snapshot=None):
        """Sample before env.step; snapshot may be the teacher's live pre-action read."""
        state = np.asarray(state, dtype=float)
        if state.shape != (50,) or not np.isfinite(state).all():
            raise ValueError("Expected finite (50,) observed state for labels")
        if snapshot is None:
            snapshot = self.sensors.snapshot(len(self.rows["priv_held"]))
        plate = np.asarray(snapshot["pose"][:3], dtype=np.float32)
        self.rows["priv_plate_pos"].append(plate)
        self.rows["priv_plate_rel"].append(pelvis_relative(plate, state))
        self.rows["priv_target_site_rel"].append(pelvis_relative(snapshot["site_xyz"], state))
        self.rows["priv_plate_tilt_deg"].append(snapshot["tilt_deg"])
        self.rows["priv_held"].append(snapshot["held"])

    def arrays(self) -> dict[str, np.ndarray]:
        result = dict(self.offsets)
        for key, rows in self.rows.items():
            array = np.asarray(rows, dtype=np.bool_ if key == "priv_held" else np.float32)
            if key in STEP_KEYS[:3]:
                array = array.reshape(-1, 3)
            if not np.isfinite(array).all():
                raise ValueError(f"Nonfinite privileged label: {key}")
            result[key] = array
        return result


def save_npz_atomic(destination: Path, **arrays) -> Path:
    """Publish a complete archive only after compression, flush and fsync."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.stem}_",
            suffix=".npz",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination
