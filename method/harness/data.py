"""Dataset guard shared by training-only and offline tools; never imported by policy."""

import numpy as np

LABEL_KEYS = ("priv_plate_rel", "priv_target_site_rel", "priv_plate_tilt_deg", "priv_held")
RESERVED_SEEDS = ((2_600_000, 2_700_000), (7_310_000, 7_320_000))


def check_seed(seed):
    seed = int(seed)
    if any(start <= seed < stop for start, stop in RESERVED_SEEDS):
        raise ValueError(f"Reserved confirm/test seed refused: {seed}")
    return seed


def audit_files(files):
    """Run before cache construction, so forbidden data cannot enter normalization."""
    seeds = set()
    for path in files:
        try:
            with np.load(path, allow_pickle=False) as episode:
                seed = check_seed(episode["seed"])
        except (OSError, EOFError, ValueError, KeyError) as exc:
            raise ValueError(f"{path}: cannot audit seed: {exc}") from exc
        if seed in seeds:
            raise ValueError(f"Duplicate seed {seed}: could leak across episode splits")
        seeds.add(seed)


def pelvis_relative(world_xyz, state):
    """Exact teacher labels.py definition, including float64 state arithmetic."""
    state = np.asarray(state, dtype=float)
    offset = np.asarray(world_xyz, dtype=float) - state[18:21]
    c, s = np.cos(state[21]), np.sin(state[21])
    return np.array(
        [c * offset[0] + s * offset[1], -s * offset[0] + c * offset[1], offset[2]],
        dtype=np.float32,
    )


def snapshot_labels(snapshot, state):
    # Teacher rounds plate xyz to float32 BEFORE the pelvis transform, but leaves
    # site xyz as float64. Preserve this distinction for bit-for-bit equality.
    return {
        "priv_plate_rel": pelvis_relative(np.asarray(snapshot["pose"][:3], np.float32), state),
        "priv_target_site_rel": pelvis_relative(snapshot["site_xyz"], state),
        "priv_plate_tilt_deg": np.float32(snapshot["tilt_deg"]),
        "priv_held": np.bool_(snapshot["held"]),
    }
