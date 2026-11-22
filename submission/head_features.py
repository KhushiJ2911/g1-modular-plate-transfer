"""Frozen v1b features plus observed pickup and rack geometry."""

import numpy as np


def head_inputs(features, state, plate, site, held, anchor):
    return np.concatenate(
        (
            features,
            state[..., :9],
            state[..., 18:22],
            state[..., 44:45],
            plate - anchor,
            plate - site,
            np.asarray(held)[..., None],
        ),
        axis=-1,
    )
