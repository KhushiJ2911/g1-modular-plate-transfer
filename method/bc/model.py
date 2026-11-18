"""Training facade for the inference network covered by the submission digest."""

from submission.model import (
    ACTIVE_DIMS,
    PROPRIO_DIMS,
    BCModel,
    auxiliary_loss,
    load_bc_weights,
    masked_l1,
    validate_proprio_dims,
)

__all__ = [
    "ACTIVE_DIMS",
    "PROPRIO_DIMS",
    "BCModel",
    "auxiliary_loss",
    "load_bc_weights",
    "masked_l1",
    "validate_proprio_dims",
]
