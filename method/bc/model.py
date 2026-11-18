"""Training facade for the inference network covered by the submission digest."""

from submission.model import (
    ACTIVE_DIMS,
    GRIPPER_INDEX,
    PROPRIO_DIMS,
    BCModel,
    action_loss_terms,
    action_masks,
    auxiliary_loss,
    load_bc_weights,
    masked_l1,
    validate_gripper_config,
    validate_proprio_dims,
)

__all__ = [
    "ACTIVE_DIMS",
    "GRIPPER_INDEX",
    "PROPRIO_DIMS",
    "BCModel",
    "action_loss_terms",
    "action_masks",
    "auxiliary_loss",
    "load_bc_weights",
    "masked_l1",
    "validate_gripper_config",
    "validate_proprio_dims",
]
