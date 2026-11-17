"""Training facade for the inference network covered by the submission digest."""

from submission.model import ACTIVE_DIMS, PROPRIO_DIMS, BCModel, masked_l1

__all__ = ["ACTIVE_DIMS", "PROPRIO_DIMS", "BCModel", "masked_l1"]
