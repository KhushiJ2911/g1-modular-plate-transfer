"""Observation-only controller with a supervised continuous perception correction."""

import os
from pathlib import Path

from method.modular.refined import attach

from .policy_modular import make_policy as make_baseline


def make_policy():
    default = Path(__file__).resolve().parents[1] / "checkpoints/astra/normal_contact_v1.npz"
    return attach(make_baseline(), os.environ.get("ADV_REFINEMENT") or default)
