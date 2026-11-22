"""Use the calibrated release fallback followed by open-hand base clearance."""

from method.modular.base_clearance import ClearancePolicy
from method.modular.fallback_seating import make_policy as make_seated_policy


def make_policy():
    return ClearancePolicy(make_seated_policy())
