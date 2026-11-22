"""Merged pickup policy with optional continuous placement corrections."""

import os

from method.modular.site_estimator import PlacementEstimate

from .policy_modular_heads import make_policy as make_pickup


def _make(groups):
    policy = make_pickup()
    policy.task = PlacementEstimate(policy.task, policy.controller, os.environ["ADV_HEADS"], groups)
    policy.controller.task_state_source = policy.task
    return policy


def make_site():
    return _make(("site",))


def make_plate():
    return _make(("plate",))


def make_policy():
    return _make(("site", "plate"))
