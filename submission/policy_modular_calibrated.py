"""Calibrate the loaded grasp of the contact-head controller from CARRY onward."""

import os

import numpy as np

from .policy_modular_heads import make_policy as make_contact_policy


def make_policy():
    policy = make_contact_policy()
    with np.load(os.environ["ADV_HEADS"], allow_pickle=False) as checkpoint:
        policy.controller.set_loaded_grasp_normal(checkpoint["loaded_normal"])
    return policy


def make_plate_policy():
    """Use the corrected visual plate center during transport and placement."""
    from method.modular.site_estimator import PlacementEstimate

    policy = make_policy()
    policy.task = PlacementEstimate(
        policy.task, policy.controller, os.environ["ADV_HEADS"], ("plate",)
    )
    policy.controller.task_state_source = policy.task
    return policy
