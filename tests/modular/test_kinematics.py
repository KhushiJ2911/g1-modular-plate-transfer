"""Fast analytic checks on the independent model. No environment rollout."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Candidate methods live in the worktree, outside the evaluator wheel.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from method.modular import layout
from method.modular.kinematics import (
    G1Kinematics,
    pelvis_to_world,
    rotation_vector,
    world_to_pelvis,
)


@pytest.fixture(scope="module")
def kin():
    return G1Kinematics()


@pytest.fixture
def state():
    s = np.zeros(50)
    s[layout.LEFT_ARM] = [-0.2, 0.25, 0.1, 0.65, -0.2, 0.15, 0.1]
    s[layout.RIGHT_ARM] = [-0.3, -0.3, -0.05, 0.85, 0.1, -0.05, 0.2]
    s[18:22] = [1.1, -0.3, 0.74, np.pi / 2]
    return s


def test_layout_landmarks(state):
    norm = layout.normalize_action(layout.ACTION_LOW)
    np.testing.assert_array_equal(norm, -np.ones(20))
    np.testing.assert_array_equal(layout.normalize_action(layout.ACTION_HIGH), np.ones(20))
    raw = layout.denormalize_action(np.linspace(-1, 1, 20))
    np.testing.assert_allclose(layout.normalize_action(raw), np.linspace(-1, 1, 20), atol=2e-7)
    p = np.array([0.4, 0.25, 0.12])
    np.testing.assert_allclose(pelvis_to_world(state, p), [0.85, 0.1, 0.86])
    np.testing.assert_allclose(world_to_pelvis(state, pelvis_to_world(state, p)), p)


@pytest.mark.parametrize("side,arm", [("left", layout.LEFT_ARM), ("right", layout.RIGHT_ARM)])
def test_jacobian(kin, state, side, arm):
    kin.set_state(state)
    jac = kin.jacobian(side)
    eps = 1e-5
    finite_difference = []
    for joint in range(7):
        delta = np.zeros_like(state)
        delta[arm.start + joint] = eps
        kin.set_state(state + delta)
        p_plus, r_plus = kin.hand_pose(side)
        kin.set_state(state - delta)
        p_minus, r_minus = kin.hand_pose(side)
        finite_difference.append(
            np.r_[p_plus - p_minus, rotation_vector(r_plus @ r_minus.T)] / (2 * eps)
        )
    np.testing.assert_allclose(jac, np.array(finite_difference).T, atol=1e-7)


@pytest.mark.parametrize(
    "side,arm,action",
    [
        ("left", layout.LEFT_ARM, layout.LEFT_ARM_ACTION),
        ("right", layout.RIGHT_ARM, layout.RIGHT_ARM_ACTION),
    ],
)
def test_ik(kin, state, side, arm, action):
    kin.set_state(state)
    p, r = kin.hand_pose(side)
    target_state = state.copy()
    target_state[arm] += [0.1, -0.12, 0.08, 0.19, -0.09, 0.04, -0.15]
    kin.set_state(target_state)
    goal_p, goal_r = kin.hand_pose(side)
    kin.set_state(state)
    for orientation in (None, goal_r):
        q = kin.ik(side, goal_p, orientation)
        np.testing.assert_array_equal(kin.hand_pose(side)[0], p)
        np.testing.assert_array_equal(kin.hand_pose(side)[1], r)
        check = state.copy()
        check[arm] = q
        kin.set_state(check)
        actual_p, actual_r = kin.hand_pose(side)
        np.testing.assert_allclose(actual_p, goal_p, atol=2e-5)
        if orientation is not None:
            np.testing.assert_allclose(actual_r, goal_r, atol=5e-5)
        kin.set_state(state)
    bounded = kin.ik(side, [10.0, 10.0, 10.0], q_init=np.full(7, 9.0))
    assert np.all(bounded >= layout.ACTION_LOW[action])
    assert np.all(bounded <= layout.ACTION_HIGH[action])
