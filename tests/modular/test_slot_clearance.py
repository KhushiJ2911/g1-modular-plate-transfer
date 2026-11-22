from types import SimpleNamespace

import numpy as np
import pytest

from method.modular import layout
from method.modular.phases import Phase


def test_opening_feedback_tracks_rack_normal_and_resets_reference():
    pytest.importorskip("torch")
    from submission.policy_slot_clearance import OpeningPlaneFeedback

    class Arm:
        def set_state(self, state):
            self.hand = state[:3]

        def hand_pose(self, side):
            return self.hand, np.eye(3)

        def jacobian(self, side):
            return np.eye(6, 7)

    class Policy:
        def __init__(self):
            self.controller = SimpleNamespace(kin=Arm(), target_rot=np.eye(3)[:, [2, 0, 1]])

        def reset(self):
            self.controller.phase = Phase.LOWER
            self.controller.last_raw = (layout.ACTION_LOW + layout.ACTION_HIGH) / 2

        def __call__(self, obs):
            return layout.normalize_action(self.controller.last_raw)

    policy = OpeningPlaneFeedback(Policy())
    ctl = policy.controller
    state = np.zeros(50)
    state[layout.PELVIS_YAW] = np.pi / 2
    obs = {"state": state}
    original = ctl.last_raw.copy()
    baseline = layout.normalize_action(original)
    np.testing.assert_array_equal(policy(obs), baseline)
    ctl.phase = Phase.RELEASE
    np.testing.assert_array_equal(policy(obs), baseline)
    state[1] += 0.04  # World -x, within the rack plane.
    np.testing.assert_allclose(policy(obs), baseline, atol=1e-7)
    ctl.phase = Phase.SETTLE
    state[0] += 0.04  # World +y, along the rack normal.
    corrected = policy(obs)
    expected = original.copy()
    expected[4] -= 0.012
    np.testing.assert_allclose(corrected, layout.normalize_action(expected), atol=1e-7)
    ctl.phase = Phase.RETREAT
    np.testing.assert_array_equal(policy(obs), corrected)
    policy.reset()
    ctl.phase = Phase.RELEASE
    np.testing.assert_array_equal(policy(obs), baseline)
