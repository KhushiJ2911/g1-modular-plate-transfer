"""Tracking contracts, with fake environments only; no physics pose injection."""

from types import SimpleNamespace

import numpy as np
import pytest

from method.teacher.teacher import Teacher
from method.teacher.test_teacher import reset_geometry
from method.teacher.track_label import (
    TrackConfig,
    assert_state_exact,
    match_phase,
    publish_npz,
    replay_episode,
    strict_gripper_label,
    tracking_action,
    validate_seed,
)


def trajectory(n=20):
    states = np.zeros((n, 50), dtype=np.float32)
    states[:, :7] = np.arange(n)[:, None] / 10
    return states


def test_monotone_phase_for_backtracking_observations():
    demo = trajectory()
    previous = 0
    phases = []
    for i in [0, 3, 1, 9, 2, 15, 6]:
        phase, _ = match_phase(demo[i], demo, previous, TrackConfig(window=4))
        assert previous <= phase <= min(previous + 4, len(demo) - 1)
        phases.append(phase)
        previous = phase
    assert phases == [0, 3, 3, 7, 7, 11, 11]


def test_forward_window_inclusive_and_clipped():
    demo = trajectory()
    assert match_phase(demo[19], demo, 2, TrackConfig(window=4))[0] == 6
    assert match_phase(demo[19], demo, 18, TrackConfig(window=4))[0] == 19
    assert match_phase(demo[0], demo, 18, TrackConfig(window=4))[0] == 18
    assert match_phase(demo[19], demo, 2, TrackConfig(window=0))[0] == 2


def test_ties_choose_earliest_and_gripper_affects_metric():
    demo = np.zeros((10, 50), dtype=np.float32)
    assert match_phase(demo[0], demo, 3, TrackConfig())[0] == 3
    demo[6, 44] = 0.6
    assert match_phase(demo[6], demo, 3, TrackConfig())[0] == 6


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window": -1},
        {"window": 1.5},
        {"drop_cost": -1},
        {"arm_weight": float("nan")},
        {"arm_scale_rad": 0},
        {"arm_indices": (44,)},
        {"gripper_index": 60},
    ],
)
def test_bad_config(kwargs):
    with pytest.raises(ValueError):
        TrackConfig(**kwargs)


@pytest.mark.parametrize("seed", [2600000, 2699999, 7310000, 7319999])
def test_forbidden_seed(seed):
    with pytest.raises(ValueError, match="Refusing"):
        validate_seed(seed)


def test_exact_state_requires_equal_bits():
    state = np.zeros(50, dtype=np.float32)
    assert_state_exact(state.copy(), state, seed=1, step=0)
    actual = state.copy()
    actual[4] = -0.0
    with pytest.raises(RuntimeError, match=r"step 3.*dims=\[4\]"):
        assert_state_exact(actual, state, seed=1, step=3)


def snapshot(**overrides):
    return {
        "contacts": {"target_rack": False},
        "site_distance": 0.04,
        "tilt_deg": 10.0,
        "held": True,
        "pose": [0, 0, 1.04],
        **overrides,
    }


def test_strict_gripper_contact_distance_tilt_and_closed_rule():
    assert strict_gripper_label(snapshot(), 1.0) == 1.0
    assert strict_gripper_label(snapshot(contacts={"target_rack": True}), 1.0) == -1.0
    assert strict_gripper_label(snapshot(contacts={"target_rack": True}, tilt_deg=10.1), 1) is None
    assert (
        strict_gripper_label(snapshot(contacts={"target_rack": True}, site_distance=0.041), 1)
        is None
    )
    assert strict_gripper_label(snapshot(held=False), 1) is None
    assert strict_gripper_label(snapshot(pose=[0, 0, 1.02]), 1) is None


def test_action_uses_original_base_math_next_arm_and_no_waits():
    teacher = Teacher(mode="v1", max_wait=0)
    teacher.reset(reset_geometry(teacher))
    teacher._act_replay(SimpleNamespace(low_dim_obs=teacher.states[0][0]))
    phase = teacher.meta[0]["grasp_step"]
    state = teacher.states[0][phase].copy()
    state[18] -= 0.05
    ts = SimpleNamespace(low_dim_obs=state)
    teacher.demo_step = phase
    original = teacher._act_replay(ts)
    action, grip = tracking_action(teacher, ts, phase, snapshot(held=False), 1)
    np.testing.assert_array_equal(action[:4], original[:4])
    np.testing.assert_array_equal(action[4:], teacher.actions[0][phase + 1, 4:])
    assert grip is None
    assert teacher.wait_steps == {"grasp": 0, "release": 0}
    action, _ = tracking_action(teacher, ts, phase, snapshot(), 1)
    assert action[18] == 1
    action, _ = tracking_action(teacher, ts, len(teacher.actions[0]) - 1, snapshot(held=False), 1)
    np.testing.assert_array_equal(action[4:], teacher.actions[0][-1, 4:])


class ReplayEnv:
    def __init__(self, teacher, states, corrupt_at=None):
        self.inner_env = reset_geometry(teacher).inner_env
        self.states = states
        self.corrupt_at = corrupt_at
        self.executed = []

    def timestep(self):
        state = self.states[min(self.index, len(self.states) - 1)].copy()
        if self.index == self.corrupt_at:
            state[0] += 1e-5
        return SimpleNamespace(low_dim_obs=state, last=lambda: self.index >= len(self.states))

    def reset(self, seed):
        self.index = 0
        return self.timestep()

    def step(self, action):
        self.executed.append(action.copy())
        self.index += 1
        return self.timestep()


def test_replay_only_executes_original_actions_and_masks_whole_rows():
    teacher = Teacher(max_wait=0)
    states = teacher.states[0][:3].copy()
    states[1:, :7] += 2
    executed = np.full((3, 20), 0.25, dtype=np.float32)
    arrays = {
        "seed": np.int64(10),
        "state": states,
        "action_executed": executed,
        "action": executed.copy(),
    }
    env = ReplayEnv(teacher, states)
    output, report = replay_episode(env, teacher, arrays, TrackConfig(drop_cost=4))
    np.testing.assert_array_equal(env.executed, executed)
    np.testing.assert_array_equal(arrays["action"], executed)
    assert output["label_mask"][0].all()
    assert not output["label_mask"][1:].any()
    assert report["verified_states"] == 3
    with pytest.raises(RuntimeError, match="NOT bit-for-bit"):
        replay_episode(ReplayEnv(teacher, states, corrupt_at=1), teacher, arrays, TrackConfig())


def test_atomic_publication_never_overwrites(tmp_path):
    path = tmp_path / "episode.npz"
    publish_npz(path, {"test": np.array([1])})
    with pytest.raises(FileExistsError):
        publish_npz(path, {"test": np.array([2])})
    with np.load(path) as data:
        np.testing.assert_array_equal(data["test"], [1])
    assert len(list(tmp_path.iterdir())) == 1
