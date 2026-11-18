"""CPU synthetic collector checks. Fake sensors are not policy rollout evidence."""

import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .collect import append_summary, collect_episode, parse_seeds, save_npz_atomic
from .diagnostics import gripper_label, outcome_category


def snapshot(step):
    return {
        "step": step,
        "pose": [0, 0, 0 if step == 0 else 0.04, 1, 0, 0, 0],
        "site_distance": 0.05 if step >= 2 else 0.2,
        "site_xy_distance": 0.02,
        "tilt_deg": 20,
        "held": step in (1, 2),
        "contacts": {"target_rack": step >= 2, "floor": False, "table": False},
    }


class FakeSensors:
    def __init__(self, inner):
        pass

    def snapshot(self, step):
        return snapshot(step)


class FakeTimestep:
    def __init__(self, step):
        self.low_dim_obs = np.full(50, step, np.float32)
        self.rgb_obs = np.full((3, 3, 84, 84), step, np.uint8)

    def last(self):
        return False


class FakeEnvironment:
    def __init__(self):
        self.inner_env = self
        self._advaitic_initial_slot = 3
        self.actions = []

    def reset(self, seed):
        self.actions = []
        return FakeTimestep(0)

    def step(self, action):
        self.actions.append(action.copy())
        return FakeTimestep(len(self.actions))

    def episode_fell(self):
        return False

    def episode_succeeded(self):
        return False

    def episode_termination(self):
        return "fake_timeout"


class FakeProcess:
    def reset(self):
        self.calls = []

    def act(self, obs):
        self.calls.append(int(obs["state"][0]))
        return np.full((3, 20), 0.2, np.float32)


def main():
    assert list(parse_seeds("2400000:2400002")) == [2400000, 2400001]
    for spec in ("1999999:2000001", "7310000:7310030", "3000000:3000001", "2400001:2400000"):
        try:
            parse_seeds(spec)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted non-training seeds: {spec}")
    s = snapshot(1)
    s["pose"][2] = 0.03
    assert gripper_label(s, 0) == 1
    s["pose"][2] = 0.029
    assert gripper_label(s, 0) is None
    s["pose"][2] = 0.04
    s["held"] = False
    assert gripper_label(s, 0) is None
    s = snapshot(2)
    assert gripper_label(s, 0) == -1  # Equality on both open thresholds.
    for key, value in (("site_distance", 0.05001), ("tilt_deg", 20.001)):
        bad = copy.deepcopy(s)
        bad[key] = value
        assert gripper_label(bad, 0) is None  # Rack contact excludes the closed label.
    s["contacts"]["target_rack"] = False
    assert gripper_label(s, 0) == 1
    assert outcome_category(False, False, 0, None, snapshot(0), 0) == "no_grasp"
    assert outcome_category(False, False, 0.04, 2, snapshot(2), 0) == "placed_not_released"
    assert outcome_category(False, False, 0.04, 2, snapshot(3), 0) == "reached_target_then_failed"

    with patch("method.dagger.collect.PrivilegedSensors", FakeSensors):
        for mode in (False, True):
            env, process = FakeEnvironment(), FakeProcess()
            record, arrays = collect_episode(
                env, process, 2400000, max_steps=10, stop_after_placed=2, relabel_execute=mode
            )
            assert process.calls == [0, 3]  # Full chunks; stop in the second chunk.
            assert record["steps"] == 4 and record["termination"] == "stop_after_placed"
            assert record["label_counts"] == {"open": 2, "closed": 1, "undefined": 1}
            assert record["label_fractions"] == {"open": 0.5, "closed": 0.25, "undefined": 0.25}
            assert arrays["state"][:, 0].tolist() == [0, 1, 2, 3]
            assert arrays["rgb"][:, 0, 0, 0, 0].tolist() == [0, 2]
            assert arrays["action"][:, 18].tolist()[1:] == [1, -1, -1]
            assert arrays["label_mask"][:, 18].tolist() == [False, True, True, True]
            assert arrays["label_mask"].sum() == 3
            np.testing.assert_array_equal(arrays["action_executed"], env.actions)
            if mode:
                np.testing.assert_array_equal(arrays["action_executed"], arrays["action"])
            else:
                assert (arrays["action_executed"] == np.float32(0.2)).all()
            assert bool(arrays["relabel_execute"]) == mode
        record, arrays = collect_episode(
            FakeEnvironment(),
            FakeProcess(),
            2400001,
            max_steps=1,
            stop_after_placed=150,
            relabel_execute=False,
        )
        assert record["steps"] == 1 and record["termination"] == "max_steps"
    with tempfile.TemporaryDirectory(prefix="dagger-smoke-") as temporary:
        root = Path(temporary)
        path = root / "ep_2400001.npz"
        save_npz_atomic(path, **arrays, summary_json=np.asarray(json.dumps(record)))
        with np.load(path, allow_pickle=False) as episode:
            assert episode["label_mask"].dtype == np.bool_
            recovered = json.loads(episode["summary_json"].item())
        summary = root / "summary.jsonl"
        summary.write_text('{"interrupted":')
        append_summary(summary, recovered)
        append_summary(summary, recovered)
        assert len(summary.read_text().splitlines()) == 2
        assert json.loads(summary.read_text().splitlines()[-1]) == recovered
    print(
        "DAgger synthetic smoke passed: oracle, chunks, pre-action alignment, execution modes, cutoffs, atomic save/log recovery. No task-success evidence."
    )


if __name__ == "__main__":
    main()
