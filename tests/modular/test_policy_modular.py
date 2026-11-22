"""Local end-to-end policy smoke with the released cameras and worker protocol."""

import os
import sys
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest

from advaitic_eval.environment import make_environment, observation, validate_actions
from advaitic_eval.policy_process import PolicyProcess


@pytest.mark.integration
def test_modular_worker(tmp_path):
    root = Path(__file__).resolve().parents[2]
    checkpoint = Path(os.environ.get("ADV_ESTIMATOR") or root / "checkpoints/est_v1b/last.pt")
    if not checkpoint.is_file():
        pytest.skip("requires the est_v1b checkpoint")
    if not os.environ.get("ADV_POLICY_PYTHON") and find_spec("torch") is None:
        pytest.skip("set ADV_POLICY_PYTHON to the policy environment with PyTorch")
    env = make_environment()
    proc = PolicyProcess(
        "submission.policy_modular:make_policy",
        root,
        tmp_path / "policy.log",
        python=os.environ.get("ADV_POLICY_PYTHON", sys.executable),
    )
    try:
        timestep = env.reset(seed=1_000_000)
        proc.reset()
        first = observation(timestep)
        for step in range(4):
            actions = validate_actions(proc.act(observation(timestep)))
            assert actions.shape == (1, 20)
            if step == 0:
                first_action = actions.copy()
            timestep = env.step(actions[0])
        proc.reset()
        np.testing.assert_allclose(validate_actions(proc.act(first)), first_action, atol=1e-6)
    finally:
        proc.close()
        env.close()
