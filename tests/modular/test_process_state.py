"""Check process-wide inference invariants. test_process_isolation.sh compares two interpreters."""

import hashlib
import pickle
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pytest
from test_combined_reset import HEAD, _exercise


def _global_data():
    """Snapshot candidate module/class data, including geometric tables and cache statistics."""
    snapshot = {}
    containers = (np.ndarray, list, tuple, set, frozenset, dict, deque)
    scalars = (str, bytes, bool, int, float, np.generic, type(None))

    def record(namespace, name, value):
        key = namespace + "." + name
        if isinstance(value, containers + scalars):
            snapshot[key] = hashlib.sha256(pickle.dumps(value, protocol=5)).digest()
        elif callable(getattr(value, "cache_info", None)):
            snapshot[key] = value.cache_info()
        if callable(value) and isinstance(getattr(value, "__defaults__", None), tuple):
            snapshot[key + " defaults"] = pickle.dumps(value.__defaults__, protocol=5)

    for name, module in sys.modules.copy().items():
        if not name.startswith(("method.modular.", "submission.")) or module is None:
            continue
        for attr, value in vars(module).items():
            if attr.startswith("__"):
                continue
            record(name, attr, value)
            if isinstance(value, type) and value.__module__ == name:
                for field, data in vars(value).items():
                    if not field.startswith("__"):
                        record(name + "." + attr, field, data)
    return snapshot


def _rng_state(torch):
    return (
        random.getstate(),
        pickle.dumps(np.random.get_state(), protocol=5),
        torch.get_rng_state().numpy().tobytes(),
    )


def _torch_settings(torch):
    return (
        torch.get_default_dtype(),
        torch.get_num_threads(),
        torch.get_num_interop_threads(),
        torch.is_grad_enabled(),
        torch.is_inference_mode_enabled(),
        torch.are_deterministic_algorithms_enabled(),
        torch.get_float32_matmul_precision(),
        torch.backends.mkldnn.enabled,
    )


def _probe(warmup, observations=None, full_prior=False):
    import torch

    from submission.policy_modular_combined import make_policy

    policy = make_policy()
    model = policy.task.estimator.model
    assert all(not layer.training for layer in model.modules())
    buffers = {key: data.clone() for key, data in model.named_buffers()}
    features = []

    def observe_forward(layer, args, output):
        assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
        assert all(not child.training for child in layer.modules())
        features.append(output.cpu().numpy().copy())

    hook = model.register_forward_hook(observe_forward)
    try:
        rng_before = _rng_state(torch)
        settings_before = _torch_settings(torch)
        globals_before = _global_data()
        if warmup:
            policy.reset()
            if full_prior:
                # One full evaluator horizon of public calls, with no forced
                # phase transitions or access to the controller's state.
                rng = np.random.default_rng(17)
                state = np.zeros(50, dtype=np.float32)
                state[20] = state[48] = 0.78
                for _ in range(1700):
                    state[:18] = rng.uniform(-0.16, 0.16, 18)
                    rgb = rng.integers(256, size=(3, 3, 84, 84), dtype=np.uint8)
                    policy({"state": state, "rgb": rgb})
            else:
                _exercise(policy, 17)
            features.clear()
        policy.reset()
        if observations is None:
            actions = _exercise(policy, 29)
        else:
            with np.load(observations, allow_pickle=False) as frames:
                actions = np.stack(
                    [
                        policy({"state": state, "rgb": rgb})
                        for state, rgb in zip(frames["state"], frames["rgb"], strict=True)
                    ]
                )
        assert actions.shape[0] == len(features)
        assert _rng_state(torch) == rng_before
        assert _torch_settings(torch) == settings_before
        after = _global_data()
        assert globals_before.keys() == after.keys()
        assert [key for key in after if after[key] != globals_before[key]] == []
        for key, data in model.named_buffers():
            assert torch.equal(buffers[key], data), key
        return actions, np.concatenate(features)
    finally:
        hook.remove()


def test_process_inference_invariants(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    # The evaluator's T4 renders cameras; the policy image installs CPU Torch.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    folder = Path(__file__).resolve().parents[2] / "checkpoints"
    est = folder / "est_v1b/last.pt"
    head = folder / "astra" / f"{HEAD}.npz"
    if not (est.is_file() and head.is_file()):
        pytest.skip("requires frozen policy weights")
    monkeypatch.setenv("ADV_ESTIMATOR", str(est))
    monkeypatch.setenv("ADV_HEADS", str(head))
    for used, fresh in zip(_probe(True), _probe(False), strict=True):
        np.testing.assert_array_equal(used, fresh)


if __name__ == "__main__":
    # The shell starts these workers sequentially, allowing kernel- and library-level
    # caches to be cold on B in one worker and warm from A in the other.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=("fresh", "reused"))
    parser.add_argument("output")
    parser.add_argument("observations", nargs="?")
    parser.add_argument("--full-prior", action="store_true")
    args = parser.parse_args()
    actions, features = _probe(args.case == "reused", args.observations, args.full_prior)
    with open(args.output, "wb") as stream:
        np.save(stream, actions, allow_pickle=False)
        np.save(stream, features, allow_pickle=False)
