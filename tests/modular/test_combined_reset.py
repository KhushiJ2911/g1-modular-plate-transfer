"""Exercise the complete checkpoint-backed policy without a simulator or camera renderer."""

import hashlib
from collections import deque
from pathlib import Path
from types import FunctionType

import mujoco
import numpy as np
import pytest

from method.modular.kinematics import pelvis_rotation
from method.modular.phases import Phase

HEAD = "64dfe105e7f716c0ab5b7a9c487b7a8a1cac06565070b390e75992dfbee4deff"


def _snapshot(policy, torch):
    """Include the Python object graph, hook ownership, weights and native robot state."""
    fields = {}
    seen = {}

    def visit(path, value):
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
            fields[path] = (str(value.device), value.requires_grad, _array_signature(array))
        elif isinstance(value, np.ndarray):
            fields[path] = _array_signature(value)
        elif isinstance(value, (mujoco.MjModel, mujoco.MjData)):
            # MuJoCo's own serialized state omits pointers and unused arena memory.
            fields[path] = (type(value), hashlib.sha256(value.__getstate__()).digest())
            if isinstance(value, mujoco.MjData):
                for field in ("qpos", "xpos", "xmat", "site_xpos", "site_xmat", "subtree_com"):
                    visit(f"{path}.{field}", getattr(value, field))
        elif isinstance(value, dict):
            fields[path] = type(value)
            # PyTorch assigns hook IDs process-wide. Their order and bound owners matter.
            items = (
                enumerate(value.values()) if path.endswith("._forward_pre_hooks") else value.items()
            )
            for key, child in items:
                visit(f"{path}[{key!r}]", child)
        elif isinstance(value, (tuple, list, deque)):
            fields[path] = (type(value), getattr(value, "maxlen", None))
            for index, child in enumerate(value):
                visit(f"{path}[{index}]", child)
        elif isinstance(value, FunctionType):
            fields[path] = value.__code__
            for index, cell in enumerate(value.__closure__ or ()):
                visit(f"{path}.closure[{index}]", cell.cell_contents)
        elif value is None or isinstance(
            value, (str, bytes, float, int, set, torch.device, np.generic)
        ):
            fields[path] = (type(value), value)
        elif id(value) in seen:
            fields[path] = ("ref", seen[id(value)])
        else:
            seen[id(value)] = path
            fields[path] = type(value)
            for key, child in sorted(vars(value).items()):
                visit(f"{path}.{key}", child)

    visit("policy", policy)
    return fields


def _array_signature(value):
    return value.dtype.str, value.shape, hashlib.sha256(value.tobytes()).digest()


def _exercise(policy, seed):
    """Call the real policy on synthetic observations, entering late phases explicitly."""
    rng = np.random.default_rng(seed)
    ctl = policy.controller
    actions = []
    for phase in Phase:
        state = np.zeros(50, dtype=np.float32)
        state[:18] = rng.uniform(-0.16, 0.16, 18)
        state[18:22] = (0.13, -0.21, 0.78, 0.15)
        state[46:50] = state[18:22]
        if phase in (Phase.GRASP, Phase.LIFT, Phase.CARRY, Phase.ALIGN, Phase.LOWER):
            state[44] = 0.95
        if ctl.phase != phase:
            # Static images cannot complete a physical transfer. Invoke the real
            # entry handlers so even a prior late-release episode is represented.
            ctl.machine._move(phase, "reset test")
            ctl.kin.set_state(state)
            hand, rot = ctl.kin.hand_pose("left")
            basis = pelvis_rotation(state)
            if phase == Phase.RELEASE:
                # Represent a preceding episode that used the late FK fallback.
                ctl.proprio_arming = True
                ctl.fk_plate_z = ctl.target_xyz[2] - 0.03
            ctl._enter(policy.task.snapshot(last_state), state[18:21] + basis @ hand, basis @ rot)
        if phase == Phase.HOLD:
            ctl.machine.elapsed = 110
        for _ in range(7 if phase == Phase.APPROACH else 2):
            rgb = rng.integers(256, size=(3, 3, 84, 84), dtype=np.uint8)
            actions.append(policy({"rgb": rgb, "state": state}).copy())
        last_state = state
    assert ctl.fallback_releases and ctl.site_contacts.site_filter.site is not None
    assert ctl.lift_rot is not None and policy.recovery.samples
    return np.stack(actions)


def test_combined_reset_equals_fresh(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from submission.policy_modular_combined import make_policy

    checkpoints = Path(__file__).resolve().parents[2] / "checkpoints"
    estimator = checkpoints / "est_v1b/last.pt"
    heads = checkpoints / "astra" / f"{HEAD}.npz"
    if not estimator.is_file() or not heads.is_file():
        pytest.skip("requires the frozen est_v1b and calibrated contact checkpoints")
    monkeypatch.setenv("ADV_ESTIMATOR", str(estimator))
    monkeypatch.setenv("ADV_HEADS", str(heads))
    fresh = make_policy()
    baseline = _snapshot(fresh, torch)
    used = make_policy()
    _exercise(used, 17)
    used.reset()
    after = _snapshot(used, torch)
    missing = object()
    differences = [
        k
        for k in sorted(after.keys() | baseline.keys())
        if after.get(k, missing) != baseline.get(k, missing)
    ]
    np.testing.assert_array_equal(_exercise(used, 29), _exercise(fresh, 29))
    assert not differences, "\n".join(differences)
