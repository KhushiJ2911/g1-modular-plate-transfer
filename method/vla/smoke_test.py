"""CPU-only synthetic checks; no simulator rollout or success-rate claim."""

import copy
import json
import os
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from method.bc.dataset import EpisodeStore
from method.bc.train import atomic_save
from method.vla.model import (
    ACTIVE_DIMS,
    GRIPPER_INDEX,
    STATE_SIGMAS,
    MicroVLA,
    flow_loss_terms,
    instruction_tokens,
    model_config,
    parameter_counts,
    prepare_images,
    token_counts,
    valid_mask,
)
from method.vla.train import Config, train
from submission.policy_vla import make_policy

TINY = dict(
    width=32,
    depth=1,
    heads=2,
    head_dim=16,
    mlp=64,
    expert_width=32,
    expert_mlp=64,
    vit_width=24,
    vit_depth=1,
    vit_heads=3,
)


def synthetic_store(root):
    rng = np.random.default_rng(13)
    directories = []
    for source in range(2):
        directory = root / f"source{source}"
        directory.mkdir()
        directories.append(str(directory))
        for ep in range(4):
            length = 9
            action = np.zeros((length, 20), np.float32)
            action[:, 18] = 1
            # Rare open mode with identical observations (one open in 9 labels).
            action[-1, 18] = -1
            mask = np.ones_like(action, dtype=bool)
            if source:
                mask[:] = False
                mask[:, 18] = True
                action[:, 4] = 0.937  # Untrusted student arm labels.
            np.savez(
                directory / f"ep_{ep}.npz",
                rgb=np.full((5, 3, 3, 84, 84), 120, np.uint8),
                state=rng.normal(size=(length, 50)).astype(np.float32),
                action=action,
                label_mask=mask,
                rgb_stride=2,
            )
    # Only fake data get a new, private, temporary BC-format fixture cache.
    return EpisodeStore(directories, root / "fixture_cache", val_episodes=1)


def checks(root):
    torch.manual_seed(123)
    store = synthetic_store(root)
    before = {
        p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in (root / "fixture_cache").iterdir()
    }
    reopened = EpisodeStore(
        [str(root / "source0"), str(root / "source1")],
        root / "fixture_cache",
        val_episodes=1,
        read_only=True,
    )
    assert reopened.digest == store.digest
    assert before == {
        p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in (root / "fixture_cache").iterdir()
    }
    try:
        EpisodeStore([str(root / "source0")], root / "absent", val_episodes=1, read_only=True)
    except FileNotFoundError:
        assert not (root / "absent").exists()
    else:
        raise AssertionError("Missing read-only cache did not fail")
    data = store.dataset("train", 20)
    assert data.sample_weights([1, 3]).shape == (len(data),)
    groups = np.repeat(store.directory_ids[data.episode_ids], np.diff(np.r_[0, data.ends]))
    assert np.isclose(data.sample_weights([1, 3])[groups == 1].sum(), 0.75)
    full = data[0]
    partial = data[len(data) // 2]
    assert partial["label_mask"][:, :GRIPPER_INDEX].sum() == 0
    assert full["pad_mask"].sum() > 0
    batch = {k: torch.from_numpy(np.stack([full[k], partial[k]])) for k in full}
    model = MicroVLA(model_config(**TINY)).train()
    images = prepare_images(batch["rgb"], 96)
    state = (batch["proprio"] / STATE_SIGMAS).clamp(-1, 1)
    instr = instruction_tokens(2, "cpu")
    mask = valid_mask(batch)
    noise = torch.randn_like(batch["target"])
    t = torch.tensor([0.4, 0.7])

    def terms(actions, active_mask=mask):
        return flow_loss_terms(model, images, state, instr, actions, active_mask, noise=noise, t=t)

    first, n = terms(batch["target"])
    corrupted = batch["target"].clone()
    corrupted[~mask] = float("nan")
    assert torch.equal(first, terms(corrupted)[0])
    assert int(n) == int(mask.sum())
    targets = batch["target"].clone().requires_grad_()
    total, _ = terms(targets)
    total.backward()
    assert targets.grad[~mask].abs().sum() == 0
    model.zero_grad(set_to_none=True)
    empty, count = terms(corrupted, torch.zeros_like(mask))
    empty.backward()
    assert empty.item() == 0 and count.item() == 0
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in model.parameters())

    # A logical-batch denominator also gives the exact full-batch gradient.
    model.zero_grad(set_to_none=True)
    (terms(batch["target"])[0] / n).backward()
    grads = {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    for j in range(2):
        loss_sum, _ = flow_loss_terms(
            model,
            images[j : j + 1],
            state[j : j + 1],
            instr[j : j + 1],
            batch["target"][j : j + 1],
            mask[j : j + 1],
            noise=noise[j : j + 1],
            t=t[j : j + 1],
        )
        (loss_sum / n).backward()
    for k, p in model.named_parameters():
        if k in grads:
            torch.testing.assert_close(p.grad, grads[k], atol=2e-6, rtol=2e-4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    initial = float(terms(batch["target"])[0].detach() / n)
    for _ in range(45):
        optimizer.zero_grad(set_to_none=True)
        loss_sum, n = terms(batch["target"])
        (loss_sum / n).backward()
        optimizer.step()
    final = float(terms(batch["target"])[0].detach() / n)
    assert final < initial * 0.35, (initial, final)
    model.eval()
    with torch.no_grad():
        cached = model.sample(images, state, instr, noise=noise, num_steps=3, cached=True)
        uncached = model.sample(images, state, instr, noise=noise, num_steps=3, cached=False)
    torch.testing.assert_close(cached, uncached, atol=1e-5, rtol=1e-4)

    config = Config(
        data_dirs=tuple(str(root / f"source{i}") for i in range(2)),
        data_weights=(1, 3),
        cache_dir=str(root / "fixture_cache"),
        run_dir=str(root / "continuous"),
        steps=3,
        batch_size=4,
        microbatch_size=2,
        val_episodes=1,
        val_batches=1,
        sample_batches=1,
        val_every=2,
        sample_every=2,
        save_every=2,
        log_every=1,
        warmup_steps=0,
        flow_steps=2,
        device="cpu",
        model_overrides=TINY,
    )
    continuous = train(config)
    split = replace(config, run_dir=str(root / "resumed"))
    train(split, stop_after=1)
    resumed = train(split, resume=True)
    a, b = [torch.load(p, weights_only=False) for p in (continuous, resumed)]
    assert a["step"] == b["step"] == 3
    for key in ("model", "ema"):
        for name in a[key]:
            torch.testing.assert_close(a[key][name], b[key][name], rtol=0, atol=0)
    for param_id, values in a["optimizer"]["state"].items():
        for key, value in values.items():
            torch.testing.assert_close(
                value, b["optimizer"]["state"][param_id][key], rtol=0, atol=0
            )
    assert torch.equal(a["rng"]["torch"], b["rng"]["torch"])
    # Make the sampled output reliably negative to verify open sign + mapping.
    policy_checkpoint = copy.deepcopy(b)
    policy_checkpoint["ema"]["action_out.weight"].zero_()
    policy_checkpoint["ema"]["action_out.bias"].zero_()
    policy_checkpoint["ema"]["action_out.bias"][GRIPPER_INDEX] = 10
    policy_path = root / "policy.pt"
    atomic_save(policy_checkpoint, policy_path)
    with patch.dict(os.environ, ADV_CKPT=str(policy_path), ADV_FLOW_STEPS="3"):
        policy = make_policy()
        assert policy.flow_steps == 3
        obs = dict(rgb=np.zeros((3, 3, 84, 84), np.uint8), state=np.zeros(50, np.float32))
        with (
            patch.object(policy.model, "prefix_cache", wraps=policy.model.prefix_cache) as prefix,
            patch.object(
                policy.model, "cached_velocity", wraps=policy.model.cached_velocity
            ) as velocity,
        ):
            action = policy(obs)
            assert prefix.call_count == 1 and velocity.call_count == 3
        assert action.shape == (10, 20) and action.dtype == np.float32
        assert np.isfinite(action).all() and np.abs(action).max() <= 1
        assert np.all(action[:, 18] == -1)
        inactive = [d for d in range(20) if d not in ACTIVE_DIMS]
        np.testing.assert_array_equal(
            action[:, inactive],
            np.broadcast_to(store.stats["action_mean"][inactive], (10, len(inactive))),
        )
        policy.reset()
        np.testing.assert_array_equal(action, policy(obs))
        assert not np.array_equal(action, policy(obs))
    # Check config identity enforcement.
    try:
        train(replace(split, horizon=21), resume=True)
    except ValueError as exc:
        assert "Resume config changed" in str(exc)
    else:
        raise AssertionError("Changed resume config accepted")

    # No labels means no Adam state, weight decay, momentum or EMA update.
    empty_store = copy.copy(store)
    empty_store.label_mask = np.zeros_like(store.label_mask)
    empty_config = replace(config, steps=1, run_dir=str(root / "empty"))
    torch.manual_seed(empty_config.seed)
    initial_model = MicroVLA(model_config(**TINY)).state_dict()
    with patch("method.vla.train.EpisodeStore", return_value=empty_store):
        empty_checkpoint = torch.load(train(empty_config), weights_only=False)
    assert not empty_checkpoint["optimizer"]["state"]
    for name, value in initial_model.items():
        torch.testing.assert_close(value, empty_checkpoint["model"][name], rtol=0, atol=0)
        torch.testing.assert_close(value, empty_checkpoint["ema"][name], rtol=0, atol=0)

    class FakeTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.ones(1), requires_grad=False)
            self.config = SimpleNamespace(hidden_size=12)

        def forward(self, pixel_values, **kwargs):
            patches = F.adaptive_avg_pool2d(pixel_values.float(), (12, 12))
            return SimpleNamespace(
                last_hidden_state=patches.flatten(2).transpose(1, 2).repeat(1, 1, 4)
            )

    # Exercise optional phase/resume without transformers, weights or downloads.
    from torch.nn import functional as F

    with patch("method.vla.train.load_teacher", side_effect=lambda c, d: FakeTeacher().to(d)):
        dc = replace(config, distill=True, distill_steps=2, run_dir=str(root / "distill"), steps=1)
        train(dc, stop_after=1)
        distilled = torch.load(train(dc, resume=True), weights_only=False)
        assert distilled["distill_step"] == 2 and distilled["step"] == 1
        assert distilled["phase"] == "train"
    print(
        json.dumps(
            dict(
                checks="PASS",
                tiny_overfit_initial=initial,
                tiny_overfit_final=final,
                resume="bitwise model/EMA/optimizer/RNG match",
                policy_shape=list(action.shape),
            )
        )
    )


def estimate():
    model = MicroVLA(model_config()).train()
    params = parameter_counts(model)
    images = torch.zeros(2, 3, 3, 96, 96)
    state = torch.zeros(2, 19)
    actions = torch.zeros(2, 20, 10)
    started = time.perf_counter()
    loss, n = flow_loss_terms(
        model,
        images,
        state,
        instruction_tokens(2, "cpu"),
        actions,
        torch.ones_like(actions, dtype=torch.bool),
    )
    (loss / n).backward()
    elapsed = time.perf_counter() - started
    # fp32 model + EMA + gradients + Adam first/second moments, no activations.
    static_gib = params["total"] * 20 / 2**30
    c = model.config
    dense_attention_mib = 2 * c.heads * token_counts(c)["total"] ** 2 * c.depth * 4 / 2**20
    result = dict(
        params=params,
        tokens=token_counts(c),
        cpu_threads=torch.get_num_threads(),
        cpu_microbatch_2_forward_backward_seconds=elapsed,
        cpu_batch_64_steps_per_second_estimate=1 / (elapsed * 32),
        training_static_gib=static_gib,
        dense_prefix_attention_scores_mib_microbatch_2=dense_attention_mib,
        cuda_training_budget_gib_estimate=[0.6, 1.8],
        cuda_estimate_note="Analytical budget only, excludes eval; measure CUDA peak before use",
    )
    print(json.dumps(result))
    return result


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Run CPU smoke with CUDA_VISIBLE_DEVICES='' (no GPU access)")
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix="microvla-smoke-") as directory:
        checks(Path(directory))
    estimate()


if __name__ == "__main__":
    main()
