"""Fast CPU-only Muon tests; fake tensors/store, no dataset or cache writes.

Run: python -B -m unittest method.vla.test_muon -v
"""

import copy
import io
import json
import os
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn

from method.vla.model import (
    MicroVLA,
    flow_loss_terms,
    instruction_tokens,
    model_config,
)
from method.vla.muon import HybridMuon, build_optimizer, hybrid_groups, muon_update, newton_schulz
from method.vla.train import (
    Config,
    schedule_optimizer,
    train,
    validate_config,
    validate_resume_config,
)

TINY = dict(
    image_size=16,
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


def fake_problem():
    torch.manual_seed(123)
    model = MicroVLA(model_config(horizon=4, **TINY))
    images = torch.randint(0, 256, (4, 3, 3, 16, 16)).float()
    state = torch.rand(4, 19) * 2 - 1
    actions = torch.rand(4, 4, 10) * 2 - 1
    noise = torch.randn_like(actions)
    t = torch.tensor([0.2, 0.4, 0.6, 0.8])
    return model, (images, state, instruction_tokens(4, "cpu"), actions), noise, t


def overfit_curves():
    initial, args, noise, t = fake_problem()
    curves = {}
    # Matched LR is an explicit control. The candidate LR is conservative (2x
    # v1); this numerical check does not claim a fair-budget algorithm win.
    for name, lr in (("adamw", 3e-4), ("muon", 3e-4), ("muon", 6e-4)):
        model = copy.deepcopy(initial)
        optimizer = build_optimizer(model, optimizer=name, lr=lr)
        curve = []
        for step in range(31):
            optimizer.zero_grad(set_to_none=True)
            total, n = flow_loss_terms(
                model, *args, torch.ones_like(args[-1], dtype=torch.bool), noise=noise, t=t
            )
            loss = total / n
            curve.append(float(loss.detach()))
            if step == 30:
                break
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        curves[f"{name}_{lr:g}"] = curve
    return curves


class MemoryDataset(torch.utils.data.Dataset):
    def __init__(self):
        generator = torch.Generator().manual_seed(71)
        self.batch = dict(
            rgb=torch.randint(0, 256, (4, 3, 3, 16, 16), dtype=torch.uint8, generator=generator),
            proprio=torch.randn(4, 19, generator=generator),
            target=torch.rand(4, 4, 10, generator=generator) * 2 - 1,
            label_mask=torch.ones(4, 4, 10, dtype=torch.bool),
            pad_mask=torch.zeros(4, 4, dtype=torch.bool),
        )

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.batch.items()}

    def sample_weights(self, weights):
        return torch.full((4,), 0.25, dtype=torch.double)


class MemoryStore:
    """EpisodeStore interface in RAM; no synthetic image cache is created."""

    digest = "synthetic-in-memory"
    image_path = "in-memory"
    stats = {}
    files = ("fake-train", "fake-val")
    train_ids, val_ids = (0,), (1,)

    def dataset(self, split, horizon):
        assert horizon == 4
        return MemoryDataset()


class FakeTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.ones(1), requires_grad=False)
        self.config = SimpleNamespace(hidden_size=12)

    def forward(self, pixel_values, **kwargs):
        patches = F.avg_pool2d(pixel_values, 16).flatten(2).transpose(1, 2)
        return SimpleNamespace(last_hidden_state=patches.repeat(1, 1, 4))


class MuonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def assert_tree_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assert_tree_equal(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            self.assertEqual(len(a), len(b))
            for left, right in zip(a, b, strict=True):
                self.assert_tree_equal(left, right)
        else:
            self.assertEqual(a, b)

    def test_singular_values_and_cpu_dtype(self):
        measurements = []
        for shape in ((8, 8), (16, 16), (32, 32), (8, 32), (32, 8)):
            torch.manual_seed(42)
            matrix = torch.randn(shape)
            original = matrix.clone()
            with torch.autocast("cpu", dtype=torch.bfloat16):
                result = newton_schulz(matrix)
            self.assertEqual(result.dtype, torch.float32)
            self.assertTrue(torch.equal(matrix, original))
            singular = torch.linalg.svdvals(result)
            self.assertGreater(float(singular.min()), 0.6)
            self.assertLess(float(singular.max()), 1.2)
            measurements.append([shape, float(singular.min()), float(singular.max())])
        self.assertTrue(torch.equal(newton_schulz(torch.zeros(4, 8)), torch.zeros(4, 8)))
        # Five iterations cannot promise a lower bound for rank-deficient input.
        self.assertLess(float(torch.linalg.svdvals(newton_schulz(torch.ones(4, 8)))[-1]), 1e-6)
        print("NS_SINGULAR_VALUES=" + json.dumps(measurements))

    def test_rms_matching(self):
        measurements = []
        for shape in ((8, 8), (8, 32), (32, 8)):
            torch.manual_seed(42)
            grad = torch.randn(shape)
            p = nn.Parameter(torch.zeros(shape))
            optimizer = torch.optim.AdamW([p], lr=1.0, weight_decay=0)
            p.grad = grad.clone()
            optimizer.step()
            # Actual first AdamW step / lr: bias-corrected m/sqrt(v) ~= sign(g).
            # Exclude decay, measure entrywise RMS, and compare the Muon direction.
            adam_rms = float(p.detach().square().mean().sqrt())
            rms = float(muon_update(grad).square().mean().sqrt())
            self.assertAlmostEqual(rms / adam_rms, 0.2, delta=0.04)
            measurements.append(dict(shape=shape, muon_rms=rms, adamw_rms=adam_rms))
        print("UPDATE_RMS=" + json.dumps(measurements))

    def test_fused_projection_split(self):
        torch.manual_seed(5)
        for splits in ((16, 16, 16), (32, 16, 16)):
            matrix = torch.randn(sum(splits), 16)
            actual = muon_update(matrix, split_sizes=splits)
            expected = torch.cat([muon_update(p) for p in matrix.split(splits)])
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertGreater(float((actual - muon_update(matrix)).abs().max()), 0.01)

    def test_complete_exclusive_groups_and_exclusions(self):
        model = MicroVLA(model_config(horizon=4, **TINY))
        for head in (None, nn.Linear(32, 12)):
            groups = hybrid_groups(model, distill_head=head)
            actual = Counter(id(p) for g in groups for p in g["params"])
            expected = (
                [*model.vision.parameters(), *head.parameters()]
                if head is not None
                else list(model.parameters())
            )
            self.assertEqual(actual, Counter(id(p) for p in expected))
            self.assertTrue(all(n == 1 for n in actual.values()))
            muon_ids = {id(p) for g in groups if g["use_muon"] for p in g["params"]}
            self.assertEqual(len(muon_ids), 4 if head is not None else 12)
            for name, p in model.named_parameters():
                if id(p) in muon_ids:
                    self.assertEqual(p.ndim, 2)
                    self.assertTrue(name.endswith("weight"))
                    self.assertNotIn("modulation", name)
                if any(s in name for s in ("modulation", "norm", "position", "embedding")):
                    self.assertNotIn(id(p), muon_ids)
                if name.startswith(("tokens.", "action_in.", "action_out.", "time_mlp.")):
                    self.assertNotIn(id(p), muon_ids)
            self.assertNotIn(id(model.vision.project.weight), muon_ids)
            if head is not None:
                self.assertNotIn(id(head.weight), muon_ids)

    def test_nesterov_decay_and_serialized_momentum(self):
        torch.manual_seed(7)
        p = nn.Parameter(torch.randn(8, 16))
        initial, reference = p.detach().clone(), p.detach().clone()
        optimizer = HybridMuon([dict(params=[p], use_muon=True)], lr=0.01, weight_decay=0.1)
        buffer = torch.zeros_like(p)
        for _ in range(2):
            grad = torch.randn_like(p)
            p.grad = grad.clone()
            buffer = 0.95 * buffer + 0.05 * grad
            reference = reference * 0.999 - 0.01 * muon_update(0.05 * grad + 0.95 * buffer)
            optimizer.step()
            torch.testing.assert_close(p, reference)
            self.assertTrue(torch.equal(p.grad, grad))
        self.assertFalse(torch.equal(initial, p))
        stream = io.BytesIO()
        torch.save(optimizer.state_dict(), stream)
        stream.seek(0)
        clone = nn.Parameter(p.detach().clone())
        resumed = HybridMuon([dict(params=[clone], use_muon=True)], lr=0.01, weight_decay=0.1)
        resumed.load_state_dict(torch.load(stream, weights_only=False))
        grad = torch.randn_like(p)
        p.grad, clone.grad = grad.clone(), grad.clone()
        optimizer.step()
        resumed.step()
        self.assert_tree_equal(optimizer.state_dict(), resumed.state_dict())
        self.assert_tree_equal(p, clone)
        # None means no momentum/decay update; an explicit zero gradient decays.
        before = clone.detach().clone()
        clone.grad = None
        resumed.step()
        self.assert_tree_equal(clone, before)
        zero = nn.Parameter(torch.ones(4, 8))
        zero.grad = torch.zeros_like(zero)
        decay = HybridMuon([dict(params=[zero], use_muon=True)], lr=0.01, weight_decay=0.1)
        decay.step()
        torch.testing.assert_close(zero, torch.full_like(zero, 0.999))

    def test_schedules_and_legacy_resume(self):
        model = MicroVLA(model_config(horizon=4, **TINY))
        config = Config(optimizer="muon", muon_lr=9e-4, warmup_steps=2, steps=10, distill_steps=6)
        optimizer = build_optimizer(model, optimizer="muon", muon_lr=config.muon_lr)
        for phase, end in (("train", 10), ("distill", 6)):
            for step, factor in ((0, 0.5), (1, 1), (2, 1), (end, 0)):
                schedule_optimizer(optimizer, config, phase, step)
                for group in optimizer.param_groups:
                    base = 9e-4 if group["use_muon"] else 3e-4
                    self.assertAlmostEqual(group["lr"], base * factor)
        config = Config(device="cpu")
        saved = asdict(config)
        for key in ("optimizer", "muon_lr", "muon_momentum", "muon_ns_steps"):
            saved.pop(key)
        previous = dict(version="microvla-1", config=saved)
        validate_resume_config(config, previous)
        with self.assertRaisesRegex(ValueError, "optimizer"):
            validate_resume_config(replace(config, optimizer="muon"), previous)
        legacy = build_optimizer(model)
        direct = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.assert_tree_equal(legacy.state_dict(), direct.state_dict())
        schedule_optimizer(legacy, config, "distill", 0)
        self.assertEqual(legacy.param_groups[0]["lr"], config.lr)

    def test_invalid_config(self):
        for settings in (
            dict(optimizer="unknown"),
            dict(muon_momentum=1),
            dict(muon_momentum=float("nan")),
            dict(muon_ns_steps=0),
            dict(muon_ns_steps=1.5),
            dict(optimizer="muon", muon_lr=-1),
            dict(optimizer="muon", muon_lr=float("nan")),
            dict(muon_lr=0.001),
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                validate_config(Config(device="cpu", **settings))

    def test_30_step_fake_overfit(self):
        curves = overfit_curves()
        adam, muon = curves["adamw_0.0003"], curves["muon_0.0006"]
        self.assertLess(muon[-1], 0.65 * muon[0])
        self.assertLess(adam[-1], 0.8 * adam[0])
        self.assertTrue(all(m <= a + 1e-6 for m, a in zip(muon, adam, strict=True)))
        print("OVERFIT_CURVES=" + json.dumps(curves))

    def test_train_phase_resume_and_optimizer_rejection(self):
        with (
            tempfile.TemporaryDirectory(prefix="microvla-muon-", dir="/tmp") as directory,
            patch("method.vla.train.EpisodeStore", return_value=MemoryStore()) as store,
            patch("method.vla.train.load_teacher", side_effect=lambda c, d: FakeTeacher().to(d)),
            redirect_stdout(io.StringIO()),
        ):
            root = Path(directory)
            for name in ("adamw", "muon"):
                config = Config(
                    optimizer=name,
                    device="cpu",
                    horizon=4,
                    exec_steps=2,
                    model_overrides=TINY,
                    steps=3,
                    distill_steps=2,
                    batch_size=2,
                    microbatch_size=1,
                    warmup_steps=1,
                    val_batches=1,
                    sample_batches=1,
                    flow_steps=1,
                    val_every=3,
                    sample_every=3,
                    save_every=3,
                    log_every=3,
                    color_jitter=False,
                )
                for distill in (False, True):
                    config = replace(config, distill=distill)
                    config = replace(config, run_dir=str(root / f"{name}-{distill}-full"))
                    full = torch.load(train(config), weights_only=False)
                    for stop in (1, 2, 3) if distill else (1,):
                        split = replace(config, run_dir=str(root / f"{name}-{distill}-{stop}"))
                        partial = torch.load(train(split, stop_after=stop), weights_only=False)
                        if name == "muon":
                            self.assertTrue(
                                any(
                                    "momentum_buffer" in s
                                    for s in partial["optimizer"]["state"].values()
                                )
                            )
                        if name == "adamw":
                            # Exercise an actual on-disk legacy checkpoint resume.
                            for key in ("optimizer", "muon_lr", "muon_momentum", "muon_ns_steps"):
                                partial["config"].pop(key)
                            torch.save(partial, Path(split.run_dir) / "last.pt")
                        resumed = torch.load(train(split, resume=True), weights_only=False)
                        for key in ("model", "ema", "optimizer", "step", "distill_step"):
                            self.assert_tree_equal(full[key], resumed[key])
                        self.assert_tree_equal(full["rng"]["torch"], resumed["rng"]["torch"])
                        store.reset_mock()
                        with self.assertRaisesRegex(ValueError, "optimizer"):
                            train(
                                replace(split, optimizer="muon" if name == "adamw" else "adamw"),
                                resume=True,
                            )
                        store.assert_not_called()


if __name__ == "__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES='' for CPU-only checks")
    unittest.main(verbosity=2)
