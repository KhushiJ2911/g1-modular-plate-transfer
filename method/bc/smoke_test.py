"""CPU-only synthetic wiring check; no policy/task-success evidence."""

import argparse
import csv
import math
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from submission.policy import BCPolicy, make_policy

from .dataset import PRIV_LABELS, EpisodeStore
from .model import (
    ACTIVE_DIMS,
    GRIPPER_INDEX,
    PROPRIO_DIMS,
    BCModel,
    action_loss_terms,
    auxiliary_loss,
    masked_l1,
)
from .train import ActionMetrics, Config, StepBatches, atomic_save, initialize_from, train


def fake_data(root):
    rng = np.random.default_rng(7)
    directories, episodes = [], {}
    for group, lengths in enumerate(((7, 9, 5), (3, 5, 7), (5, 7, 9))):
        directory = root / ("human60", "teacher_v1", "teacher_v2")[group]
        directory.mkdir()
        directories.append(str(directory))
        for index, length in enumerate(lengths):
            state = rng.normal(size=(length, 50)).astype(np.float32) + 10 * index
            action = np.broadcast_to(np.linspace(-0.8, 0.8, 20), (length, 20)).copy()
            action[:, ACTIVE_DIMS] = rng.uniform(-1, 1, (length, 10))
            action[:, 18] = np.where(np.arange(length) < 4, -1, 1)
            action = action.astype(np.float32)
            rgb = rng.integers(0, 256, ((length + 1) // 2, 3, 3, 84, 84), dtype=np.uint8)
            labels = {
                "priv_plate_rel": rng.normal(size=(length, 3)).astype(np.float32),
                "priv_target_site_rel": rng.normal(size=(length, 3)).astype(np.float32),
                "priv_plate_tilt_deg": rng.uniform(0, 90, length).astype(np.float32),
                "priv_held": (np.arange(length) >= 4).astype(np.float32),
                "priv_target_rack_offset": np.asarray([index, group], np.float32),
                "priv_start_rack_offset": np.asarray([group, index], np.float32),
            }
            if index == 0:  # Validation-only outliers must not affect normalization.
                labels["priv_plate_rel"] += 1000
            if (group, index) in ((0, 2), (2, 0)):
                labels = {}
            if (group, index) == (1, 2):
                labels = {"priv_held": labels["priv_held"]}
            path = directory / f"ep_{index}.npz"
            action_labels = {}
            if group == 2 and index in (1, 2):
                label_mask = np.zeros((length, 20), dtype=bool)
                if index == 1:
                    label_mask[[1, 4], 18] = True
                    label_mask[:, 19] = True  # Inactive dimensions must still be excluded.
                action_labels["label_mask"] = label_mask
            np.savez_compressed(
                path,
                rgb=rgb,
                rgb_stride=2,
                state=state,
                action=action,
                action_executed=np.full_like(action, 0.99),
                **action_labels,
                **(labels if group else {}),
            )
            if not group and labels:
                np.savez(path.with_suffix(".priv.npz"), **labels)
            if (group, index) == (1, 1):
                np.savez(
                    path.with_suffix(".priv.npz"), priv_plate_rel=labels["priv_plate_rel"] + 900
                )
            episodes[group, index] = (rgb, state, action, labels)
    return directories, episodes


def check_dataset(root, directories, episodes):
    store = EpisodeStore(directories, root / "cache", val_episodes=1)
    cached = EpisodeStore(directories, root / "cache", val_episodes=1)
    assert store.image_path == cached.image_path
    assert len(list((root / "cache").glob("*.u8"))) == 1
    assert store.val_ids == [0, 3, 6] and store.train_ids == [1, 2, 4, 5, 7, 8]
    dataset = store.dataset("train", horizon=20)
    # Last stored frame of human ep_1 is aligned with its final clean action.
    last = dataset[4]
    assert np.array_equal(last["rgb"], episodes[0, 1][0][-1])
    assert last["pad_mask"].tolist() == [False] + [True] * 19
    expected = np.broadcast_to(episodes[0, 1][2][-1, list(ACTIVE_DIMS)], (20, 10))
    np.testing.assert_array_equal(last["target"], expected)
    assert last["aux_mask"].all() and last["aux_target"][-1] == 1
    assert not dataset[5]["aux_mask"].any() and not dataset[5]["aux_target"].any()
    # Partial episodes retain supervision on their available held label only.
    assert dataset[11]["aux_mask"].tolist() == [False] * 8 + [True]
    # Default full supervision, and per-step partial labels through chunking/padding.
    assert last["label_mask"].tolist() == [[True] * 10] + [[False] * 10] * 19
    partial = dataset[15]  # teacher_v2 ep_1, t=0, length 7.
    expected_mask = np.zeros((20, len(ACTIVE_DIMS)), dtype=bool)
    expected_mask[[1, 4], ACTIVE_DIMS.index(18)] = True
    np.testing.assert_array_equal(partial["label_mask"], expected_mask)
    later = dataset[16]  # Same episode at t=2; the odd-step label is now behind us.
    assert np.flatnonzero(later["label_mask"][:, ACTIVE_DIMS.index(18)]).tolist() == [2]
    assert not dataset[19]["label_mask"].any()  # An entirely unlabeled episode.
    prediction = torch.full((1, 20, 10), 0.25, requires_grad=True)
    target = torch.from_numpy(partial["target"])[None]
    pad_mask = torch.from_numpy(partial["pad_mask"])[None]
    mask = torch.from_numpy(partial["label_mask"])[None]
    loss = masked_l1(prediction, target, pad_mask, mask)
    torch.testing.assert_close(loss, (prediction - target).abs()[mask].mean())
    loss.backward()
    assert (prediction.grad[~mask] == 0).all() and (prediction.grad[mask] != 0).all()
    # Microbatch normalization is by labeled elements, including empty microbatches.
    predictions = torch.cat((prediction.detach(), prediction.detach()))
    targets = torch.cat((target, target))
    pads = torch.cat((pad_mask, pad_mask))
    masks = torch.cat((mask, torch.zeros_like(mask)))
    accumulated = sum(
        masked_l1(predictions[i : i + 1], targets[i : i + 1], pads[i : i + 1], masks[i : i + 1])
        * masks[i].sum()
        / masks.sum()
        for i in range(2)
    )
    torch.testing.assert_close(accumulated, masked_l1(predictions, targets, pads, masks))
    assert masked_l1(prediction, target, pad_mask, torch.zeros_like(mask)) == 0
    action_rows = np.concatenate(
        [np.arange(store.steps[i], store.steps[i + 1]) for i in store.train_ids]
    )
    expected_mean = (
        store.action[action_rows].astype(np.float64) * store.label_mask[action_rows]
    ).sum(0) / store.label_mask[action_rows].sum(0)
    np.testing.assert_allclose(store.stats["action_mean"], expected_mean, rtol=1e-6, atol=1e-7)
    states = np.concatenate([episodes[g, i][1] for g in range(3) for i in (1, 2)])
    np.testing.assert_allclose(
        store.stats["proprio_mean"], states[:, PROPRIO_DIMS].mean(0), rtol=1e-6
    )
    plate = np.concatenate(
        [episodes[g, i][3]["priv_plate_rel"] for g, i in ((0, 1), (1, 1), (2, 1), (2, 2))]
    )
    np.testing.assert_allclose(store.stats["aux_mean"][:3], plate.mean(0), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(store.stats["aux_std"][:3], plate.std(0), rtol=1e-6)
    assert store.stats["aux_mean"][8] == 0 and store.stats["aux_std"][8] == 1
    assert set(store.priv) == set(PRIV_LABELS)
    # Sidecars are excluded from episode enumeration and included in cache identity.
    sidecar = Path(directories[0]) / "ep_2.priv.npz"
    np.savez(sidecar, priv_held=np.zeros(5, np.float32))
    changed = EpisodeStore(directories, root / "cache", val_episodes=1)
    assert changed.digest != store.digest and changed.dataset("train", 20)[5]["aux_mask"][-1]
    sidecar.unlink()
    # Default weights are uniform over frames; explicit weights allocate source mass.
    np.testing.assert_allclose(dataset.sample_weights(), 1 / len(dataset))
    counts = np.diff(np.r_[0, dataset.ends])
    groups = np.repeat(store.directory_ids[dataset.episode_ids], counts)
    weights = dataset.sample_weights((2, 1, 1))
    np.testing.assert_allclose([weights[groups == i].sum() for i in range(3)], (0.5, 0.25, 0.25))
    assert not dataset.sample_weights((1, 0, 0))[groups != 0].any()
    for invalid in ((1, 2), (0, 0, 0), (-1, 1, 1), (float("nan"), 1, 1)):
        try:
            dataset.sample_weights(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted bad data_weights: {invalid}")
    draws = list(StepBatches(weights, 10000, 0, 1, 9))[0]
    np.testing.assert_allclose(
        np.bincount(groups[draws]) / len(draws), (0.5, 0.25, 0.25), atol=0.02
    )
    all_batches = list(StepBatches(weights, 8, 0, 4, 7))
    assert list(StepBatches(weights, 8, 2, 4, 7)) == all_batches[2:]
    return store


def check_losses():
    prediction = torch.tensor([[[1.0] * 10, [1000.0] * 10]])
    assert masked_l1(prediction, torch.zeros_like(prediction), torch.tensor([[False, True]])) == 1
    aux = torch.zeros(3, 9, requires_grad=True)
    target = torch.ones_like(aux)
    mask = torch.tensor([[True] * 9, [False] * 9, [False] * 8 + [True]])
    loss = auxiliary_loss(aux, target, mask)
    torch.testing.assert_close(loss, torch.tensor((8 + 2 * math.log(2)) / 10))
    loss.backward()
    assert (aux.grad[~mask] == 0).all() and (aux.grad[mask] != 0).all()
    # Accumulation must use the labeled count, independently of action padding.
    accumulated = sum(
        auxiliary_loss(aux[i : i + 1], target[i : i + 1], mask[i : i + 1])
        * mask[i].sum()
        / mask.sum()
        for i in range(3)
    )
    torch.testing.assert_close(accumulated, loss)
    assert auxiliary_loss(aux, target, torch.zeros_like(mask)) == 0


def check_gripper_losses():
    torch.manual_seed(17)
    prediction = torch.randn(4, 3, len(ACTIVE_DIMS), requires_grad=True)
    target = torch.ones_like(prediction)
    target[0, 0, GRIPPER_INDEX] = -1
    target[1, :, GRIPPER_INDEX] = -1
    mask = torch.zeros_like(target, dtype=torch.bool)
    mask[0] = True  # Full sample.
    mask[1, :, GRIPPER_INDEX] = True  # Gripper-only sample.
    mask[2, :, :2] = True  # No gripper labels.
    pad = torch.tensor([[False, False, True]] * 4)
    terms = action_loss_terms(
        prediction, target, pad, mask, gripper_loss="bce", gripper_pos_weight=3
    )
    assert terms["l1_count"] == 22 and terms["gripper_count"] == 4
    assert terms["gripper_open_count"] == 3
    valid = mask & ~pad.unsqueeze(-1)
    gripper = valid[..., GRIPPER_INDEX]
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        prediction[..., GRIPPER_INDEX][gripper],
        (target[..., GRIPPER_INDEX][gripper] < 0).float(),
        pos_weight=torch.tensor(3.0),
    )
    torch.testing.assert_close(terms["gripper_bce_sum"] / terms["gripper_count"], expected)
    weight = 2.5
    loss = terms["l1_sum"] / terms["l1_count"] + weight * expected
    gradient = torch.autograd.grad(loss, prediction, retain_graph=True)[0]
    assert (gradient[~valid] == 0).all() and (gradient[valid] != 0).all()
    metrics = ActionMetrics()
    accumulated = prediction.sum() * 0
    for i in range(4):
        micro = action_loss_terms(
            prediction[i : i + 1],
            target[i : i + 1],
            pad[i : i + 1],
            mask[i : i + 1],
            gripper_loss="bce",
            gripper_pos_weight=3,
        )
        metrics.add(micro)
        accumulated = (
            accumulated
            + micro["l1_sum"] / terms["l1_count"]
            + weight * micro["gripper_bce_sum"] / terms["gripper_count"]
        )
    torch.testing.assert_close(accumulated, loss)
    torch.testing.assert_close(torch.autograd.grad(accumulated, prediction)[0], gradient)
    torch.testing.assert_close(torch.tensor(metrics.means()["gripper_bce"]), expected)
    empty = action_loss_terms(prediction, target, pad, torch.zeros_like(mask), gripper_loss="bce")
    zero = empty["l1_sum"] + empty["gripper_bce_sum"]
    assert zero == 0
    assert not torch.autograd.grad(zero, prediction)[0].any()
    # Legacy mode is exactly the existing all-active-dimension masked L1.
    old = action_loss_terms(prediction, target, pad, mask)
    torch.testing.assert_close(
        old["l1_sum"] / old["l1_count"], masked_l1(prediction, target, pad, mask)
    )


def check_rare_open_overfit():
    """Fit the actual BC action head on synthetic features, without image training."""
    torch.manual_seed(19)
    model = BCModel(horizon=3, pretrained=False).eval()
    # Fixed ambiguous observations: 1/32 labels open, with no separating cue.
    # pos_weight lets BCE overcome that prior; L1 retains its closed median.
    with torch.no_grad():
        features = model.head[:-1](torch.zeros(32, model.head[0].in_features))
    target = torch.ones(32, 3, len(ACTIVE_DIMS))
    target[0, :, GRIPPER_INDEX] = -1
    mask = torch.zeros_like(target, dtype=torch.bool)
    mask[..., GRIPPER_INDEX] = True
    pad = torch.zeros(32, 3, dtype=torch.bool)
    results = {}
    for mode, initial in (("bce", -2.0), ("l1", 0.93)):
        head = model.head[-1]
        with torch.no_grad():
            head.weight.zero_()
            head.bias.fill_(initial)
        optimizer = torch.optim.Adam(head.parameters(), lr=0.03)
        history = []
        for step in range(101):
            prediction = head(features).reshape_as(target)
            terms = action_loss_terms(
                prediction, target, pad, mask, gripper_loss=mode, gripper_pos_weight=62
            )
            loss = terms["l1_sum"] / terms["l1_count"].clamp_min(1) + terms[
                "gripper_bce_sum"
            ] / terms["gripper_count"].clamp_min(1)
            if step in (0, 100):
                history.append((float(loss.detach()), float(terms["gripper_open_correct"] / 3)))
            if step < 100:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        results[mode] = history
    assert results["bce"][1][0] < results["bce"][0][0]
    assert results["bce"][0][1] == 0 and results["bce"][1][1] == 1
    assert results["l1"][1][1] == 0
    print(f"Rare-open synthetic head overfit (loss, open accuracy): {results}")


def check_residual_head():
    torch.manual_seed(19)
    rgb = torch.randint(0, 256, (2, 3, 3, 84, 84), dtype=torch.uint8)
    for options, block_count in (({}, 3), ({"head_blocks": 2, "fuse_skip": True}, 2)):
        model = BCModel(horizon=3, pretrained=False, head_type="resmlp", **options)
        head = model.head
        assert len(head.blocks) == block_count
        with torch.no_grad():
            fused = torch.randn(2, head.input.in_features)
            assert torch.equal(head(fused), head.output(head.norm(head.input(fused))))
        proprio = torch.randn(2, len(PROPRIO_DIMS), requires_grad=True)
        actions = model(rgb, proprio)
        assert actions.shape == (2, 3, len(ACTIVE_DIMS))
        actions.square().mean().backward()
        assert proprio.grad.shape == proprio.shape and proprio.grad.any()
        assert head.input.weight.grad.any()
        for block in head.blocks:
            assert not block.project.weight.any() and not block.project.bias.any()
            assert block.project.weight.grad.any()
            if options.get("fuse_skip"):
                assert block.project.weight.grad[:, 1024:].any()


def check_bce_policy(root, checkpoint, obs):
    class FixedChunk(torch.nn.Module):
        def __init__(self, logits):
            super().__init__()
            self.chunk = torch.zeros(1, 20, len(ACTIVE_DIMS))
            self.chunk[0, :, GRIPPER_INDEX] = logits

        def forward(self, *args):
            return self.chunk

    saved = dict(checkpoint)
    saved["config"] = {**checkpoint["config"], "gripper_loss": "bce", "gripper_threshold": 0.8}
    path = root / "bce-policy.pt"
    atomic_save(saved, path)
    logits = torch.tensor([-4, -1, 0, 1, 2, 4, 0, 2, -4, 4] * 2, dtype=torch.float32)
    with patch.dict(os.environ):
        os.environ.pop("ADV_GRIPPER_THRESHOLD", None)
        policy = BCPolicy(path)
        policy.model = FixedChunk(logits)
        expected = np.where(logits[:10].sigmoid().numpy() > 0.8, -1, 1)
        np.testing.assert_array_equal(policy(obs)[:, 18], expected)
        os.environ["ADV_GRIPPER_THRESHOLD"] = "0.5"
        policy = BCPolicy(path)
        policy.model = FixedChunk(logits)
        np.testing.assert_array_equal(policy(obs)[:, 18], np.where(logits[:10] > 0, -1, 1))
        # Strict threshold: probability == 0.5 closes. Legacy ignores even a bad override.
        os.environ["ADV_GRIPPER_THRESHOLD"] = "invalid"
        old_path = root / "legacy.pt"
        old = BCPolicy(old_path)
        old.model = FixedChunk(logits)
        np.testing.assert_array_equal(old(obs)[:, 18], np.where(logits[:10] >= 0, 1, -1))
        for invalid in ("nan", "-0.1", "1.1", "invalid"):
            os.environ["ADV_GRIPPER_THRESHOLD"] = invalid
            try:
                BCPolicy(path)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Accepted invalid threshold {invalid}")
        os.environ.pop("ADV_GRIPPER_THRESHOLD")
        saved["config"]["temporal_ensemble"] = True
        atomic_save(saved, path)
        policy = BCPolicy(path)
        policy.model = FixedChunk(torch.full((20,), 4.0))
        assert (policy(obs)[:, 18] == -1).all()
        policy.model = FixedChunk(torch.full((20,), -1.0))
        # The overlapping logits average to ~1.51, above logit(0.8), at every step.
        assert (policy(obs)[:, 18] == -1).all()


def check_initialization(checkpoint, path, legacy_path):
    stats = {key: value.copy() for key, value in checkpoint["stats"].items()}
    stats["proprio_mean"] += 0.5
    stats["proprio_std"] *= 1.3
    stats["aux_mean"][:8] += 0.2
    stats["aux_std"][:8] *= 1.2
    original = BCModel(pretrained=False, aux_head=True).eval()
    original.load_state_dict(checkpoint["ema"])
    initialized = BCModel(pretrained=False, aux_head=True).eval()
    initialize_from(initialized, path, stats)
    rgb = torch.zeros(1, 3, 3, 84, 84, dtype=torch.uint8)
    raw = np.zeros(19, np.float32)
    with torch.inference_mode():
        old_action, old_aux = original(
            rgb,
            torch.from_numpy(
                (raw - checkpoint["stats"]["proprio_mean"]) / checkpoint["stats"]["proprio_std"]
            )[None],
            return_aux=True,
        )
        new_action, new_aux = initialized(
            rgb,
            torch.from_numpy((raw - stats["proprio_mean"]) / stats["proprio_std"])[None],
            return_aux=True,
        )
    torch.testing.assert_close(old_action, new_action, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(
        old_aux * torch.from_numpy(checkpoint["stats"]["aux_std"])
        + torch.from_numpy(checkpoint["stats"]["aux_mean"]),
        new_aux * torch.from_numpy(stats["aux_std"]) + torch.from_numpy(stats["aux_mean"]),
        atol=2e-6,
        rtol=2e-5,
    )
    initialize_from(initialized, legacy_path, checkpoint["stats"])
    for name, value in initialized.state_dict().items():
        if not name.startswith("aux_head."):
            torch.testing.assert_close(value, checkpoint["ema"][name], rtol=0, atol=0)
    # An architectural mismatch must fail explicitly, rather than silently drop weights.
    try:
        initialize_from(BCModel(horizon=10, pretrained=False), path, stats)
    except ValueError:
        pass
    else:
        raise AssertionError("Accepted incompatible init horizon")


def check_policy(root, store, checkpoint, path, episodes):
    previous = os.environ.get("ADV_CKPT")
    os.environ["ADV_CKPT"] = str(path)
    try:
        policy = make_policy()
        assert policy.model.aux_head is None
        obs = {"rgb": episodes[0, 0][0][0], "state": episodes[0, 0][1][0]}
        actions = policy(obs)
        assert actions.shape == (10, 20) and actions.dtype == np.float32
        assert np.isfinite(actions).all() and np.abs(actions).max() <= 1
        assert set(np.unique(actions[:, 18])) <= {-1, 1}
        inactive = [i for i in range(20) if i not in ACTIVE_DIMS]
        np.testing.assert_allclose(
            actions[:, inactive],
            np.broadcast_to(store.stats["action_mean"][inactive], (10, len(inactive))),
        )
        policy.reset()
        np.testing.assert_array_equal(policy(obs), actions)
        # Exactly the old inference schema: no aux tensors, dims or new config keys.
        legacy = {
            "version": 1,
            "config": {"horizon": 20, "exec_steps": 10},
            "ema": {
                key: value
                for key, value in checkpoint["ema"].items()
                if not key.startswith("aux_head.")
            },
            "stats": {
                key: value
                for key, value in checkpoint["stats"].items()
                if not key.startswith("aux_")
            },
        }
        legacy_path = root / "legacy.pt"
        atomic_save(legacy, legacy_path)
        np.testing.assert_array_equal(BCPolicy(legacy_path)(obs), actions)
        legacy["config"]["temporal_ensemble"] = True
        ensemble_path = root / "ensemble.pt"
        atomic_save(legacy, ensemble_path)
        ensemble = BCPolicy(ensemble_path)
        for _ in range(4):
            assert ensemble(obs).shape == (10, 20)
        ensemble.reset()
        assert ensemble.time == 0 and not ensemble.predictions
    finally:
        if previous is None:
            os.environ.pop("ADV_CKPT", None)
        else:
            os.environ["ADV_CKPT"] = previous
    return legacy_path, obs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-workers", type=int, choices=(0, 1, 2), default=0)
    workers = parser.parse_args().num_workers
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix="bc-smoke-") as temporary:
        root = Path(temporary)
        directories, episodes = fake_data(root)
        store = check_dataset(root, directories, episodes)
        check_losses()
        check_gripper_losses()
        check_rare_open_overfit()
        check_residual_head()
        config = Config(
            data_dirs=tuple(directories),
            data_weights=(2, 1, 1),
            cache_dir=str(root / "cache"),
            run_dir=str(root / "interrupted"),
            device="cpu",
            steps=20,
            batch_size=2,
            microbatch_size=1,
            num_workers=workers,
            val_episodes=1,
            warmup_steps=3,
            save_every=10,
            val_every=10,
            log_every=10,
            pretrained=False,
        )
        train(config, stop_after=10)
        # Emulate a version-2 checkpoint/log written before head config and gripper metrics.
        old_path = root / "interrupted" / "last.pt"
        old = torch.load(old_path, map_location="cpu", weights_only=False)
        for key in (
            "head_type",
            "head_blocks",
            "fuse_skip",
            "gripper_loss",
            "gripper_weight",
            "gripper_pos_weight",
            "gripper_threshold",
        ):
            old["config"].pop(key)
        old.pop("val_metrics")
        atomic_save(old, old_path)
        del old
        log_path = root / "interrupted" / "log.csv"
        with log_path.open(newline="") as stream:
            old_rows = list(csv.reader(stream))
        with log_path.open("w", newline="") as stream:
            csv.writer(stream).writerows(row[:7] for row in old_rows)
        checkpoint_path = train(replace(config, num_workers=0))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert checkpoint["step"] == 20 and checkpoint["optimizer"]["state"]
        reference_path = train(replace(config, run_dir=str(root / "uninterrupted")))
        reference = torch.load(reference_path, map_location="cpu", weights_only=False)
        for key in ("model", "ema"):
            for name, value in checkpoint[key].items():
                torch.testing.assert_close(value, reference[key][name], rtol=0, atol=0)
        del reference
        with (root / "interrupted" / "log.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert [row["step"] for row in rows] == ["10", "20"]
        assert all(
            float(row["train_aux_loss"]) > 0 and float(row["val_aux_loss"]) > 0 for row in rows
        )
        assert any(
            not torch.equal(checkpoint["model"][key], value)
            for key, value in checkpoint["ema"].items()
        )
        legacy_path, obs = check_policy(root, store, checkpoint, checkpoint_path, episodes)
        check_bce_policy(root, checkpoint, obs)
        check_initialization(checkpoint, checkpoint_path, legacy_path)
        # Fine-tuning an old checkpoint creates fresh optimizer state and step count.
        init_path = train(
            replace(
                config,
                run_dir=str(root / "finetune"),
                steps=1,
                num_workers=0,
                pretrained=True,
                gripper_loss="bce",
                gripper_pos_weight=3,
            ),
            init_from=legacy_path,
        )
        initialized = torch.load(init_path, map_location="cpu", weights_only=False)
        assert initialized["step"] == 1 and "aux_head.weight" in initialized["ema"]
        assert initialized["config"]["gripper_loss"] == "bce"
        assert initialized["val_metrics"]["gripper_bce"] > 0
        assert BCPolicy(init_path)(obs).shape == (10, 20)
        with (root / "finetune" / "log.csv").open() as stream:
            row = next(csv.DictReader(stream))
        for split in ("train", "val"):
            assert float(row[f"{split}_gripper_bce"]) > 0
            assert 0 <= float(row[f"{split}_gripper_open_accuracy"]) <= 1
        assert all(int(state["step"]) == 1 for state in initialized["optimizer"]["state"].values())
        del initialized
        # Training with aux disabled and a reordered proprio subset also serves inference.
        reduced_path = train(
            replace(
                config,
                run_dir=str(root / "reduced"),
                steps=1,
                num_workers=0,
                aux_weight=0,
                gripper_loss="bce",
                proprio_dims=(44, 0, 18),
            )
        )
        reduced = BCPolicy(reduced_path)
        assert reduced.proprio_dims == (44, 0, 18) and reduced(obs).shape == (10, 20)
        saved = torch.load(reduced_path, map_location="cpu", weights_only=False)
        assert "aux_head.weight" not in saved["ema"] and saved["val_aux_loss"] == 0
        # Entirely unlabeled batches must not trigger Adam momentum/weight decay.
        empty_dir = root / "unlabeled"
        empty_dir.mkdir()
        for index in range(2):
            np.savez_compressed(
                empty_dir / f"ep_{index}.npz",
                rgb=np.zeros((1, 3, 3, 84, 84), np.uint8),
                rgb_stride=2,
                state=np.zeros((1, 50), np.float32),
                action=np.ones((1, 20), np.float32),
                label_mask=np.zeros((1, 20), bool),
            )
        empty_path = train(
            replace(
                config,
                data_dirs=(str(empty_dir),),
                data_weights=(),
                run_dir=str(root / "empty-run"),
                steps=1,
                num_workers=0,
                aux_weight=0,
                gripper_loss="bce",
            )
        )
        empty = torch.load(empty_path, map_location="cpu", weights_only=False)
        assert not empty["optimizer"]["state"] and empty["val_loss"] == 0
        assert not empty["stats"]["action_mean"].any()
        for key in empty["model"]:
            torch.testing.assert_close(empty["model"][key], empty["ema"][key], rtol=0, atol=0)
        print(
            f"CPU smoke passed: mixed/partial labels, BCE rare-open fit, independent microbatch losses, thresholds/ensembling, weighted sources, 20-step exact resume with {workers}/0 workers, L1-to-BCE init-from, old/new policies and custom proprio. Wiring only; no task-success evidence."
        )


if __name__ == "__main__":
    main()
