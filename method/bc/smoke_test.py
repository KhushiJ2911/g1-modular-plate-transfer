"""CPU-only synthetic wiring check; no policy/task-success evidence."""

import argparse
import csv
import math
import os
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from submission.policy import BCPolicy, make_policy

from .dataset import PRIV_LABELS, EpisodeStore
from .model import ACTIVE_DIMS, PROPRIO_DIMS, BCModel, auxiliary_loss, masked_l1
from .train import Config, StepBatches, atomic_save, initialize_from, train


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
            np.savez_compressed(
                path,
                rgb=rgb,
                rgb_stride=2,
                state=state,
                action=action,
                action_executed=np.full_like(action, 0.99),
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
        check_initialization(checkpoint, checkpoint_path, legacy_path)
        # Fine-tuning an old checkpoint creates fresh optimizer state and step count.
        init_path = train(
            replace(
                config, run_dir=str(root / "finetune"), steps=1, num_workers=0, pretrained=True
            ),
            init_from=legacy_path,
        )
        initialized = torch.load(init_path, map_location="cpu", weights_only=False)
        assert initialized["step"] == 1 and "aux_head.weight" in initialized["ema"]
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
                proprio_dims=(44, 0, 18),
            )
        )
        reduced = BCPolicy(reduced_path)
        assert reduced.proprio_dims == (44, 0, 18) and reduced(obs).shape == (10, 20)
        saved = torch.load(reduced_path, map_location="cpu", weights_only=False)
        assert "aux_head.weight" not in saved["ema"] and saved["val_aux_loss"] == 0
        print(
            f"CPU smoke passed: mixed/partial labels, weighted sources, 20-step exact resume with {workers}/0 workers, init-from, old/new policies and custom proprio. Wiring only; no task-success evidence."
        )


if __name__ == "__main__":
    main()
