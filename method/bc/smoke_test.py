"""CPU-only synthetic wiring check; no policy/task-success evidence."""

import csv
import os
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from submission.policy import make_policy

from .dataset import EpisodeStore
from .model import ACTIVE_DIMS, PROPRIO_DIMS, masked_l1
from .train import Config, atomic_save, train


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix="bc-smoke-") as temporary:
        root = Path(temporary)
        data = root / "episodes"
        data.mkdir()
        rng = np.random.default_rng(7)
        episodes = []
        for index, length in enumerate((7, 9)):
            state = rng.normal(size=(length, 50)).astype(np.float32) + 10 * index
            action = np.broadcast_to(np.linspace(-0.8, 0.8, 20), (length, 20)).copy()
            action[:, ACTIVE_DIMS] = rng.uniform(-1, 1, (length, 10))
            action[:, 18] = np.where(np.arange(length) < 4, -1, 1)
            action = action.astype(np.float32)
            rgb = rng.integers(0, 256, ((length + 1) // 2, 3, 3, 84, 84), dtype=np.uint8)
            np.savez_compressed(
                data / f"ep_{index}.npz",
                rgb=rgb,
                rgb_stride=2,
                state=state,
                action=action,
                seed=index,
                slot=-1,
                success=True,
            )
            episodes.append((rgb, state, action))
        store = EpisodeStore([data], root / "cache", val_episodes=1)
        cached = EpisodeStore([data], root / "cache", val_episodes=1)
        assert store.image_path == cached.image_path
        assert len(list((root / "cache").glob("*.u8"))) == 1
        dataset = store.dataset("train", horizon=20)
        last = dataset[len(dataset) - 1]
        assert np.array_equal(last["rgb"], episodes[1][0][-1])
        assert last["pad_mask"].tolist() == [False] + [True] * 19
        expected = np.broadcast_to(episodes[1][2][-1, list(ACTIVE_DIMS)], (20, 10))
        np.testing.assert_array_equal(last["target"], expected)
        np.testing.assert_allclose(
            store.stats["proprio_mean"], episodes[1][1][:, PROPRIO_DIMS].mean(0), rtol=1e-6
        )
        prediction = torch.tensor([[[1.0] * 10, [1000.0] * 10]])
        assert (
            masked_l1(prediction, torch.zeros_like(prediction), torch.tensor([[False, True]])) == 1
        )
        config = Config(
            data_dirs=(str(data),),
            cache_dir=str(root / "cache"),
            run_dir=str(root / "interrupted"),
            device="cpu",
            steps=20,
            batch_size=2,
            microbatch_size=1,
            num_workers=0,
            val_episodes=1,
            warmup_steps=3,
            save_every=10,
            val_every=10,
            log_every=10,
            pretrained=False,  # Synthetic wiring check must work without downloads.
        )
        train(config, stop_after=10)
        checkpoint_path = train(config)  # Automatically restores all state.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert checkpoint["step"] == 20 and checkpoint["optimizer"]["state"]
        # Compare against uninterrupted execution to test RNG, schedule and EMA resume.
        reference_path = train(replace(config, run_dir=str(root / "uninterrupted")))
        reference = torch.load(reference_path, map_location="cpu", weights_only=False)
        for key in ("model", "ema"):
            for name, value in checkpoint[key].items():
                torch.testing.assert_close(value, reference[key][name], rtol=0, atol=0)
        with (root / "interrupted" / "log.csv").open() as stream:
            assert [row["step"] for row in csv.DictReader(stream)] == ["10", "20"]
        assert any(
            not torch.equal(checkpoint["model"][key], value)
            for key, value in checkpoint["ema"].items()
        )
        previous = os.environ.get("ADV_CKPT")
        os.environ["ADV_CKPT"] = str(checkpoint_path)
        try:
            policy = make_policy()
            obs = {"rgb": episodes[0][0][0], "state": episodes[0][1][0]}
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
            checkpoint["config"]["temporal_ensemble"] = True
            atomic_save(checkpoint, checkpoint_path)
            ensemble = make_policy()
            for _ in range(4):
                assert ensemble(obs).shape == (10, 20)
            ensemble.reset()
            assert ensemble.time == 0 and not ensemble.predictions
        finally:
            if previous is None:
                os.environ.pop("ADV_CKPT", None)
            else:
                os.environ["ADV_CKPT"] = previous
        print(
            "CPU smoke passed: 20-step train/resume matches uninterrupted training; policy contract OK."
        )


if __name__ == "__main__":
    main()
