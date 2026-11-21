"""Tiny CPU plumbing check on two teacher episodes. This is NOT estimator training evidence."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from method.bc.dataset import episode_files
from method.harness.estimator import Estimator
from method.harness.offline_replay import replay
from method.harness.train_estimator import main as train_main
from method.harness.train_estimator import prepare_data
from submission.harness import DEFAULT_CONFIG
from submission.policy_harness import HarnessPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True, type=Path, help="New scratch directory")
    parser.add_argument("--bc-checkpoint", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=False)
    data = args.out / "two_episodes"
    data.mkdir()
    paths = episode_files([args.data_dir])[:2]
    if len(paths) != 2:
        raise ValueError("Need two teacher episodes")
    for path in paths:
        (data / path.name).symlink_to(path)
    run, cache = args.out / "run", args.out / "cache"
    # Exercise the actual CLI, including the launcher's required arguments.
    import sys
    from unittest.mock import patch

    cli = [
        "train_estimator",
        "--device",
        "cpu",
        "--cache-dir",
        str(cache),
        "--run-dir",
        str(run),
        "--data-dirs",
        str(data),
        "--steps",
        "2",
        "--batch-size",
        "4",
        "--microbatch-size",
        "2",
        "--val-episodes",
        "1",
        "--val-every",
        "2",
        "--num-workers",
        "0",
        "--no-pretrained",
        "--max-train-rows",
        "8",
        "--max-val-rows",
        "16",
    ]
    with patch.object(sys, "argv", cli):
        train_main()
    saved = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    config = saved["config"]
    store, stats, training, validation = prepare_data(
        argparse.Namespace(**{key: value for key, value in config.items() if key != "proprio_dims"})
    )
    assert set(store.train_ids).isdisjoint(store.val_ids)
    train_rows = np.concatenate(
        [np.arange(store.steps[i], store.steps[i + 1]) for i in store.train_ids]
    )
    np.testing.assert_allclose(
        stats["target_mean"], training.target[train_rows, :7].mean(0), rtol=1e-5, atol=1e-6
    )
    # Verify odd/even RGB mapping against the original compressed archive.
    training.rows = np.arange(store.steps[store.train_ids[0]], store.steps[store.train_ids[0]] + 4)
    with np.load(store.files[store.train_ids[0]], allow_pickle=False) as episode:
        for t in range(4):
            np.testing.assert_array_equal(training[t]["rgb"], episode["rgb"][t // 2])
    with np.load(paths[0], allow_pickle=False) as episode:
        obs = {"rgb": episode["rgb"][0], "state": episode["state"][0]}
    estimator = Estimator(run / "last.pt", "cpu")
    estimate = estimator(obs)
    assert np.isfinite(
        np.r_[
            estimate["plate_rel"],
            estimate["target_site_rel"],
            estimate["tilt_deg"],
            estimate["held"],
        ]
    ).all()
    # Real bce1 loading/forward, both ensemble settings, repeatability after reset.
    for ensemble in (False, True):
        policy = HarnessPolicy(
            args.bc_checkpoint, run / "last.pt", {**DEFAULT_CONFIG, "temporal_ensemble": ensemble}
        )
        first = policy({**obs, "seed": "must be ignored", "success": "must be ignored"})
        second = policy(obs)
        assert first.shape == second.shape == (DEFAULT_CONFIG["exec_steps"], 20)
        assert first.dtype == second.dtype == np.float32
        assert np.isfinite(second).all() and np.abs(second).max() <= 1
        policy.reset()
        np.testing.assert_array_equal(policy(obs), first)
    offline = replay(paths[0], estimator, DEFAULT_CONFIG, max_steps=6)
    record = {
        "smoke_only": True,
        "train_rows_sampled": 8,
        "val_rows_sampled": len(validation),
        "updates": 2,
        "pretrained": False,
        "metrics": saved["val_metrics"],
        "offline_prefix": offline,
        "checks": [
            "train-only normalization",
            "episode split",
            "odd/even RGB t//2",
            "checkpoint reload",
            "real bce1 wrapper and reset",
            "optional temporal ensemble",
            "offline replay prefix",
        ],
    }
    (args.out / "smoke.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps(record, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
