"""One shared disk image cache, small in-memory state/action index, episode splits."""

import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from .model import ACTIVE_DIMS, PROPRIO_DIMS

FRAME_SHAPE = (3, 3, 84, 84)
FRAME_BYTES = int(np.prod(FRAME_SHAPE))


def episode_files(directories):
    files = []
    for directory in directories:
        episodes = sorted(Path(directory).resolve().glob("ep_*.npz"), key=lambda p: int(p.stem[3:]))
        if not episodes:
            raise ValueError(f"No ep_<index>.npz files in {directory}")
        files.extend(episodes)
    if len(set(files)) != len(files):
        raise ValueError("Episode directories overlap")
    return files


def build_cache(files, cache_dir):
    """Stream one compressed episode at a time; publish index last as commit marker."""
    signature = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    digest = hashlib.sha256(json.dumps([1, signature]).encode()).hexdigest()[:24]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    image_path = cache_dir / f"{digest}.u8"
    index_path = cache_dir / f"{digest}.npz"
    with (cache_dir / f"{digest}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if image_path.exists() and index_path.exists():
            with np.load(index_path, allow_pickle=False) as index:
                expected = int(index["frame_offsets"][-1]) * FRAME_BYTES
            if image_path.stat().st_size == expected:
                return image_path, index_path, digest
        states, actions, lengths, frame_counts = [], [], [], []
        with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                for path in files:
                    with np.load(path, allow_pickle=False) as episode:
                        action = np.asarray(episode["action"], dtype=np.float32)
                        state = np.asarray(episode["state"], dtype=np.float32)
                        rgb = episode["rgb"]
                        length = len(action)
                        if int(episode["rgb_stride"]) != 2:
                            raise ValueError(f"{path}: expected rgb_stride=2")
                        if length < 1 or action.shape != (length, 20):
                            raise ValueError(f"{path}: invalid action shape")
                        if state.shape != (length, 50):
                            raise ValueError(f"{path}: state must have exactly T rows")
                        if rgb.dtype != np.uint8 or rgb.shape != ((length + 1) // 2, *FRAME_SHAPE):
                            raise ValueError(f"{path}: invalid RGB dtype/shape")
                        if not np.isfinite(state).all() or not np.isfinite(action).all():
                            raise ValueError(f"{path}: nonfinite state/action")
                        if np.abs(action).max() > 1.00001:
                            raise ValueError(f"{path}: actions must be normalized")
                        rgb.tofile(stream)
                        states.append(state)
                        actions.append(action)
                        lengths.append(length)
                        frame_counts.append(len(rgb))
                stream.flush()
                os.fsync(stream.fileno())
                os.replace(temporary, image_path)
            finally:
                temporary.unlink(missing_ok=True)
        with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                np.savez(
                    stream,
                    state=np.concatenate(states),
                    action=np.concatenate(actions),
                    step_offsets=np.r_[0, np.cumsum(lengths)],
                    frame_offsets=np.r_[0, np.cumsum(frame_counts)],
                    files=np.asarray([str(p) for p in files]),
                )
                stream.flush()
                os.fsync(stream.fileno())
                os.replace(temporary, index_path)
            finally:
                temporary.unlink(missing_ok=True)
    return image_path, index_path, digest


class EpisodeStore:
    def __init__(self, directories, cache_dir, val_episodes=6):
        files = episode_files(directories)
        if not 1 <= val_episodes < len(files):
            raise ValueError("Need at least one training and one validation episode")
        self.image_path, index_path, self.digest = build_cache(files, cache_dir)
        with np.load(index_path, allow_pickle=False) as index:
            self.state = index["state"]
            self.action = index["action"]
            self.steps = index["step_offsets"]
            self.frames = index["frame_offsets"]
        self.files = [str(p) for p in files]
        self.val_ids = np.linspace(0, len(files) - 1, val_episodes, dtype=int).tolist()
        self.train_ids = [i for i in range(len(files)) if i not in self.val_ids]
        train_rows = np.concatenate(
            [np.arange(self.steps[i], self.steps[i + 1]) for i in self.train_ids]
        )
        proprio = self.state[train_rows][:, PROPRIO_DIMS].astype(np.float64)
        self.stats = {
            "proprio_mean": proprio.mean(0).astype(np.float32),
            "proprio_std": np.maximum(proprio.std(0), 1e-3).astype(np.float32),
            "action_mean": self.action[train_rows].mean(0, dtype=np.float64).astype(np.float32),
        }

    def dataset(self, split, horizon):
        if split not in ("train", "val"):
            raise ValueError("split must be train or val")
        return ChunkDataset(self, self.train_ids if split == "train" else self.val_ids, horizon)


class ChunkDataset(Dataset):
    def __init__(self, store, episode_ids, horizon):
        self.store = store
        self.horizon = horizon
        self.episode_ids = np.asarray(episode_ids)
        self.ends = np.cumsum([store.frames[i + 1] - store.frames[i] for i in episode_ids])
        self.images = None

    def __len__(self):
        return int(self.ends[-1])

    def __getstate__(self):
        # Spawned workers reopen this same read-only mapping, never duplicate image arrays.
        return {**self.__dict__, "images": None}

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.images is None:
            self.images = np.memmap(
                self.store.image_path,
                mode="r",
                dtype=np.uint8,
                shape=(int(self.store.frames[-1]), *FRAME_SHAPE),
            )
        position = int(np.searchsorted(self.ends, index, side="right"))
        episode = self.episode_ids[position]
        frame = index - (int(self.ends[position - 1]) if position else 0)
        t = 2 * frame
        start, end = self.store.steps[episode : episode + 2]
        steps = t + np.arange(self.horizon)
        pad_mask = steps >= end - start
        target = self.store.action[start + np.minimum(steps, end - start - 1)][:, ACTIVE_DIMS]
        proprio = self.store.state[start + t, list(PROPRIO_DIMS)]
        proprio = (proprio - self.store.stats["proprio_mean"]) / self.store.stats["proprio_std"]
        return {
            "rgb": self.images[self.store.frames[episode] + frame].copy(),
            "proprio": proprio.astype(np.float32),
            "target": target.copy(),
            "pad_mask": pad_mask,
        }
