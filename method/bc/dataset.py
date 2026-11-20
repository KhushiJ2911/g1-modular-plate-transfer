"""Shared disk image cache, optional privileged labels, and per-source splits."""

import fcntl
import hashlib
import json
import os
import re
import tempfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from .model import ACTIVE_DIMS, PROPRIO_DIMS, validate_proprio_dims

FRAME_SHAPE = (3, 3, 84, 84)
FRAME_BYTES = int(np.prod(FRAME_SHAPE))
# Shape suffix and whether the label is constant over the episode.
PRIV_LABELS = {
    "priv_plate_rel": ((3,), False),
    "priv_target_site_rel": ((3,), False),
    "priv_plate_tilt_deg": ((), False),
    "priv_held": ((), False),
    "priv_target_rack_offset": ((2,), True),
    "priv_start_rack_offset": ((2,), True),
}
AUX_LABELS = ("priv_plate_rel", "priv_target_site_rel", "priv_target_rack_offset", "priv_held")


def episode_files(directories):
    files = []
    for directory in directories:
        episodes = sorted(
            (
                p
                for p in Path(directory).resolve().glob("ep_*.npz")
                if re.fullmatch(r"ep_\d+", p.stem)
            ),
            key=lambda p: int(p.stem[3:]),
        )
        if not episodes:
            raise ValueError(f"No ep_<index>.npz files in {directory}")
        files.extend(episodes)
    if len(set(files)) != len(files):
        raise ValueError("Episode directories overlap")
    return files


def sidecar_path(path):
    return path.with_suffix(".priv.npz")


def read_priv(path, episode, length):
    """Embedded labels take precedence per key; absent dimensions are masked."""
    labels, available = {}, {}
    with ExitStack() as stack:
        sidecar = sidecar_path(path)
        extra = (
            stack.enter_context(np.load(sidecar, allow_pickle=False)) if sidecar.exists() else {}
        )
        for key, (suffix, constant) in PRIV_LABELS.items():
            source = episode if key in episode else extra
            present = key in source
            shape = suffix if constant else (length, *suffix)
            value = (
                np.asarray(source[key], dtype=np.float32)
                if present
                else np.zeros(shape, np.float32)
            )
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{path}: invalid {key} shape/values, expected {shape}")
            if key == "priv_held" and not np.isin(value, (0, 1)).all():
                raise ValueError(f"{path}: priv_held must be binary")
            labels[key] = value
            available[key] = present
    return labels, available


def build_cache(files, cache_dir, *, read_only=False):
    """Stream one compressed episode at a time; publish index last as commit marker."""
    signature = [
        (str(p), p.stat().st_size, p.stat().st_mtime_ns)
        for episode in files
        for p in (episode, sidecar_path(episode))
        if p.exists()
    ]
    digest = hashlib.sha256(json.dumps([3, signature]).encode()).hexdigest()[:24]
    cache_dir = Path(cache_dir)
    image_path = cache_dir / f"{digest}.u8"
    index_path = cache_dir / f"{digest}.npz"
    if read_only:
        if image_path.is_file() and index_path.is_file():
            with np.load(index_path, allow_pickle=False) as index:
                expected = int(index["frame_offsets"][-1]) * FRAME_BYTES
            if image_path.stat().st_size == expected:
                return image_path, index_path, digest
        raise FileNotFoundError(
            f"Read-only BC cache missing/incomplete: {image_path}, {index_path}. "
            "Point cache_dir at an existing schema-3 BC cache; no cache was written."
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    with (cache_dir / f"{digest}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if image_path.exists() and index_path.exists():
            with np.load(index_path, allow_pickle=False) as index:
                expected = int(index["frame_offsets"][-1]) * FRAME_BYTES
            if image_path.stat().st_size == expected:
                return image_path, index_path, digest
        states, actions, label_masks, lengths, frame_counts = [], [], [], [], []
        privileged = {key: [] for key in PRIV_LABELS}
        available = {key: [] for key in PRIV_LABELS}
        with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                for path in files:
                    with np.load(path, allow_pickle=False) as episode:
                        # Executed teacher actions can contain controller corrections.
                        # Supervision always uses the clean action array.
                        action = np.asarray(episode["action"], dtype=np.float32)
                        state = np.asarray(episode["state"], dtype=np.float32)
                        rgb = episode["rgb"]
                        length = len(action)
                        if int(episode["rgb_stride"]) != 2:
                            raise ValueError(f"{path}: expected rgb_stride=2")
                        if length < 1 or action.shape != (length, 20):
                            raise ValueError(f"{path}: invalid action shape")
                        label_mask = (
                            episode["label_mask"]
                            if "label_mask" in episode
                            else np.ones((length, 20), dtype=bool)
                        )
                        if label_mask.shape != (length, 20) or label_mask.dtype != np.bool_:
                            raise ValueError(f"{path}: label_mask must be bool (T,20)")
                        if state.shape != (length, 50):
                            raise ValueError(f"{path}: state must have exactly T rows")
                        if rgb.dtype != np.uint8 or rgb.shape != ((length + 1) // 2, *FRAME_SHAPE):
                            raise ValueError(f"{path}: invalid RGB dtype/shape")
                        if not np.isfinite(state).all() or not np.isfinite(action).all():
                            raise ValueError(f"{path}: nonfinite state/action")
                        if np.abs(action).max() > 1.00001:
                            raise ValueError(f"{path}: actions must be normalized")
                        labels, presence = read_priv(path, episode, length)
                        for key in PRIV_LABELS:
                            privileged[key].append(labels[key])
                            available[key].append(presence[key])
                        rgb.tofile(stream)
                        states.append(state)
                        actions.append(action)
                        label_masks.append(label_mask)
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
                    label_mask=np.concatenate(label_masks),
                    step_offsets=np.r_[0, np.cumsum(lengths)],
                    frame_offsets=np.r_[0, np.cumsum(frame_counts)],
                    files=np.asarray([str(p) for p in files]),
                    **{
                        key: np.stack(values) if PRIV_LABELS[key][1] else np.concatenate(values)
                        for key, values in privileged.items()
                    },
                    **{key + "_available": np.asarray(values) for key, values in available.items()},
                )
                stream.flush()
                os.fsync(stream.fileno())
                os.replace(temporary, index_path)
            finally:
                temporary.unlink(missing_ok=True)
    return image_path, index_path, digest


class EpisodeStore:
    def __init__(
        self, directories, cache_dir, val_episodes=6, proprio_dims=PROPRIO_DIMS, *, read_only=False
    ):
        files = episode_files(directories)
        self.proprio_dims = validate_proprio_dims(proprio_dims)
        self.directory_ids = np.asarray(
            [
                next(i for i, d in enumerate(directories) if p.parent == Path(d).resolve())
                for p in files
            ]
        )
        self.val_ids = []
        for directory_id, directory in enumerate(directories):
            ids = np.flatnonzero(self.directory_ids == directory_id)
            if not 1 <= val_episodes < len(ids):
                raise ValueError(
                    f"{directory}: need more than val_episodes={val_episodes} episodes"
                )
            self.val_ids.extend(ids[np.linspace(0, len(ids) - 1, val_episodes, dtype=int)].tolist())
        self.train_ids = [i for i in range(len(files)) if i not in self.val_ids]
        self.image_path, index_path, self.digest = build_cache(
            files, cache_dir, read_only=read_only
        )
        with np.load(index_path, allow_pickle=False) as index:
            self.state = index["state"]
            self.action = index["action"]
            self.label_mask = index["label_mask"]
            self.steps = index["step_offsets"]
            self.frames = index["frame_offsets"]
            self.priv = {key: index[key] for key in PRIV_LABELS}
            self.priv_available = {key: index[key + "_available"] for key in PRIV_LABELS}
        self.files = [str(p) for p in files]
        row_episodes = np.repeat(np.arange(len(files)), np.diff(self.steps))
        targets, masks = [], []
        for key in AUX_LABELS:
            values = self.priv[key]
            if PRIV_LABELS[key][1]:
                values = values[row_episodes]
            values = values.reshape(len(self.state), -1)
            targets.append(values)
            masks.append(
                np.broadcast_to(self.priv_available[key][row_episodes, None], values.shape)
            )
        self.aux_target = np.concatenate(targets, axis=1)
        self.aux_mask = np.concatenate(masks, axis=1).copy()
        train_rows = np.concatenate(
            [np.arange(self.steps[i], self.steps[i + 1]) for i in self.train_ids]
        )
        proprio = self.state[train_rows][:, self.proprio_dims].astype(np.float64)
        aux_mean, aux_std = np.zeros(9, np.float32), np.ones(9, np.float32)
        for dim in range(8):
            rows = train_rows[self.aux_mask[train_rows, dim]]
            if len(rows):
                values = self.aux_target[rows, dim].astype(np.float64)
                aux_mean[dim], aux_std[dim] = values.mean(), max(values.std(), 1e-3)
        # Inactive policy defaults must not learn from untrusted student actions.
        action_mask = self.label_mask[train_rows]
        action_mean = (
            (self.action[train_rows].astype(np.float64) * action_mask).sum(0)
            / np.maximum(action_mask.sum(0), 1)
        ).astype(np.float32)
        self.stats = {
            "proprio_mean": proprio.mean(0).astype(np.float32),
            "proprio_std": np.maximum(proprio.std(0), 1e-3).astype(np.float32),
            "action_mean": action_mean,
            "aux_mean": aux_mean,
            "aux_std": aux_std,
        }
        # Held stays binary; zero masked targets for easier inspection and safe losses.
        self.aux_target = np.where(self.aux_mask, (self.aux_target - aux_mean) / aux_std, 0)

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

    def sample_weights(self, data_weights=()):
        """Each directory gets its requested total probability, uniform within it."""
        counts = np.diff(np.r_[0, self.ends])
        groups = np.repeat(self.store.directory_ids[self.episode_ids], counts)
        sizes = np.bincount(groups, minlength=len(np.unique(self.store.directory_ids)))
        weights = np.asarray(data_weights if len(data_weights) else sizes, dtype=np.float64)
        if (
            weights.shape != sizes.shape
            or not np.isfinite(weights).all()
            or (weights < 0).any()
            or weights.sum() <= 0
        ):
            raise ValueError("data_weights must have one finite nonnegative weight per directory")
        return (weights / weights.sum() / sizes)[groups]

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
        rows = start + np.minimum(steps, end - start - 1)
        target = self.store.action[rows][:, ACTIVE_DIMS]
        label_mask = self.store.label_mask[rows][:, ACTIVE_DIMS] & ~pad_mask[:, None]
        proprio = self.store.state[start + t, list(self.store.proprio_dims)]
        proprio = (proprio - self.store.stats["proprio_mean"]) / self.store.stats["proprio_std"]
        return {
            "rgb": self.images[self.store.frames[episode] + frame].copy(),
            "proprio": proprio.astype(np.float32),
            "target": target.copy(),
            "label_mask": label_mask.copy(),
            "pad_mask": pad_mask,
            "aux_target": self.store.aux_target[start + t].copy(),
            "aux_mask": self.store.aux_mask[start + t].copy(),
        }
