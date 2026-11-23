"""Convert our demo .npz episodes to a LeRobotDataset (lerobot 0.7.0), for VLA fine-tuning.

Usage: python lerobot_convert.py OUT_ROOT DATA_DIR [DATA_DIR ...] [--limit N]

Same observation/action contract as our own policies:
  observation.images.cam0..cam2  3 cameras, 84x84 RGB (uint8 HWC); images are stored every
                                 2nd control step (rgb_stride=2), odd steps reuse the previous image
  observation.state              the BC/microVLA 19 proprio dims (PROPRIO_DIMS)
  action                         the 10 active dims (ACTIVE_DIMS), gripper as +-1
fps = 50 (control rate). Only fully labelled episodes are accepted (no DAgger masks).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ACTIVE_DIMS = (0, 1, 4, 5, 6, 7, 8, 9, 10, 18)  # submission/model.py
PROPRIO_DIMS = (*range(9), *range(18, 22), *range(40, 44), 44, 45)
TASK = "move the plate from the left rack to the right rack"  # method/vla/model.py INSTRUCTION
CAMS = ("cam0", "cam1", "cam2")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("dirs", nargs="+", type=Path)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--repo-id", default="advaitic/plate_transfer")
    p.add_argument("--shard", default="0/1", help="i/n: every n-th episode starting at i")
    a = p.parse_args()
    features = {f"observation.images.{c}": {"dtype": "video", "shape": (84, 84, 3),
                                             "names": ["height", "width", "channels"]}
                for c in CAMS}
    features["observation.state"] = {"dtype": "float32", "shape": (len(PROPRIO_DIMS),),
                                     "names": [f"s{d}" for d in PROPRIO_DIMS]}
    features["action"] = {"dtype": "float32", "shape": (len(ACTIVE_DIMS),),
                          "names": [f"a{d}" for d in ACTIVE_DIMS]}
    ds = LeRobotDataset.create(repo_id=a.repo_id, fps=50, features=features, root=a.out,
                               robot_type="unitree_g1_dex1", use_videos=True)
    manifest = []
    files = []
    for d in a.dirs:
        files += sorted(d.glob("ep_*.npz"), key=lambda f: int(f.stem[3:]))
    i, n = map(int, a.shard.split("/"))
    files = files[i::n]
    if a.limit:
        files = files[: a.limit]
    for f in files:
        z = np.load(f, allow_pickle=False)
        action, state, rgb = z["action"], z["state"], z["rgb"]
        T = len(action)
        if "label_mask" in z.files and not z["label_mask"].all():
            raise ValueError(f"{f}: partially labelled episode (DAgger); not supported here")
        if int(z["rgb_stride"]) != 2 or rgb.shape[0] != (T + 1) // 2 or rgb.shape[1:] != (3, 3, 84, 84):
            raise ValueError(f"{f}: unexpected rgb layout {rgb.shape}")
        if state.shape != (T, 50) or action.shape != (T, 20):
            raise ValueError(f"{f}: unexpected state/action shape")
        for t in range(T):
            img = rgb[t // 2]  # (cams, C, H, W)
            frame = {f"observation.images.{c}": np.ascontiguousarray(img[i].transpose(1, 2, 0))
                     for i, c in enumerate(CAMS)}
            frame["observation.state"] = state[t, list(PROPRIO_DIMS)].astype(np.float32)
            frame["action"] = action[t, list(ACTIVE_DIMS)].astype(np.float32)
            frame["task"] = TASK
            ds.add_frame(frame)
        ds.save_episode()
        manifest.append({"file": str(f), "frames": int(T)})
        print(f"{f.parent.name}/{f.name}: {T} frames", flush=True)
    ds.finalize()
    (a.out / "advaitic_manifest.json").write_text(json.dumps(
        {"task": TASK, "active_dims": ACTIVE_DIMS, "proprio_dims": PROPRIO_DIMS,
         "episodes": manifest, "total_frames": sum(m["frames"] for m in manifest)}, indent=1))
    print("episodes", len(manifest), "frames", sum(m["frames"] for m in manifest))


if __name__ == "__main__":
    main()
