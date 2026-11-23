"""Make a GIF (3 cameras + gripper/step overlay) and a time-series plot for one episode.

Usage: view_episode.py SET EPISODE_ID [--every N]
  SET: human60 | teacher_v1 | teacher_v1_dart0 | dagger | dagger2
  e.g. view_episode.py human60 0 ; view_episode.py teacher_v1 2000001
Writes /home/keyu/advaitic/viz/<SET>_<ID>.gif and .png
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

DATA = Path("/home/keyu/advaitic/data")
OUT = Path("/home/keyu/advaitic/viz")
CAMS = ("head", "left wrist", "right wrist")

p = argparse.ArgumentParser()
p.add_argument("set")
p.add_argument("episode")
p.add_argument("--every", type=int, default=3, help="use every Nth stored frame (stored frames are every 2nd step)")
a = p.parse_args()
OUT.mkdir(exist_ok=True)
z = np.load(DATA / a.set / f"ep_{a.episode}.npz")
rgb, state, action = z["rgb"], z["state"], z["action"]
stride = int(z["rgb_stride"]) if "rgb_stride" in z.files else 2
mask = z["label_mask"] if "label_mask" in z.files else None
T = len(action)
grip = action[:, 18]
closes = np.where(np.diff((grip > 0).astype(int)) == 1)[0] + 1
opens = np.where(np.diff((grip > 0).astype(int)) == -1)[0] + 1
name = f"{a.set}_{a.episode}"

# ---- GIF ----
frames = []
for i in range(0, len(rgb), a.every):
    t = i * stride
    row = np.concatenate([rgb[i, c].transpose(1, 2, 0) for c in range(rgb.shape[1])], 1)
    img = Image.fromarray(row).resize((row.shape[1] * 3, row.shape[0] * 3), Image.NEAREST)
    d = ImageDraw.Draw(img)
    g = "CLOSED" if grip[min(t, T - 1)] > 0 else "open"
    d.rectangle([0, 0, img.width, 16], fill=(0, 0, 0))
    d.text((4, 2), f"{name}  step {t}/{T}  gripper {g}", fill=(255, 255, 0) if g == "CLOSED" else (255, 255, 255))
    frames.append(img)
frames[0].save(OUT / f"{name}.gif", save_all=True, append_images=frames[1:], duration=80, loop=0)

# ---- plot ----
fig, ax = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
ax[0].plot(grip, label="gripper cmd (+1 closed)")
if mask is not None:
    ax[0].plot(np.where(mask[:, 18], grip, np.nan), "r.", ms=2, label="DAgger supervised")
ax[0].legend(loc="upper right"); ax[0].set_ylabel("gripper")
ax[1].plot(action[:, 0], label="vx"); ax[1].plot(action[:, 1], label="vy"); ax[1].legend(); ax[1].set_ylabel("base cmd")
for k in range(4, 11):
    ax[2].plot(action[:, k], lw=0.8, label=f"arm{k-4}")
ax[2].legend(ncol=7, fontsize=7); ax[2].set_ylabel("left arm cmd")
ax[3].plot(state[:, 18] - state[0, 18], label="pelvis dx"); ax[3].plot(state[:, 19] - state[0, 19], label="pelvis dy")
ax[3].legend(); ax[3].set_ylabel("m"); ax[3].set_xlabel("step (50 Hz)")
for x in ax:
    for c in closes:
        x.axvline(c, color="g", ls="--", lw=0.8)
    for o in opens:
        x.axvline(o, color="m", ls="--", lw=0.8)
succ = bool(z["success"]) if "success" in z.files else None
fig.suptitle(f"{name}  T={T}  slot={int(z['slot']) if 'slot' in z.files else '?'}  success={succ}  (green=close, magenta=open)")
fig.tight_layout(); fig.savefig(OUT / f"{name}.png", dpi=90)
print(f"{OUT/name}.gif  {OUT/name}.png  T={T} closes={closes.tolist()} opens={opens.tolist()}")
