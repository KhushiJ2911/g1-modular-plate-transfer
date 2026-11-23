"""Replay a saved eval episode's actions and film it with free cameras (analysis only).

Usage: record_episode.py RESULT_DIR EPISODE [--every 4] [--size 480x360]
Writes /home/keyu/advaitic/viz/<result>_ep<EP>.gif (+ .mp4 if ffmpeg is available).
Physics, controllers and success checks are untouched; only actions are replayed.
"""
import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

import onnxruntime

onnxruntime.disable_telemetry_events()
from bigym.loco import make  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("result_dir")
p.add_argument("episode", type=int)
p.add_argument("--every", type=int, default=4)
p.add_argument("--size", default="480x360")
a = p.parse_args()
W, H = map(int, a.size.split("x"))
res = Path(a.result_dir)
row = next(r for r in csv.DictReader(open(res / "episodes.csv")) if int(r["episode"]) == a.episode)
acts = np.load(res / f"episode_{a.episode:03d}.actions.npz")["actions"]

env = make("move_plate", camera_keys=())
env.reset(seed=int(row["seed"]))
inner = env.inner_env
model, data = inner.model, inner.data
model.vis.global_.offwidth = max(model.vis.global_.offwidth, W)
model.vis.global_.offheight = max(model.vis.global_.offheight, H)
renderer = mujoco.Renderer(model, H, W)
plate = inner.plates[0]
sites = inner.rack_target.sites


def cam(lookat, distance, azimuth, elevation):
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:] = lookat
    c.distance, c.azimuth, c.elevation = distance, azimuth, elevation
    return c


def frame(t):
    pos = np.asarray(plate.get_position())
    site_xyz = [inner.data.bind(s).xpos.copy() for s in sites] if hasattr(inner.data, "bind") else None
    target_center = np.mean(site_xyz, 0) if site_xyz else np.array([0.55, -0.24, 0.75])
    d = min(np.linalg.norm(pos - s) for s in site_xyz) if site_xyz else float("nan")
    views = []
    # wide view of the whole scene, and a close-up on the target rack from the side
    for c in (cam([0.45, 0.0, 0.85], 1.9, 200, -30), cam(target_center, 0.55, 180, -10),
              cam(target_center, 0.45, 180, -89)):
        renderer.update_scene(data, camera=c)
        views.append(renderer.render())
    img = Image.fromarray(np.concatenate(views, 1))
    dr = ImageDraw.Draw(img)
    g = "CLOSED" if acts[min(t, len(acts) - 1), 18] > 0 else "open"
    dr.rectangle([0, 0, img.width, 18], fill=(0, 0, 0))
    dr.text((6, 3), f"ep {a.episode} seed {row['seed']} slot {row['source_slot']} | step {t} | gripper {g} | "
            f"plate->nearest slot centre {d*100:.1f} cm (limit 5.0) | plate z {pos[2]:.3f}",
            fill=(255, 255, 0))
    return img


frames = [frame(0)]
for t, act in enumerate(acts):
    ts = env.step(act)
    if (t + 1) % a.every == 0:
        frames.append(frame(t + 1))
    if ts.last():
        break
out = Path("/home/keyu/advaitic/viz")
out.mkdir(exist_ok=True)
stem = out / f"{res.name}_ep{a.episode}"
frames[0].save(f"{stem}.gif", save_all=True, append_images=frames[1:], duration=40 * a.every // 2, loop=0)
try:
    import imageio.v2 as iio

    iio.mimsave(f"{stem}.mp4", [np.asarray(f) for f in frames], fps=50 // a.every * 2)
    print(f"{stem}.mp4")
except Exception as e:  # ffmpeg backend may be missing
    print("mp4 skipped:", type(e).__name__)
print(f"{stem}.gif  frames={len(frames)}")
pos = np.asarray(plate.get_position())
for k, s_ in enumerate(sites):
    x = inner.data.bind(s_).xpos
    print(f"site {k}: dx {100*(pos[0]-x[0]):+5.1f}  dy {100*(pos[1]-x[1]):+5.1f}  dz {100*(pos[2]-x[2]):+5.1f} cm  dist {100*np.linalg.norm(pos-x):.1f}")
