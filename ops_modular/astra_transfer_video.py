"""Film a verified generation action archive for presentation, independently of policy inference."""

import inspect
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import modal

from ops_modular.modular_sim import image, volume

app = modal.App("modular-astra-transfer-video")
records = modal.Volume.from_name("modular-runs")


def film(directory, destination, label):
    import mujoco
    import numpy as np
    from bigym.loco import make
    from PIL import Image, ImageDraw

    folder = Path(directory)
    report = json.loads((folder / "report.json").read_text())
    ep = report["episodes"][0]
    if not report["integrity_ok"] or not ep["completed"] or ep["policy_error"]:
        raise ValueError("An integrity-valid completed rollout is required")
    with np.load(folder / "episode_000.actions.npz", allow_pickle=False) as f:
        actions = f["actions"]
    if len(actions) != ep["steps"]:
        raise ValueError("Action/report length mismatch")
    env = make("move_plate", camera_keys=())
    renderer = None
    width, height = 352, 224
    try:
        ts = env.reset(seed=ep["seed"])
        inner = env.inner_env
        inner.model.vis.global_.offwidth = max(width, inner.model.vis.global_.offwidth)
        inner.model.vis.global_.offheight = max(height, inner.model.vis.global_.offheight)
        renderer = mujoco.Renderer(inner.model, height, width)
        center = np.mean([inner.data.bind(s).xpos.copy() for s in inner.rack_target.sites], axis=0)

        def camera(at, distance, azimuth, elevation):
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = at
            cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
            return cam

        # Same wide / side-rack viewpoints as analysis/record_episode.py.
        views = [camera([0.45, 0, 0.85], 1.9, 200, -30), camera(center, 0.59, 180, -12)]
        frames = []

        def frame(step):
            panels = []
            for cam in views:
                renderer.update_scene(inner.data, camera=cam)
                panels.append(renderer.render().copy())
            im = Image.new("RGB", (width * 2, height + 34))
            im.paste(Image.fromarray(np.concatenate(panels, axis=1)), (0, 34))
            draw = ImageDraw.Draw(im)
            draw.text(
                (7, 2),
                f"{label}   |   seed {ep['seed']}   |   step {step} of {len(actions)}",
                fill="white",
            )
            status = (
                ep["termination"]
                if step == len(actions)
                else ("gripper open" if actions[step, 18] < 0 else "gripper closed")
            )
            draw.text(
                (7, 17),
                f"Action replay from the recorded public rollout   |   {status}",
                fill=(210, 223, 240),
            )
            frames.append(im)

        frame(0)
        for i, action in enumerate(actions, 1):
            ts = env.step(action)
            # Preserve normal 50-Hz simulator timing during manipulation. For
            # a long terminal hold, append its endpoint after the active window.
            if (i % 6 == 0 and i <= 900) or i == len(actions):
                frame(i)
        if (
            not ts.last()
            or bool(env.episode_succeeded()) != ep["success"]
            or bool(env.episode_fell()) != ep["fell"]
            or env.episode_termination() != ep["termination"]
        ):
            raise ValueError("Physical replay did not match the evaluator")
        durations = [120] * len(frames)
        durations[-1] = 1400
        frames[0].save(
            destination,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=0,
            optimize=False,
        )
        return dict(
            seed=ep["seed"],
            steps=len(actions),
            success=ep["success"],
            termination=ep["termination"],
            frames=len(frames),
            last_active_window=900,
            source=directory,
            policy=report["policy"],
        )
    finally:
        if renderer is not None:
            renderer.close()
        env.close()


@app.function(
    image=image,
    cpu=1,
    memory=3072,
    timeout=1200,
    max_containers=1,
    volumes={"/runs": volume, "/records": records},
)
def clip(seed, arm, folder):
    if (
        not 3001600 <= seed < 3001750
        or arm not in ("base", "fk")
        or not re.fullmatch(rf"/runs/modular/sol_fk_full_{arm}-{seed}(-r[0-9]+)?/eval", folder)
    ):
        raise ValueError("Expected a fresh, paired generation archive")
    records.reload()
    volume.reload()
    dest = Path("/records/astra-integration/video")
    dest.mkdir(exist_ok=True, parents=True)
    stem = dest / f"{seed}_{arm}"
    if stem.with_suffix(".json").exists():
        return json.loads(stem.with_suffix(".json").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        f = str(Path(tmp) / "episode.gif")
        title = "Calibrated placement" if arm == "base" else "Wrist-based seating"
        src = (
            "import json\nfrom pathlib import Path\n"
            + inspect.getsource(film)
            + f"\nprint(json.dumps(film({folder!r},{f!r},{title!r})))"
        )
        start = time.monotonic()
        p = subprocess.run(
            ["/root/policy/bin/python", "-c", src],
            cwd="/root/code",
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "MUJOCO_GL": "osmesa",
                "PYOPENGL_PLATFORM": "osmesa",
                "LP_NUM_THREADS": "2",
                "OMP_NUM_THREADS": "1",
            },
        )
        if p.returncode:
            raise RuntimeError(p.stderr[-1600:])
        row = json.loads(p.stdout.splitlines()[-1])
        row["seconds"] = time.monotonic() - start
        stem.with_suffix(".gif").write_bytes(Path(f).read_bytes())
        stem.with_suffix(".json").write_text(json.dumps(row))
    records.commit()
    return row
