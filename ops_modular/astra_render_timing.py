"""Measure OSMesa CPU allocation on identical ordinary evaluator observations."""

import inspect
import json
import os
import subprocess
from pathlib import Path

import modal

from ops_modular.modular_sim import image, volume

app = modal.App("modular-astra-render-timing")


def timed(folder):
    import hashlib
    import time

    import numpy as np

    from advaitic_eval.environment import make_environment, observation

    path = Path(folder)
    result = json.loads((path / "report.json").read_text())
    if not result["integrity_ok"] or result["episodes"][0]["seed"] != 3001600:
        raise ValueError("Expected the generation comparison smoke")
    with np.load(path / "episode_000.actions.npz", allow_pickle=False) as f:
        actions = f["actions"][:64]
    env = make_environment()
    rgb, state = hashlib.sha256(), hashlib.sha256()
    try:
        ts = env.reset(seed=3001600)
        times = []
        for action in actions:
            start = time.monotonic()
            obs = observation(ts)
            rgb.update(obs["rgb"].tobytes())
            state.update(obs["state"].tobytes())
            ts = env.step(action)
            times.append(time.monotonic() - start)
    finally:
        env.close()
    return dict(
        frames=len(times),
        rgb=rgb.hexdigest(),
        state=state.hexdigest(),
        seconds=sum(times),
        median=float(np.median(times)),
    )


def launch(cores):
    volume.reload()
    src = "import json\nfrom pathlib import Path\n" + inspect.getsource(timed)
    src += "\nprint(json.dumps(timed('/runs/modular/sol_fk_full_base-3001600/eval')))"
    settings = {
        **os.environ,
        "MUJOCO_GL": "osmesa",
        "PYOPENGL_PLATFORM": "osmesa",
        "OMP_NUM_THREADS": "1",
    }
    if cores > 2:
        settings["LP_NUM_THREADS"] = str(cores)
    proc = subprocess.run(
        ["/root/policy/bin/python", "-c", src],
        cwd="/root/code",
        text=True,
        env=settings,
        capture_output=True,
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr[-1800:])
    return {**json.loads(proc.stdout.splitlines()[-1]), "requested_cpu": cores}


options = dict(image=image, memory=2048, timeout=600, max_containers=1, volumes={"/runs": volume})


@app.function(**options, cpu=2)
def standard():
    return launch(2)


@app.function(**options, cpu=8)
def eight():
    return launch(8)
