"""One camera-free M1 episode on Modal CPU using the worktree lockfile."""

import json
import subprocess
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
CODE = "/root/code"
app = modal.App("modular-astra-reach")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git",
        "libegl1",
        "libgl1",
        "libgles2",
        "libegl-mesa0",
        "libgl1-mesa-dri",
        "libosmesa6",
        "libglib2.0-0",
    )
    .pip_install("uv==0.9.5")
    .env({"MUJOCO_GL": "egl", "ORT_DISABLE_TELEMETRY": "1", "OPENBLAS_NUM_THREADS": "1"})
)
if modal.is_local():
    for name in ("pyproject.toml", "uv.lock", "README.md"):
        image = image.add_local_file(ROOT / name, f"{CODE}/{name}", copy=True)
    image = image.add_local_dir(
        ROOT / "src", f"{CODE}/src", copy=True, ignore=["**/__pycache__"]
    ).run_commands(f"cd {CODE} && uv sync --frozen --no-dev")
    for name in ("__init__.py", "layout.py", "kinematics.py", "ik_reach_eval.py"):
        image = image.add_local_file(
            ROOT / "method/modular" / name, f"{CODE}/method/modular/{name}"
        )


@app.function(image=image, cpu=1, memory=2048, timeout=300, max_containers=1)
def smoke(seed: int = 3_000_100):
    if not (3_000_000 <= seed < 2**32) or 7_310_000 <= seed < 7_320_000:
        raise ValueError("Protected seed block or invalid privileged development seed")
    start = time.monotonic()
    proc = subprocess.run(
        [
            "uv",
            "run",
            "--frozen",
            "--no-dev",
            "python",
            "-m",
            "method.modular.ik_reach_eval",
            "reach",
            "--start",
            str(seed),
        ],
        cwd=CODE,
        capture_output=True,
        text=True,
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr[-3000:])
    row = json.loads(proc.stdout.splitlines()[-1])
    if row["seed"] != seed or row["fell"] or row["settled_steps"] != 120:
        raise RuntimeError(f"Invalid smoke outcome {row}")
    print(json.dumps({"seconds": time.monotonic() - start, **row}), flush=True)
    return row
