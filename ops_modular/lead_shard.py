"""Sharded DEV evaluation (lead). Each planned seed of the official balanced plan runs as a full-length
1-episode --smoke evaluator run in its own container (no --max-steps), so a 30-90 episode dev read takes one
episode of wall time. Non-qualifying by construction; the qualifying run stays a single eval_run report.
Deploy: MODAL_CONFIG_PATH=~/.modal_modular.toml modal deploy ops_modular/lead_shard.py"""

import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
CODE = "/root/code"
OUT = Path("/runs/modular")
CHECKPOINT = Path("/runs/est_v1b/last.pt")
WEIGHTS_SHA = "bd8ed33c87383b3b208ba251d8fa0b6f957d9f03a35cf6c8d23a81ce9cd867cd"
app = modal.App("modular-shard")
volume = modal.Volume.from_name("advaitic-runs")
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
    ).run_commands(
        f"cd {CODE} && uv sync --frozen --no-dev",
        f"cd {CODE} && uv run --frozen --no-dev advaitic-check",
        "uv venv /root/mlenv --python 3.12",
        "uv pip install --python /root/mlenv/bin/python torch==2.5.1 torchvision==0.20.1 "
        "--index-url https://download.pytorch.org/whl/cpu",
        "uv venv /root/policy --python 3.12",
        "printf '%s\\n' /root/code/src /root/code/.venv/lib/python3.12/site-packages "
        "/root/mlenv/lib/python3.12/site-packages > /root/policy/lib/python3.12/site-packages/robot_ml.pth",
    )
    for directory in ("submission", "method/modular"):
        image = image.add_local_dir(
            ROOT / directory, f"{CODE}/{directory}", ignore=["**/__pycache__", "**/*.pyc"]
        )

OPTIONS = dict(
    image=image, gpu="T4", cpu=2, memory=4096, volumes={"/runs": volume}, scaledown_window=10
)


def run_candidate(args):
    if hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != WEIGHTS_SHA:
        raise ValueError("Unexpected est_v1b checkpoint digest")
    env = {
        **os.environ,
        "ADV_ESTIMATOR": str(CHECKPOINT),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NVIDIA_DRIVER_CAPABILITIES": "all",
    }
    return subprocess.run(
        args, cwd=CODE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )


def job_dir(name):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", name):
        raise ValueError("Invalid run name")
    volume.reload()
    dest = OUT / name
    dest.mkdir(parents=True, exist_ok=False)
    return dest


CPU_OPTIONS = dict(image=image, cpu=2, memory=4096, volumes={"/runs": volume}, scaledown_window=10)


def _episode(name: str, seed: int, gl: str):
    if not 1_000_000 <= seed < 1_100_000:
        raise ValueError("Only development seeds supported")
    from advaitic_eval.protocol import make_plan

    # A 1-episode smoke plan starting at `seed` is exactly that seed.
    assert make_plan(30, seed)[0].seed == seed
    dest = job_dir(f"{name}-{seed}")
    args = [
        f"{CODE}/.venv/bin/advaitic-eval",
        "--policy",
        "submission.policy_modular:make_policy",
        "--policy-python",
        "/root/policy/bin/python",
        "--method",
        name,
        "--checkpoint",
        f"est_v1b/last.pt@sha256:{WEIGHTS_SHA}",
        "--episodes",
        "1",
        "--seed-start",
        str(seed),
        "--smoke",
        "--out",
        str(dest / "eval"),
    ]
    os.environ["MUJOCO_GL"] = gl
    if gl == "osmesa":
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    start = time.monotonic()
    proc = run_candidate(args)
    (dest / "run.log").write_text(proc.stdout)
    volume.commit()
    report_path = dest / "eval/report.json"
    if not report_path.is_file():
        raise RuntimeError(proc.stdout[-4000:])
    report = json.loads(report_path.read_text())
    if not report["integrity_ok"] or len(report["episodes"]) != 1 or report["infrastructure_error"]:
        raise RuntimeError(f"Incomplete or invalid episode in {dest}")
    return {"seed": seed, "seconds": time.monotonic() - start, "gl": gl, **report["episodes"][0]}


@app.function(**OPTIONS, timeout=1800, max_containers=60)
def episode_gpu(name: str, seed: int):
    return _episode(name, seed, "egl")


@app.function(**CPU_OPTIONS, timeout=3600, max_containers=200)
def episode_cpu(name: str, seed: int):
    return _episode(name, seed, "osmesa")


@app.function(image=image, timeout=600)
def plan(episodes: int, seed_start: int):
    from advaitic_eval.protocol import make_plan

    return [(e.seed, e.source_slot) for e in make_plan(episodes, seed_start)]
