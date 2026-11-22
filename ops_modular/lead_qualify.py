"""QUALIFYING-FORMAT evaluation (lead): one unmodified `advaitic-eval` run per call — a complete, balanced, scored report
(no --smoke, no sharding), exactly the command in README "Evaluate". Seed starts are restricted to the confirmation block
2600000 and the reviewer block 7310000 (each run ONCE on the frozen final submission), plus a 1-episode smoke option.
Deploy: modal deploy ops_modular/lead_qualify.py (from a git-archive export of the frozen commit)."""

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
app = modal.App("modular-qualify")
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


@app.function(**OPTIONS, timeout=6 * 3600, max_containers=2)
def qualify(
    name: str, seed_start: int, policy: str, heads: str, episodes: int = 30, smoke: bool = False
):
    dev = 1_000_000 <= seed_start < 1_100_000
    if seed_start not in (2_600_000, 7_310_000) and not (smoke or dev):
        raise ValueError(
            "Qualifying-format runs only on dev blocks, 2600000 (confirm) or 7310000 (reviewer)"
        )
    if smoke and not 1_000_000 <= seed_start < 1_100_000:
        raise ValueError("Smoke only on development seeds")
    dest = job_dir(name)
    args = [
        f"{CODE}/.venv/bin/advaitic-eval",
        "--policy",
        policy,
        "--policy-python",
        "/root/policy/bin/python",
        "--method",
        "modular-controller",
        "--checkpoint",
        f"est_v1b@sha256:{WEIGHTS_SHA}+heads:{os.path.basename(heads)}",
        "--episodes",
        str(1 if smoke else episodes),
        "--seed-start",
        str(seed_start),
        "--out",
        str(dest / "eval"),
    ]
    if smoke:
        args += ["--smoke", "--max-steps", "10"]
    os.environ["ADV_HEADS"] = heads
    start = time.monotonic()
    proc = run_candidate(args)
    (dest / "run.log").write_text(proc.stdout)
    volume.commit()
    report_path = dest / "eval/report.json"
    if not report_path.is_file():
        raise RuntimeError(proc.stdout[-4000:])
    report = json.loads(report_path.read_text())
    keys = ("status", "successes", "integrity_ok", "infrastructure_error", "submission_sha256")
    return {
        "output": str(dest),
        "seconds": time.monotonic() - start,
        **{k: report.get(k) for k in keys},
        "episodes": len(report["episodes"]),
        "per_slot": report.get("per_slot") or report.get("slots"),
    }
