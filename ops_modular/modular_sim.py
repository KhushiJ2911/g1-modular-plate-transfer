"""Deploy with modal deploy ops_modular/modular_sim.py. Uses the advaitic-runs volume."""

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
CODE = "/root/code"
OUT = Path("/runs/modular")
CHECKPOINT = Path("/runs/est_v1b/last.pt")
WEIGHTS_SHA = "bd8ed33c87383b3b208ba251d8fa0b6f957d9f03a35cf6c8d23a81ce9cd867cd"
app = modal.App("modular-sim")
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


@app.function(**OPTIONS, timeout=7200, max_containers=3)
def eval_run(name: str, seed_start: int, episodes: int = 30, smoke: bool = False):
    # The release planner may scan beyond seed_start+episodes to balance slots.
    if not (1_000_000 <= seed_start and seed_start + episodes * 100 <= 1_100_000):
        raise ValueError("Only development seeds supported")
    if episodes != (1 if smoke else 30):
        raise ValueError("Choose one smoke episode or 30 scored episodes")
    dest = job_dir(name)
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
        str(episodes),
        "--seed-start",
        str(seed_start),
        "--out",
        str(dest / "eval"),
    ]
    if smoke:
        args += ["--smoke", "--max-steps", "10"]
    start = time.monotonic()
    proc = run_candidate(args)
    (dest / "run.log").write_text(proc.stdout)
    volume.commit()
    report_path = dest / "eval/report.json"
    if not report_path.is_file():
        raise RuntimeError(proc.stdout[-4000:])
    report = json.loads(report_path.read_text())
    records = report["episodes"]
    if not report["integrity_ok"] or len(records) != episodes or report["infrastructure_error"]:
        raise RuntimeError(f"Incomplete or invalid evaluation in {dest}")
    if smoke and (records[0]["steps"] != 10 or records[0]["policy_error"] or records[0]["fell"]):
        raise RuntimeError(f"Policy smoke failed: {records[0]}")
    return {
        "output": str(dest),
        "seconds": time.monotonic() - start,
        "status": report["status"],
        "episodes": len(records),
        "successes": report["successes"],
        "first_episode": records[0],
        "submission_sha256": report["submission_sha256"],
    }


@app.function(**OPTIONS, timeout=600, max_containers=5)
def phase_episode(name: str, seed: int):
    if not 3_000_000 <= seed < 3_100_000:
        raise ValueError("Only privileged development seeds supported")
    dest = job_dir(f"{name}-{seed}")
    start = time.monotonic()
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / f"{seed}.jsonl.gz"
        proc = run_candidate(
            [
                "/root/policy/bin/python",
                "-m",
                "method.modular.phase_errors",
                "run",
                str(seed),
                str(log),
            ]
        )
        (dest / "run.log").write_text(proc.stdout)
        if proc.returncode:
            volume.commit()
            raise RuntimeError(proc.stdout[-4000:])
        (dest / log.name).write_bytes(log.read_bytes())
    volume.commit()
    return {"seed": seed, "seconds": time.monotonic() - start, "output": str(dest)}
