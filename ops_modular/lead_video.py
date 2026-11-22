"""Record full-length development episodes from a frozen policy export on one T4.

From the export root (with the intended Modal account selected via MODAL_CONFIG_PATH):
    uv run --frozen --with modal modal run -m ops_modular.lead_video::record \
        --name video_414f15a_dev --seed-start 1042000 --video-episodes 4 \
        --policy submission.policy_modular_combined:make_policy \
        --heads /runs/heads/64dfe105e7f716c0ab5b7a9c487b7a8a1cac06565070b390e75992dfbee4deff.npz

The evaluator writes report.json, MP4s and actions to advaitic-runs:/modular/<name>/eval.
"""

import json
import os
from pathlib import Path

import modal

from ops_modular.lead_qualify import CODE, ROOT, WEIGHTS_SHA, job_dir, run_candidate, volume

app = modal.App("modular-grasp-lead-video")
# Match lead_qualify's cached evaluator and policy environments, then install
# the encoder into the evaluator environment before mounting the frozen policy.
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
    image = image.run_commands(f"cd {CODE} && uv sync --frozen --extra video --no-dev")
    for directory in ("submission", "method/modular"):
        image = image.add_local_dir(
            ROOT / directory, f"{CODE}/{directory}", ignore=["**/__pycache__", "**/*.pyc"]
        )


@app.function(
    image=image,
    gpu="T4",
    cpu=2,
    memory=4096,
    volumes={"/runs": volume},
    max_containers=1,
    scaledown_window=10,
    timeout=3600,
)
def record(name: str, seed_start: int, policy: str, heads: str, video_episodes: int = 4):
    if not 1_000_000 <= seed_start < 1_100_000:
        raise ValueError("Choose a development seed start in [1000000, 1100000).")
    if not 1 <= video_episodes <= 30:
        raise ValueError("The evaluator supports 1–30 episodes in a development smoke run.")
    dest = job_dir(name)
    output = dest / "eval"
    os.environ["ADV_HEADS"] = heads
    try:
        proc = run_candidate(
            [
                f"{CODE}/.venv/bin/advaitic-eval",
                "--smoke",
                "--policy",
                policy,
                "--policy-python",
                "/root/policy/bin/python",
                "--method",
                "modular-controller",
                "--checkpoint",
                f"est_v1b@sha256:{WEIGHTS_SHA}+heads:{Path(heads).name}",
                "--episodes",
                str(video_episodes),
                "--video-episodes",
                str(video_episodes),
                "--seed-start",
                str(seed_start),
                "--out",
                str(output),
            ]
        )
        (dest / "run.log").write_text(proc.stdout)
        if proc.returncode:
            raise RuntimeError(proc.stdout[-4000:])
        report = json.loads((output / "report.json").read_text())
        if (
            not report["integrity_ok"]
            or report["infrastructure_error"]
            or report["completed_episodes"] != video_episodes
            or any(not ep["completed"] or ep["policy_error"] for ep in report["episodes"])
        ):
            raise RuntimeError(f"Incomplete or invalid video run. See {output}/report.json")
        for i in range(video_episodes):
            if not (output / f"episode_{i:03}.mp4").stat().st_size:
                raise RuntimeError(f"Empty recording for episode {i}")
        return {
            "output": str(output),
            "videos": video_episodes,
            "successes": report["successes"],
            "submission_sha256": report["submission_sha256"],
        }
    finally:
        volume.commit()
