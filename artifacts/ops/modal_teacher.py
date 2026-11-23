"""Generate teacher_v1 (DART 0) demos on Modal T4s, using NVIDIA EGL.

Usage (from /home/keyu/advaitic/ops):
  modal run modal_teacher.py::check --start 2800000
  modal run modal_teacher.py::gen --start 2800000 --count 240 --per-shard 10

The check writes teacher_v1_gen2_check; gen writes teacher_v1_gen2 on advaitic-data.
Only a clean wt-teacher worktree may be snapshotted; its SHA is baked into .git_sha.
"""

import json
import shutil
import subprocess
import time
from pathlib import Path

import modal

import os
SOURCE = os.environ.get("ADV_TEACHER_CODE", "/home/keyu/advaitic/wt-teacher")
CODE = "/root/code"
GL_ENV = {
    "MUJOCO_GL": "egl",
    "LIBGL_ALWAYS_SOFTWARE": "1",
    "__EGL_VENDOR_LIBRARY_FILENAMES": "/usr/share/glvnd/egl_vendor.d/50_mesa.json",
    "EGL_PLATFORM": "surfaceless",
    "ORT_DISABLE_TELEMETRY": "1",
    "OMP_NUM_THREADS": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}
GPU_GL = ("env -u LIBGL_ALWAYS_SOFTWARE -u __EGL_VENDOR_LIBRARY_FILENAMES -u EGL_PLATFORM "
          "MUJOCO_GL=egl ")

# Do not run local git commands when Modal imports this module inside a container.
image = None
if modal.is_local():
    if Path(SOURCE, ".git_sha").exists() and not Path(SOURCE, ".git").exists():
        sha = Path(SOURCE, ".git_sha").read_text().strip()  # clean `git archive` export
    else:
        dirty = subprocess.check_output(["git", "-C", SOURCE, "status", "--porcelain"], text=True)
        if dirty:
            raise RuntimeError(f"Refusing to snapshot dirty wt-teacher:\n{dirty}")
        sha = subprocess.check_output(["git", "-C", SOURCE, "rev-parse", "--short", "HEAD"],
                                      text=True).strip()
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git", "libegl1", "libgl1", "libgles2", "libegl-mesa0",
                     "libgl1-mesa-dri", "libosmesa6", "libglib2.0-0")
        .pip_install("uv==0.9.5")
        .env(GL_ENV)
        .add_local_dir(SOURCE, CODE, copy=True,
                       ignore=["**/__pycache__", "**/checkpoints", ".git", ".venv", "results"])
        .run_commands(
            f"printf '%s\\n' {sha} > {CODE}/.git_sha",
            f"cd {CODE} && uv sync --frozen --no-dev",
            f"cd {CODE} && uv run bigym-download --task move_plate",
            f"cd {CODE} && uv run advaitic-check",
            "uv venv /root/mlenv --python 3.12",
            "uv pip install --python /root/mlenv/bin/python torch==2.5.1 torchvision==0.20.1 "
            "--index-url https://download.pytorch.org/whl/cpu",
            "uv pip install --python /root/mlenv/bin/python numpy==2.5.5 pyyaml==6.0.3 "
            "pillow==13.0.0 safetensors==0.8.0",
        )
    )

data = modal.Volume.from_name("advaitic-data", create_if_missing=True)
app = modal.App("advaitic-teacher", image=image)
# Match collect_shard_gpu's CPU and memory; shorten its timeout to two hours.
SHARD_OPTIONS = dict(gpu="T4", cpu=4, memory=8192, timeout=2 * 3600,
                     volumes={"/data": data})


def assert_seeds(start: int, stop: int):
    assert 0 <= start < stop <= 2**32, "Seeds must be a nonempty uint32 range"
    for lo, hi, name in (
        (1000000, 1100000, "dev"),
        (2600000, 2700000, "confirm"),
        (7310000, 7320000, "test"),
    ):
        assert stop <= lo or start >= hi, f"Seeds overlap protected {name} block [{lo}, {hi})"


def refuse_existing(destination: Path, start: int, stop: int):
    for seed in range(start, stop):
        path = destination / f"ep_{seed}.npz"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")


def _generate(start: int, stop: int, dataset: str, extra: str = ""):
    assert_seeds(start, stop)
    destination = Path("/data") / dataset
    data.reload()
    refuse_existing(destination, start, stop)

    # Containers can be reused. Clear previous local output so no stale demos/logs count.
    work = Path("/tmp/gen")
    shutil.rmtree(work, ignore_errors=True)
    demos = work / "demos"
    demos.mkdir(parents=True)
    attempts = work / f"attempts_{start}.jsonl"
    log = work / f"log_{start}.txt"
    cmd = (
        f"{GPU_GL}uv run python -m method.teacher.run_teacher "
        f"--mode v1 --dart-noise 0 --seeds {start}:{stop} "
        f"--out {attempts} --save-demos {demos} {extra}"
    )
    t0 = time.monotonic()
    with log.open("w") as stream:
        stream.write(f"git_sha={Path(CODE, '.git_sha').read_text().strip()}\n{cmd}\n")
        stream.flush()
        rc = subprocess.run(cmd, shell=True, cwd=CODE, stdout=stream,
                            stderr=subprocess.STDOUT).returncode

    # Preserve diagnostics even when the generator fails before its first episode.
    attempts.touch(exist_ok=True)
    rows = [json.loads(line) for line in attempts.read_text().splitlines() if line.strip()]
    data.reload()
    refuse_existing(destination, start, stop)
    meta = destination / "_meta"
    meta.mkdir(parents=True, exist_ok=True)
    for demo in sorted(demos.glob("ep_*.npz")):
        # Exclusive creation also protects against a file appearing since the check.
        with demo.open("rb") as source, (destination / demo.name).open("xb") as target:
            shutil.copyfileobj(source, target)
    for path in (attempts, log):
        shutil.copy2(path, meta / path.name)
    data.commit()
    return start, stop, len(rows), sum(row["success"] for row in rows), rc, round(time.monotonic() - t0, 1)


@app.function(**SHARD_OPTIONS)
def gen_shard(start: int, stop: int, dataset: str = "teacher_v1_gen2", extra: str = ""):
    return _generate(start, stop, dataset, extra)


@app.function(**SHARD_OPTIONS)
def check_shard(start: int, stop: int):
    return _generate(start, stop, "teacher_v1_gen2_check")


@app.local_entrypoint()
def gen(start: int = 2800000, count: int = 240, per_shard: int = 10, dataset: str = "teacher_v1_gen2",
        extra: str = ""):
    assert count > 0 and per_shard > 0, "count and per_shard must be positive"
    stop = start + count
    assert_seeds(start, stop)  # Validate the entire block before submitting any work.
    shards = [(s, min(s + per_shard, stop), dataset, extra) for s in range(start, stop, per_shard)]
    attempts = successes = failures = 0
    for result in gen_shard.starmap(shards, order_outputs=False):
        print(result, flush=True)
        _, _, n_attempts, n_success, rc, _ = result
        attempts += n_attempts
        successes += n_success
        failures += rc != 0
    yield_rate = successes / attempts if attempts else 0.0
    print(f"TOTAL attempts={attempts} successes={successes} yield={yield_rate:.1%}", flush=True)
    if failures:
        raise RuntimeError(f"{failures} shard(s) failed; inspect _meta logs")


@app.local_entrypoint()
def check(start: int = 2800000):
    assert_seeds(start, start + 1)
    result = check_shard.remote(start, start + 1)
    print(result, flush=True)
    if result[4] != 0:
        raise RuntimeError("Check failed; inspect teacher_v1_gen2_check/_meta")
