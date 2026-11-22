"""Deploy and collect camera-free modular development rollouts on Modal CPU.

Run the smoke, deploy the app, then use launch and collect as local entrypoints.
Only the selected controller sources and the pinned simulator package are uploaded.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
CODE = Path("/root/code")
APP_NAME = "modular-sol-privileged"
MODULES = (
    "__init__.py",
    "controller.py",
    "kinematics.py",
    "layout.py",
    "phases.py",
    "privileged.py",
    "run_privileged.py",
)
MAX_CONTAINERS = 12
MAX_SHARD_EPISODES = 20

app = modal.App(APP_NAME)
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
    .env(
        {
            "MUJOCO_GL": "egl",
            "ORT_DISABLE_TELEMETRY": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        }
    )
)
if modal.is_local():
    for name in ("pyproject.toml", "uv.lock", "README.md"):
        image = image.add_local_file(ROOT / name, str(CODE / name), copy=True)
    image = image.add_local_dir(
        ROOT / "src", str(CODE / "src"), copy=True, ignore=["**/__pycache__"]
    ).run_commands(f"cd {CODE} && uv sync --frozen --no-dev")
    for name in MODULES:
        source = Path("method/modular") / name
        image = image.add_local_file(ROOT / source, str(CODE / source))


def check_seeds(start: int, count: int):
    stop = start + count
    if not 3_000_000 <= start < stop <= 2**32:
        raise ValueError("Use a nonempty privileged development seed range beginning at 3000000+")
    for lo, hi in ((2_600_000, 2_700_000), (7_310_000, 7_320_000)):
        if start < hi and stop > lo:
            raise ValueError("Requested seeds overlap a protected block")


def source_digests(root: Path) -> dict[str, str]:
    return {
        name: hashlib.sha256((root / "method/modular" / name).read_bytes()).hexdigest()
        for name in MODULES
    }


@app.function(image=image, cpu=1, memory=2048, timeout=300, max_containers=MAX_CONTAINERS)
def shard(seed_start: int, episodes: int) -> dict:
    check_seeds(seed_start, episodes)
    if episodes > MAX_SHARD_EPISODES:
        raise ValueError(f"Each container can run at most {MAX_SHARD_EPISODES} episodes")
    start = time.monotonic()
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / "episodes.jsonl"
        proc = subprocess.run(
            [
                "uv",
                "run",
                "--frozen",
                "--no-dev",
                "--offline",
                "--no-sync",
                "python",
                "-m",
                "method.modular.run_privileged",
                "--policy",
                "modular",
                "--seed-start",
                str(seed_start),
                "--episodes",
                str(episodes),
                "--out",
                str(out),
            ],
            cwd=CODE,
            capture_output=True,
            text=True,
        )
        if proc.returncode:
            raise RuntimeError(f"{proc.stdout[-1000:]}\n{proc.stderr[-4000:]}")
        rows = [json.loads(line) for line in out.read_text().splitlines()]
    if sorted(r["seed"] for r in rows) != list(range(seed_start, seed_start + episodes)):
        raise RuntimeError("Shard output does not cover its assigned seeds")
    return {"rows": rows, "seconds": time.monotonic() - start, "sha256": source_digests(CODE)}


@app.local_entrypoint()
def smoke(seed: int, out: str):
    result = shard.remote(seed, 1)
    row = result["rows"][0]
    if result["sha256"] != source_digests(ROOT):
        raise RuntimeError("Modal source differs from the local worktree")
    if not (row["success"] and row["m2"] and row["termination"] == "success"):
        raise RuntimeError(f"Smoke did not complete the task: {row}")
    with Path(out).open("x") as stream:
        json.dump(result, stream, indent=2)
    print(f"{APP_NAME}: smoke success seed={seed}, {row['steps']} steps, {result['seconds']:.1f}s")


@app.local_entrypoint()
def launch(seed_start: int, episodes: int, out: str, per_shard: int = MAX_SHARD_EPISODES):
    """Spawn calls on the deployed app. The jobs survive this client exiting."""
    check_seeds(seed_start, episodes)
    if not 1 <= per_shard <= MAX_SHARD_EPISODES or episodes > MAX_CONTAINERS * per_shard:
        raise ValueError("Use at most twelve shards of 1–20 episodes")
    deployed = modal.Function.from_name(APP_NAME, "shard")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    stop = seed_start + episodes
    with Path(out).open("x") as stream:
        meta = {
            "app": APP_NAME,
            "git_sha": sha,
            "sha256": source_digests(ROOT),
            "time": time.time(),
        }
        stream.write(json.dumps(meta) + "\n")
        for start in range(seed_start, stop, per_shard):
            count = min(per_shard, stop - start)
            call = deployed.spawn(start, count)
            stream.write(json.dumps({"start": start, "count": count, "id": call.object_id}) + "\n")
            stream.flush()
            print(start, count, call.object_id, flush=True)


@app.local_entrypoint()
def collect(calls: str, out: str):
    """Retrieve and validate the detached results before publishing aggregate files."""
    path = Path(out)
    meta, *manifest = [json.loads(line) for line in Path(calls).read_text().splitlines()]
    rows = []
    shard_seconds = []
    for call in manifest:
        result = modal.FunctionCall.from_id(call["id"]).get(timeout=360)
        if result["sha256"] != meta["sha256"]:
            raise RuntimeError("Deployed app source differs from the launch worktree")
        rows.extend(result["rows"])
        shard_seconds.append(result["seconds"])
        print(f"Retrieved {call['start']}–{call['start'] + call['count'] - 1}", flush=True)
    rows.sort(key=lambda r: r["seed"])
    expected = [
        seed for call in manifest for seed in range(call["start"], call["start"] + call["count"])
    ]
    if not rows or [r["seed"] for r in rows] != expected:
        raise RuntimeError("Sweep results contain missing or duplicate seeds")
    successes = sum(row["success"] for row in rows)
    summary = {
        **meta,
        "seed_start": rows[0]["seed"],
        "seed_end": rows[-1]["seed"],
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows),
        "m2_episodes": sum(row["m2"] for row in rows),
        "failure_buckets": dict(Counter(row["failure_bucket"] for row in rows)),
        "containers_requested": len(manifest),
        "shard_seconds": shard_seconds,
        "elapsed_until_collection_s": time.time() - meta["time"],
    }
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    with path.with_suffix(".summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))
