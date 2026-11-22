"""Detached rendered input-substitution sweeps on the dedicated Modal account."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
CODE = Path("/root/code")
CHECKPOINT = Path("/runs/est_v1b/last.pt")
CHECKPOINT_SHA = "bd8ed33c87383b3b208ba251d8fa0b6f957d9f03a35cf6c8d23a81ce9cd867cd"
APP = "modular-sol-inputs"
CPU_RENDER = os.environ.get("SOL_INPUTS_CPU_RENDER") == "1"
VARIANTS = (
    "truth",
    "source_xyz",
    "placement_xyz",
    "destination",
    "orientation",
    "tilt_only",
    "held",
    "contacts",
    "destination_contacts",
    "destination_contacts_locked",
    "all",
)
MODULES = tuple(
    f"method/modular/{name}.py"
    for name in (
        "__init__",
        "controller",
        "estimated",
        "input_ablation",
        "kinematics",
        "observed_controller",
        "orientation_feedback",
        "layout",
        "phases",
        "privileged",
        "run_privileged",
    )
) + ("submission/estimator.py", "submission/model.py")

app = modal.App(APP)
volume = modal.Volume.from_name("modular-runs")
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
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NVIDIA_DRIVER_CAPABILITIES": "all",
        }
    )
)
if CPU_RENDER:
    image = image.env(
        {
            "LIBGL_ALWAYS_SOFTWARE": "1",
            "EGL_PLATFORM": "surfaceless",
            "LP_NUM_THREADS": "2",
            "CUDA_VISIBLE_DEVICES": "",
        }
    )
if modal.is_local():
    for name in ("pyproject.toml", "uv.lock", "README.md"):
        image = image.add_local_file(ROOT / name, str(CODE / name), copy=True)
    image = image.add_local_dir(
        ROOT / "src", str(CODE / "src"), copy=True, ignore=["**/__pycache__"]
    ).run_commands(
        f"cd {CODE} && uv sync --frozen --no-dev",
        "uv venv /root/mlenv --python 3.12",
        "uv pip install --python /root/mlenv/bin/python torch==2.5.1 torchvision==0.20.1 "
        "--index-url https://download.pytorch.org/whl/cpu",
    )
    for name in MODULES:
        image = image.add_local_file(ROOT / name, str(CODE / name))


def verify_range(start, count):
    if not 1 <= count <= 60 or not 3_000_000 <= start < start + count <= 2**32:
        raise ValueError("Use 1–60 privileged development seeds per sweep")
    for lo, hi in ((2_600_000, 2_700_000), (7_310_000, 7_320_000)):
        if start < hi and start + count > lo:
            raise ValueError("Protected seed range")


def digests(root):
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in MODULES}


@app.function(
    image=image,
    volumes={"/runs": volume},
    gpu=None if CPU_RENDER else os.environ.get("SOL_INPUTS_GPU", "T4"),
    cpu=2,
    memory=4096,
    max_containers=16,
    timeout=900,
    scaledown_window=10,
)
def episode(seed: int, variants: tuple[str, ...]):
    verify_range(seed, 1)
    volume.reload()
    if hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != CHECKPOINT_SHA:
        raise RuntimeError("Unexpected estimator checkpoint digest")
    start = time.monotonic()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "rows.jsonl"
        result = subprocess.run(
            [
                str(CODE / ".venv/bin/python"),
                "-m",
                "method.modular.input_ablation",
                "--start",
                str(seed),
                "--count",
                "1",
                "--out",
                str(out),
                "--checkpoint",
                str(CHECKPOINT),
                "--variants",
                *variants,
            ],
            env={**os.environ, "PYTHONPATH": "/root/mlenv/lib/python3.12/site-packages"},
            cwd=CODE,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(f"{result.stdout[-3000:]}\n{result.stderr[-4000:]}")
        rows = [json.loads(line) for line in out.read_text().splitlines()]
    return {"rows": rows, "seconds": time.monotonic() - start, "sha256": digests(CODE)}


@app.local_entrypoint()
def smoke(seed: int, out: str, variant: str = "truth"):
    expected = digests(ROOT)
    data = episode.remote(seed, (variant,))
    if data["sha256"] != expected:
        raise RuntimeError("Smoke code hashes differ")
    row = data["rows"][0]
    if variant == "truth" and (not row["success"] or not row["m2"]):
        raise RuntimeError(f"Truth-path smoke did not complete: {row}")
    with Path(out).open("x") as f:
        json.dump(data, f, indent=2)
    print(APP, seed, variant, row["success"], row["steps"], data["seconds"], flush=True)


@app.local_entrypoint()
def launch(start: int, count: int, out: str, variants: str = "", deployment: str = APP):
    verify_range(start, count)
    groups = tuple(filter(None, variants.split(","))) if variants else VARIANTS
    if not groups or len(set(groups)) != len(groups):
        raise ValueError("Choose nonempty, distinct group names")
    if not deployment.startswith("modular-sol-inputs"):
        raise ValueError("Use a modular-sol-inputs deployment")
    func = modal.Function.from_name(deployment, "episode")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    meta = {
        "app": deployment,
        "backend": "CPU software EGL" if CPU_RENDER else os.environ.get("SOL_INPUTS_GPU", "T4"),
        "git_sha": sha,
        "start": start,
        "count": count,
        "variants": groups,
        "sha256": digests(ROOT),
        "checkpoint_sha256": CHECKPOINT_SHA,
        "time": time.time(),
    }
    with Path(out).open("x") as f:
        f.write(json.dumps(meta) + "\n")
        for seed in range(start, start + count):
            call = func.spawn(seed, groups)
            f.write(json.dumps({"seed": seed, "id": call.object_id}) + "\n")
            f.flush()
            print(seed, call.object_id, flush=True)


@app.local_entrypoint()
def collect(calls: str, out: str):
    meta, *manifest = [json.loads(line) for line in Path(calls).read_text().splitlines()]
    expected_keys = {
        (seed, variant)
        for seed in range(meta["start"], meta["start"] + meta["count"])
        for variant in meta["variants"]
    }
    planned_keys = [
        (item["seed"], variant)
        for item in manifest
        for variant in item.get("variants", meta["variants"])
    ]
    if len(planned_keys) != len(expected_keys) or set(planned_keys) != expected_keys:
        raise RuntimeError("Incomplete or overlapping call manifest")
    path = Path(out)

    def fetch(item):
        try:
            return item, modal.FunctionCall.from_id(item["id"]).get(timeout=0)
        except (TimeoutError, ConnectionError, modal.exception.ConnectionError):
            return item, None

    with path.open("a+", encoding="utf-8") as stream, ThreadPoolExecutor(max_workers=12) as pool:
        stream.seek(0)
        done = {(r["seed"], r["variant"]) for r in map(json.loads, stream)}
        pending = manifest
        while pending:
            outstanding = []
            for item, data in pool.map(fetch, pending):
                seed = item["seed"]
                expected = {(seed, variant) for variant in item.get("variants", meta["variants"])}
                if expected <= done:
                    continue
                if data is None:
                    outstanding.append(item)
                    continue
                if data["sha256"] != meta["sha256"]:
                    raise RuntimeError("Deployed code differs from source recorded at launch")
                rows = data["rows"]
                if (
                    len(rows) != len(expected)
                    or {(r["seed"], r["variant"]) for r in rows} != expected
                ):
                    raise RuntimeError(f"Incomplete episode at seed {seed}")
                for row in rows:
                    if (row["seed"], row["variant"]) not in done:
                        row["shard_seconds"] = data["seconds"]
                        row["shard_id"] = item["id"]
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                done.update(expected)
                print(
                    seed,
                    data["seconds"],
                    [row["variant"] for row in rows if row["success"]],
                    flush=True,
                )
            pending = outstanding
            if pending:
                time.sleep(15)
        stream.seek(0)
        rows = list(map(json.loads, stream))
        by_shard = {row.get("shard_id", row["seed"]): row["shard_seconds"] for row in rows}
    path.with_suffix(".run.json").write_text(
        json.dumps(
            {
                **meta,
                "retrieved": len({row["seed"] for row in rows}),
                "shards": len(manifest),
                "max_containers": 16,
                "container_work_s": sum(by_shard.values()),
                "collection_wall_s": time.time() - meta["time"],
            },
            indent=2,
        )
    )
