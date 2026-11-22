"""Dedicated GPU trainer for the v2 observation estimator. No live environment in this image."""

import json
import subprocess
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("modular-astra-v2-train")
data = modal.Volume.from_name("modular-data")
runs = modal.Volume.from_name("modular-runs")
base = modal.Volume.from_name("advaitic-runs")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1", index_url="https://download.pytorch.org/whl/cu121"
    )
    .pip_install("numpy==2.2.6")
    .env({"PYTHONPATH": "/root/code", "ORT_DISABLE_TELEMETRY": "1", "OMP_NUM_THREADS": "2"})
)
if modal.is_local():
    for key in (
        "submission/__init__.py",
        "submission/model.py",
        "submission/estimator.py",
        "submission/estimator_v2.py",
        "method/modular/__init__.py",
        "method/modular/train_estimator_v2.py",
    ):
        image = image.add_local_file(ROOT / key, "/root/code/" + key)


@app.function(
    image=image,
    gpu="A10G",
    cpu=4,
    memory=24576,
    volumes={"/data": data, "/runs": runs, "/base": base},
    timeout=7200,
    max_containers=1,
    scaledown_window=10,
)
def train(name, start, count):
    smoke = name == "gpu_smoke"
    if not (
        name in ("gpu_smoke", "est_v2") and start == 3100000 and count == (1 if smoke else 300)
    ):
        raise ValueError("Expected one smoke episode or the predeclared 300-episode training block")
    data.reload()
    runs.reload()
    out = Path("/runs/estimator-v2") / name
    if (out / "report.json").exists():
        report = json.loads((out / "report.json").read_text())
        return {"sha256": report["sha256"], "cached": True}
    out.mkdir(parents=True, exist_ok=True)
    t = time.monotonic()
    args = [
        "python",
        "-m",
        "method.modular.train_estimator_v2",
        "/data/est_v2",
        str(out),
        "/base/est_v1b/last.pt",
        "--start",
        str(start),
        "--count",
        str(count),
    ]
    if smoke:
        args += ["--smoke", "--steps", "3", "--batch", "8"]
    with (out / "train.log").open("w") as f:
        proc = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT)
    runs.commit()
    if proc.returncode:
        raise RuntimeError((out / "train.log").read_text()[-3000:])
    report = json.loads((out / "report.json").read_text())
    return {
        "seconds": time.monotonic() - t,
        "sha256": report["sha256"],
        "step": report["step"],
        "gpu": report["gpu"],
    }
