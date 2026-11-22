"""Run the estimator smoke on T4 and the short fine-tune on A10."""

from pathlib import Path
from time import monotonic

import modal

REMOTE_INIT = "/root/est_v1b.pt"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1", index_url="https://download.pytorch.org/whl/cu121"
    )
    .pip_install("numpy==2.5.5")
    .env({"PYTHONPATH": "/root/est2", "ORT_DISABLE_TELEMETRY": "1", "OMP_NUM_THREADS": "2"})
)
if modal.is_local():
    repo = Path(__file__).resolve().parents[2]
    image = (
        image.add_local_dir(repo / "method", "/root/est2/method", ignore=["**/__pycache__"])
        .add_local_dir(repo / "submission", "/root/est2/submission", ignore=["**/__pycache__"])
        .add_local_file(repo.parent / "ckpts" / "est_v1b.pt", REMOTE_INIT)
    )

app = modal.App("advaitic-est2-finetune", image=image)
data = modal.Volume.from_name("advaitic-data")
runs = modal.Volume.from_name("advaitic-runs")


@app.function(
    gpu="T4",
    cpu=2,
    memory=4096,
    timeout=180,
    scaledown_window=5,
    volumes={"/data": data.with_mount_options(read_only=True)},
)
def smoke():
    start = monotonic()
    import numpy as np
    import torch

    from method.harness.finetune_estimator import finetune
    from submission.estimator import Estimator

    dataset = Path("/data/est2_brief22_smoke")
    checkpoint_dir = Path("/tmp/est2-smoke")
    result = finetune(dataset, Path(REMOTE_INIT), checkpoint_dir, 5, 16, 5, smoke=True)
    estimator = Estimator(checkpoint_dir / "last.pt", "cuda")
    with np.load(next((dataset / "dev").glob("ep_*.npz")), allow_pickle=False) as episode:
        output = estimator({"rgb": episode["rgb"][0], "state": episode["state"][0]})
    print(torch.cuda.get_device_name(), output, flush=True)
    return {**result, "function_seconds": monotonic() - start}


@app.function(
    gpu="A10",
    cpu=4,
    memory=8192,
    timeout=1800,
    scaledown_window=5,
    volumes={"/data": data.with_mount_options(read_only=True), "/runs": runs},
)
def train():
    start = monotonic()
    import torch

    from method.harness.finetune_estimator import finetune

    print(torch.cuda.get_device_name(), flush=True)
    result = finetune(
        Path("/data/est2_brief22"), Path(REMOTE_INIT), Path("/runs/est_v2"), 2000, 64, 250
    )
    runs.commit()
    return {**result, "function_seconds": monotonic() - start}


@app.local_entrypoint()
def main(stage: str):
    jobs = {"smoke": smoke, "train": train}
    print(jobs[stage].remote(), flush=True)
