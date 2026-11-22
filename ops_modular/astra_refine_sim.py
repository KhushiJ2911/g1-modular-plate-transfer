"""Pinned observation-head evaluation, isolated from in-flight baseline workers."""

import hashlib
from pathlib import Path

import modal

from ops_modular.astra_perception import data, image, run_episode, volume

app = modal.App("modular-astra-refinement")
HEAD_DIGEST = "ef87f233a748c13bbe2a6dce4a4669b1a659526b7cf39411cd1caae1b17a0ae7"


@app.function(
    image=image,
    gpu="T4",
    cpu=2,
    memory=4096,
    volumes={"/runs": volume, "/data": data},
    max_containers=8,
    timeout=900,
    scaledown_window=10,
)
def episode(kind, seed, index=-1, block=0, model=""):
    p = Path("/root/code/method/modular/refined.py")
    if hashlib.sha256(p.read_bytes()).hexdigest() != HEAD_DIGEST:
        raise RuntimeError("Fresh refinement worker has the wrong adapter digest")
    if kind != "refined":
        raise ValueError("This deployment is for held-out refinement comparisons")
    return run_episode(kind, seed, index, block, model)
