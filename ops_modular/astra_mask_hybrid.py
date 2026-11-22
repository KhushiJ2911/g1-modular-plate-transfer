"""Matched single-prefix placement smoke for the two-view calibration against 9a2eca9."""

import hashlib
import json
import os
from pathlib import Path

import modal

from ops_modular.modular_sim import image, run_candidate, volume

DIGEST = "409e9b7f694de4cfe2ab07e60eb06435ff16394be04ddd0e7e95f8c8ee4b5cfa"
app = modal.App("modular-astra-mask-hybrid-0b010d5")
if modal.is_local():
    image = image.add_local_file(
        Path(__file__).with_name("astra_integration_hybrid.py"),
        "/root/code/astra_integration_hybrid.py",
    )


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=7200,
    max_containers=2,
    single_use_containers=True,
    volumes={"/runs": volume},
)
def smoke(arm):
    if arm not in ("visual", "mask"):
        raise ValueError("Use the predeclared placement comparison")
    checkpoint = Path("/runs/heads") / (DIGEST + ".npz")
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != DIGEST:
        raise ValueError("Mask digest mismatch")
    os.environ.update(
        ADV_HEADS=str(checkpoint),
        MUJOCO_GL="osmesa",
        PYOPENGL_PLATFORM="osmesa",
        LP_NUM_THREADS="2",
    )
    p = run_candidate(["/root/policy/bin/python", "-m", "astra_integration_hybrid", "3001100", arm])
    if p.returncode:
        raise RuntimeError(p.stdout[-1900:])
    return json.loads(p.stdout.splitlines()[-1])
