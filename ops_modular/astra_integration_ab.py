"""Detached CPU placement hybrid jobs with the frozen integration candidate."""

import hashlib
import json
import os
from pathlib import Path

import modal

from ops_modular.modular_sim import image, run_candidate, volume

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("modular-astra-integration-ab")
if modal.is_local():
    image = image.add_local_file(
        ROOT / "ops_modular/astra_integration_hybrid.py", "/root/code/astra_integration_hybrid.py"
    )
data = modal.Volume.from_name("modular-data")


@app.function(
    image=image,
    cpu=4,
    memory=4096,
    timeout=7200,
    max_containers=40,
    volumes={"/data": data, "/runs": volume},
)
def run(seed, arm, digest):
    if not 3000800 <= seed < 3001000 or arm not in ("base", "site", "plate", "both"):
        raise ValueError("Invalid integration diagnostic arm")
    if not len(digest) == 64:
        raise ValueError("Expected checkpoint identity")
    data.reload()
    out = Path("/data/astra-integration/hybrid") / digest
    out.mkdir(exist_ok=True, parents=True)
    dest = out / f"{seed}-{arm}.json"
    if dest.exists():
        return json.loads(dest.read_text())
    path = Path("/data/astra-integration") / (digest + ".npz")
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("Bad combined checkpoint")
    os.environ.update(
        ADV_HEADS=str(path), MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa", LP_NUM_THREADS="4"
    )
    proc = run_candidate(
        ["/root/policy/bin/python", "-m", "astra_integration_hybrid", str(seed), arm]
    )
    if proc.returncode:
        raise RuntimeError(proc.stdout[-1800:])
    rec = json.loads(proc.stdout.splitlines()[-1])
    dest.write_text(json.dumps(rec))
    data.commit()
    return rec
