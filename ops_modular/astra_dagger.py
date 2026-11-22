"""CPU rendered data from Sol's grasp controller. One detached call per seed."""

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import modal

from ops_modular.modular_sim import image, run_candidate, volume

app = modal.App("modular-astra-dagger")
data = modal.Volume.from_name("modular-data")


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=5400,
    max_containers=100,
    volumes={"/runs": volume, "/data": data},
    scaledown_window=5,
)
def episode(seed, arm="sol", digest=""):
    return run_episode(seed, arm, digest)


def run_episode(seed, arm, digest):
    if not (
        3110000 <= seed < 3110300 and arm == "sol" or 3111000 <= seed < 3111030 and arm == "heads"
    ):
        raise ValueError("Only the predeclared training and closed-loop test blocks")
    os.environ.update(MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa", LP_NUM_THREADS="2")
    data.reload()
    if arm == "heads":
        path = Path("/data/astra-dagger/heads.npz")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Head checkpoint mismatch")
        os.environ["ADV_HEADS"] = str(path)
    folder = Path("/data/astra-dagger") / (arm if not digest else digest)
    folder.mkdir(exist_ok=True, parents=True)
    rec = folder / f"{seed}.json"
    if rec.exists():
        return json.loads(rec.read_text())
    start = time.monotonic()
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "record.npz"
        p = run_candidate(
            [
                "/root/policy/bin/python",
                "-m",
                "method.modular.v2_rollouts",
                str(seed),
                str(dest),
                "--factory",
                arm,
            ]
        )
        if p.returncode:
            raise RuntimeError(p.stdout[-4000:])
        result = dict(
            **json.loads(p.stdout.splitlines()[-1]), seconds=time.monotonic() - start, arm=arm
        )
        (folder / f"{seed}.npz").write_bytes(dest.read_bytes())
    rec.write_text(json.dumps(result))
    data.commit()
    return result
