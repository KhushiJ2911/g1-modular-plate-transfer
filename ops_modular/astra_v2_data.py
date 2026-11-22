"""Dedicated-account rendered collection, one generation episode per invocation."""

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import modal

from ops_modular.modular_sim import image, run_candidate, volume

app = modal.App("modular-astra-v2-data")
data = modal.Volume.from_name("modular-data")
root = Path("/data/est_v2/onpolicy")


def run(seed, gl):
    if not 3100000 <= seed < 3200000:
        raise ValueError("Only fresh v2 generation seeds")
    for file, digest in FROZEN.items():
        if hashlib.sha256((Path("/root/code") / file).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Collector policy source mismatch: {file}")
    os.environ["MUJOCO_GL"] = gl
    if gl == "osmesa":
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    data.reload()
    root.mkdir(parents=True, exist_ok=True)
    dest = root / f"{seed}.npz"
    record = root / f"{seed}.json"
    if record.exists() and dest.exists():
        return json.loads(record.read_text())
    start = time.monotonic()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "episode.npz"
        p = run_candidate(
            ["/root/policy/bin/python", "-m", "method.modular.v2_rollouts", str(seed), str(out)]
        )
        if p.returncode:
            raise RuntimeError(p.stdout[-3000:])
        result = dict(
            **json.loads(p.stdout.splitlines()[-1]),
            seconds=time.monotonic() - start,
            gl=gl,
            size=out.stat().st_size,
            sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
        )
        dest.write_bytes(out.read_bytes())
        record.write_text(json.dumps(result))
    data.commit()
    return result


OPTIONS = dict(
    image=image, cpu=2, memory=4096, volumes={"/runs": volume, "/data": data}, scaledown_window=10
)


@app.function(**OPTIONS, gpu="T4", max_containers=6, timeout=900)
def episode_gpu(seed):
    return run(seed, "egl")


@app.function(**OPTIONS, max_containers=48, timeout=3600)
def episode_cpu(seed):
    return run(seed, "osmesa")


FROZEN = {
    "submission/policy_modular.py": "6816aa8745de82c4a6827e826da0f05d34e4f8ad1746f7c7680859d884b2b4b9",
    "submission/estimator.py": "72cfcbad1a9eba05cafb11d18005f5fdd8422f792fb11fc2c4ef5535647a6c84",
    "submission/model.py": "f688737626e9674d132edf8d7e9032815c4fde86998dc8b3ab637d194fd20506",
    "method/modular/controller.py": "2d90ca7f22c02449d7c1398523bf9502529b3e35eb70ed1a783986f227e26623",
    "method/modular/phases.py": "6a97aecc3c128f07428783c7fc0708a8f5ca6ad5308945b8dcdab7d5d5dec7d1",
    "method/modular/estimated.py": "fc8a263d609bdc7390cb593d70289fe1673dac79b1eecee54d5e85ac9f3c51f9",
    "method/modular/kinematics.py": "3ea4c8948449d0c5b094f203a6985856b55b7eba52a953fb42954050be31e401",
    "method/modular/layout.py": "3946e69e1ade8d5a0c59668bc4be4a3ffbc28847b762ba03274d55ad00db1d78",
}
