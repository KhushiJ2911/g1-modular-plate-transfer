"""GPU-only trainer for the contact, held and pickup residual heads."""

import json
import subprocess
import time
from pathlib import Path

import modal

from ops_modular.astra_v2_train import ROOT, base, data, image

app = modal.App("modular-astra-dagger-fit")
if modal.is_local():
    for f in ("method/modular/train_dagger_heads.py", "submission/head_features.py"):
        image = image.add_local_file(ROOT / f, "/root/code/" + f)


@app.function(
    image=image,
    gpu="T4",
    cpu=2,
    memory=8192,
    timeout=2400,
    max_containers=1,
    volumes={"/data": data, "/base": base},
)
def fit(smoke=False):
    data.reload()
    folder = Path("/data/astra-dagger")
    stem = "smoke" if smoke else "heads"
    dest = folder / (stem + ".npz")
    report = folder / (stem + ".json")
    if report.exists():
        return json.loads(report.read_text())
    t = time.monotonic()
    cmd = ["python", "-m", "method.modular.train_dagger_heads", str(dest)]
    proc = subprocess.run(cmd + (["--smoke"] if smoke else []), capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(proc.stderr[-4500:])
    r = dict(**json.loads(proc.stdout.splitlines()[-1]), seconds=time.monotonic() - t)
    report.write_text(json.dumps(r))
    data.commit()
    return r
