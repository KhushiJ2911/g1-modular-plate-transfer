"""Detached fresh-data jobs and recorded-action replays on the dedicated account."""

import csv
import hashlib
import json
import time
from pathlib import Path

import modal

from ops_modular.modular_sim import image, run_candidate, volume

app = modal.App("modular-astra-perception")
data = modal.Volume.from_name("modular-data", create_if_missing=True)


def run_episode(kind: str, seed: int, index: int = -1, block: int = 0, model: str = ""):
    if kind not in ("estimate", "truth", "replay", "refined"):
        raise ValueError(kind)
    volume.reload()
    data.reload()
    out = Path("/data/astra-r3")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{kind}-{seed}{'-' + model if model else ''}.npz"
    if path.exists():
        raise FileExistsError(path)
    driver = "truth" if kind == "truth" else "estimate"
    args = [
        "/root/policy/bin/python",
        "-m",
        "method.modular.perception_audit",
        str(seed),
        driver,
        str(path),
    ]
    official = None
    if kind == "replay":
        if block not in (1000000, 1010000, 1020000) or not 0 <= index < 30:
            raise ValueError("Expected one official development trace")
        folder = Path(f"/runs/modular/lead_e588ae9_dev_{block}/eval")
        if (folder / "report.json").exists():
            official = json.loads((folder / "report.json").read_text())["episodes"][index]
        else:
            with (folder / "episodes.csv").open() as stream:
                official = list(csv.DictReader(stream))[index]
            for key in ("steps", "seed", "episode"):
                official[key] = int(official[key])
            for key in ("success", "fell"):
                official[key] = official[key] == "True"
        frozen = Path("/root/code/method/modular/controller.py")
        if (
            hashlib.sha256(frozen.read_bytes()).hexdigest()
            != "2d90ca7f22c02449d7c1398523bf9502529b3e35eb70ed1a783986f227e26623"
        ):
            raise RuntimeError("Shadow replay requires frozen controller J")
        if official["success"] or official["seed"] != seed or official["episode"] != index:
            raise ValueError("Expected failed recorded episode")
        args += ["--actions", str(folder / f"episode_{index:03}.actions.npz")]
    elif not 3000000 <= seed < 3100000:
        raise ValueError("Unreserved privileged-development seeds only")
    if kind == "refined":
        if not model.isidentifier():
            raise ValueError("Invalid checkpoint name")
        args += ["--refined", str(out / (model + ".npz"))]
    start = time.monotonic()
    proc = run_candidate(args)
    if proc.returncode:
        raise RuntimeError(proc.stdout[-4000:])
    info = json.loads(proc.stdout.splitlines()[-1])
    if official and any(
        info["outcome"][k] != official[k] for k in ("steps", "fell", "success", "termination")
    ):
        raise RuntimeError(f"Replay disagrees with official record: {info} vs {official}")
    data.commit()
    return {**info, "seconds": time.monotonic() - start}


episode = app.function(
    image=image,
    gpu="T4",
    cpu=2,
    memory=4096,
    volumes={"/runs": volume, "/data": data},
    max_containers=16,
    timeout=900,
    scaledown_window=10,
    name="episode",
)(run_episode)
