"""Read-only slot geometry replays of the two frozen official candidates."""

import json
import os
import re
import subprocess
from pathlib import Path

import modal

from ops_modular.modular_sim import CODE, ROOT, image, volume

app = modal.App("modular-grasp-slot-geometry")
observability = modal.App("modular-grasp-slot-observability")
REPLAY_ENV = {"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"}
if modal.is_local():
    project = ROOT.parent
    image = (
        image.add_local_file(
            ROOT / "ops_modular/grasp_slot_geometry.py", "/root/analysis/slot_geometry.py"
        )
        .add_local_file(ROOT / "ops_modular/grasp_slot_drift.py", "/root/analysis/slot_drift.py")
        .add_local_file(project / "analysis/replay_eval.py", "/root/analysis/replay_eval.py")
        .add_local_file(
            project / "wt-teacher/method/teacher/diagnostics.py",
            "/home/keyu/advaitic/wt-teacher/method/teacher/diagnostics.py",
        )
    )


@app.function(
    image=image,
    cpu=1,
    memory=1536,
    max_containers=4,
    volumes={"/runs": volume},
    timeout=360,
    scaledown_window=5,
)
def replay(folder):
    if not re.fullmatch(
        r"astra_(?:calibrated_bd951bc|height_6b02f33)_dev-10[012][0-9]{4}(?:-r[0-9]+)?", folder
    ):
        raise ValueError("Expected a frozen development archive")
    src = Path("/runs/modular", folder, "eval")
    p = subprocess.run(
        [f"{CODE}/.venv/bin/python", "/root/analysis/slot_geometry.py", str(src)],
        env={**os.environ, **REPLAY_ENV},
        text=True,
        capture_output=True,
    )
    if p.returncode:
        raise RuntimeError(p.stderr[-2500:])
    return json.loads(p.stdout.splitlines()[-1])


@app.function(
    image=image,
    cpu=1,
    memory=1536,
    max_containers=1,
    volumes={"/runs": volume},
    timeout=1800,
    scaledown_window=5,
)
def compare(pairs):
    rows = []
    for base, clearance in pairs:
        for arm, path in (("base", base), ("clearance", clearance)):
            if not re.fullmatch(rf"/runs/slot-development/{arm}-30009\d\d-[\w-]+/eval", path):
                raise ValueError("Expected a fresh public archive")
        p = subprocess.run(
            [f"{CODE}/.venv/bin/python", "/root/analysis/slot_geometry.py", base, clearance],
            env={**os.environ, **REPLAY_ENV},
            text=True,
            capture_output=True,
        )
        if p.returncode:
            raise RuntimeError(p.stderr[-2500:])
        rows.append(json.loads(p.stdout.splitlines()[-1]))
    return rows


@observability.function(
    image=image,
    cpu=1,
    memory=2048,
    max_containers=1,
    volumes={"/runs": volume},
    timeout=1800,
    scaledown_window=5,
)
def batch(jobs, intervention=None):
    args = ["/root/policy/bin/python", "/root/analysis/slot_drift.py", json.dumps(jobs)]
    if intervention is not None:
        if intervention not in ("unload", "feedback"):
            raise ValueError("Expected an opening intervention")
        args.append(intervention)
    p = subprocess.run(
        args,
        cwd=CODE,
        env={**os.environ, **REPLAY_ENV, "PYTHONPATH": CODE},
        text=True,
        capture_output=True,
    )
    if p.returncode:
        raise RuntimeError(p.stderr[-2500:])
    return json.loads(p.stdout.splitlines()[-1])
