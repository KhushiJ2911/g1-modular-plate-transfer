"""Run the three complete scored reports after a frozen candidate passes sharded development.

Copy this file into the frozen export beside lead_shard.py and deploy by module from that
directory. The exported frozen.json pins the candidate and the base and sidecar are verified
before evaluation. Start each block once using balanced.spawn(name, seed_start).
"""

import hashlib
import json
import os
import re
import time
from pathlib import Path

import modal

from ops_modular.lead_shard import (
    CODE,
    CPU_OPTIONS,
    ROOT,
    WEIGHTS_SHA,
    image,
    run_candidate,
    volume,
)

app = modal.App("modular-astra-balanced")
records = modal.Volume.from_name("modular-runs")
if modal.is_local():
    image = image.add_local_file(ROOT / "frozen.json", f"{CODE}/frozen.json")
options = {
    **CPU_OPTIONS,
    "image": image,
    "volumes": {"/runs": volume, "/records": records},
}


@app.function(**options, timeout=86400, max_containers=3, single_use_containers=True)
def balanced(name, seed_start):
    if not re.fullmatch(r"astra_[a-z0-9_]+", name) or seed_start not in (1000000, 1010000, 1020000):
        raise ValueError("Expected an integrator development block")
    frozen = json.loads(Path(CODE, "frozen.json").read_text())
    source = Path("/records/astra-integration") / name
    candidate = json.loads((source / "candidate.json").read_text())
    score = json.loads((source / "summary.json").read_text())
    if (
        candidate["app"] != frozen["app"]
        or score["decision"] != "pass"
        or score["done"] != 90
        or any(score["blocks"][str(b)][0] < 27 for b in (1000000, 1010000, 1020000))
        or any(n != 30 for _, n in score["blocks"].values())
    ):
        raise ValueError("The complete sharded development evidence must match this frozen export")
    head = Path(candidate["checkpoint"])
    if (
        head.parent != Path("/runs/heads")
        or hashlib.sha256(head.read_bytes()).hexdigest() != frozen["digest"]
    ):
        raise ValueError("Perception weights do not match the frozen export")
    volume.reload()
    root = Path("/runs/modular")
    prefix = f"{name}-balanced-{seed_start}"
    # Preemptions may persist the directory before the evaluator writes its
    # report. Preserve interrupted attempts and reuse any finished scored run.
    attempt = 0
    while True:
        dest = root / (prefix if attempt == 0 else f"{prefix}-r{attempt}")
        path = dest / "eval/report.json"
        if not dest.exists():
            break
        if path.exists() and not json.loads(path.read_text()).get("infrastructure_error"):
            break
        attempt += 1
    if not path.exists():
        dest.mkdir(exist_ok=False)
        volume.commit()
        os.environ.update(
            MUJOCO_GL="osmesa",
            PYOPENGL_PLATFORM="osmesa",
            ADV_HEADS=str(head),
        )
        args = [
            f"{CODE}/.venv/bin/advaitic-eval",
            "--policy",
            candidate["policy"],
            "--policy-python",
            "/root/policy/bin/python",
            "--method",
            name + "-balanced",
            "--checkpoint",
            f"{frozen['sha']} + est_v1b@{WEIGHTS_SHA} + heads@{frozen['digest']}",
            "--episodes",
            "30",
            "--seed-start",
            str(seed_start),
            "--out",
            str(path.parent),
        ]
        started = time.monotonic()
        proc = run_candidate(args)
        (dest / "run.log").write_text(proc.stdout)
        (dest / "wall_seconds").write_text(str(time.monotonic() - started))
        volume.commit()
        if not path.is_file():
            raise RuntimeError(proc.stdout[-2400:])
    report = json.loads(path.read_text())
    expected = [(e["seed"], e["slot"]) for e in candidate["plan"] if e["block"] == seed_start]
    actual = [(e["seed"], e["source_slot"]) for e in report["episodes"]]
    if (
        report["smoke"]
        or not report["integrity_ok"]
        or report["infrastructure_error"]
        or actual != expected
        or len(actual) != 30
        or not report["qualifying_run"]
        or not all(e["completed"] for e in report["episodes"])
        or report["policy"] != candidate["policy"]
    ):
        raise ValueError(f"Incomplete or mismatched report at {path}")
    return {
        "path": str(path),
        "successes": report["successes"],
        "meets_success_criterion": report["meets_success_criterion"],
        "submission_sha256": report["submission_sha256"],
        "worker_seconds": (
            float((dest / "wall_seconds").read_text()) if (dest / "wall_seconds").exists() else None
        ),
    }
