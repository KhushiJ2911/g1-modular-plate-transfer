"""Annotate original indexed action archives from the three complete scored dev blocks."""

import hashlib
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

import modal

from ops_modular.astra_integration_replay import image, volume

app = modal.App("modular-astra-scored-replay")
name = "astra_height_6b02f33_dev"


@app.function(image=image, cpu=1, timeout=600, max_containers=1, volumes={"/runs": volume})
def episode(folder, index, report_digest):
    pattern = rf"{name}-balanced-(1000000|1010000|1020000)(-r[0-9]+)?"
    match = re.fullmatch(pattern, folder)
    if not match or not 0 <= index < 30:
        raise ValueError("Expected a frozen development block and an original episode index")
    volume.reload()
    src = Path("/runs/modular") / folder / "eval"
    raw = (src / "report.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != report_digest:
        raise ValueError("Archived report differs from the validated receipt")
    report = json.loads(raw)
    row = report["episodes"][index]
    if (
        not report["integrity_ok"]
        or report["smoke"]
        or not report["qualifying_run"]
        or report["infrastructure_error"]
        or report["completed_episodes"] != 30
        or report["seed_start"] != int(match[1])
        or report["method"] != name + "-balanced"
        or report["policy"] != "method.modular.fk_seating:make_policy"
        or row["episode"] != index
        or not row["completed"]
        or row["success"]
        or row["policy_error"]
    ):
        raise ValueError("Expected a completed physical failure in its original official report")
    start = time.monotonic()
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "event.json"
        subprocess.run(
            [
                "/root/code/.venv/bin/python",
                "/root/analysis/event_replay.py",
                str(src),
                str(out),
                str(index),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(out.read_text())
    if (result["episode"], result["seed"], result["slot"], result["replay_success"]) != (
        index,
        row["seed"],
        row["source_slot"],
        False,
    ):
        raise ValueError("Camera-free annotation and scored report disagree")
    return {
        **result,
        "report_sha256": report_digest,
        "actions_sha256": hashlib.sha256(
            (src / f"episode_{index:03}.actions.npz").read_bytes()
        ).hexdigest(),
        "seconds": time.monotonic() - start,
    }
