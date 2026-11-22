"""Retrospective camera-free labels and observed decisions for the retired bd951bc signal."""

import json
import re
import time
from pathlib import Path

import modal

from ops_modular.astra_gate_trace import options, sha, trace_archive
from ops_modular.astra_integration_replay import image, replay_episode, volume
from ops_modular.lead_shard import CODE, ROOT

app = modal.App("modular-astra-signal-audit-bd951bc")
NAME = "sig1_bd951bc"
FACTORY = "submission.policy_modular_calibrated:make_policy"
HEAD = "/runs/heads/64dfe105e7f716c0ab5b7a9c487b7a8a1cac06565070b390e75992dfbee4deff.npz"
if modal.is_local():
    image = image.add_local_file(ROOT / "frozen.json", f"{CODE}/frozen.json")


def original_failure(folder):
    if sha != "bd951bc":
        raise ValueError("Use the lead's frozen bd951bc policy export")
    match = re.fullmatch(r"sig1_bd951bc-(10[3-8]00[0-9]{2})(-r[0-9]+)?", folder)
    if not match:
        raise ValueError("Expected an existing retired six-block bd951bc archive")
    volume.reload()
    src = Path("/runs/modular") / folder / "eval"
    report = json.loads((src / "report.json").read_text())
    row = report["episodes"][0]
    if (
        report["method"] != NAME
        or report["policy"] != FACTORY
        or not report["integrity_ok"]
        or report["infrastructure_error"]
        or not report["smoke"]
        or len(report["episodes"]) != 1
        or row["seed"] != int(match[1])
        or row["success"]
        or not row["completed"]
        or row["policy_error"]
    ):
        raise ValueError("Expected a completed physical failure from the original frozen policy")
    return src


@app.function(image=image, cpu=1, timeout=600, max_containers=4, volumes={"/runs": volume})
def physical(folder):
    start = time.monotonic()
    result = replay_episode(original_failure(folder))
    return {**result, "seconds": time.monotonic() - start}


@app.function(**options, timeout=7200, max_containers=8, single_use_containers=True)
def decisions(folder):
    original_failure(folder)
    return trace_archive(NAME, folder, HEAD)
