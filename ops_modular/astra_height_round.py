"""Advance from the frozen loaded-normal policy to FK seating on the same official dev plan."""

import json
import time
from pathlib import Path

import modal

app = modal.App("modular-astra-height-decision")
records = modal.Volume.from_name("modular-runs")
ROOT = Path("/records/astra-integration")
PREVIOUS = "astra_calibrated_bd951bc_dev"
NAME = "astra_height_6b02f33_dev"
SHARDER = "modular-shard-astra-6b02f33a-64dfe105"
POLICY = "method.modular.fk_seating:make_policy"
HEAD = "/runs/heads/64dfe105e7f716c0ab5b7a9c487b7a8a1cac06565070b390e75992dfbee4deff.npz"


def launch_once(path, function, *args, **kwargs):
    if path.exists():
        return modal.FunctionCall.from_id(path.read_text().strip())
    call = function.spawn(*args, **kwargs)
    path.write_text(call.object_id)
    records.commit()
    return call


def qualify(name, balanced_app):
    folder = ROOT / name
    fn = modal.Function.from_name(balanced_app, "balanced")
    calls = {
        block: launch_once(folder / f"balanced_{block}.call", fn, name, block).object_id
        for block in (1000000, 1010000, 1020000)
    }
    return {"qualified_calls": calls, "name": name}


@app.function(cpu=0.125, memory=256, timeout=86400, volumes={"/records": records}, max_containers=1)
def advance(prior_call, diagnostic_smoke):
    result = modal.FunctionCall.from_id(prior_call).get()
    records.reload()
    previous = ROOT / PREVIOUS
    if (previous / "official.call").read_text().strip() != prior_call:
        raise ValueError("Supervisor does not belong to the frozen predecessor")
    if result["decision"] == "pass":
        return qualify(PREVIOUS, "modular-astra-balanced")
    if result["decision"] != "no-go":
        raise ValueError("Expected a terminal dev decision")
    plan = json.loads((previous / "candidate.json").read_text())["plan"]
    folder = ROOT / NAME
    folder.mkdir(parents=True, exist_ok=True)
    supervisor = modal.Function.from_name("modular-astra-integration-round", "supervise")
    next_round = launch_once(
        folder / "official.call", supervisor, NAME, SHARDER, POLICY, HEAD, plan, {}, [], smoke=True
    )
    diagnostic = modal.Function.from_name("modular-astra-fault-inputs", "monitor")
    launch_once(
        folder / "faults.call", diagnostic, NAME, HEAD, "astra_visual_9a2eca9_dev", diagnostic_smoke
    )
    while True:
        records.reload()
        summary = folder / "summary.json"
        if summary.is_file() and json.loads(summary.read_text())["done"]:
            # The supervisor's initial autoscaler setting has taken effect and
            # its separate official smoke completed under the public policy.
            modal.Function.from_name(SHARDER, "episode_cpu").autoscaler.update(max_containers=35)
            break
        try:
            next_round.get(timeout=0)
        except TimeoutError:
            time.sleep(45)
        else:
            break
    verdict = next_round.get()
    if verdict["decision"] == "pass":
        return qualify(NAME, "modular-astra-balanced-6b02f33")
    return dict(name=NAME, decision=verdict["decision"], official=next_round.object_id)
