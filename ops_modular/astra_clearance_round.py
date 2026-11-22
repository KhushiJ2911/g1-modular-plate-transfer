"""Run an independently validated clearance candidate after the pending FK round, if needed."""

import json
import re
import time

import modal

from ops_modular.astra_height_round import HEAD, ROOT, launch_once, qualify, records

app = modal.App("modular-astra-clearance-decision")
PREVIOUS = "astra_height_6b02f33_dev"


@app.function(cpu=0.125, memory=256, timeout=86400, max_containers=1, volumes={"/records": records})
def advance(prior_call, low_site_call, name, sharder, policy, balanced_app, max_shards):
    if not re.fullmatch(r"astra_clearance_[0-9a-f]{7,8}_dev", name):
        raise ValueError("Expected a frozen clearance development round")
    sha = name.split("_")[2]
    if (
        not re.fullmatch(rf"modular-shard-astra-{sha}[0-9a-f]?-64dfe105", sharder)
        or policy
        not in (
            "method.modular.base_clearance:make_policy",
            "method.modular.base_clearance:make_fk_policy",
        )
        or balanced_app != "modular-astra-balanced-clearance-" + sha
        or not 1 <= max_shards <= 35
    ):
        raise ValueError("Unexpected candidate or capacity")
    previous = ROOT / PREVIOUS
    records.reload()
    if (previous / "official.call").read_text().strip() != prior_call:
        raise ValueError("Previous round identity mismatch")
    verdict = modal.FunctionCall.from_id(prior_call).get()
    if verdict["decision"] == "pass":
        return {"skipped": "FK-seating round passed and owns the qualifying runs"}
    if verdict["decision"] != "no-go":
        raise ValueError("Expected terminal FK-seating decision")
    # The lead scheduled the low-site combination next. It either stops on
    # its generation comparison, or runs its own complete dev round.
    low_site = modal.FunctionCall.from_id(low_site_call).get()
    if low_site.get("qualified_calls") or low_site.get("decision") == "pass":
        return {"skipped": "Low-site candidate owns the qualifying reports"}
    if low_site.get("decision") not in ("paired regression", "no-go"):
        raise ValueError("Expected a terminal low-site result")

    records.reload()
    folder = ROOT / name
    evidence = json.loads((folder / "promotion.json").read_text())
    if (
        evidence["policy"] != policy
        or evidence["sharder"] != sharder
        or evidence["observed_generation_gains"] <= evidence["observed_generation_losses"]
        or not evidence["full_policy_smoke"]
    ):
        raise ValueError("Expected independent positive observed-generation evidence")
    plan = json.loads((previous / "candidate.json").read_text())["plan"]
    runner = modal.Function.from_name("modular-astra-integration-round", "supervise")
    call = launch_once(
        folder / "official.call", runner, name, sharder, policy, HEAD, plan, {}, [], smoke=True
    )
    while True:
        records.reload()
        path = folder / "summary.json"
        if path.exists() and json.loads(path.read_text())["done"]:
            modal.Function.from_name(sharder, "episode_cpu").autoscaler.update(
                max_containers=max_shards
            )
            break
        try:
            call.get(timeout=0)
            break
        except TimeoutError:
            time.sleep(45)
    terminal = call.get()
    if terminal["decision"] == "pass":
        return qualify(name, balanced_app)
    return {"name": name, "decision": terminal["decision"], "official": call.object_id}
