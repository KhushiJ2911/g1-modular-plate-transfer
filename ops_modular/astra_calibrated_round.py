"""Promote a frozen grasp-calibration ablation only after independent paired evidence."""

import json
import time
from pathlib import Path

import modal

app = modal.App("modular-astra-calibrated-decision")
records = modal.Volume.from_name("modular-runs")
CHECKPOINT = "64dfe105e7f716c0ab5b7a9c487b7a8a1cac06565070b390e75992dfbee4deff"


@app.function(cpu=0.125, memory=256, timeout=86400, volumes={"/records": records})
def schedule(name, sharder, plan):
    if not name.startswith("astra_calibrated_") or not sharder.startswith("modular-shard-astra-"):
        raise ValueError("Expected a frozen integrator calibration")
    root = Path("/records/astra-integration")
    preceding = root / "astra_visual_9a2eca9_dev/summary.json"
    study = root / "grasp-transfer/v1"
    folder = root / name
    folder.mkdir(exist_ok=True, parents=True)
    call = folder / "official.call"
    verdict = folder / "promotion.json"
    while True:
        records.reload()
        if call.exists():
            return {"existing": call.read_text().strip()}
        if preceding.is_file():
            previous = json.loads(preceding.read_text())["decision"]
            if previous == "pass":
                return {"skipped": "preceding candidate passed"}
            if previous == "no-go":
                results = [json.loads(p.read_text()) for p in study.glob("[0-9]*.json")]
                gains = sum(
                    z["candidate"]["success"] and not z["prior"]["success"] for z in results
                )
                losses = sum(
                    z["prior"]["success"] and not z["candidate"]["success"] for z in results
                )
                remaining = 30 - len(results)
                if remaining < 0:
                    raise RuntimeError("More than 30 paired episodes")
                evidence = dict(
                    complete=len(results), gains=gains, losses=losses, remaining=remaining
                )
                if gains - losses > remaining:
                    verdict.write_text(json.dumps(dict(**evidence, decision="promote")))
                    records.commit()
                    break
                if losses - gains >= remaining:
                    verdict.write_text(json.dumps(dict(**evidence, decision="no gain")))
                    records.commit()
                    return evidence
        time.sleep(45)
    fn = modal.Function.from_name("modular-astra-integration-round", "supervise")
    launched = fn.spawn(
        name,
        sharder,
        "submission.policy_modular_calibrated:make_policy",
        "/runs/heads/" + CHECKPOINT + ".npz",
        plan,
        {},
        [],
        smoke=True,
    )
    call.write_text(launched.object_id)
    records.commit()
    return {"official": launched.object_id, **evidence}
