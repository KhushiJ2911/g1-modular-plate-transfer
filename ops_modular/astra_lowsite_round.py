"""Compare the frozen low-site candidate on 30 shared generation episodes before dev promotion."""

import json
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

from ops_modular.astra_height_round import HEAD, ROOT, launch_once, qualify, records

app = modal.App("modular-astra-lowsite-decision")
archives = modal.Volume.from_name("advaitic-runs")
GEN = "modular-astra-low-site-generation-b5a9d72"
DEV = "modular-shard-astra-b5a9d720-64dfe105"
POLICY = "method.modular.site_seating:make_policy"
PREVIOUS = "astra_height_6b02f33_dev"
NAME = "astra_lowsite_b5a9d72_dev"
PAIRED = "astra_lowsite_b5a9d72_paired"


@app.function(
    cpu=0.125,
    memory=512,
    max_containers=1,
    timeout=86400,
    volumes={"/records": records, "/runs": archives},
)
def evaluate(plan, preceding):
    records.reload()
    archives.reload()
    prev = ROOT / PREVIOUS
    if (prev / "official.call").read_text().strip() != preceding:
        raise ValueError("Unexpected preceding dev round")
    if (
        len(plan) != 30
        or len(set(z["seed"] for z in plan)) != 30
        or Counter(z["source_slot"] for z in plan) != Counter({i: 5 for i in range(6)})
    ):
        raise ValueError("Expected the same balanced generation comparison")
    for z in plan:
        s = z["seed"]
        if not 3001600 <= s < 3001750 or not re.fullmatch(
            rf"/runs/modular/sol_fk_full_base(?:_retry)?-{s}(?:-r[0-9]+)?/eval", z["archive"]
        ):
            raise ValueError("Unexpected baseline archive")
        report = json.loads((Path(z["archive"]) / "report.json").read_text())
        e = report["episodes"][0]
        if (
            not report["integrity_ok"]
            or not e["completed"]
            or e["policy_error"]
            or (e["seed"], e["source_slot"], e["success"]) != (s, z["source_slot"], z["success"])
        ):
            raise ValueError("Paired baseline disagrees with evaluator")
    folder = ROOT / NAME
    folder.mkdir(exist_ok=True, parents=True)
    (folder / "paired_plan.json").write_text(json.dumps(plan))
    fn = modal.Function.from_name(GEN, "episode_cpu")
    calls = {}

    def start(row):
        s = row["seed"]
        calls[s] = launch_once(folder / f"paired_{s}.call", fn, PAIRED, s, POLICY, HEAD)

    def finish(row):
        s = row["seed"]
        result = calls[s].get()
        if (result["seed"], result["source_slot"], result["completed"], result["policy_error"]) != (
            s,
            row["source_slot"],
            True,
            None,
        ):
            raise ValueError(f"Invalid generation report for {s}")
        return {**result, "baseline": row["success"]}

    def write(rows):
        (folder / "paired.json").write_text(json.dumps(rows))
        records.commit()

    start(plan[0])
    rows = [finish(plan[0])]
    write(rows)
    for row in plan[1:]:
        start(row)
    with ThreadPoolExecutor(15) as pool:
        from concurrent.futures import as_completed

        futures = [pool.submit(finish, row) for row in plan[1:]]
        for f in as_completed(futures):
            rows.append(f.result())
            write(rows)
    if sum(r["success"] for r in rows) < sum(r["baseline"] for r in rows) or any(
        r["fell"] for r in rows
    ):
        return {
            "decision": "paired regression",
            "successes": sum(r["success"] for r in rows),
            "total": len(rows),
        }
    previous = modal.FunctionCall.from_id(preceding).get()
    if previous["decision"] not in ("pass", "no-go"):
        raise ValueError("Expected terminal preceding round")
    records.reload()
    dev_plan = json.loads((prev / "candidate.json").read_text())["plan"]
    runner = modal.Function.from_name("modular-astra-integration-round", "supervise")
    call = launch_once(
        folder / "official.call", runner, NAME, DEV, POLICY, HEAD, dev_plan, {}, [], smoke=True
    )
    while True:
        records.reload()
        path = folder / "summary.json"
        if path.exists() and json.loads(path.read_text())["done"]:
            modal.Function.from_name(DEV, "episode_cpu").autoscaler.update(max_containers=60)
            break
        try:
            call.get(timeout=0)
            break
        except TimeoutError:
            time.sleep(45)
    final = call.get()
    if final["decision"] == "pass":
        return qualify(NAME, "modular-astra-balanced-lowsite-b5a9d72")
    return final
