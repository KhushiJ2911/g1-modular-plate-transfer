"""Publish a terminal development read from rescanned evaluator reports."""

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

from ops_modular.astra_integration_round import count

RUNS = Path("/home/keyu/advaitic/plans/modular_controller/runs")


def report(name, supervisor):
    if os.environ.get("MODAL_CONFIG_PATH") != "/home/keyu/.modal_modular.toml":
        raise ValueError("Select the dedicated account")
    records = modal.Volume.from_name("modular-runs")
    root = f"astra-integration/{name}"

    def read(filename):
        return json.loads(b"".join(records.read_file(f"{root}/{filename}")))

    final = modal.FunctionCall.from_id(supervisor).get(timeout=0)
    if final.get("decision") not in ("no-go", "pass"):
        raise ValueError("Expected the terminal supervisor of this development round")
    candidate = read("candidate.json")
    plan = candidate["plan"]
    scorer = modal.Function.from_name("modular-astra-integration-scores", "scores")
    rows = scorer.remote(name, [(p["block"], p["seed"], p["slot"]) for p in plan])
    score = count(rows, plan)
    if score["decision"] != final["decision"]:
        raise ValueError("Disagreement between the terminal supervisor and committed reports")
    replays = read("replays.json")
    fn = modal.Function.from_name("modular-astra-integration-replay", "replay")
    missing = [
        row
        for row in rows
        if not row["success"]
        and not row["policy_error"]
        and "bucket" not in replays.get(str(row["seed"]), {})
    ]

    def replay(row):
        return str(row["seed"]), fn.remote(row["folder"])

    with ThreadPoolExecutor(max_workers=2) as workers:
        replays.update(workers.map(replay, missing))
    failures = [row for row in rows if not row["success"]]
    for row in failures:
        if row["policy_error"]:
            replays[str(row["seed"])] = {"bucket": "policy_error", "detail": row["policy_error"]}
    if {str(r["seed"]) for r in failures} != set(replays):
        raise ValueError("Unclassified or extraneous failure")
    outcomes = {}
    for row in failures:
        audit = replays[str(row["seed"])]
        key = audit["bucket"]
        if key.startswith("D1") and "open_at_target" in audit:
            key += " after opening" if audit["open_at_target"] is not None else " without opening"
        outcomes[key] = outcomes.get(key, 0) + 1

    def rates(counts):
        return ", ".join(f"{a}/{n}" for a, n in counts.values())

    body = (
        f"{candidate['policy']} • {candidate['app']} • {candidate['checkpoint']}. "
        f"{score['successes']}/{score['done']} complete, blocks 1000000 / 1010000 / 1020000 "
        f"({rates(score['blocks'])}), slots 0–5 ({rates(score['slots'])}). "
        f"Decision {score['decision']}. Failure counts "
        + ", ".join(f"{k} ({v})" for k, v in sorted(outcomes.items()))
        + "."
    )
    report = dict(
        name=name, candidate=candidate, **score, outcomes=outcomes, rows=rows, replays=replays
    )
    output = RUNS / (name + ".final.json")
    output.write_text(json.dumps(report, indent=2))
    # The digest distinguishes a late committed report from an identical rescan.
    marker = hashlib.sha256(body.encode()).hexdigest()[:12]
    line = f"2026-11-22 · Terminal sharded dev read {name} [{marker}]"
    for p in (RUNS / "astra_findings.md", Path("EXPERIMENTS.md")):
        if line not in p.read_text():
            with p.open("a") as f:
                f.write(
                    f"\n{line}\n\n{body} Complete source and replay diagnostics in {output.name}.\n"
                )
    print(body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument(
        "supervisor", help="ID of the completed supervisor, or its local .call file"
    )
    a = parser.parse_args()
    s = a.supervisor if a.supervisor.startswith("fc-") else Path(a.supervisor).read_text().strip()
    report(a.name, s)
