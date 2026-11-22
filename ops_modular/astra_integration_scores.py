"""Read existing reports for the three official dev plans, including Modal retry directories."""

import json
import re
from pathlib import Path

import modal

app = modal.App("modular-astra-integration-scores")
volume = modal.Volume.from_name("advaitic-runs")


@app.function(volumes={"/runs": volume}, cpu=0.125, memory=256, timeout=300)
def scores(name, plan):
    if re.fullmatch(r"[a-zA-Z0-9_-]+", name) is None:
        raise ValueError("Invalid run name")
    volume.reload()
    root = Path("/runs/modular")
    by_seed = {}
    for folder in root.glob(name + "-*"):
        suffix = folder.name[len(name) + 1 :]
        m = re.fullmatch(r"(\d+)(?:-r(\d+))?", suffix)
        if m:
            by_seed.setdefault(int(m[1]), []).append((int(m[2] or 0), folder))
    results = []
    for block, seed, slot in plan:
        attempts = []
        for _, folder in sorted(by_seed.get(seed, [])):
            path = folder / "eval/report.json"
            if not path.exists():
                continue
            rec = json.loads(path.read_text())
            if (
                not rec["integrity_ok"]
                or rec.get("infrastructure_error")
                or len(rec["episodes"]) != 1
            ):
                continue
            e = rec["episodes"][0]
            if (
                e["seed"] != seed
                or e["source_slot"] != slot
                or not e["completed"]
            ):
                continue
            attempts.append(
                dict(
                    block=block,
                    seed=seed,
                    slot=slot,
                    success=e["success"],
                    fell=e["fell"],
                    termination=e["termination"],
                    policy_error=e["policy_error"],
                    folder=folder.name,
                )
            )
        if attempts:
            if len({a["success"] for a in attempts}) != 1:
                raise ValueError(f"Conflicting complete attempts for {seed}")
            results.append(attempts[0])
    return results


if __name__ == "__main__":
    import os
    import sys
    from collections import Counter

    if os.environ.get("MODAL_CONFIG_PATH") != "/home/keyu/.modal_modular.toml":
        raise ValueError("Select dedicated account")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from advaitic_eval.protocol import make_plan

    name, output = sys.argv[1:]
    plan = [
        (block, e.seed, e.source_slot)
        for block in [1000000, 1010000, 1020000]
        for e in make_plan(30, block)
    ]
    rows = modal.Function.from_name("modular-astra-integration-scores", "scores").remote(name, plan)

    def counts(key):
        n = Counter(r[key] for r in rows)
        ok = Counter(r[key] for r in rows if r["success"])
        return {k: [ok[k], n[k]] for k in sorted(n)}

    report = dict(
        name=name,
        blocks=counts("block"),
        slots=counts("slot"),
        success=sum(r["success"] for r in rows),
        done=len(rows),
        rows=rows,
    )
    Path(output).write_text(json.dumps(report, indent=2))
    print({k: v for k, v in report.items() if k != "rows"})
