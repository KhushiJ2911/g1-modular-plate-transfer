"""Summarize matched, complete episodes from the Sol variants of the lead sharder."""

import argparse
import json
from collections import Counter
from pathlib import Path

from advaitic_eval.protocol import make_plan

VARIANTS = ("previous", "pickup", "rigid", "late_site", "seating")


def summarize(path, seed_start, expected, prefix="sol_rb", variants=VARIANTS):
    if expected not in (6, 12, 18, 24, 30):
        raise ValueError("Expected an equal number of episodes per slot")
    counts = Counter()
    plan = {}
    for episode in make_plan(30, seed_start):
        counts[episode.source_slot] += 1
        if counts[episode.source_slot] <= expected // 6:
            plan[episode.seed] = episode.source_slot
    seen = {}
    reports = {}
    for variant in variants:
        file = path / f"{prefix}_{variant}.jsonl"
        rows = [json.loads(line) for line in file.read_text().splitlines()] if file.exists() else []
        if len({row["seed"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate seed in {file}")
        wrong_plan = [
            row
            for row in rows
            if row.get("block") != seed_start
            or row["seed"] not in plan
            or row.get("planned_slot") != plan[row["seed"]]
        ]
        if wrong_plan:
            raise ValueError(f"Unexpected plan in {file}")
        good = {
            row["seed"]: row
            for row in rows
            if row.get("completed")
            and not (row.get("policy_error") or row.get("error"))
            and row.get("source_slot") == row.get("planned_slot")
        }
        seen[variant] = good
        lifted = [
            row for row in good.values() if any(phase["to"] == "CARRY" for phase in row["phases"])
        ]
        opened = [
            row for row in good.values() if any(phase["to"] == "RELEASE" for phase in row["phases"])
        ]
        terminal = Counter(
            (row["phases"][-1]["from"] if row["phases"] else "APPROACH")
            for row in good.values()
            if not row["success"]
        )
        reports[variant] = {
            "terminal_failure_phases": dict(terminal),
            "entered_carry": len(lifted),
            "entered_release": len(opened),
            "success_after_carry": sum(row["success"] for row in lifted),
            "complete": len(good),
            "successes": sum(row["success"] for row in good.values()),
            "successes_by_slot": dict(
                Counter(row["source_slot"] for row in good.values() if row["success"])
            ),
            "infra_or_policy_errors": len(rows) - len(good),
            "unreported": sorted(plan.keys() - {r["seed"] for r in rows}),
        }
    builds = {row["snapshot"] for rows in seen.values() for row in rows.values()}
    if len(builds) > 1:
        raise ValueError("Mixed snapshots")
    for previous, current in zip(variants, variants[1:], strict=False):
        left, right = seen[previous], seen[current]
        paired = left.keys() & right.keys()
        old = {s for s in paired if left[s]["success"]}
        new = {s for s in paired if right[s]["success"]}
        failures = Counter()
        for seed in paired - new:
            phases = right[seed]["phases"]
            failure = next((x for x in phases if x["reason"] != "exit_check"), None)
            failures[
                failure["from"] if failure else (phases[-1]["to"] if phases else "APPROACH")
            ] += 1
        reports[current]["paired_with_previous"] = {
            "episodes": len(paired),
            "previous_successes": len(old),
            "current_successes": len(new),
            "gains": sorted(new - old),
            "losses": sorted(old - new),
            "first_failure_phases": dict(failures),
        }
    return {
        "start": seed_start,
        "expected": expected,
        "snapshot": sorted(builds),
        "results": reports,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--start", type=int, default=3_000_460)
    parser.add_argument("--expected", type=int, default=30)
    parser.add_argument("--prefix", default="sol_rb")
    parser.add_argument("--variants", nargs="+", default=VARIANTS)
    args = parser.parse_args()
    print(
        json.dumps(
            summarize(args.path, args.start, args.expected, args.prefix, args.variants), indent=2
        )
    )
