"""Paired seed comparison of completed teacher JSONLs (standard library only)."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def load_records(path: Path) -> dict[int, dict]:
    records = {}
    run_ids = set()
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if row.get("completed") is not True or type(row.get("seed")) is not int:
                raise ValueError("expected a completed seed record")
            if type(row.get("success")) is not bool or type(row.get("slot")) is not int:
                raise ValueError("expected boolean success and integer source slot")
            if not 0 <= row["slot"] < 6:
                raise ValueError("source slot outside [0, 5]")
            if row["seed"] in records:
                raise ValueError(f"duplicate seed {row['seed']}")
            records[row["seed"]] = row
            run_ids.add(row.get("run_id"))
        except (ValueError, AttributeError) as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
    if not records:
        raise ValueError(f"{path}: no completed records")
    if len(run_ids) > 1:
        raise ValueError(f"{path}: mixed run IDs")
    return records


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """Two-sided exact conditional binomial test; concordant pairs are excluded."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(a_only, b_only) + 1))
    return min(1.0, (2 * tail) / (1 << n))


def compare(a, b):
    if a.keys() != b.keys():
        raise ValueError(
            f"Seed sets differ: A-only={sorted(a.keys() - b.keys())}; "
            f"B-only={sorted(b.keys() - a.keys())}"
        )
    seeds = sorted(a)
    if any(a[s]["slot"] != b[s]["slot"] for s in seeds):
        raise ValueError("Paired seeds have different source slots")
    a_only = [s for s in seeds if a[s]["success"] and not b[s]["success"]]
    b_only = [s for s in seeds if b[s]["success"] and not a[s]["success"]]
    print(f"Paired seeds: {len(seeds)}")
    print(
        f"Successes: A={sum(a[s]['success'] for s in seeds)}/{len(seeds)} "
        f"B={sum(b[s]['success'] for s in seeds)}/{len(seeds)}"
    )
    print(f"Discordant: A success/B failure={len(a_only)}; A failure/B success={len(b_only)}")
    print(f"  A-only seeds: {a_only}")
    print(f"  B-only seeds: {b_only}")
    print(f"Exact McNemar p (two-sided): {mcnemar_exact(len(a_only), len(b_only)):.10g}")
    print("slot  n  A_success  B_success  A_only  B_only")
    for slot in range(6):
        subset = [s for s in seeds if a[s]["slot"] == slot]
        print(
            f"{slot:4d} {len(subset):2d} {sum(a[s]['success'] for s in subset):10d} "
            f"{sum(b[s]['success'] for s in subset):10d} "
            f"{sum(s in a_only for s in subset):7d} {sum(s in b_only for s in subset):7d}"
        )
    categories = []
    for rows in (a, b):
        categories.append(
            Counter(
                rows[s].get("failure_category", "unknown") for s in seeds if not rows[s]["success"]
            )
        )
    print("Failure category               A    B")
    for category in sorted(categories[0].keys() | categories[1].keys()):
        print(f"{category:28s} {categories[0][category]:4d} {categories[1][category]:4d}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    args = parser.parse_args()
    try:
        print(f"A: {args.a}\nB: {args.b}")
        compare(load_records(args.a), load_records(args.b))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
