"""Launch a sharded DEV evaluation: official balanced plans, one container per episode.

Usage (Modal's python, dedicated account):
  MODAL_CONFIG_PATH=~/.modal_modular.toml python ops_modular/lead_shard_run.py NAME {cpu|gpu} OUT.jsonl module:factory HEADS_PATH_OR_"" 1000000 1010000 1020000

Each seed_start contributes one balanced 30-episode plan (5 per source slot), exactly as advaitic-eval would
pick it. Results are dev-only and non-qualifying; they are aggregated per block and per slot.
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from advaitic_eval.protocol import make_plan  # noqa: E402


def main():
    name, kind, out, policy, heads, *starts = sys.argv[1:]
    fn = modal.Function.from_name(os.environ.get("SHARD_APP", "modular-shard"), f"episode_{kind}")
    jobs = [(int(s), e.seed, e.source_slot) for s in starts for e in make_plan(30, int(s))]
    assert all(make_plan(30, seed)[0].seed == seed for _, seed, _ in jobs)
    # Optional SLICE="i/n": run only every n-th planned episode (offset i), so one round can be split
    # across several launchers (e.g. GPU + CPU, or two Modal accounts) without overlap.
    if os.environ.get("SLICE"):
        i, n = map(int, os.environ["SLICE"].split("/"))
        jobs = jobs[i::n]
    print(f"{name}: {len(jobs)} episodes on {kind}", flush=True)
    calls = [(block, seed, slot, fn.spawn(name, seed, policy, heads)) for block, seed, slot in jobs]
    rows = []
    with open(out, "w") as fh:
        for block, seed, slot, call in calls:
            try:
                r = call.get()
            except (
                Exception
            ) as exc:  # infrastructure/policy crash counts as failure, but is flagged
                r = {
                    "seed": seed,
                    "success": False,
                    "error": f"{type(exc).__name__}: {str(exc)[-500:]}",
                }
            r.update(block=block, planned_slot=slot)
            rows.append(r)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
    for block in sorted({r["block"] for r in rows}):
        b = [r for r in rows if r["block"] == block]
        per = Counter(r["planned_slot"] for r in b if r["success"])
        errs = sum("error" in r for r in b)
        print(
            f"block {block}: {sum(r['success'] for r in b)}/{len(b)}  per-slot {dict(sorted(per.items()))}  infra_errors={errs}"
        )
    print(f"TOTAL {sum(r['success'] for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
