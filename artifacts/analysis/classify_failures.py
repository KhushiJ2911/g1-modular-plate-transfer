"""Failure buckets from replay_eval.py output (same rules as the 11-21 analysis).

Usage: classify_failures.py RUN.replay.jsonl [...]   -> one table row per file, plus D2/E detail
"""
import collections
import json
import sys

ORDER = ["success", "A never grasped", "B grasped, never lifted", "C lifted, never reached rack",
         "D1 on rack, never released", "D2 on rack, not settled", "E reached rack, then dropped"]


def cat(r):
    if r["orig_success"] == "True":
        return "success"
    if r["first_held"] is None:
        return "A never grasped"
    if r["max_lift_m"] < 0.05:
        return "B grasped, never lifted"
    t = r["min_target_xy_while_lifted"]
    if t is None or t > 0.05:
        return "C lifted, never reached rack"
    if r["end_contacts"]["target_rack"] and r["held_steps"] > 1300:
        return "D1 on rack, never released"
    if r["end_contacts"]["target_rack"]:
        return "D2 on rack, not settled"
    return "E reached rack, then dropped"


print("| run | n | " + " | ".join(o.split(",")[0] for o in ORDER) + " |")
for f in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f) if l.strip()]
    c = collections.Counter(cat(r) for r in rows)
    print(f"| {f.split('/')[-1].replace('.replay.jsonl', '')} | {len(rows)} | "
          + " | ".join(str(c[o]) for o in ORDER) + " |")
    d2 = [r for r in rows if cat(r) == "D2 on rack, not settled"]
    if d2:
        print("   D2 end_site_dist/end_tilt:", [(r["end_site_dist"], r["end_tilt"]) for r in d2])
