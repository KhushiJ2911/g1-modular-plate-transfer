"""Resume deferred dev seeds after earlier inputs drain, under the shared worker cap."""

import concurrent.futures
import json
import os
import time
from pathlib import Path

import modal

D = Path("/home/keyu/advaitic/plans/modular_controller/runs")
NAME = "astra_pickup_20139df_dev"


def main():
    assert os.environ.get("MODAL_CONFIG_PATH") == "/home/keyu/.modal_modular.toml"
    journal = D / (NAME + ".jsonl.calls")
    calls = [json.loads(line) for line in (D / (NAME + ".original_calls")).read_text().splitlines()]
    keep = json.loads((D / (NAME + ".deferred.json")).read_text())["keep"]

    def terminal(row):
        c = modal.FunctionCall.from_id(row[3])
        try:
            c.get(timeout=0)
        except (TimeoutError, modal.exception.ConnectionError):
            return False
        except Exception:
            return True
        return True

    with concurrent.futures.ThreadPoolExecutor(12) as pool:
        while not all(pool.map(terminal, keep)):
            if (D / (NAME + ".stop.json")).exists():
                return
            time.sleep(50)
    scores = modal.Function.from_name("modular-astra-integration-scores", "scores")
    rows = scores.remote(NAME, [r[:3] for r in calls])
    failed = [
        sum(not r["success"] for r in rows if r["block"] == b) for b in (1000000, 1010000, 1020000)
    ]
    if max(failed) > 3 or (D / (NAME + ".stop.json")).exists():
        return
    import subprocess

    live = json.loads(subprocess.check_output(["modal", "container", "list", "--json"]))
    own = sum("astra" in c["app_name"] for c in live)
    capacity = min(23, 35 - own, 100 - len(live))
    if capacity < 1:
        raise RuntimeError("No recovery capacity available")
    fn = modal.Function.from_name("modular-shard-astra-20139df9-51257d33-recovery", "episode_cpu")
    fn.autoscaler.update(max_containers=capacity)
    done = {r["seed"] for r in rows}
    with (D / (NAME + ".capacity_retries.jsonl")).open("x") as out:
        for row in calls:
            if row[1] in done:
                continue
            c = fn.spawn(
                NAME,
                row[1],
                "submission.policy_modular_heads:make_policy",
                "/runs/heads/contact_v1_51257d33.npz",
            )
            print(json.dumps([row[1], row[3], c.object_id]), file=out, flush=True)
            row[3] = c.object_id
    tmp = journal.with_suffix(".next")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in calls))
    tmp.replace(journal)


if __name__ == "__main__":
    main()
