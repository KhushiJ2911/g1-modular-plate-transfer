"""Stop only this integrator's unfinished dev calls once a balanced block cannot pass."""

import argparse
import concurrent.futures
import json
import os
import time
from pathlib import Path

import modal


def monitor(name, manifest, folder):
    if (
        not name.startswith("astra_")
        or manifest.name != name + ".jsonl.calls"
        or os.environ.get("MODAL_CONFIG_PATH") != "/home/keyu/.modal_modular.toml"
    ):
        raise ValueError("Expected this integrator's own dedicated-account journal")
    fn = modal.Function.from_name("modular-astra-integration-scores", "scores")
    while True:
        calls = [json.loads(s) for s in manifest.read_text().splitlines()]
        if len(calls) != 90:
            time.sleep(30)
            continue
        try:
            rows = fn.remote(name, [c[:3] for c in calls])
        except modal.exception.ConnectionError:
            time.sleep(30)
            continue
        failed = {
            b: sum(not r["success"] for r in rows if r["block"] == b)
            for b in (1000000, 1010000, 1020000)
        }
        (folder / (name + ".scores.json")).write_text(json.dumps(rows, indent=2))
        if max(failed.values()) > 3 or len(rows) == 90:
            done = {r["seed"] for r in rows}
            pending = [c for _, s, _, c in calls if s not in done]

            def cancel(call_id):
                c = modal.FunctionCall.from_id(call_id)
                try:
                    c.get(timeout=0)
                except TimeoutError:
                    c.cancel(terminate_containers=True)
                    return call_id
                except Exception:
                    # A canceled or failed invocation has no active worker to stop.
                    pass
                return None

            with concurrent.futures.ThreadPoolExecutor(10) as pool:
                canceled = list(filter(None, pool.map(cancel, pending)))
            (folder / (name + ".stop.json")).write_text(
                json.dumps(dict(failed=failed, completed=len(rows), canceled=canceled), indent=2)
            )
            return
        time.sleep(50)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("runs", type=Path)
    a = p.parse_args()
    monitor(a.name, a.runs / (a.name + ".jsonl.calls"), a.runs)
