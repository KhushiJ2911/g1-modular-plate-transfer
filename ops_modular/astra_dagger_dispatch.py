"""Dispatch detached generation or closed-loop audit after a successful single-episode smoke."""

import argparse
import concurrent.futures
import json
import os
from pathlib import Path

import modal

RUN = Path("/home/keyu/advaitic/plans/modular_controller/runs/astra_dagger")


def main(arm, digest):
    if os.environ.get("MODAL_CONFIG_PATH") != "/home/keyu/.modal_modular.toml":
        raise ValueError("Dedicated account required")
    start, n = (3110000, 300) if arm == "sol" else (3111000, 30)
    fn = modal.Function.from_name("modular-astra-dagger", "episode")
    callfile = RUN / ("smoke.call" if arm == "sol" else f"{arm}_{digest}.call")
    if not callfile.exists():
        callfile.write_text(fn.spawn(start, arm, digest).object_id)
    first = modal.FunctionCall.from_id(callfile.read_text().strip()).get()
    assert first["seed"] == start and first["frames"] > 10 and 1 <= first["steps"] <= 1700
    log = RUN / (arm + "_calls.jsonl")
    calls = {start: callfile.read_text().strip()}
    if log.exists():
        calls.update(json.loads(line) for line in log.read_text().splitlines())
    with concurrent.futures.ThreadPoolExecutor(10) as pool, log.open("a") as f:

        def submit(seed):
            return seed, fn.spawn(seed, arm, digest).object_id

        for lo in range(start + 1, start + n, 60):
            todo = [s for s in range(lo, min(start + n, lo + 60)) if s not in calls]
            for s, c in pool.map(submit, todo):
                calls[s] = c
                print(json.dumps([s, c]), file=f, flush=True)

    def get(item):
        seed, c = item
        result = modal.FunctionCall.from_id(c).get()
        assert seed == result["seed"]
        return result

    with (
        concurrent.futures.ThreadPoolExecutor(32) as pool,
        (RUN / (arm + "_episodes.jsonl")).open("w") as f,
    ):
        for r in pool.map(get, sorted(calls.items())):
            print(json.dumps(r), file=f, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("arm", choices=["sol", "heads"])
    p.add_argument("--digest", default="")
    a = p.parse_args()
    main(a.arm, a.digest)
