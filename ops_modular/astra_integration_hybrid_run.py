"""Run a balanced paired placement comparison, reusing detached calls after a restart."""

import concurrent.futures
import json
import os
import time
from pathlib import Path

import modal

from advaitic_eval.protocol import make_plan

D = Path("/home/keyu/advaitic/plans/modular_controller/runs")


def retrieve(c):
    while True:
        try:
            return c.get(timeout=15)
        except (TimeoutError, modal.exception.ConnectionError):
            time.sleep(15)


def main():
    assert os.environ.get("MODAL_CONFIG_PATH") == "/home/keyu/.modal_modular.toml"
    digest = (D / "astra_site_combined.sha256").read_text()
    fn = modal.Function.from_name("modular-astra-integration-ab", "run")
    arms = ("base", "both")
    for arm in arms:
        row = retrieve(modal.FunctionCall.from_id((D / f"astra_ab_smoke_{arm}.call").read_text()))
        assert row["seed"] == 3000800 and row["carried"] and not row["fell"]
        (D / f"astra_ab_smoke_{arm}.json").write_text(json.dumps(row))
    # Do not add another wave while the prior dev wave holds the account.
    while not (D / "astra_pickup_20139df_dev.stop.json").exists():
        time.sleep(45)
    import subprocess

    usage = json.loads(subprocess.check_output(["modal", "container", "list", "--json"]))
    other = sum(
        r["app_name"].startswith("modular-astra")
        and r["app_name"] != "modular-astra-integration-ab"
        for r in usage
    )
    cap = max(
        1,
        min(
            26,
            35 - other,
            100 - len(usage) + sum(r["app_name"] == "modular-astra-integration-ab" for r in usage),
        ),
    )
    fn.autoscaler.update(max_containers=cap)
    plan = make_plan(30, 3000810)
    manifest = D / "astra_placement_hybrid.calls.jsonl"
    calls = {}
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            seed, arm, c = json.loads(line)
            calls[seed, arm] = c
    with manifest.open("a") as f:
        for ep in plan:
            for arm in arms:
                if (ep.seed, arm) not in calls:
                    c = fn.spawn(ep.seed, arm, digest).object_id
                    print(json.dumps([ep.seed, arm, c]), file=f, flush=True)
                    calls[ep.seed, arm] = c

    def read(item):
        (seed, arm), c = item
        row = retrieve(modal.FunctionCall.from_id(c))
        assert row["seed"] == seed and row["arm"] == arm
        return row

    with (
        concurrent.futures.ThreadPoolExecutor(16) as pool,
        (D / "astra_placement_hybrid.jsonl").open("w") as f,
    ):
        for row in pool.map(read, sorted(calls.items())):
            print(json.dumps(row), file=f, flush=True)


if __name__ == "__main__":
    main()
