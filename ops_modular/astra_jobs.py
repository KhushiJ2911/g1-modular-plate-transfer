"""Submit analysis jobs to the dedicated account; collect existing action traces only."""

import argparse
import concurrent.futures
import csv
import json
import os
from pathlib import Path

import modal

RUNS = Path("/home/keyu/advaitic/plans/modular_controller/runs")


def main():
    if os.environ.get("MODAL_CONFIG_PATH") != "/home/keyu/.modal_modular.toml":
        raise ValueError("Select the dedicated modular Modal configuration")
    cmd = argparse.ArgumentParser(description=__doc__)
    cmd.add_argument("operation", choices=["spawn", "collect"])
    cmd.add_argument("name")
    cmd.add_argument("kind", choices=["truth", "estimate", "refined", "replay"])
    cmd.add_argument("start", type=int)
    cmd.add_argument("count", type=int)
    cmd.add_argument("--model", default="")
    cmd.add_argument("--archived", action="store_true")
    a = cmd.parse_args()
    if not 1 <= a.count <= 60:
        raise ValueError("Maximum 60 episodes per request")
    local = RUNS / "astra_r3" / a.name
    local.mkdir(exist_ok=True, parents=True)
    ids = local / "calls.json"
    if a.operation == "spawn":
        jobs = [(a.kind, seed, -1, 0, a.model) for seed in range(a.start, a.start + a.count)]
        if a.kind == "replay":
            if a.start not in (1000000, 1010000, 1020000) or a.count != 30:
                raise ValueError("Use the original 30-episode block")
            if a.archived:
                folder = RUNS / "astra_r3/actions" / str(a.start)
                with (folder / "episodes.csv").open() as f:
                    recs = [
                        dict(
                            seed=int(r["seed"]),
                            episode=int(r["episode"]),
                            success=r["success"] == "True",
                        )
                        for r in csv.DictReader(f)
                    ]
            else:
                volume = modal.Volume.from_name("advaitic-runs")
                report = json.loads(
                    b"".join(
                        volume.read_file(f"modular/lead_e588ae9_dev_{a.start}/eval/report.json")
                    )
                )
                (local / "official.json").write_text(json.dumps(report, indent=2))
                recs = report["episodes"]
            submitted = set()
            for previous in (RUNS / "astra_r3").glob("replay*/calls.json"):
                submitted.update(seed for seed, _ in json.loads(previous.read_text()))
            jobs = [
                ("replay", r["seed"], r["episode"], a.start, "")
                for r in recs
                if not r["success"] and r["seed"] not in submitted
            ]
        with ids.open("x") as f:
            app = "modular-astra-refinement" if a.kind == "refined" else "modular-astra-perception"
            fn = modal.Function.from_name(app, "episode")
            with concurrent.futures.ThreadPoolExecutor(8) as ex:
                entries = list(ex.map(lambda j: (j[1], fn.spawn(*j).object_id), jobs))
            json.dump(entries, f)
        print(len(jobs), "submitted", flush=True)
        return
    volume = modal.Volume.from_name("modular-data")

    def get(pair):
        seed, call = pair
        outcome = modal.FunctionCall.from_id(call).get()
        (local / f"{seed}.json").write_text(json.dumps(outcome, indent=2))
        name = f"{a.kind}-{seed}{'-' + a.model if a.model else ''}.npz"
        (local / name).write_bytes(b"".join(volume.read_file("astra-r3/" + name)))
        print(seed, outcome["outcome"], round(outcome["seconds"], 1), flush=True)

    with concurrent.futures.ThreadPoolExecutor(8) as ex:
        list(ex.map(get, json.loads(ids.read_text())))


if __name__ == "__main__":
    main()
