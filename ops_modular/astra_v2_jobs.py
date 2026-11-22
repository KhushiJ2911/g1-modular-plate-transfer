"""Submit/collect detached generation shards and the GPU training call."""

import argparse
import concurrent.futures
import json
import os
from pathlib import Path

import modal

RUNS = Path("/home/keyu/advaitic/plans/modular_controller/runs/astra_v2")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["spawn", "collect"])
    p.add_argument("name")
    p.add_argument("kind", choices=["gpu", "cpu", "train"])
    p.add_argument("start", type=int)
    p.add_argument("count", type=int)
    args = p.parse_args()
    if os.environ.get("MODAL_CONFIG_PATH") != "/home/keyu/.modal_modular.toml":
        raise ValueError("Select the dedicated account")
    if not 1 <= args.count <= 60:
        raise ValueError("Use shards of 1–60 episodes")
    RUNS.mkdir(parents=True, exist_ok=True)
    manifest = RUNS / (args.name + ".calls.jsonl")
    if args.mode == "spawn":
        if args.kind == "train":
            fn = modal.Function.from_name("modular-astra-v2-train", "train")
        else:
            fn = modal.Function.from_name("modular-astra-v2-data", "episode_" + args.kind)

        def launch(seed):
            c = (
                fn.spawn(args.name, args.start, args.count)
                if args.kind == "train"
                else fn.spawn(seed)
            )
            return {"seed": seed, "call": c.object_id}

        with manifest.open("x") as f, concurrent.futures.ThreadPoolExecutor(6) as pool:
            seeds = (
                [args.start] if args.kind == "train" else range(args.start, args.start + args.count)
            )
            for row in pool.map(launch, seeds):
                print(json.dumps(row), file=f, flush=True)
        return

    def get(row):
        return modal.FunctionCall.from_id(row["call"]).get()

    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    with (
        (RUNS / (args.name + ".jsonl")).open("w") as f,
        concurrent.futures.ThreadPoolExecutor(12) as pool,
    ):
        for row in pool.map(get, rows):
            print(json.dumps(row), file=f, flush=True)
            print(row.get("seed"), row.get("frames"), row.get("seconds"), flush=True)


if __name__ == "__main__":
    main()
