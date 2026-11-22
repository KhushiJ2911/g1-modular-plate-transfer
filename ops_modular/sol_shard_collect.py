"""Recover detached lead-sharder calls after the local stream drops."""

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import modal


def read_rows(path):
    if not path.exists():
        return {}
    return {row["seed"]: row for row in map(json.loads, path.read_text().splitlines())}


def fetch(call_id):
    try:
        return modal.FunctionCall.from_id(call_id).get(timeout=0)
    except (modal.exception.FunctionTimeoutError, modal.exception.OutputExpiredError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    except (TimeoutError, OSError, modal.exception.TimeoutError, modal.exception.ConnectionError):
        return None
    except Exception as exc:
        if type(exc).__name__ in {"StreamTerminatedError", "GRPCError"}:
            return None
        return {"error": f"{type(exc).__name__}: {str(exc)[-500:]}"}


def collect(source, dest, snapshot, names, wait):
    dest.mkdir(parents=True, exist_ok=True)
    outstanding = []
    counts = defaultdict(int)
    for name in names:
        old = read_rows(source / name)
        current = read_rows(dest / name)
        manifest = (source / f"{name}.calls").read_text().splitlines()
        for block, seed, slot, call_id in map(json.loads, manifest):
            if seed in current:
                continue
            row = old.get(seed)
            if row and row.get("snapshot") == snapshot and row.get("completed"):
                with (dest / name).open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                counts[name] += 1
            else:
                outstanding.append((name, block, seed, slot, call_id))
    with ThreadPoolExecutor(max_workers=6) as pool:
        while outstanding:
            pending = {pool.submit(fetch, item[-1]): item for item in outstanding}
            outstanding = []
            for future in as_completed(pending):
                name, block, seed, slot, call_id = pending[future]
                row = future.result()
                if row is None:
                    outstanding.append((name, block, seed, slot, call_id))
                    continue
                if "error" not in row and (
                    row["seed"] != seed or row["source_slot"] != slot or row["snapshot"] != snapshot
                ):
                    raise ValueError(f"Inconsistent evaluator evidence for {name} seed {seed}")
                row.update(block=block, planned_slot=slot, seed=seed, call_id=call_id)
                with (dest / name).open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                counts[name] += 1
                print(name, seed, row.get("success"), row.get("error"), flush=True)
            if not wait or not outstanding:
                break
            print(f"{sum(counts.values())} newly collected, {len(outstanding)} pending", flush=True)
            time.sleep(45)
    print(f"{sum(counts.values())} newly collected, {len(outstanding)} pending", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("dest", type=Path)
    parser.add_argument("snapshot")
    parser.add_argument("names", nargs="+")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    collect(args.source, args.dest, args.snapshot, args.names, args.wait)
