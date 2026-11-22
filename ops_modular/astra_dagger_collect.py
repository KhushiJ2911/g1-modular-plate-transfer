"""Detached CPU collector orchestration. Each manifest covers at most 60 seeds."""

import concurrent.futures
import json
import time
from pathlib import Path

import modal

app = modal.App("modular-astra-dagger-collect")
data = modal.Volume.from_name("modular-data")


@app.function(cpu=0.125, memory=512, timeout=86400, volumes={"/data": data})
def collect(smoke_id):
    t = time.monotonic()
    smoke = modal.FunctionCall.from_id(smoke_id).get()
    if smoke["seed"] != 3110000 or smoke["frames"] < 10 or smoke["steps"] > 1700:
        raise ValueError(f"Invalid collection smoke {smoke}")
    fn = modal.Function.from_name("modular-astra-dagger", "episode")
    data.reload()
    folder = Path("/data/astra-dagger/calls")
    folder.mkdir(parents=True, exist_ok=True)
    calls = {3110000: smoke_id}
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for start in range(3110001, 3110300, 60):
            shard = folder / f"{start}.json"
            if not shard.is_file():
                seeds = list(range(start, min(start + 60, 3110300)))
                ids = list(pool.map(lambda s: fn.spawn(s, "sol", "").object_id, seeds))
                shard.write_text(json.dumps(list(zip(seeds, ids, strict=True))))
                data.commit()
            calls.update(json.loads(shard.read_text()))
    assert set(calls) == set(range(3110000, 3110300))

    def get(pair):
        seed, call_id = pair
        c = modal.FunctionCall.from_id(call_id)
        try:
            r = c.get()
        except Exception as exc:
            print(f"Retrying {seed} after {type(exc).__name__}", flush=True)
            r = fn.spawn(seed, "sol", "").get()
        assert r["seed"] == seed
        return r

    with concurrent.futures.ThreadPoolExecutor(32) as pool:
        rows = list(pool.map(get, sorted(calls.items())))
    rec = dict(seconds=time.monotonic() - t, episodes=rows)
    (folder.parent / "generation.json").write_text(json.dumps(rec))
    data.commit()
    return dict(count=len(rows), frames=sum(r["frames"] for r in rows))
