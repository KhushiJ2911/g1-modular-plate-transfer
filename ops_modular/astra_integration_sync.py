"""Copy the small durable round summaries to the coordination directory."""

import json
import os
from collections import Counter
from pathlib import Path

import modal

RUNS = Path("/home/keyu/advaitic/plans/modular_controller/runs")
NAMES = ("astra_pickup_20139df_dev", "astra_visual_9a2eca9_dev")


def fetch(volume, remote, local):
    try:
        data = b"".join(volume.read_file(remote))
    except FileNotFoundError:
        return None
    local.write_bytes(data)
    return json.loads(data)


def main():
    assert os.environ.get("MODAL_CONFIG_PATH") == "/home/keyu/.modal_modular.toml"
    v = modal.Volume.from_name("modular-runs")
    for n in NAMES:
        try:
            files = {Path(e.path).name for e in v.listdir("astra-integration/" + n)}
        except FileNotFoundError:
            continue
        for name in ("summary", "replays", "smoke", "hybrid"):
            f = name + ".json"
            if f not in files:
                continue
            r = fetch(v, f"astra-integration/{n}/{f}", RUNS / (n + "." + f))
            if name == "summary":
                print(n, {k: w for k, w in r.items() if k != "rows"})
            elif name == "replays":
                print(
                    "failure buckets",
                    dict(Counter(z.get("bucket", "replay error") for z in r.values())),
                )
            else:
                print(name, [(z["arm"], z["seed"], z["success"]) for z in r])
    dv = modal.Volume.from_name("modular-data")
    df = {Path(e.path).name for e in dv.listdir("astra-dagger")}
    for name in ("generation", "heads", "closed_loop"):
        f = name + ".json"
        if f in df:
            local = RUNS / "astra_dagger" / f
            if not local.exists():
                r = fetch(dv, "astra-dagger/" + f, local)
                print(
                    "dagger",
                    f,
                    r.get("sha256", r.get("checkpoint", "")),
                    len(r.get("episodes", [])),
                )
    for p in [
        RUNS / (NAMES[0] + ".remote.call"),
        RUNS / (NAMES[1] + ".pipeline.call"),
        RUNS / "astra_dagger/supervisor.call",
    ]:
        c = modal.FunctionCall.from_id(p.read_text().strip())
        try:
            out = c.get(timeout=0)
            print(p.name, "complete", str(out)[:160])
        except TimeoutError:
            print(p.name, "active")
        except Exception as e:
            print(p.name, type(e).__name__, str(e)[-350:])


if __name__ == "__main__":
    main()
