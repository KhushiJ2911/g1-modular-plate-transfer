"""Copy completed official failures for replay, leaving the original evaluation untouched."""

import concurrent.futures
import csv
import io
import json
import os

import modal

from ops_modular.astra_jobs import RUNS


def main():
    if os.environ.get("MODAL_CONFIG_PATH") != "/home/keyu/.modal_modular.toml":
        raise ValueError("Use the dedicated modular account")
    volume = modal.Volume.from_name("advaitic-runs")
    for start in (1000000, 1010000, 1020000):
        prefix = f"modular/lead_e588ae9_dev_{start}/eval"
        dest = RUNS / "astra_r3/actions" / str(start)
        dest.mkdir(parents=True, exist_ok=True)
        data = b"".join(volume.read_file(f"{prefix}/episodes.csv"))
        rows = list(csv.DictReader(io.StringIO(data.decode())))
        (dest / "episodes.csv").write_bytes(data)
        failures = [r for r in rows if r["success"] == "False"]

        def fetch(rec, dest=dest, prefix=prefix):
            name = f"episode_{int(rec['episode']):03}.actions.npz"
            path = dest / name
            if not path.exists():
                path.write_bytes(b"".join(volume.read_file(prefix + "/" + name)))

        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            list(pool.map(fetch, failures))
        print(
            json.dumps(dict(start=start, completed=len(rows), failures=len(failures))), flush=True
        )


if __name__ == "__main__":
    main()
