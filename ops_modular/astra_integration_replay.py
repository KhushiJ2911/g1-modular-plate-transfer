"""Replay a sharded development failure camera-free and verify its terminal outcome."""

import json
import re
import subprocess
import tempfile
from pathlib import Path

import modal

from ops_modular.modular_sim import image, volume

app = modal.App("modular-astra-integration-replay")
if modal.is_local():
    project = Path("/home/keyu/advaitic")
    image = image.add_local_file(
        project / "analysis/replay_eval.py", "/root/analysis/replay_eval.py"
    ).add_local_file(
        project / "wt-teacher/method/teacher/diagnostics.py",
        "/home/keyu/advaitic/wt-teacher/method/teacher/diagnostics.py",
    )


@app.function(image=image, cpu=1, timeout=300, max_containers=35, volumes={"/runs": volume})
def replay(folder):
    if re.fullmatch(r"astra_[a-zA-Z0-9_]+-10[0-9]{5}(-r[0-9]+)?", folder) is None:
        raise ValueError("Expected an integrator development run")
    volume.reload()
    src = Path("/runs/modular") / folder / "eval"
    original = json.loads((src / "report.json").read_text())
    assert (
        original["integrity_ok"]
        and not original["infrastructure_error"]
        and not original["successes"]
    )
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "metrics.jsonl"
        subprocess.run(
            ["/root/code/.venv/bin/python", "/root/analysis/replay_eval.py", str(src), str(p)],
            check=True,
        )
        r = json.loads(p.read_text())
    assert r["seed"] == original["episodes"][0]["seed"] and not r["replay_success"]
    if r["first_held"] is None:
        label = "A never grasped"
    elif r["max_lift_m"] < 0.05:
        label = "B grasp lost"
    elif r["min_target_xy_while_lifted"] is None or r["min_target_xy_while_lifted"] > 0.06:
        label = "C target not reached"
    elif r["held_steps"] > 1000:
        label = "D1 held on rack" if r["end_contacts"]["target_rack"] else "D2 hovering"
    else:
        label = (
            "D3 released on rack, unsettled"
            if r["end_contacts"]["target_rack"]
            else "E dropped off rack"
        )
    return dict(**r, bucket=label)


if __name__ == "__main__":
    import argparse
    import concurrent.futures
    import os
    import time

    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("directory", type=Path)
    a = p.parse_args()
    assert os.environ["MODAL_CONFIG_PATH"] == "/home/keyu/.modal_modular.toml"
    source = a.directory / (a.run + ".scores.json")
    out = a.directory / (a.run + ".replays.jsonl")
    fn = modal.Function.from_name("modular-astra-integration-replay", "replay")
    seen = {json.loads(line)["seed"] for line in out.read_text().splitlines()} if out.exists() else set()
    while True:
        rows = json.loads(source.read_text())
        failures = [r for r in rows if not r["success"] and r["seed"] not in seen]

        def run(r):
            return fn.remote(r["folder"])

        with out.open("a") as f, concurrent.futures.ThreadPoolExecutor(12) as pool:
            for rec in pool.map(run, failures):
                print(json.dumps(rec), file=f, flush=True)
                seen.add(rec["seed"])
        if (a.directory / (a.run + ".stop.json")).exists() and all(
            r["success"] or r["seed"] in seen for r in json.loads(source.read_text())
        ):
            break
        time.sleep(50)
