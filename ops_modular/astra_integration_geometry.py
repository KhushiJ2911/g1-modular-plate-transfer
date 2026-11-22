"""Isolate the extra positional feedback on the new merged visual-release controller."""

import hashlib
import json
import os
import time
from pathlib import Path

import modal

from ops_modular.modular_sim import image, run_candidate, volume

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("modular-astra-geometry-hybrid")
records = modal.Volume.from_name("modular-runs")
digest = "0f4053833b6aac4f187d95bb17f0daa3d3a81d2c4e461ee320b3f6b3f17ffcad"
if modal.is_local():
    image = image.add_local_file(
        ROOT / "ops_modular/astra_integration_hybrid.py", "/root/code/astra_integration_hybrid.py"
    )


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=7200,
    max_containers=2,
    single_use_containers=True,
    volumes={"/runs": volume},
)
def episode(seed, arm):
    if not (3000868 <= seed < 3001000 and arm in ("visual", "geometry")):
        raise ValueError("Fresh geometry hybrid seeds only")
    checkpoint = Path("/runs/heads") / (digest + ".npz")
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != digest:
        raise ValueError("Bad combined checkpoint")
    os.environ.update(
        MUJOCO_GL="osmesa",
        PYOPENGL_PLATFORM="osmesa",
        ADV_HEADS=str(checkpoint),
        LP_NUM_THREADS="2",
    )
    p = run_candidate(["/root/policy/bin/python", "-m", "astra_integration_hybrid", str(seed), arm])
    if p.returncode:
        raise RuntimeError(p.stdout[-3000:])
    return json.loads(p.stdout.splitlines()[-1])


@app.function(cpu=0.125, memory=256, volumes={"/records": records}, timeout=86400)
def schedule(plan):
    if len(plan) != 30 or len(set(plan)) != 30 or not all(3000870 <= s < 3001000 for s in plan):
        raise ValueError("Predeclared diagnostic plan required")
    root = Path("/records/astra-integration")
    previous = root / "astra_visual_9a2eca9_dev"
    folder = root / "visual_geometry_hybrid"
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "calls.json"
    if manifest.exists():
        raise ValueError("Existing journal")
    while True:
        records.reload()
        if (previous / "hybrid.json").exists():
            break
        time.sleep(50)
    calls = {}

    def dispatch(seed):
        for arm in ("visual", "geometry"):
            calls[f"{seed}_{arm}"] = episode.spawn(seed, arm).object_id
        manifest.write_text(json.dumps(calls))
        records.commit()

    dispatch(3000868)

    def poll(cid):
        try:
            return modal.FunctionCall.from_id(cid).get(timeout=0)
        except (TimeoutError, modal.exception.ConnectionError):
            return None

    while True:
        smoke = [poll(c) for c in calls.values()]
        if all(smoke):
            break
        time.sleep(45)
    if not all(r["carried"] and r["steps"] > 100 for r in smoke):
        raise ValueError("Invalid paired smoke")
    (folder / "smoke.json").write_text(json.dumps(smoke))
    records.commit()
    # Reserve most shared capacity for the official go/no-go decision.
    while True:
        records.reload()
        p = previous / "summary.json"
        result = json.loads(p.read_text())["decision"] if p.exists() else "pending"
        if result == "pass":
            return "target met"
        if result == "no-go":
            break
        time.sleep(50)
    episode.autoscaler.update(max_containers=18)
    for s in plan:
        dispatch(s)
    done = {}
    while len(done) < len(calls):
        for k, c in calls.items():
            if k not in done:
                r = poll(c)
                if r is not None:
                    done[k] = r
        (folder / "episodes.json").write_text(json.dumps(done))
        records.commit()
        if len(done) < len(calls):
            time.sleep(50)
    base = sum(done[f"{s}_visual"]["success"] for s in plan)
    cand = sum(done[f"{s}_geometry"]["success"] for s in plan)
    gains = sum(
        done[f"{s}_geometry"]["success"] and not done[f"{s}_visual"]["success"] for s in plan
    )
    summary = dict(base=base, candidate=cand, gains=gains, losses=base + gains - cand, seeds=plan)
    (folder / "paired.json").write_text(json.dumps(summary))
    records.commit()
    return summary
