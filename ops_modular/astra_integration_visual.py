"""The visual-release privileged hybrid and its dependency on the preceding dev decision."""

import hashlib
import json
import os
import time
from pathlib import Path

import modal

from ops_modular.modular_sim import image, run_candidate, volume

ROOT = Path(__file__).resolve().parents[1]
NAME = "astra_visual_9a2eca9_dev"
PRIOR = "astra_pickup_20139df_dev"
DIGEST = "d81acb2fe6f1cb75af5acbd0fa39cda58c17fdc14486094ce1ed1312e65b723c"
HEAD = "/runs/heads/" + DIGEST + ".npz"
app = modal.App("modular-astra-visual-hybrid-9a2eca9")
records = modal.Volume.from_name("modular-runs")
if modal.is_local():
    image = image.add_local_file(
        ROOT / "ops_modular/astra_integration_hybrid.py", "/root/code/astra_integration_hybrid.py"
    )


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    volumes={"/runs": volume},
    timeout=7200,
    max_containers=6,
    single_use_containers=True,
)
def hybrid(seed, arm):
    if not 3000800 <= seed < 3001000 or arm not in {"base", "visual"}:
        raise ValueError("Expected reserved hybrid seeds")
    if hashlib.sha256(Path(HEAD).read_bytes()).hexdigest() != DIGEST:
        raise ValueError("Combined head digest mismatch")
    os.environ.update(
        ADV_HEADS=HEAD, MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa", LP_NUM_THREADS="2"
    )
    p = run_candidate(["/root/policy/bin/python", "-m", "astra_integration_hybrid", str(seed), arm])
    if p.returncode:
        raise RuntimeError(p.stdout[-3000:])
    return json.loads(p.stdout.splitlines()[-1])


@app.function(cpu=0.125, memory=256, volumes={"/records": records}, timeout=86400)
def schedule(plan, hybrid_seeds):
    fold = Path("/records/astra-integration") / NAME
    fold.mkdir(exist_ok=True, parents=True)
    manifest = fold / "hybrid_calls.json"
    if manifest.exists():
        raise ValueError("Existing hybrid journal")
    calls = {}
    smoke_seed = 3000802

    def spawn(seed):
        for arm in ("base", "visual"):
            calls[f"{seed}_{arm}"] = hybrid.spawn(seed, arm).object_id
        manifest.write_text(json.dumps(calls))
        records.commit()

    spawn(smoke_seed)

    def result(cid):
        while True:
            try:
                return modal.FunctionCall.from_id(cid).get(timeout=15)
            except (TimeoutError, modal.exception.ConnectionError):
                time.sleep(20)

    smoke = [result(c) for c in calls.values()]
    if not all(r["steps"] >= 100 and r["carried"] for r in smoke):
        raise RuntimeError(f"Unusable hybrid smoke {smoke}")
    (fold / "smoke.json").write_text(json.dumps(smoke))
    records.commit()
    for seed in hybrid_seeds:
        spawn(seed)
    rows = [result(c) for c in calls.values()]
    (fold / "hybrid.json").write_text(json.dumps(rows, indent=2))
    records.commit()
    old = fold.parent / PRIOR / "summary.json"
    while True:
        records.reload()
        if old.exists() and json.loads(old.read_text())["decision"] != "pending":
            break
        time.sleep(45)
    fn = modal.Function.from_name("modular-astra-integration-round", "supervise")
    c = fn.spawn(
        NAME,
        "modular-shard-astra-9a2eca9a-d81acb2f",
        "submission.policy_modular_integrated:make_visual_release",
        HEAD,
        plan,
        {},
        [],
        smoke=True,
    )
    (fold / "official.call").write_text(c.object_id)
    records.commit()
    return c.object_id
