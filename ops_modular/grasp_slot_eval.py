"""Full-horizon, observation-only paired slot diagnostics against the frozen bd951bc policy."""

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import modal

from ops_modular.modular_sim import CHECKPOINT, CODE, ROOT, WEIGHTS_SHA, image, volume

HEAD_SHA = "64dfe105e7f716c0ab5b7a9c487b7a8a1cac06565070b390e75992dfbee4deff"
SOURCE_DIGEST = {
    "base": "00c51e2b264ef6762ca4876bd92de9329ca489921d2d2cb53674dd8504ca2432",
    "clearance": "bc8a7048bdb3f6c75fed44512b25965c24813c79fd768552bc0bba8c585079b7",
}
app_name = os.environ.get("SLOT_EVAL_APP", "modular-grasp-slot-baseline")
app = modal.App(app_name)
frozen = "/root/slot-policy"
if modal.is_local():
    image = image.add_local_dir(ROOT / "results/slot0/frozen", frozen)
    if app_name == "modular-grasp-slot-candidate":
        extension = ROOT / "submission/policy_slot_clearance.py"
        image = image.add_local_file(extension, frozen + "/submission/policy_slot_clearance.py")


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    volumes={"/runs": volume},
    timeout=7200,
    max_containers=4,
    scaledown_window=5,
)
def episode(seed, arm):
    if not 3_000_900 <= seed < 3_001_000 or arm not in {"base", "clearance"}:
        raise ValueError("Expected a fresh slot diagnostic seed")
    head = Path("/runs/heads") / (HEAD_SHA + ".npz")
    if (
        hashlib.sha256(head.read_bytes()).hexdigest() != HEAD_SHA
        or hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != WEIGHTS_SHA
    ):
        raise RuntimeError("Checkpoint mismatch")
    root = Path("/runs/slot-development")
    root.mkdir(exist_ok=True)
    out = Path(tempfile.mkdtemp(dir=root, prefix=f"{arm}-{seed}-")) / "eval"
    policy = (
        "submission.policy_modular_calibrated"
        if arm == "base"
        else "submission.policy_slot_clearance"
    ) + ":make_policy"
    env = {
        **os.environ,
        "MUJOCO_GL": "osmesa",
        "PYOPENGL_PLATFORM": "osmesa",
        "ADV_HEADS": str(head),
        "ADV_ESTIMATOR": str(CHECKPOINT),
        "MKL_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "LP_NUM_THREADS": "2",
    }
    try:
        p = subprocess.run(
            [
                f"{CODE}/.venv/bin/advaitic-eval",
                "--policy-root",
                frozen,
                "--policy",
                policy,
                "--policy-python",
                "/root/policy/bin/python",
                "--method",
                f"slot0-{arm}",
                "--checkpoint",
                f"bd951bc+{WEIGHTS_SHA}+{HEAD_SHA}",
                "--seed-start",
                str(seed),
                "--smoke",
                "--episodes",
                "1",
                "--out",
                str(out),
            ],
            capture_output=True,
            text=True,
            env=env,
            cwd=CODE,
        )
        if p.returncode:
            raise RuntimeError((p.stderr + p.stdout)[-2000:])
        report = json.loads((out / "report.json").read_text())
        if report["submission_sha256"] != SOURCE_DIGEST[arm]:
            raise ValueError("Unexpected policy source digest")
        e = report["episodes"][0]
        if (
            not report["integrity_ok"]
            or report["infrastructure_error"]
            or e["policy_error"]
            or not e["completed"]
            or e["seed"] != seed
        ):
            raise RuntimeError(str(e))
        return dict(**e, arm=arm, archive=str(out), integrity=True)
    finally:
        volume.commit()
