"""Frozen CPU-only replay and observation-only evaluation for the hover investigation."""

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import modal

from ops_modular.grasp_slot_eval import HEAD_SHA
from ops_modular.modular_sim import CHECKPOINT, CODE, ROOT, WEIGHTS_SHA, image, volume

role = os.environ.get("HOVER_ROLE", "trace")
app = modal.App("modular-grasp-hover-" + role)
policy_root = "/root/hover-policy"
if modal.is_local():
    image = image.add_local_dir(ROOT / "results/slot0/frozen", policy_root)
    image = image.add_local_file(
        ROOT / "ops_modular/grasp_hover_trace.py", "/root/analysis/hover.py"
    )
    if role in ("audit", "screen", "effect"):
        image = image.add_local_file(
            ROOT / "ops_modular/grasp_hover_audit.py", "/root/analysis/hover_audit.py"
        )
    if role in ("candidate", "confirm-candidate", "screen"):
        image = image.add_local_file(
            ROOT / "submission/policy_hover.py", policy_root + "/submission/policy_hover.py"
        )

options = dict(
    image=image,
    cpu=2,
    memory=4096,
    timeout=7200,
    volumes={"/runs": volume},
    scaledown_window=5,
    max_containers={
        "trace": 4,
        "base": 7,
        "candidate": 7,
        "audit": 1,
        "effect": 1,
        "screen": 2,
        "confirm-base": 1,
        "confirm-candidate": 1,
    }[role],
)


def run(args):
    head = Path("/runs/heads", HEAD_SHA + ".npz")
    for path, expected in ((head, HEAD_SHA), (CHECKPOINT, WEIGHTS_SHA)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Unexpected checkpoint {path.name}")
    env = {
        **os.environ,
        "MUJOCO_GL": "osmesa",
        "PYOPENGL_PLATFORM": "osmesa",
        "PYTHONPATH": policy_root,
        "ADV_HEADS": str(head),
        "ADV_ESTIMATOR": str(CHECKPOINT),
        "MKL_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "LP_NUM_THREADS": "2",
    }
    p = subprocess.run(args, cwd=policy_root, env=env, text=True, capture_output=True)
    if p.returncode:
        raise RuntimeError((p.stderr + p.stdout)[-2300:])
    return p.stdout


def json_result(args):
    # BiGym writes initialization messages before the analysis program's JSON.
    return json.loads(run(args).splitlines()[-1])


@app.function(**options)
def trace(seed, mode):
    if seed not in (1030010, 1070027) or mode not in ("physics", "public"):
        raise ValueError("Expected one of the requested failure archives")
    path = f"/runs/modular/sig1_bd951bc-{seed}/eval"
    return json_result(["/root/policy/bin/python", "/root/analysis/hover.py", path, mode])


@app.function(**options)
def audit(jobs, reach=False):
    args = ["/root/policy/bin/python", "/root/analysis/hover_audit.py", json.dumps(jobs)]
    if reach:
        args.append("reach")
    return json_result(args)


@app.function(**options)
def effect(pairs):
    return json_result(
        ["/root/policy/bin/python", "/root/analysis/hover_audit.py", json.dumps(pairs), "effect"]
    )


@app.function(**options)
def screen(seed):
    if not 3_003_800 <= seed < 3_003_900:
        raise ValueError("Expected a fresh coverage seed")
    return json_result(
        ["/root/policy/bin/python", "/root/analysis/hover_audit.py", str(seed), "screen"]
    )


@app.function(**options)
def episode(seed, candidate):
    if not (3_001_000 <= seed < 3_001_100 or 3_003_800 <= seed < 3_003_900):
        raise ValueError("Expected a fresh hover development seed")
    root = Path("/runs/hover-development")
    root.mkdir(exist_ok=True)
    arm = "candidate" if candidate else "base"
    out = Path(tempfile.mkdtemp(dir=root, prefix=f"{arm}-{seed}-")) / "eval"
    policy = "submission.policy_hover" if candidate else "submission.policy_modular_calibrated"
    try:
        run(
            [
                f"{CODE}/.venv/bin/advaitic-eval",
                "--policy-root",
                policy_root,
                "--policy",
                policy + ":make_policy",
                "--policy-python",
                "/root/policy/bin/python",
                "--method",
                "hover-" + arm,
                "--checkpoint",
                f"bd951bc+{WEIGHTS_SHA}+{HEAD_SHA}",
                "--seed-start",
                str(seed),
                "--smoke",
                "--episodes",
                "1",
                "--out",
                str(out),
            ]
        )
        report = json.loads((out / "report.json").read_text())
        e = report["episodes"][0]
        assert (
            report["integrity_ok"]
            and not report["infrastructure_error"]
            and e["completed"]
            and not e["policy_error"]
            and e["seed"] == seed
        )
        return dict(**e, archive=str(out), arm=arm, source=report["submission_sha256"])
    finally:
        volume.commit()
