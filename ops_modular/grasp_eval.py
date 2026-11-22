"""Deploy a CPU OSMesa worker and run detached paired development episodes."""

import concurrent.futures
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import modal
from grpclib.exceptions import StreamTerminatedError

from ops_modular.modular_sim import WEIGHTS_SHA, image, run_candidate

if modal.is_local():
    image = image.add_local_file(
        Path(__file__).with_name("modular_sim.py"), "/root/ops_modular/modular_sim.py"
    )

APP = os.environ.get("GRASP_EVAL_APP", "modular-grasp-feedback")
if not APP.startswith("modular-grasp-"):
    raise ValueError("Use a dedicated pickup app")
app = modal.App(APP)
runs = modal.Volume.from_name("modular-runs")
data = modal.Volume.from_name("modular-data")
HEADS = Path("/data/astra-r3/contact_v1.npz")
HEAD_SHA = "51257d3363a1d8eaae21ba863c36c166299e4e5bf2e0842a69ae81bbd8b7c86e"
EVAL = "method.modular.pickup_validation"
POLICY_PYTHON = "/root/policy/bin/python"


def run(args):
    p = run_candidate(args)
    if p.returncode:
        raise RuntimeError(p.stdout[-4000:])
    return p.stdout


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    max_containers=30,
    volumes={"/runs": runs, "/data": data},
    timeout=7200,
    scaledown_window=10,
)
def episode(seed: int, arm: str):
    if not 3000500 <= seed < 3002000 or arm not in ("base", "pickup", "obs_base", "obs_pickup"):
        raise ValueError("Expected a fresh pickup development seed and arm")
    os.environ.update(
        MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa", LP_NUM_THREADS="2", ADV_HEADS=str(HEADS)
    )
    if hashlib.sha256(HEADS.read_bytes()).hexdigest() != HEAD_SHA:
        raise ValueError("Unexpected perception head weights")
    start = time.monotonic()
    if arm.startswith("obs_"):
        factory = (
            EVAL + ":baseline"
            if arm == "obs_base"
            else "submission.policy_modular_heads:make_policy"
        )
        parent = Path("/runs/grasp-validation")
        parent.mkdir(exist_ok=True)
        bundle = Path(tempfile.mkdtemp(prefix=f"{arm}-{seed}-", dir=parent))
        dest = bundle / "eval"
        try:
            run(
                [
                    "/root/code/.venv/bin/advaitic-eval",
                    "--policy-python",
                    POLICY_PYTHON,
                    "--policy",
                    factory,
                    "--seed-start",
                    str(seed),
                    "--episodes",
                    "1",
                    "--smoke",
                    "--checkpoint",
                    f"est_v1b@{WEIGHTS_SHA}+contact_v1@{HEAD_SHA}",
                    "--method",
                    arm,
                    "--out",
                    str(dest),
                ]
            )
            report = json.loads((dest / "report.json").read_text())
            row = report["episodes"][0]
            if (
                not report["integrity_ok"]
                or report["infrastructure_error"]
                or row["policy_error"]
                or row["seed"] != seed
            ):
                raise ValueError(str(row))
            audit = json.loads(
                run(
                    [
                        POLICY_PYTHON,
                        "-m",
                        EVAL,
                        str(seed),
                        "audit",
                        "--archive",
                        str(dest / "episode_000.actions.npz"),
                    ]
                ).splitlines()[-1]
            )
            if any(row[k] != audit[k] for k in ("success", "fell")):
                raise ValueError("Official replay mismatch")
            result = dict(
                **row,
                **{k: audit[k] for k in ("lift", "secure", "early_losses")},
                integrity_ok=True,
                variant=arm,
                archive=str(dest),
            )
        finally:
            runs.commit()
    else:
        result = json.loads(run([POLICY_PYTHON, "-m", EVAL, str(seed), arm]).splitlines()[-1])
    return {**result, "seconds": time.monotonic() - start}


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    max_containers=16,
    volumes={"/runs": runs, "/data": data},
    timeout=5400,
    scaledown_window=10,
)
def pickup_trace(seed: int):
    documented_failures = {1010018, 1010029, 1010033, 1020026, 1020043}
    if seed not in documented_failures and not 3000500 <= seed < 3002000:
        raise ValueError("Expected a development diagnostic seed")
    os.environ.update(
        MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa", LP_NUM_THREADS="2", ADV_HEADS=str(HEADS)
    )
    if hashlib.sha256(HEADS.read_bytes()).hexdigest() != HEAD_SHA:
        raise ValueError("Unexpected perception head weights")
    out = Path("/runs/grasp-validation") / f"{APP}-{seed}.npz"
    if out.exists():
        raise FileExistsError(out)
    start = time.monotonic()
    result = json.loads(
        run([POLICY_PYTHON, "-m", "method.modular.pickup_trace", str(seed), str(out)]).splitlines()[
            -1
        ]
    )
    runs.commit()
    return dict(**result, seconds=time.monotonic() - start, archive=str(out), variant="trace")


@app.local_entrypoint()
def sweep(start: int, out: str, mode: str = "hybrid", smoke: bool = False):
    from advaitic_eval.protocol import make_plan

    arms = {"hybrid": ("base", "pickup"), "obs": ("obs_base", "obs_pickup")}[mode]
    jobs = [(e.seed, arm) for e in make_plan(30, start) for arm in arms]
    if smoke:
        jobs = jobs[:2]
    path = Path(out)
    fn = modal.Function.from_name(APP, "episode")
    with (
        concurrent.futures.ThreadPoolExecutor(8) as pool,
        path.with_suffix(".calls").open("x") as manifest,
    ):

        def spawn(job):
            seed, arm = job
            return seed, arm, fn.spawn(seed, arm).object_id

        calls = []
        for record in pool.map(spawn, jobs):
            calls.append(record)
            manifest.write(json.dumps(record) + "\n")
            manifest.flush()
    collect(str(path))


def collect(out):
    path = Path(out)
    calls = list(map(json.loads, path.with_suffix(".calls").read_text().splitlines()))

    def check(record):
        _, _, call_id = record
        try:
            return modal.FunctionCall.from_id(call_id).get(timeout=0)
        except (TimeoutError, OSError, StreamTerminatedError, modal.exception.ConnectionError):
            return None

    with path.open("a+") as f, concurrent.futures.ThreadPoolExecutor(12) as pool:
        f.seek(0)
        done = {(r["seed"], r["variant"]) for r in map(json.loads, f)}
        pending = [r for r in calls if tuple(r[:2]) not in done]
        while pending:
            outstanding = []
            for record, result in zip(pending, pool.map(check, pending), strict=True):
                if result is None:
                    outstanding.append(record)
                    continue
                if [result["seed"], result["variant"]] != record[:2]:
                    raise ValueError("Unexpected result")
                f.write(json.dumps(result) + "\n")
                f.flush()
                print(
                    result["seed"],
                    result["variant"],
                    result["secure"],
                    result["success"],
                    flush=True,
                )
            pending = outstanding
            if pending:
                time.sleep(15)
