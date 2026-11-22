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
    fresh = 3000500 <= seed < 3002000 or 3055000 <= seed < 3057000
    if seed not in documented_failures and not fresh:
        raise ValueError("Expected a development diagnostic seed")
    os.environ.update(
        MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa", LP_NUM_THREADS="2", ADV_HEADS=str(HEADS)
    )
    if hashlib.sha256(HEADS.read_bytes()).hexdigest() != HEAD_SHA:
        raise ValueError("Unexpected perception head weights")
    parent = Path("/runs/grasp-validation")
    parent.mkdir(exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=f"trace-{seed}-", dir=parent)) / "trace.npz"
    start = time.monotonic()
    result = json.loads(
        run([POLICY_PYTHON, "-m", "method.modular.pickup_trace", str(seed), str(out)]).splitlines()[
            -1
        ]
    )
    runs.commit()
    return dict(**result, seconds=time.monotonic() - start, archive=str(out), variant="trace")


@app.function(
    image=image,
    cpu=2,
    memory=2048,
    volumes={"/runs": runs},
    timeout=1200,
    max_containers=1,
    scaledown_window=5,
)
def archived_handoff(seed: int, arm: str, archive: str):
    folder = Path(archive)
    if folder.parent.parent != Path("/runs/grasp-validation") or folder.name != "eval":
        raise ValueError("Expected a pickup validation archive")
    runs.reload()
    os.environ.update(MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa")
    row = json.loads((folder / "report.json").read_text())["episodes"][0]
    if row["seed"] != seed:
        raise ValueError("Archive seed mismatch")
    args = [
        POLICY_PYTHON,
        "-m",
        EVAL,
        str(seed),
        "handoff",
        "--archive",
        str(folder / "episode_000.actions.npz"),
    ]
    result = json.loads(run(args).splitlines()[-1])
    if any(result[k] != row[k] for k in ("success", "fell", "steps")):
        raise ValueError("Official handoff replay mismatch")
    return {**result, "variant": arm, "archive": archive}


@app.local_entrypoint()
def sweep(
    start: int, out: str, mode: str = "hybrid", smoke: bool = False, candidate_only: bool = False
):
    from advaitic_eval.protocol import make_plan

    arms = {"hybrid": ("base", "pickup"), "obs": ("obs_base", "obs_pickup")}[mode]
    if candidate_only:
        arms = arms[-1:]
    jobs = [(e.seed, arm) for e in make_plan(30, start) for arm in arms]
    if smoke:
        jobs = jobs[: len(arms)]
    path = Path(out)
    fn = modal.Function.from_name(APP, "episode")
    with (
        concurrent.futures.ThreadPoolExecutor(8) as pool,
        path.with_suffix(".calls").open("a+") as manifest,
    ):
        manifest.seek(0)
        existing = [(seed, arm) for seed, arm, _ in map(json.loads, manifest)]
        if len(existing) != len(set(existing)) or not set(existing) <= set(jobs):
            raise ValueError("Call manifest does not match the plan")
        missing = [job for job in jobs if job not in existing]

        def spawn(job):
            seed, arm = job
            return seed, arm, fn.spawn(seed, arm).object_id

        for record in pool.map(spawn, missing):
            manifest.write(json.dumps(record) + "\n")
            manifest.flush()
    collect(str(path))


def collect(out):
    path = Path(out)
    calls = list(map(json.loads, path.with_suffix(".calls").read_text().splitlines()))
    # Keep the original manifest immutable. Replacements are recorded only
    # after checking the failed input's remote status and its volume archive.
    retry_path = path.with_suffix(".retry.calls")
    if retry_path.exists():
        by_job = {tuple(record[:2]): record for record in calls}
        for seed, arm, new_id, old_id, _reason in map(
            json.loads, retry_path.read_text().splitlines()
        ):
            if by_job[seed, arm][2] != old_id:
                raise ValueError("Replacement does not match the preceding call")
            by_job[seed, arm] = [seed, arm, new_id]
        calls = list(by_job.values())

    def check(record):
        _, _, call_id = record
        call = modal.FunctionCall.from_id(call_id)
        try:
            return call.get(timeout=0)
        except modal.exception.FunctionTimeoutError:
            raise
        except (TimeoutError, OSError, StreamTerminatedError, modal.exception.ConnectionError):
            # A preempted attempt can show FAILURE while its original call resumes.
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
