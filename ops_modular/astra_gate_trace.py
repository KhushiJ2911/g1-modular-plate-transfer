"""Recover actual stateful controller inputs at failed placement decisions using saved actions."""

import hashlib
import inspect
import json
import os
import re
import subprocess
import time
from pathlib import Path

import modal

from ops_modular.lead_shard import CHECKPOINT, CODE, CPU_OPTIONS, ROOT, WEIGHTS_SHA, volume

records = modal.Volume.from_name("modular-runs")
if modal.is_local():
    sha = json.loads((ROOT / "frozen.json").read_text())["sha"][:7]
    image = CPU_OPTIONS["image"].add_local_file(ROOT / "frozen.json", f"{CODE}/frozen.json")
else:
    sha = json.loads(Path(CODE, "frozen.json").read_text())["sha"][:7]
    image = CPU_OPTIONS["image"]
options = {
    **CPU_OPTIONS,
    "image": image,
    "volumes": {"/runs": volume, "/records": records},
}
app = modal.App("modular-astra-gate-trace-" + sha)


def measure(directory, name):
    import importlib

    import numpy as np

    from advaitic_eval.environment import make_environment, observation, validate_actions
    from method.modular.privileged import PrivilegedTaskState

    path = Path(directory)
    report = json.loads((path / "report.json").read_text())
    episode = report["episodes"][0]
    seed = episode["seed"]
    if (
        not report["integrity_ok"]
        or not episode["completed"]
        or episode["success"]
        or episode["policy_error"]
        or report["method"] != name
    ):
        raise ValueError("Expected a completed frozen development failure")
    module, factory = report["policy"].split(":")
    policy = getattr(importlib.import_module(module), factory)()
    policy.reset()
    controller = policy.controller
    with np.load(path / "episode_000.actions.npz", allow_pickle=False) as f:
        logged = f["actions"]
    states = []
    before_contacts = controller.site_contacts
    ready_to_release = controller._ready_to_release

    class Observe:
        task = None
        ready = None

        def update(self, state, task):
            self.task = before_contacts.update(state, task)
            return self.task

        def release(self, task):
            self.ready = bool(ready_to_release(task))
            return self.ready

    seen = Observe()
    controller.site_contacts = seen
    controller._ready_to_release = seen.release
    env = make_environment()
    try:
        ts = env.reset(seed=seed)
        labels = PrivilegedTaskState(env)
        for step, archived in enumerate(logged):
            obs = observation(ts)
            actual = labels(obs["state"])
            phase = controller.phase.value
            if phase == "HOLD":
                break
            seen.ready = None
            issued = validate_actions(policy(obs))[0]
            err = float(np.abs(issued - archived).max())
            if err > 1e-5:
                raise ValueError(f"Candidate does not reproduce archived action {step}: {err}")
            task = seen.task
            states.append(
                dict(
                    step=step,
                    phase=phase,
                    next_phase=controller.phase.value,
                    max_action_error=err,
                    issued_grip=float(issued[18]),
                    ready=seen.ready,
                    est_held=bool(task["held"]),
                    true_held=bool(actual["held"]),
                    est_rack=bool(task["contacts"]["target_rack"]),
                    true_rack=bool(actual["contacts"]["target_rack"]),
                    est_tilt=float(task["tilt_deg"]),
                    true_tilt=float(actual["tilt_deg"]),
                    est_gap=(task["plate_xyz"] - controller.target_xyz).tolist(),
                    true_gap=(actual["plate_xyz"] - actual["site_xyz"]).tolist(),
                    plate_error=(task["plate_xyz"] - actual["plate_xyz"]).tolist(),
                    site_error=(controller.target_xyz - actual["site_xyz"]).tolist(),
                )
            )
            ts = env.step(archived)
    finally:
        env.close()
    return dict(
        seed=seed, policy=report["policy"], transitions=controller.transitions, states=states
    )


@app.function(**options, timeout=7200, max_containers=2, single_use_containers=True)
def trace(name, folder, head):
    candidates = {
        "bd951bc": "astra_calibrated_bd951bc_dev",
        "6b02f33": "astra_height_6b02f33_dev",
    }
    if name != candidates[sha] or not re.fullmatch(
        re.escape(name) + r"-10[0-9]{5}(-r[0-9]+)?", folder
    ):
        raise ValueError("Unexpected frozen run directory")
    path = Path(head)
    if (
        path.parent != Path("/runs/heads")
        or path.stem != "64dfe105e7f716c0ab5b7a9c487b7a8a1cac06565070b390e75992dfbee4deff"
        or hashlib.sha256(path.read_bytes()).hexdigest() != path.stem
        or hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != WEIGHTS_SHA
    ):
        raise ValueError("Incorrect policy checkpoint")
    records.reload()
    dest = Path("/records/astra-integration") / name / "gate_traces" / (folder + ".json")
    if dest.exists():
        return {"path": str(dest)}
    volume.reload()
    directory = str(Path("/runs/modular") / folder / "eval")
    src = "import json\nfrom pathlib import Path\n" + inspect.getsource(measure)
    src += f"\nprint(json.dumps(measure({directory!r},{name!r})))"
    start = time.monotonic()
    out = subprocess.run(
        ["/root/policy/bin/python", "-c", src],
        cwd=CODE,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "MUJOCO_GL": "osmesa",
            "PYOPENGL_PLATFORM": "osmesa",
            "ADV_HEADS": head,
            "ADV_ESTIMATOR": str(CHECKPOINT),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
    )
    if out.returncode:
        raise RuntimeError(out.stderr[-2200:])
    result = json.loads(out.stdout.splitlines()[-1])
    result["seconds"] = time.monotonic() - start
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result))
    records.commit()
    return {"path": str(dest), "seed": result["seed"], "seconds": result["seconds"]}
