"""Test delayed clearance from recorded public generation handoffs with fresh live feedback."""

import inspect
import json
import os
import re
import subprocess
import time
from pathlib import Path

import modal

from ops_modular.modular_sim import image, volume

app = modal.App("modular-astra-late-hybrid")
records = modal.Volume.from_name("modular-runs")
data = modal.Volume.from_name("modular-data")


def run(seed, directory, rendered, phase_aware=False, trace_motion=False):
    import mujoco
    import numpy as np
    from bigym.loco import make
    from bigym.loco.config import DEFAULT_CAMERA_KEYS

    from method.modular.base_clearance import QUIET_STEPS, LateClearance
    from method.modular.estimated import SITE_ROTATION, EstimatedTaskState
    from method.modular.kinematics import G1Kinematics
    from method.modular.privileged import PrivilegedTaskState
    from submission.estimator import Estimator

    previous = None
    if directory is None:
        with np.load(f"/data/astra-dagger/sol/{seed}.npz", allow_pickle=False) as f:
            ep = json.loads(str(f["meta"]))
            actions = f["actions"]
            h = f["step"][f["phase"] == "HOLD"]
            op = f["step"][f["phase"] == "RELEASE"]
            hold = int(h[0]) if len(h) and len(op) and op[0] < h[0] else None
            if hold is not None:
                previous = str(f["phase"][f["step"] < hold][-1])
    else:
        path = Path(directory)
        rep = json.loads((path / "report.json").read_text())
        ep = rep["episodes"][0]
        if (
            not rep["integrity_ok"]
            or ep["seed"] != seed
            or ep["policy_error"]
            or not ep["completed"]
        ):
            raise ValueError("Invalid public archive")
        with np.load(path / "episode_000.actions.npz", allow_pickle=False) as f:
            actions = f["actions"]
        trans = [
            json.loads(s.removeprefix("SOL_PHASE "))
            for s in (path / "policy.stderr.log").read_text().splitlines()
            if s.startswith("SOL_PHASE ")
        ]
        hold = next(
            (
                t["step"] + 1
                for t in trans
                if t["to"] == "HOLD" and t["from"] in ("SETTLE", "RETREAT")
            ),
            None,
        )
        if hold is not None:
            previous = next(t["from"] for t in trans if t["to"] == "HOLD")

    quiet = 0 if phase_aware and previous == "SETTLE" else QUIET_STEPS
    result = dict(seed=seed, baseline=ep["success"], hold=hold, rendered=rendered, quiet=quiet)
    if hold is None or len(actions) <= hold + max(quiet - 1, 0):
        return dict(**result, candidate=ep["success"], reason="identical prefix", active=0)
    pause_action = actions[hold]
    if np.max(np.abs(actions[hold:] - pause_action)) > 1e-5 or pause_action[18] >= 0:
        raise ValueError("Expected constant open posture during the archived HOLD")
    env = make("move_plate", camera_keys=())
    renderer = None
    kin = G1Kinematics()
    motion = LateClearance(quiet)
    try:
        ts = env.reset(seed=seed)
        inner = env.inner_env
        if rendered:
            estimator = EstimatedTaskState(Estimator("/runs/est_v1b/last.pt", device="cpu"), kin)
            renderer = mujoco.Renderer(inner.model, height=84, width=84)
            cams = [
                next(
                    i
                    for i in range(inner.model.ncam)
                    if inner.model.camera(i).name.split("/")[-1] == k
                )
                for k in DEFAULT_CAMERA_KEYS
            ]
        labels = PrivilegedTaskState(env)
        init_sites = []

        def observe(state):
            if not rendered:
                return labels(state)
            frames = []
            for cam in cams:
                renderer.update_scene(inner.data, camera=cam)
                frames.append(renderer.render().transpose(2, 0, 1))
            return estimator.update({"state": state, "rgb": np.stack(frames)})

        steps = 0
        motion_start = None
        for t in range(1700):
            if ts.last():
                break
            state = ts.low_dim_obs
            age = t - hold + 1
            if t < 6:
                init_sites.append(observe(state)["site_xyz"])
                target = np.median(init_sites, axis=0)
            action = actions[t] if t < hold else pause_action
            if age >= max(1, quiet - 5) and not motion.reason:
                task = observe(state)
                # The true-state diagnostic uses the task's site nearest this
                # plate. The observed fork uses its stationary six-frame lock.
                site = target if rendered else task["site_xyz"]
                action = motion.apply(action, state, kin, task, site, SITE_ROTATION, age)
                if motion.elapsed == 1 and "start_step" not in result:
                    result["start_step"] = t
                    result["start_true_tilt"] = labels(state)["tilt_deg"]
                if trace_motion and motion.elapsed:
                    physical = labels(state)
                    center = np.median(motion.samples, axis=0)
                    if motion_start is None:
                        motion_start = (physical["plate_xyz"].copy(), task["site_xyz"].copy())
                    # Also record at a successful terminal if no guard fires.
                    result["motion"] = {
                        "step": t,
                        "true_delta_cm": (100 * (physical["plate_xyz"] - motion_start[0])).tolist(),
                        "observed_delta_cm": (100 * (center - motion.plate)).tolist(),
                        "observed_site_delta_cm": (
                            100 * (task["site_xyz"] - motion_start[1])
                        ).tolist(),
                        "initial_observed_tilt": motion.first_tilt,
                        "observed_tilt": task["tilt_deg"],
                        "true_tilt": physical["tilt_deg"],
                    }
            ts = env.step(action)
            steps += 1
        if not ts.last():
            raise ValueError("Fork did not terminate")
        if not motion.elapsed and bool(env.episode_succeeded()) != ep["success"]:
            raise ValueError("Unchanged trajectory did not reproduce")
        return dict(
            **result,
            candidate=bool(env.episode_succeeded()),
            fell=bool(env.episode_fell()),
            active=motion.elapsed,
            reason=motion.reason or "terminal",
            steps=steps,
        )
    finally:
        if renderer is not None:
            renderer.close()
        env.close()


@app.function(
    image=image,
    cpu=2,
    memory=2500,
    max_containers=4,
    timeout=1000,
    volumes={"/runs": volume, "/data": data, "/records": records},
)
def trial(seed, arm, archive, rendered, phase_aware=False, trace_motion=False):
    if archive is None:
        if arm != "sol" or not 3110180 <= seed < 3110240:
            raise ValueError("Expected the historical generation cohort")
        data.reload()
    elif not (
        3001600 <= seed < 3001750
        and arm in ("base", "fk")
        and re.fullmatch(
            rf"/runs/modular/sol_fk_full_{arm}(?:_retry)?-{seed}(?:-r[0-9]+)?/eval", archive
        )
    ):
        raise ValueError("Expected the paired current generation cohort")
    volume.reload()
    records.reload()
    folder = Path("/records/astra-integration") / (
        "clearance-motion"
        if trace_motion
        else ("phase-clearance" if phase_aware else "late-clearance")
    )
    folder.mkdir(exist_ok=True, parents=True)
    dest = folder / f"{arm}_{seed}_{int(rendered)}.json"
    if dest.exists():
        return json.loads(dest.read_text())
    script = "import json\nfrom pathlib import Path\n" + inspect.getsource(run)
    script += f"\nprint(json.dumps(run({seed},{archive!r},{rendered!r},{phase_aware!r},{trace_motion!r})))"
    start = time.monotonic()
    p = subprocess.run(
        ["/root/policy/bin/python", "-c", script],
        cwd="/root/code",
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "MUJOCO_GL": "osmesa",
            "PYOPENGL_PLATFORM": "osmesa",
            "OMP_NUM_THREADS": "1",
            "LP_NUM_THREADS": "2",
        },
    )
    if p.returncode:
        raise RuntimeError(p.stderr[-1800:])
    result = dict(**json.loads(p.stdout.splitlines()[-1]), seconds=time.monotonic() - start)
    dest.write_text(json.dumps(result))
    records.commit()
    return result
