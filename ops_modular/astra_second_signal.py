"""Replay existing failed 528c854 signal archives and verify their physical endpoints."""

import json
import re
import time
from pathlib import Path

import modal

from ops_modular.astra_integration_replay import image, replay_episode, volume

app = modal.App("modular-astra-second-signal-actions")


def original_failure(folder):
    match = re.fullmatch(r"sig2_528c854-(10[3-6]10[0-9]{2})(?:-r[0-9]+)?", folder)
    if not match:
        raise ValueError("Expected an original second-signal shard")
    volume.reload()
    src = Path("/runs/modular") / folder / "eval"
    report = json.loads((src / "report.json").read_text())
    (row,) = report["episodes"]
    if (
        report["method"] != "sig2_528c854"
        or report["policy"] != "method.modular.base_clearance:make_policy"
        or not report["integrity_ok"]
        or report["infrastructure_error"]
        or row["seed"] != int(match[1])
        or not row["completed"]
        or row["success"]
        or row["policy_error"]
    ):
        raise ValueError("Only terminal archived failures from the frozen withdrawal candidate")
    return src, row


@app.function(
    image=image,
    cpu=1,
    timeout=400,
    max_containers=1,
    single_use_containers=True,
    volumes={"/runs": volume},
)
def replay(folder):
    src, _ = original_failure(folder)
    start = time.monotonic()
    result = replay_episode(src)
    return {**result, "seconds": time.monotonic() - start, "folder": folder}


def guard_inputs(directory, event):
    import mujoco
    import numpy as np
    from bigym.loco import make
    from bigym.loco.config import DEFAULT_CAMERA_KEYS

    from method.modular.base_clearance import LateClearance
    from method.modular.estimated import SITE_ROTATION, EstimatedTaskState
    from method.modular.kinematics import G1Kinematics
    from method.modular.privileged import PrivilegedTaskState
    from submission.estimator import Estimator

    with np.load(Path(directory) / "episode_000.actions.npz", allow_pickle=False) as f:
        actions = f["actions"]
    motion = event.get("open_base_motion")
    names = {i: "site lock" for i in range(6)}
    for key in ("open_at_target", "first_pad_return"):
        if event.get(key):
            names[event[key]["step"]] = key
    if motion:
        first, stop = motion["first"]["step"], motion["last"]["step"] + 1
        if stop >= len(actions):
            raise ValueError("Cannot verify a stop after the final command")
        for i in range(first - 4, stop + 1):
            names.setdefault(i, "withdrawal")
    checks = []
    if motion is None and event.get("open_at_target"):
        # An uninstrumented policy records no gate state on an unchanged HOLD
        # action. Sample its observable prerequisites through the settle/retry
        # window and at terminal, without attributing an unseen transition.
        opening = event["open_at_target"]["step"]
        checks = [i for i in range(opening + 80, min(len(actions), opening + 401), 20)] + [
            len(actions) - 1
        ]
        for i in checks:
            for j in range(i - 4, i + 1):
                names.setdefault(j, "stationary eligibility")
    selected = {}
    env = make("move_plate", camera_keys=())
    renderer = None
    kin = G1Kinematics()
    obs = EstimatedTaskState(Estimator("/runs/est_v1b/last.pt", device="cpu"), kin)
    try:
        ts = env.reset(seed=event["seed"])
        real = PrivilegedTaskState(env)
        renderer = mujoco.Renderer(env.inner_env.model, height=84, width=84)
        model = env.inner_env.model
        cams = [
            next(i for i in range(model.ncam) if model.camera(i).name.split("/")[-1] == key)
            for key in DEFAULT_CAMERA_KEYS
        ]
        for i, command in enumerate(actions):
            if i in names:
                state = ts.low_dim_obs.copy()
                views = []
                for camera in cams:
                    renderer.update_scene(env.inner_env.data, camera=camera)
                    views.append(renderer.render().transpose(2, 0, 1))
                task = obs.update({"rgb": np.stack(views), "state": state})
                selected[i] = (state, task, real(state))
            ts = env.step(command)
        report = json.loads((Path(directory) / "report.json").read_text())
        ep = report["episodes"][0]
        if (env.episode_succeeded(), env.episode_fell(), env.episode_termination()) != (
            ep["success"],
            ep["fell"],
            ep["termination"],
        ):
            raise ValueError("Sparse input audit failed the original terminal check")
    finally:
        if renderer is not None:
            renderer.close()
        env.close()
    notes = []
    for i, (_, task, actual) in selected.items():
        if names[i] in ("first_pad_return", "open_at_target"):
            notes.append(
                dict(
                    step=i,
                    event=names[i],
                    raw_held=bool(task["held"]),
                    held=bool(actual["held"]),
                    visual_tilt=task["tilt_deg"],
                    physical_tilt=actual["tilt_deg"],
                    center_error_cm=(100 * (task["plate_xyz"] - actual["plate_xyz"])).tolist(),
                )
            )
    site = np.median([selected[i][1]["site_xyz"] for i in range(6)], axis=0)
    if not motion:
        windows = []
        for i in checks:
            state, task, actual = selected[i]
            center = np.median([selected[j][1]["plate_xyz"] for j in range(i - 4, i + 1)], axis=0)
            gap = center - site
            near = np.linalg.norm(gap[:2]) < 0.065
            support = task["contacts"]["target_rack"] or (near and -0.060 < gap[2] < -0.012)
            windows.append(
                dict(
                    step=i,
                    hand_open=bool(max(state[44:46]) <= 0.08),
                    near=bool(near),
                    support=bool(support),
                    visual_tilt=task["tilt_deg"],
                    actual_tilt=actual["tilt_deg"],
                    actual_support=bool(actual["contacts"]["target_rack"]),
                    visual_xy_cm=float(100 * np.linalg.norm(gap[:2])),
                    actual_xy_cm=float(100 * actual["site_xy_distance"]),
                )
            )
        return dict(events=notes, tests=[], windows=windows)
    tests = []
    # The uninstrumented deploy did not save its HOLD transition. The five-frame
    # median could have between zero and four earlier observations at activation.
    # Try every such history and report how well its full command segment matches.
    for n in range(5):
        guard = LateClearance(0)
        guard.samples.extend(selected[i][1]["plate_xyz"] for i in range(first - n, first))
        error = 0.0
        for i in range(first, stop + 1):
            state, task, actual = selected[i]
            static = actions[i].copy()
            static[:2] = 0.0
            predicted = guard.apply(static, state, kin, task, site, SITE_ROTATION, i - first)
            error = max(error, float(np.max(np.abs(predicted[:2] - actions[i, :2]))))
        if guard.origin is None:
            tests.append(dict(history=n, max_action_error=error, reason="never armed"))
            continue
        plate = np.median(guard.samples, axis=0)
        tests.append(
            dict(
                history=n,
                max_action_error=error,
                reason=guard.reason,
                active=guard.elapsed,
                observed_motion_cm=float(np.linalg.norm(plate - guard.plate) * 100),
                physical_motion_cm=float(
                    np.linalg.norm(actual["plate_xyz"] - selected[first][2]["plate_xyz"]) * 100
                ),
                first_visual_tilt=guard.first_tilt,
                stop_visual_tilt=task["tilt_deg"],
                first_physical_tilt=selected[first][2]["tilt_deg"],
                stop_physical_tilt=actual["tilt_deg"],
                physical_support=bool(actual["contacts"]["target_rack"]),
            )
        )
    return dict(events=notes, tests=tests, first=first, stop=stop)


@app.function(
    image=image,
    cpu=1,
    memory=2304,
    timeout=300,
    max_containers=1,
    single_use_containers=True,
    volumes={"/runs": volume},
)
def observe(folder, event):
    import hashlib
    import inspect
    import os
    import subprocess

    from ops_modular.modular_sim import CHECKPOINT, WEIGHTS_SHA

    directory, episode = original_failure(folder)
    if (
        event["folder"] != folder
        or episode["seed"] != event["seed"]
        or hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != WEIGHTS_SHA
    ):
        raise ValueError("Expected a verified frozen clearance failure with v1b")
    src = "import json\nfrom pathlib import Path\n" + inspect.getsource(guard_inputs)
    src += f"\nprint(json.dumps(guard_inputs({str(directory)!r},{event!r})))"
    start = time.monotonic()
    out = subprocess.run(
        ["/root/policy/bin/python", "-c", src],
        cwd="/root/code",
        text=True,
        capture_output=True,
        env=dict(
            os.environ,
            MUJOCO_GL="osmesa",
            PYOPENGL_PLATFORM="osmesa",
            OMP_NUM_THREADS="1",
            LP_NUM_THREADS="1",
        ),
    )
    if out.returncode:
        raise RuntimeError(out.stderr[-2100:])
    return dict(
        seed=event["seed"],
        folder=folder,
        seconds=time.monotonic() - start,
        **json.loads(out.stdout.splitlines()[-1]),
    )
