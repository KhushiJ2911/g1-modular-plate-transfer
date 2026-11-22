"""Fork generation rollouts at the first open-gripper HOLD and test base clearance."""

import inspect
import json
import os
import re
import subprocess
import time
from pathlib import Path

import modal

from ops_modular.modular_sim import image

app = modal.App("modular-astra-base-clearance")
data = modal.Volume.from_name("modular-data")
records = modal.Volume.from_name("modular-runs")
archives = modal.Volume.from_name("advaitic-runs")


def fork(seed, directory=None):
    from pathlib import Path

    import numpy as np
    from bigym.loco import make

    from method.modular import layout
    from method.modular.kinematics import G1Kinematics, pelvis_rotation, pelvis_to_world
    from method.modular.privileged import PrivilegedTaskState

    expected_state = None
    if directory is None:
        with np.load(f"/data/astra-dagger/sol/{seed}.npz", allow_pickle=False) as f:
            steps, phases, actions = f["step"], f["phase"], f["actions"]
            meta = json.loads(str(f["meta"]))
            hold = np.flatnonzero(phases == "HOLD")
            opening = np.flatnonzero(phases == "RELEASE")
            fork_at = None
            if len(hold) and len(opening) and hold[0] > opening[0]:
                fork_at = int(steps[hold[0]])
                expected_state = f["state"][hold[0]]
    else:
        path = Path(directory)
        report = json.loads((path / "report.json").read_text())
        meta = report["episodes"][0]
        if (
            not report["integrity_ok"]
            or meta["seed"] != seed
            or not meta["completed"]
            or meta["policy_error"]
        ):
            raise ValueError("Invalid current-generation source")
        with np.load(path / "episode_000.actions.npz", allow_pickle=False) as f:
            actions = f["actions"]
        transitions = [
            json.loads(line.removeprefix("SOL_PHASE "))
            for line in (path / "policy.stderr.log").read_text().splitlines()
            if line.startswith("SOL_PHASE ")
        ]
        # The transition is logged on the action that entered HOLD. The policy
        # first sees its new phase on the following observation.
        fork_at = next(
            (
                e["step"] + 1
                for e in transitions
                if e["to"] == "HOLD" and e["from"] in ("RETREAT", "SETTLE")
            ),
            None,
        )
    result = dict(seed=seed, baseline=meta["success"], eligible=False)
    if fork_at is None or fork_at >= len(actions):
        return dict(**result, candidate=meta["success"], reason="identical prefix")

    env = make("move_plate", camera_keys=())
    try:
        ts = env.reset(seed=seed)
        for action in actions[:fork_at]:
            ts = env.step(action)
        start_state = ts.low_dim_obs.copy()
        if expected_state is not None:
            np.testing.assert_array_equal(expected_state, start_state)
        kin = G1Kinematics()
        labels = PrivilegedTaskState(env)
        task = labels(start_state)
        result.update(
            initial_tilt=task["tilt_deg"],
            initial_held=task["held"],
            initial_xy_cm=100 * task["site_xy_distance"],
            initial_support=task["contacts"]["target_rack"],
        )
        if (
            max(start_state[44:46]) > 0.08
            or not task["contacts"]["target_rack"]
            or task["site_xy_distance"] > 0.065
            or task["tilt_deg"] > 18
        ):
            return dict(**result, candidate=meta["success"], reason="outside guard", step=fork_at)
        result.update(eligible=True, step=fork_at)
        plate = task["plate_xyz"].copy()
        kin.set_state(start_state)
        hand = pelvis_to_world(start_state, kin.hand_pose("left")[0])
        axis = task["site_rot"][:2, 0].copy()
        axis /= np.linalg.norm(axis)
        # Withdraw along the rack plane, in the direction of the loaded wrist.
        side = np.dot(hand[:2] - plate[:2], axis)
        if abs(side) < 0.01:
            side = np.dot(start_state[18:20] - plate[:2], axis)
        axis *= 1 if side >= 0 else -1
        result.update(direction=axis.tolist(), hand_minus_plate=(hand - plate).tolist())
        start_tilt = task["tilt_deg"]
        target = layout.denormalize_action(actions[fork_at])
        reason = None
        first_free = None
        max_shift = 0.0
        travel = 0.0
        for t in range(fork_at, 1700):
            if ts.last():
                break
            state = ts.low_dim_obs
            if reason is None:
                task = labels(state)
                shift = np.linalg.norm(task["plate_xyz"] - plate)
                max_shift = max(max_shift, shift)
                travel = float((state[18:20] - start_state[18:20]) @ axis)
                if first_free is None and not task["held"]:
                    first_free = t - fork_at
                if shift > 0.014 or task["tilt_deg"] > min(22, start_tilt + 6):
                    reason = "plate motion"
                elif not task["contacts"]["target_rack"]:
                    reason = "support lost"
                elif travel >= 0.055 or t - fork_at >= 130:
                    reason = "clearance bound"
            target[:2] = 0 if reason else pelvis_rotation(state)[:2, :2].T @ (axis * 0.075)
            ts = env.step(layout.normalize_action(target))
        if not ts.last():
            raise ValueError("Fork did not reach the release terminal condition")
        end = labels(ts.low_dim_obs)
        return dict(
            **result,
            candidate=bool(env.episode_succeeded()),
            fell=bool(env.episode_fell()),
            reason=reason or "terminal",
            first_free=first_free,
            base_travel_cm=100 * travel,
            max_plate_shift_cm=100 * max_shift,
            end_tilt=end["tilt_deg"],
            end_held=end["held"],
        )
    finally:
        env.close()


def measure(seed, dest, directory):
    records.reload()
    if dest.exists():
        return json.loads(dest.read_text())
    dest.parent.mkdir(parents=True, exist_ok=True)
    script = "import json\n" + inspect.getsource(fork)
    script += f"\nprint(json.dumps(fork({seed},{directory!r})))"
    start = time.monotonic()
    result = subprocess.run(
        ["/root/policy/bin/python", "-c", script],
        cwd="/root/code",
        env={**os.environ, "MUJOCO_GL": "osmesa", "OMP_NUM_THREADS": "1"},
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr[-2200:])
    out = dict(**json.loads(result.stdout.splitlines()[-1]), seconds=time.monotonic() - start)
    dest.write_text(json.dumps(out))
    records.commit()
    return out


@app.function(
    image=image,
    cpu=1,
    memory=1536,
    max_containers=2,
    timeout=600,
    volumes={"/data": data, "/records": records},
)
def evaluate(seed):
    if not 3110180 <= seed < 3110240:
        raise ValueError("Only the declared development-generation cohort")
    data.reload()
    return measure(seed, Path("/records/astra-integration/base-clearance") / f"{seed}.json", None)


@app.function(
    image=image,
    cpu=1,
    memory=1536,
    max_containers=2,
    timeout=600,
    volumes={"/runs": archives, "/records": records},
)
def current(seed, arm, directory):
    if (
        not 3001600 <= seed < 3001750
        or arm not in ("base", "fk")
        or not re.fullmatch(
            rf"/runs/modular/sol_fk_full_{arm}(?:_retry)?-{seed}(?:-r[0-9]+)?/eval", directory
        )
    ):
        raise ValueError("Only the paired public generation cohort")
    archives.reload()
    dest = Path("/records/astra-integration/base-clearance-current") / f"{arm}_{seed}.json"
    return measure(seed, dest, directory)


@app.function(image=image, cpu=0.25, memory=512, timeout=180, volumes={"/data": data})
def eligible_smoke():
    """Find one physically applicable smoke among the fixed 60 without running a simulator."""
    data.reload()
    script = """
import json
import numpy as np
out=[]
for seed in range(3110180, 3110240):
    with np.load(f"/data/astra-dagger/sol/{seed}.npz", allow_pickle=False) as f:
        hold=np.flatnonzero(f['phase']=='HOLD')
        op=np.flatnonzero(f['phase']=='RELEASE')
        if not len(hold) or not len(op) or hold[0]<=op[0]: continue
        i=hold[0]
        y=f['truth'][i]
        if f['state'][i,44]<=0.08 and y[6]<18 and y[15]>0.5 and np.linalg.norm(y[:2]-y[3:5])<=0.065:
            out.append(dict(seed=seed,held=bool(y[7])))
print(json.dumps(out))
"""
    p = subprocess.run(
        ["/root/policy/bin/python", "-c", script], text=True, capture_output=True, check=True
    )
    return json.loads(p.stdout.splitlines()[-1])
