"""Validate the frozen open-finger detector on complete, current generation action archives."""

import inspect
import json
import os
import re
import subprocess
import time
from pathlib import Path

import modal

from ops_modular.modular_sim import image, volume

app = modal.App("modular-astra-contact-validation")
records = modal.Volume.from_name("modular-runs")


def measure(seed, folder):
    from collections import deque

    import numpy as np
    from bigym.loco import make

    from method.modular.privileged import PrivilegedTaskState
    from method.modular.recontact import OpenFingerContact

    path = Path(folder)
    saved = json.loads((path / "report.json").read_text())
    ep = saved["episodes"][0]
    if ep["seed"] != seed or ep["policy_error"] or not ep["completed"] or not saved["integrity_ok"]:
        raise ValueError("Invalid generation report")
    transitions = [
        json.loads(s.removeprefix("SOL_PHASE "))
        for s in (path / "policy.stderr.log").read_text().splitlines()
        if s.startswith("SOL_PHASE ")
    ]
    opened = next((t["step"] for t in transitions if t["to"] == "RELEASE"), None)
    out = dict(
        seed=seed,
        success=ep["success"],
        opened=opened,
        transitions=transitions,
        first_free=None,
        first_sustained=None,
        first_alarm=None,
        tp=0,
        fp=0,
        fn=0,
        tn=0,
    )
    detector = OpenFingerContact()
    history = deque(maxlen=7)
    env = make("move_plate", camera_keys=())
    try:
        ts = env.reset(seed=seed)
        labels = PrivilegedTaskState(env)
        with np.load(path / "episode_000.actions.npz", allow_pickle=False) as f:
            actions = f["actions"]
        for i, action in enumerate(actions):
            if opened is not None and opened <= i < opened + 100:
                state = ts.low_dim_obs
                alarm = detector.update(state)
                held = bool(labels(state)["held"])
                if not held and out["first_free"] is None:
                    out["first_free"] = i
                if out["first_free"] is not None and state[44] <= 0.08:
                    history.append(held)
                    if len(history) == 7 and history[0] and held and out["first_sustained"] is None:
                        out["first_sustained"] = i
                    if alarm and out["first_alarm"] is None:
                        out["first_alarm"] = i
                    out[("tp" if held else "fp") if alarm else ("fn" if held else "tn")] += 1
                else:
                    history.clear()
            ts = env.step(action)
        if not ts.last() or (bool(env.episode_succeeded()), bool(env.episode_fell())) != (
            ep["success"],
            ep["fell"],
        ):
            raise ValueError("Different physical outcome on replay")
    finally:
        env.close()
    return out


@app.function(
    image=image,
    cpu=1,
    memory=1536,
    timeout=600,
    max_containers=2,
    volumes={"/runs": volume, "/records": records},
)
def audit(seed, arm, folder):
    if (
        not 3001600 <= seed < 3001750
        or arm not in ("base", "fk")
        or not re.fullmatch(rf"/runs/modular/sol_fk_full_{arm}-{seed}(?:-r[0-9]+)?/eval", folder)
    ):
        raise ValueError("Expected the paired generation block")
    records.reload()
    root = Path("/records/astra-integration/contact-validation")
    root.mkdir(parents=True, exist_ok=True)
    dest = root / f"{arm}_{seed}.json"
    if dest.is_file():
        existing = json.loads(dest.read_text())
        if existing["archive"] != folder:
            raise ValueError("Conflicting source archive")
        return existing
    volume.reload()
    src = (
        "import json\nfrom pathlib import Path\n"
        + inspect.getsource(measure)
        + f"\nprint(json.dumps(measure({seed},{folder!r})))"
    )
    start = time.monotonic()
    p = subprocess.run(
        ["/root/policy/bin/python", "-c", src],
        cwd="/root/code",
        text=True,
        capture_output=True,
        env={**os.environ, "MUJOCO_GL": "osmesa", "OMP_NUM_THREADS": "1"},
    )
    if p.returncode:
        raise RuntimeError(p.stderr[-1800:])
    out = dict(
        **json.loads(p.stdout.splitlines()[-1]),
        arm=arm,
        archive=folder,
        seconds=time.monotonic() - start,
    )
    dest.write_text(json.dumps(out))
    records.commit()
    return out
