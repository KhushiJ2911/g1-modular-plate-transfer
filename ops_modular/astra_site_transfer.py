"""Compare two continuously estimated stationary rack sites from identical public pickups."""

import hashlib
import inspect
import json
import os
import subprocess
import time
from pathlib import Path

import modal

from ops_modular.astra_grasp_transfer import continuation
from ops_modular.modular_sim import WEIGHTS_SHA, image

app = modal.App("modular-astra-site-transfer")
records = modal.Volume.from_name("modular-runs")
weights = modal.Volume.from_name("advaitic-runs")
SITE_SHA = "0f4053833b6aac4f187d95bb17f0daa3d3a81d2c4e461ee320b3f6b3f17ffcad"


def lock(seed, archive):
    import numpy as np

    from advaitic_eval.environment import make_environment, observation
    from method.modular.privileged import PrivilegedTaskState
    from submission.policy_modular_integrated import make_site

    policy = make_site()
    with np.load(Path(archive) / "episode_000.actions.npz") as f:
        actions = f["actions"][:5]
    initial, corrected = [], []
    env = make_environment()
    try:
        ts = env.reset(seed=seed)
        true = PrivilegedTaskState(env)(ts.low_dim_obs)["site_xyz"]
        for step in range(6):
            obs = observation(ts)
            estimate = policy.task.update(obs)
            initial.append(policy.task.source.snapshot(obs["state"])["site_xyz"].copy())
            corrected.append(estimate["site_xyz"].copy())
            if step < 5:
                ts = env.step(actions[step])
    finally:
        env.close()
    return dict(
        raw=np.median(initial, axis=0).tolist(),
        corrected=np.median(corrected, axis=0).tolist(),
        true=true.tolist(),
    )


@app.function(
    image=image,
    cpu=1,
    memory=2560,
    timeout=1200,
    max_containers=2,
    volumes={"/runs": records, "/weights": weights},
    single_use_containers=True,
)
def compare(seed):
    if not 3000700 <= seed < 3000850:
        raise ValueError("Use the independent source pickup plan")
    source = Path(f"/runs/astra-integration/grasp-transfer/v1/{seed}.json")
    row = json.loads(source.read_text())
    out = Path("/runs/astra-integration/site-transfer") / f"{seed}.json"
    if out.exists():
        return json.loads(out.read_text())
    head = Path("/weights/heads") / (SITE_SHA + ".npz")
    estimator = Path("/weights/est_v1b/last.pt")
    if (
        hashlib.sha256(head.read_bytes()).hexdigest() != SITE_SHA
        or hashlib.sha256(estimator.read_bytes()).hexdigest() != WEIGHTS_SHA
    ):
        raise ValueError("Unexpected perception checkpoint")
    args = repr((seed, row["archive"]))
    src = (
        "from pathlib import Path\nimport json\n"
        + inspect.getsource(lock)
        + "\n"
        + inspect.getsource(continuation)
        + f"\nsites=lock(*{args})\n"
        + f"r=[continuation(*{args},{row['vector']!r},sites[k]) for k in ('raw','corrected')]\n"
        + "print(json.dumps([sites,*r]))"
    )
    started = time.monotonic()
    proc = subprocess.run(
        ["/root/policy/bin/python", "-c", src],
        cwd="/root/code",
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "ADV_HEADS": str(head),
            "ADV_ESTIMATOR": str(estimator),
            "MUJOCO_GL": "osmesa",
            "PYOPENGL_PLATFORM": "osmesa",
            "LP_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        },
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr[-2500:])
    sites, a, b = json.loads(proc.stdout.splitlines()[-1])
    if any(
        (arm["secure"], arm["prefix"]) != (row["candidate"]["secure"], row["candidate"]["prefix"])
        for arm in (a, b)
    ):
        raise ValueError("Paired handoff mismatch")
    result = dict(
        seed=seed,
        archive=row["archive"],
        sites=sites,
        raw=a,
        corrected=b,
        oracle=row["candidate"],
        seconds=time.monotonic() - started,
    )
    out.parent.mkdir(exist_ok=True, parents=True)
    out.write_text(json.dumps(result))
    records.commit()
    return result
