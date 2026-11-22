"""Measure the loaded grasp on public pickup archives and test the resulting transfer."""

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

app = modal.App("modular-astra-lift-capture")
records = modal.Volume.from_name("modular-runs")
weights = modal.Volume.from_name("advaitic-runs")


def estimate(seed, archive, prefix, path):
    import mujoco
    import numpy as np
    from bigym.loco import make
    from bigym.loco.config import DEFAULT_CAMERA_KEYS

    from method.modular.kinematics import G1Kinematics
    from submission.estimator import Estimator
    from submission.policy_modular_capture import LiftObservation

    with np.load(path, allow_pickle=False) as w:
        capture = LiftObservation(G1Kinematics(), w)
    source = Estimator("/weights/est_v1b/last.pt", device="cpu")
    with np.load(Path(archive) / "episode_000.actions.npz") as f:
        actions = f["actions"]
    env = make("move_plate", camera_keys=())
    renderer = None
    normal = capture.prior
    steps = []
    try:
        ts = env.reset(seed=seed)
        inner = env.inner_env
        renderer = mujoco.Renderer(inner.model, height=84, width=84)
        cams = [
            next(
                i for i in range(inner.model.ncam) if inner.model.camera(i).name.split("/")[-1] == k
            )
            for k in DEFAULT_CAMERA_KEYS
        ]
        for i in range(prefix + 1):
            state = ts.low_dim_obs
            if capture.due(state):
                views = []
                for camera in cams:
                    renderer.update_scene(inner.data, camera=camera)
                    views.append(renderer.render().transpose(2, 0, 1))
                rgb = np.stack(views)
                guess = source({"state": state, "rgb": rgb})["plate_rel"]
                normal = capture.observe(rgb, guess)
                steps.append(i)
            elif state[44] < 0.8:
                normal = capture.prior
            if i < prefix:
                ts = env.step(actions[i])
    finally:
        if renderer is not None:
            renderer.close()
        env.close()
    return dict(normal=normal.tolist(), samples=steps, visible=len(capture.measurements))


@app.function(
    image=image,
    cpu=1,
    memory=2304,
    volumes={"/runs": records, "/weights": weights},
    max_containers=2,
    timeout=1200,
    single_use_containers=True,
)
def compare(seed, digest):
    if not 3000700 <= seed < 3000850 or len(digest) != 64:
        raise ValueError("Expected the paired pickup generation block")
    input_file = Path(f"/runs/astra-integration/grasp-transfer/v1/{seed}.json")
    row = json.loads(input_file.read_text())
    head = Path("/weights/heads") / (digest + ".npz")
    if any(
        hashlib.sha256(path.read_bytes()).hexdigest() != sha
        for path, sha in ((head, digest), (Path("/weights/est_v1b/last.pt"), WEIGHTS_SHA))
    ):
        raise ValueError("Unexpected capture weights")
    out = Path("/runs/astra-integration/lift-capture") / digest / f"{seed}.json"
    if out.exists():
        return json.loads(out.read_text())
    out.parent.mkdir(parents=True, exist_ok=True)
    t = time.monotonic()
    if row["candidate"]["secure"]:
        args = repr((seed, row["archive"], row["candidate"]["prefix"], str(head)))
        src = (
            "from pathlib import Path\nimport json\n"
            + inspect.getsource(estimate)
            + "\n"
            + inspect.getsource(continuation)
            + f"\nm=estimate(*{args})\n"
            + f"r=continuation({seed},{row['archive']!r},m['normal'])\n"
            + "print(json.dumps([m,r]))"
        )
        proc = subprocess.run(
            ["/root/policy/bin/python", "-c", src],
            cwd="/root/code",
            env={
                **os.environ,
                "MUJOCO_GL": "osmesa",
                "PYOPENGL_PLATFORM": "osmesa",
                "LP_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
            capture_output=True,
            text=True,
        )
        if proc.returncode:
            raise RuntimeError(proc.stderr[-2400:])
        measurement, result = json.loads(proc.stdout.splitlines()[-1])
    else:
        measurement, result = None, row["candidate"]
    prior = row["candidate"]
    if (prior["prefix"], prior["secure"]) != (result["prefix"], result["secure"]):
        raise ValueError("The paired continuation did not reach the same physical handoff")
    record = dict(
        seed=seed,
        archive=row["archive"],
        prior=prior,
        candidate=result,
        measurement=measurement,
        seconds=time.monotonic() - t,
    )
    out.write_text(json.dumps(record))
    records.commit()
    return record


@app.function(cpu=0.125, memory=256, timeout=86400, volumes={"/runs": records})
def schedule(seeds, digest, smoke_id):
    first = modal.FunctionCall.from_id(smoke_id).get()
    if len(seeds) != 30 or len(set(seeds)) != 30 or sorted(seeds)[0] != first["seed"]:
        raise ValueError("Expected the original 30 paired generation seeds and first-seed smoke")
    if first["measurement"] is None or first["measurement"]["visible"] < 3:
        raise ValueError("The first physical handoff did not produce usable RGB measurements")
    folder = Path("/runs/astra-integration/lift-capture") / digest
    source = folder.parent.parent / "grasp-transfer/v1"
    ids = {first["seed"]: smoke_id}
    records.reload()
    if (folder / "calls.json").exists():
        raise ValueError("Existing capture study")
    while len(ids) < len(seeds):
        records.reload()
        for s in seeds:
            if s not in ids and (source / f"{s}.json").is_file():
                ids[s] = compare.spawn(s, digest).object_id
        (folder / "calls.json").write_text(json.dumps(ids))
        records.commit()
        if len(ids) < len(seeds):
            time.sleep(45)
    results = [modal.FunctionCall.from_id(c).get() for c in ids.values()]
    (folder / "complete.json").write_text(json.dumps(results))
    records.commit()
    return len(results)
