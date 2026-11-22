"""Ablate continuous plate XYZ feedback during physical transfers from recorded public lifts."""

import hashlib
import inspect
import json
import os
import subprocess
import time
from pathlib import Path

import modal

from ops_modular.astra_grasp_transfer import continuation
from ops_modular.astra_site_transfer import SITE_SHA, image, records, weights
from ops_modular.modular_sim import WEIGHTS_SHA

app = modal.App("modular-astra-plate-transfer")


class ObservedPlate:
    def __init__(self, env, head, corrected):
        import mujoco
        import numpy as np
        from bigym.loco.config import DEFAULT_CAMERA_KEYS

        from submission.estimator import Estimator

        self.inner = env.inner_env
        self.renderer = mujoco.Renderer(self.inner.model, height=84, width=84)
        self.cameras = [
            next(
                i
                for i in range(self.inner.model.ncam)
                if self.inner.model.camera(i).name.split("/")[-1] == k
            )
            for k in DEFAULT_CAMERA_KEYS
        ]
        self.source = Estimator("/weights/est_v1b/last.pt", device="cpu")

        def capture(module, args):
            self.features = args[0][0].detach().cpu().numpy()

        self.source.model.head[0].register_forward_pre_hook(capture)
        with np.load(head, allow_pickle=False) as w:
            self.weights = {k: w[k] for k in w.files if k.startswith("plate_")}
        self.corrected = corrected
        self.errors = {}

    def __call__(self, state, truth, phase):
        import numpy as np

        from method.modular.kinematics import pelvis_rotation
        from method.modular.site_estimator import plate_features, predict

        views = []
        for camera in self.cameras:
            self.renderer.update_scene(self.inner.data, camera=camera)
            views.append(self.renderer.render().transpose(2, 0, 1))
        raw = self.source({"state": state, "rgb": np.stack(views)})
        rel = raw["plate_rel"].astype(float)
        if self.corrected:
            x = plate_features(self.features, state, rel, raw["target_site_rel"])
            rel += predict(self.weights, "plate", x)
        world = state[18:21] + pelvis_rotation(state) @ rel
        self.errors.setdefault(phase.value, []).append(world - truth["plate_xyz"])
        delta = world - truth["site_xyz"]
        return {
            **truth,
            "plate_xyz": world,
            "plate_rel": rel,
            "site_distance": float(np.linalg.norm(delta)),
            "site_xy_distance": float(np.linalg.norm(delta[:2])),
        }

    def stats(self):
        import numpy as np

        out = {}
        for phase, values in self.errors.items():
            e = 100 * np.array(values)
            out[phase] = dict(
                frames=len(e),
                bias_cm=e.mean(0).tolist(),
                p95_cm=float(np.percentile(np.linalg.norm(e, axis=1), 95)),
            )
        return out

    def close(self):
        self.renderer.close()


@app.function(
    image=image,
    cpu=2,
    memory=3072,
    timeout=7200,
    max_containers=2,
    single_use_containers=True,
    volumes={"/runs": records, "/weights": weights},
)
def episode(seed, arm):
    if not 3000700 <= seed < 3000850 or arm not in ("raw", "corrected"):
        raise ValueError("Expected a paired generation transfer")
    records.reload()
    row = json.loads(Path(f"/runs/astra-integration/grasp-transfer/v1/{seed}.json").read_text())
    folder = Path("/runs/astra-integration/plate-transfer")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{seed}_{arm}.json"
    if path.exists():
        return json.loads(path.read_text())
    head = Path("/weights/heads") / (SITE_SHA + ".npz")
    if (
        hashlib.sha256(head.read_bytes()).hexdigest() != SITE_SHA
        or hashlib.sha256(Path("/weights/est_v1b/last.pt").read_bytes()).hexdigest() != WEIGHTS_SHA
    ):
        raise ValueError("Unexpected estimator or residual")
    src = (
        "from pathlib import Path\nimport json\n"
        + inspect.getsource(ObservedPlate)
        + "\n"
        + inspect.getsource(continuation)
        + f"\nob=lambda env: ObservedPlate(env,{str(head)!r},{arm == 'corrected'!r})\n"
        + f"print(json.dumps(continuation({seed},{row['archive']!r},{row['vector']!r},observe_plate=ob)))"
    )
    start = time.monotonic()
    proc = subprocess.run(
        ["/root/policy/bin/python", "-c", src],
        cwd="/root/code",
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "MUJOCO_GL": "osmesa",
            "PYOPENGL_PLATFORM": "osmesa",
            "LP_NUM_THREADS": "2",
            "OMP_NUM_THREADS": "1",
        },
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr[-2300:])
    result = json.loads(proc.stdout.splitlines()[-1])
    if (result["secure"], result["prefix"]) != (
        row["candidate"]["secure"],
        row["candidate"]["prefix"],
    ):
        raise ValueError("Mismatched physical prefix")
    record = dict(seed=seed, arm=arm, **result, seconds=time.monotonic() - start)
    path.write_text(json.dumps(record))
    records.commit()
    return record


@app.function(cpu=0.125, memory=256, timeout=86400, volumes={"/runs": records})
def schedule(seeds, smoke):
    if len(set(seeds)) != 6 or seeds[0] != 3000700 or set(smoke) != {"raw", "corrected"}:
        raise ValueError("Expected one paired smoke followed by five distinct slots")
    rows = [modal.FunctionCall.from_id(c).get() for c in smoke.values()]
    if not all(
        z["seed"] == seeds[0] and z["secure"] and "LOWER" in z["position_errors"] for z in rows
    ):
        raise ValueError("Smoke did not measure both placement feedback arms")
    folder = Path("/runs/astra-integration/plate-transfer")
    if (folder / "calls.json").exists():
        raise ValueError("Existing placement feedback experiment")
    calls = {f"{seeds[0]}_{k}": v for k, v in smoke.items()}
    for seed in seeds[1:]:
        for arm in smoke:
            calls[f"{seed}_{arm}"] = episode.spawn(seed, arm).object_id
        (folder / "calls.json").write_text(json.dumps(calls))
        records.commit()
    results = [modal.FunctionCall.from_id(cid).get() for cid in calls.values()]
    (folder / "complete.json").write_text(json.dumps(results))
    records.commit()
    return len(results)
