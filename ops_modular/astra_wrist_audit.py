"""Measure the geometry visible in the loaded wrist on archived generation observations."""

import inspect
import json
import subprocess
from pathlib import Path

import modal

from ops_modular.modular_sim import image

app = modal.App("modular-astra-loaded-wrist")
data = modal.Volume.from_name("modular-data")


def gather():
    import numpy as np

    from method.modular.kinematics import G1Kinematics

    kin = G1Kinematics()
    sample = {
        key: []
        for key in (
            "seed",
            "phase",
            "rgb",
            "head_rgb",
            "normal",
            "center",
            "estimated_center",
            "torso_pos",
            "torso_rot",
            "finger",
        )
    }
    for path in sorted(Path("/data/est_v2/onpolicy").glob("*.npz")):
        seed = int(path.stem)
        if not 3100000 <= seed < 3100300:
            continue
        with np.load(path) as f:
            truth, phase, state, baseline = f["truth"], f["phase"], f["state"], f["baseline"]
            photos = None
            for name in ("CARRY", "ALIGN", "LOWER"):
                indices = np.flatnonzero(
                    (phase == name) & (truth[:, 7] > 0.5) & (state[:, 44] > 0.8)
                )
                if len(indices) == 0:
                    continue
                if photos is None:
                    photos = f["rgb"]
                for i in indices[np.unique(np.linspace(0, len(indices) - 1, 5, dtype=int))]:
                    kin.set_state(state[i])
                    p, rot = kin.hand_pose("left")
                    sample["seed"].append(seed)
                    sample["phase"].append(name)
                    sample["rgb"].append(photos[i, 2])
                    sample["head_rgb"].append(photos[i, 0])
                    torso = kin.data.body("torso_link")
                    sample["torso_pos"].append(rot.T @ (torso.xpos - p))
                    sample["torso_rot"].append(rot.T @ torso.xmat.reshape(3, 3))
                    sample["estimated_center"].append(rot.T @ (baseline[i, :3] - p))
                    sample["normal"].append(rot.T @ np.cross(truth[i, 8:11], truth[i, 11:14]))
                    sample["center"].append(rot.T @ (truth[i, :3] - p))
                    sample["finger"].append(state[i, 7:9])
    dest = Path("/data/astra-integration/loaded_views.npz")
    np.savez_compressed(dest, **sample)
    return dict(
        seeds=len(set(sample["seed"])), images=len(sample["seed"]), bytes=dest.stat().st_size
    )


@app.function(image=image, cpu=1, memory=2048, timeout=1800, volumes={"/data": data})
def extract():
    cmd = (
        "import json\nfrom pathlib import Path\n"
        + inspect.getsource(gather)
        + "\nprint(json.dumps(gather()))"
    )
    p = subprocess.run(
        ["/root/policy/bin/python", "-c", cmd], cwd="/root/code", capture_output=True, text=True
    )
    if p.returncode:
        raise ValueError(p.stderr[-2000:])
    data.commit()
    return json.loads(p.stdout.splitlines()[-1])
