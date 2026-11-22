"""Recover the observed pinch offset from public generation actions and test rigid center tracking."""

import hashlib
import inspect
import json
import os
import subprocess
import time
from pathlib import Path

import modal

from ops_modular.astra_grasp_transfer import continuation
from ops_modular.astra_plate_transfer import ObservedPlate
from ops_modular.astra_site_transfer import SITE_SHA, image, records, weights
from ops_modular.modular_sim import WEIGHTS_SHA

app = modal.App("modular-astra-pinch-transfer")


def recover(seed, archive, prefix):
    import numpy as np

    from advaitic_eval.environment import make_environment, observation, validate_actions
    from submission.policy_modular_heads import make_policy

    policy = make_policy()
    with np.load(Path(archive) / "episode_000.actions.npz") as f:
        actions = f["actions"]
    env = make_environment()
    error = 0.0
    try:
        ts = env.reset(seed=seed)
        policy.reset()
        for command in actions[:prefix]:
            estimated = validate_actions(policy(observation(ts)))[0]
            error = max(error, float(abs(estimated - command).max()))
            if error > 1e-5:
                raise ValueError(f"Public pickup reexecution differs by {error}")
            ts = env.step(command)
        offset = policy.controller.hand_to_center
        if offset is None:
            raise ValueError("No confirmed public pinch at the secure prefix")
        return offset.copy(), error
    finally:
        env.close()


class ObservedPinch(ObservedPlate):
    def __init__(self, env, head, offset, bias):
        from method.modular.kinematics import G1Kinematics

        super().__init__(env, head, False)
        self.kin = G1Kinematics()
        self.offset = offset
        self.bias = bias

    def __call__(self, state, truth, phase):
        import numpy as np

        from method.modular.kinematics import pelvis_rotation
        from method.modular.site_estimator import HELD_PLACEMENT, grasp_center_rel

        if phase.value not in HELD_PLACEMENT or not truth["held"] or state[44] < 0.8:
            return super().__call__(state, truth, phase)
        rel = grasp_center_rel(self.kin, state, self.offset, self.bias)
        world = state[18:21] + pelvis_rotation(state) @ rel
        self.errors.setdefault(phase.value, []).append(world - truth["plate_xyz"])
        delta = world - truth["site_xyz"]
        return {
            **truth,
            "plate_rel": rel,
            "plate_xyz": world,
            "site_distance": float(np.linalg.norm(delta)),
            "site_xy_distance": float(np.linalg.norm(delta[:2])),
        }


@app.function(
    image=image,
    cpu=2,
    memory=3072,
    timeout=7200,
    max_containers=2,
    volumes={"/runs": records, "/weights": weights},
    single_use_containers=True,
)
def episode(seed, digest):
    if not 3000700 <= seed < 3000850 or len(digest) != 64:
        raise ValueError("Expected the independent pickup block and calibrated weights")
    records.reload()
    row = json.loads(Path(f"/runs/astra-integration/grasp-transfer/v1/{seed}.json").read_text())
    folder = Path("/runs/astra-integration/pinch-transfer") / digest
    dest = folder / f"{seed}.json"
    if dest.exists():
        return json.loads(dest.read_text())
    head = Path("/weights/heads") / (digest + ".npz")
    visual = Path("/weights/heads") / (SITE_SHA + ".npz")
    estimator = Path("/weights/est_v1b/last.pt")
    if any(
        hashlib.sha256(p.read_bytes()).hexdigest() != d
        for p, d in [(head, digest), (visual, SITE_SHA), (estimator, WEIGHTS_SHA)]
    ):
        raise ValueError("Wrong public policy weights")
    args = repr((seed, row["archive"], row["candidate"]["prefix"]))
    src = (
        "from pathlib import Path\nimport json,numpy as np\n"
        + "\n".join(map(inspect.getsource, [recover, ObservedPlate, ObservedPinch, continuation]))
        + f"\noffset,error=recover(*{args})\nwith np.load({str(head)!r}) as ck: bias=ck['pinch_center_bias']\n"
        + f"obs=lambda env: ObservedPinch(env,{str(visual)!r},offset,bias)\n"
        + f"r=continuation({seed},{row['archive']!r},{row['vector']!r},observe_plate=obs)\n"
        + "print(json.dumps(dict(**r,prefix_action_error=error,public_offset=offset.tolist())))"
    )
    start = time.monotonic()
    p = subprocess.run(
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
            "LP_NUM_THREADS": "2",
            "OMP_NUM_THREADS": "1",
        },
    )
    if p.returncode:
        raise RuntimeError(p.stderr[-2000:])
    z = json.loads(p.stdout.splitlines()[-1])
    if (z["prefix"], z["secure"]) != (row["candidate"]["prefix"], row["candidate"]["secure"]):
        raise ValueError("Different physical pickup")
    out = dict(seed=seed, **z, seconds=time.monotonic() - start)
    folder.mkdir(exist_ok=True, parents=True)
    dest.write_text(json.dumps(out))
    records.commit()
    return out
