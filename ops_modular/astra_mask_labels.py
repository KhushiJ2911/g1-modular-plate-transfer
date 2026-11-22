"""Exact-action generation replay for offline plate silhouette annotations."""

import inspect
import json
import time
from pathlib import Path

import modal
from ops_modular.modular_sim import image

app = modal.App("modular-astra-mask-labels")
data = modal.Volume.from_name("modular-data")


def sample(seed):
    import mujoco
    import numpy as np
    from bigym.loco import make
    from method.modular.kinematics import G1Kinematics

    with np.load(f"/data/astra-dagger/sol/{seed}.npz") as f:
        phase, truth, state, baseline, steps, actions = (
            f[k] for k in ("phase", "truth", "state", "baseline", "step", "actions")
        )
    chosen = {}
    for ph in ("LIFT", "CARRY", "ALIGN", "LOWER", "RELEASE", "SETTLE"):
        idx = np.flatnonzero(phase == ph)
        if len(idx):
            for j in np.unique(np.linspace(0, len(idx)-1, min(7,len(idx)), dtype=int)):
                chosen[int(steps[idx[j]])] = int(idx[j])
    out = Path(f"/data/astra-integration/masks/{seed}.npz")
    out.parent.mkdir(exist_ok=True, parents=True)
    if not chosen:
        np.savez_compressed(out, step=np.array([], dtype=int))
        return dict(seed=seed, samples=0)
    kin = G1Kinematics()
    env = make("move_plate", camera_keys=())
    renderer = None
    images, masks, ids, camera, hand, centers, normals = [], [], [], [], [], [], []
    try:
        ts = env.reset(seed=seed)
        inner = env.inner_env
        renderer = mujoco.Renderer(inner.model, height=84, width=84)
        cams = [next(i for i in range(inner.model.ncam) if inner.model.camera(i).name.split("/")[-1] == k) for k in ("head","left_wrist")]
        objects = [inner.model.geom(g.name).id for g in inner.plates[0].geoms]
        for t in range(max(chosen)+1):
            if t in chosen:
                i = chosen[t]
                np.testing.assert_array_equal(ts.low_dim_obs, state[i])
                kin.set_state(state[i])
                p, R = kin.hand_pose("left")
                torso = kin.data.body("torso_link")
                hand.append(np.r_[R.T @ (torso.xpos - p), (R.T @ torso.xmat.reshape(3,3)).ravel()])
                centers.append(R.T @ (truth[i,:3]-p))
                normals.append(R.T @ np.cross(truth[i,8:11],truth[i,11:14]))
                ids.append(i)
                rgb, mask = [], []
                for cam in cams:
                    renderer.disable_segmentation_rendering()
                    renderer.update_scene(inner.data, camera=cam)
                    rgb.append(renderer.render().copy())
                    renderer.enable_segmentation_rendering()
                    segmentation = renderer.render()
                    mask.append((segmentation[...,1]==mujoco.mjtObj.mjOBJ_GEOM) & np.isin(segmentation[...,0],objects))
                images.append(rgb)
                masks.append(mask)
                # The true camera frame is only for auditing FK-related measurement error.
                camera.append(np.r_[inner.data.cam_xpos[cams].ravel(),inner.data.cam_xmat[cams].ravel()])
            if t != max(chosen):
                ts = env.step(actions[t])
        near = float(inner.model.vis.map.znear*inner.model.stat.extent)
    finally:
        if renderer is not None:
            renderer.close()
        env.close()
    np.savez_compressed(out, step=steps[ids], phase=phase[ids], state=state[ids], baseline=baseline[ids], truth=truth[ids], torso=hand,
                        center=centers, normal=normals, camera=camera, rgb=images, mask=masks, near=near)
    return dict(seed=seed,samples=len(ids),near=near,cam=cams,objects=objects,visible=np.sum(masks,axis=(0,2,3)).tolist())


@app.function(image=image,cpu=1,memory=1800,timeout=1800,max_containers=4,single_use_containers=True,volumes={"/data":data})
def episode(seed):
    import os
    import subprocess
    if not 3110000 <= seed < 3110300:
        raise ValueError("Use recorded DAgger generation seeds")
    t=time.monotonic()
    src="from pathlib import Path\nimport json\n"+inspect.getsource(sample)+f"\nprint(json.dumps(sample({seed})))"
    p = subprocess.run(["/root/policy/bin/python","-c",src],cwd="/root/code",env={**os.environ,"MUJOCO_GL":"osmesa","PYOPENGL_PLATFORM":"osmesa","LP_NUM_THREADS":"1"},capture_output=True,text=True)
    if p.returncode:
        raise ValueError(p.stderr[-2500:])
    result=json.loads(p.stdout.splitlines()[-1])
    data.commit()
    return dict(**result, seconds=time.monotonic()-t)
