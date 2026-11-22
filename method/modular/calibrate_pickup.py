"""Measure wrist rim projection error on physically reached, labeled training poses."""

import argparse
import json

import mujoco
import numpy as np

from .kinematics import G1Kinematics, pelvis_rotation
from .phases import Phase
from .pickup import PickupController, rim_lateral
from .privileged import PrivilegedTaskState


def measure(start, count, vary_tilt=False):
    from bigym.loco import make

    if not 3_052_000 <= start < start + count <= 3_055_000:
        raise ValueError("Use the source calibration seed block")
    controller = PickupController(G1Kinematics())
    env = make("move_plate", camera_keys=())
    renderer = None
    errors, unreached = [], 0
    try:
        for seed in range(start, start + count):
            ts = env.reset(seed=seed)
            inner = env.inner_env
            if renderer is None:
                renderer = mujoco.Renderer(inner.model, height=84, width=84)
                (cam,) = [
                    i for i in range(inner.model.ncam) if "left_wrist" in inner.model.camera(i).name
                ]
            labels = PrivilegedTaskState(env)
            controller.reset()
            # Perturb the controller's position input to move the wrist to
            # different standoffs. Every plate and arm motion comes from env.step.
            rng = np.random.default_rng(seed)
            bias = rng.uniform([-0.025, -0.035, -0.002], [0.025, 0.035, 0.002])
            angle = np.deg2rad(rng.uniform(-9, 9)) if vary_tilt else 0.0
            c, s = np.cos(angle), np.sin(angle)
            tilt = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
            for _ in range(160):
                state = np.asarray(ts.low_dim_obs)
                truth = labels(state)
                estimated = truth["plate_xyz"] + bias
                if controller.phase == Phase.GRASP and controller.machine.elapsed == 4:
                    renderer.update_scene(inner.data, camera=cam)
                    wrist = renderer.render().transpose(2, 0, 1)
                    controller.kin.set_state(state)
                    pos, rot = controller.kin.hand_pose("left")
                    basis = pelvis_rotation(state)
                    y = rim_lateral(wrist, state[18:21] + basis @ pos, basis @ rot, estimated[2])
                    if y is not None:
                        errors.append(y - truth["plate_xyz"][1])
                    break
                task = {**truth, "plate_xyz": estimated, "plate_rot": tilt @ truth["plate_rot"]}
                ts = env.step(controller.act(state, task))
            else:
                unreached += 1
    finally:
        if renderer is not None:
            renderer.close()
        env.close()
    result = dict(start=start, count=count, measured=len(errors), unreached=unreached)
    if errors:
        result["mean_signed_m"] = float(np.mean(errors))
        result["absolute_p50_p95_m"] = np.percentile(np.abs(errors), [50, 95]).tolist()
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("start", type=int)
    p.add_argument("count", type=int)
    p.add_argument("--vary-tilt", action="store_true")
    a = p.parse_args()
    print(json.dumps(measure(a.start, a.count, a.vary_tilt)))
