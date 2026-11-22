"""Camera-free orientation hybrid for the combined grip and FK-finished release."""

import argparse
import json
import multiprocessing
import time
from pathlib import Path


def compare(seed):
    import numpy as np
    from bigym.loco import make

    from method.modular.contact_perception import PerceivedRackContact
    from method.modular.input_ablation import _first_failure
    from method.modular.kinematics import G1Kinematics
    from method.modular.pickup import PickupController
    from method.modular.privileged import PrivilegedTaskState
    from submission.policy_modular_integrated import PickupAndRelease

    env = make("move_plate", camera_keys=())
    rows = []
    try:
        for label, cls in (("pickup", PickupController), ("pickup_fk_release", PickupAndRelease)):
            ts = env.reset(seed=seed)
            ctl = cls(G1Kinematics(), None)
            ctl.site_contacts = PerceivedRackContact(ctl.site_contacts)
            truth = PrivilegedTaskState(env)
            begin = time.monotonic()
            while not ts.last():
                state = np.asarray(ts.low_dim_obs)
                task = truth(state)
                # True positions/contacts/unsigned tilt in both arms. Their
                # actual pickup verification and held-phase orientation run on
                # measured slides and FK, as they do inside the public policy.
                ts = env.step(ctl.act(state, task))
            ok = bool(env.episode_succeeded())
            rows.append(
                dict(
                    seed=seed,
                    arm=label,
                    success=ok,
                    fell=bool(env.episode_fell()),
                    steps=ctl.machine.step,
                    seconds=time.monotonic() - begin,
                    first_failure=_first_failure(ctl.transitions, ctl.phase.value, ok),
                )
            )
        return rows
    finally:
        env.close()


if __name__ == "__main__":
    from advaitic_eval.protocol import make_plan

    p = argparse.ArgumentParser()
    p.add_argument("start", type=int)
    p.add_argument("n", type=int, choices=(1, 30))
    p.add_argument("out", type=Path)
    a = p.parse_args()
    if not 3000900 <= a.start <= 3000910:
        raise ValueError("Use fresh diagnostic block")
    seeds = [a.start] if a.n == 1 else [e.seed for e in make_plan(a.n, a.start)]
    with (
        a.out.open("x") as f,
        multiprocessing.get_context("spawn").Pool(min(3, len(seeds))) as pool,
    ):
        for pair in pool.imap_unordered(compare, seeds):
            for r in pair:
                print(json.dumps(r), file=f, flush=True)
