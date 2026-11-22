"""Matched placement hybrid. Only the source object's pickup measurements are privileged."""

import argparse
import importlib
import json
import time

from method.modular.phases import Phase


def episode(seed, arm):
    from advaitic_eval.environment import (
        make_environment,
        observation,
        observed_source_slot,
        validate_actions,
    )
    from method.modular.input_ablation import _first_failure
    from method.modular.privileged import PrivilegedTaskState

    if not 3000800 <= seed < 3001000:
        raise ValueError("Fresh integration hybrid seeds only")
    factory = {
        "base": "submission.policy_modular_heads:make_policy",
        "site": "submission.policy_modular_integrated:make_site",
        "plate": "submission.policy_modular_integrated:make_plate",
        "both": "submission.policy_modular_integrated:make_policy",
    }[arm]
    mod, name = factory.split(":")
    p = getattr(importlib.import_module(mod), name)()
    env = make_environment()
    try:
        t = time.monotonic()
        ts = env.reset(seed=seed)
        p.reset()
        src = PrivilegedTaskState(env)
        slot = observed_source_slot(env)
        steps = 0
        carried = False
        while not ts.last():
            obs = observation(ts)
            task = p.task.update(obs)
            phase = p.controller.phase
            if phase in (Phase.APPROACH, Phase.PREGRASP, Phase.GRASP, Phase.LIFT):
                truth = src(obs["state"])
                task = {
                    **task,
                    **{
                        k: truth[k]
                        for k in (
                            "plate_xyz",
                            "plate_rel",
                            "plate_rot",
                            "plate_rot_rel",
                            "held",
                            "tilt_deg",
                        )
                    },
                    "contacts": {
                        **task["contacts"],
                        "source_rack": truth["contacts"]["source_rack"],
                    },
                }
            elif phase == Phase.CARRY:
                carried = True
            a = validate_actions(p.controller.act(obs["state"], task))[0]
            ts = env.step(a)
            steps += 1
        success = bool(env.episode_succeeded())
        return dict(
            seed=seed,
            arm=arm,
            slot=slot,
            steps=steps,
            success=success,
            fell=bool(env.episode_fell()),
            carried=carried,
            phase=_first_failure(p.controller.transitions, p.controller.phase.value, success),
            seconds=time.monotonic() - t,
        )
    finally:
        env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("seed", type=int)
    p.add_argument("arm", choices=["base", "site", "plate", "both"])
    a = p.parse_args()
    print(json.dumps(episode(a.seed, a.arm)))
