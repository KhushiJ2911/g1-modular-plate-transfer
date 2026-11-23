"""Conditional short continuation from a saved strict release on a Brief 21 dev seed.

Replay the saved prefix, then continue with the guarded policy. Rebuild the BC ensemble from recent
public observations and the supervisor's release-decision history from stderr. Require
the off control to reproduce the opening action before using the on branch as evidence.
This is an offline intervention, not an evaluation of the policy from reset.
"""

import argparse
import csv
import gzip
import json
import os
import re
import sys
from pathlib import Path

os.environ["ORT_DISABLE_TELEMETRY"] = "1"
import mujoco
import numpy as np
import onnxruntime

onnxruntime.disable_telemetry_events()
from bigym.loco import make  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "wt-harness"))
sys.path.append(str(BASE / "mlenv/lib/python3.12/site-packages"))
from submission.estimator import Estimator  # noqa: E402
from submission.harness import ARM_HIGH, ARM_LOW, ReleaseSupervisor  # noqa: E402
from submission.policy import BCPolicy  # noqa: E402


def at_open(config, t, entry, stable, prev_action, guard):
    sup = ReleaseSupervisor({**config, "release_height_guard": guard}, log=False)
    cadence = config["exec_steps"]
    sup.phase, sup.phase_since, sup.time, sup.last_time = "PLACE", entry, t, t - cadence
    sup.streaks = {"safe": (True, stable - cadence), "seat": (True, config["seat_steps"] - cadence)}
    sup.place_wrist_target = ARM_LOW[5] + (1 + prev_action[9]) / 2 * (ARM_HIGH[5] - ARM_LOW[5])
    return sup


def probe(env, renderer, bc, est, run, seed, traces, live_limit):
    if run not in [f"dev_s2_st10_r{i}" for i in (1, 2, 3)] or not 1000000 <= seed < 1001000:
        raise ValueError("Only the three Brief 21 baseline dev runs are supported")
    root = BASE / "results/modal/evals" / run
    with (root / "episodes.csv").open() as f:
        row = next(r for r in csv.DictReader(f) if int(r["seed"]) == seed)
    config = json.loads(
        (root / "policy.stderr.log").read_text().splitlines()[0].removeprefix("[harness config] ")
    )
    with gzip.open(traces / f"{run}_{seed}.csv.gz", "rt") as f:
        events = [r for r in csv.DictReader(f) if r["reason"]]
    decision = next(r for r in events if r["phase"] == "RELEASE")
    t0 = int(decision["t"])
    entry = max(int(r["t"]) for r in events if r["phase"] == "PLACE" and int(r["t"]) < t0)
    stable = int(re.search(r"stable=(\d+)", decision["reason"])[1])
    with np.load(root / f"episode_{int(row['episode']):03}.actions.npz") as archive:
        saved = archive["actions"]
    enabled = at_open(config, t0, entry, stable, saved[t0 - 1], True)
    control = at_open(config, t0, entry, stable, saved[t0 - 1], False)
    bc.reset()
    bc.exec_steps, bc.temporal_ensemble, bc.ensemble_decay = config["exec_steps"], True, 0.0
    ts = env.reset(seed=seed)
    inner = env.inner_env
    cameras = [inner.cameras_map[k][0] for k in env.official_config.camera_keys]

    def observation():
        rgb = []
        for cam in cameras:
            renderer.update_scene(inner.data, cam)
            rgb.append(np.moveaxis(renderer.render(), -1, 0))
        return {"rgb": np.stack(rgb), "state": ts.low_dim_obs}

    for t in range(t0):
        if t > t0 - bc.horizon and (t0 - t) % bc.exec_steps == 0:
            bc(observation())
        ts = env.step(saved[t])
        if ts.last():
            raise RuntimeError("Prefix terminated before recorded release")
    t = t0
    rec = dict(run=run, seed=seed, original_success=row["success"] == "True", original_open=t0)
    while not ts.last() and t - t0 <= live_limit:
        if enabled.release_time is None:
            obs = observation()
            proposal, estimate = bc(obs), est(obs)
            if t == t0:
                off = control.step(estimate, obs["state"], proposal)
                rec["est_z_cm"] = float(
                    100 * (estimate["plate_rel"] - estimate["target_site_rel"])[2]
                )
                rec["control_open"] = control.phase == "RELEASE"
                rec["max_control_action_error"] = float(abs(off - saved[t : t + len(off)]).max())
                if not rec["control_open"] or rec["max_control_action_error"] > 0.0001:
                    return dict(rec, outcome="unmatched control")
            actions = enabled.step(estimate, obs["state"], proposal)
            if t == t0 and enabled.release_time == t0:
                return dict(rec, outcome="same opening", new_open=t0)
        else:
            # With this config every subsequent action equals the open/hold latch.
            actions = enabled.hold[None, :]
        for action in actions:
            ts = env.step(action)
            t += 1
            if ts.last():
                break
        if enabled.release_time is not None and t - enabled.release_time >= 120:
            break
    # Read terminal geometry only for the result record, after all actions have executed.
    sys.path.insert(0, str(BASE / "wt-teacher"))
    from method.teacher.diagnostics import PrivilegedSensors

    end = PrivilegedSensors(inner).snapshot(t)
    return dict(
        rec,
        end_cm=100 * end["site_distance"],
        end_deg=end["tilt_deg"],
        end_pad=end["held"],
        end_rack=end["contacts"]["target_rack"],
        new_open=enabled.release_time,
        end=t,
        outcome="success"
        if env.episode_succeeded() and not env.episode_fell()
        else "no success in window",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("traces", type=Path)
    p.add_argument("run")
    p.add_argument("seeds", type=int, nargs="+")
    p.add_argument("--limit", type=int, default=200)
    args = p.parse_args()
    bc = BCPolicy(BASE / "ckpts/student_v1b_bce1.pt")
    est = Estimator(BASE / "ckpts/est_v1b.pt", device=bc.device)
    env = make("move_plate", camera_keys=())
    with mujoco.Renderer(env.inner_env.model, *env.official_config.camera_shape) as renderer:
        try:
            for seed in args.seeds:
                print(
                    json.dumps(
                        probe(env, renderer, bc, est, args.run, seed, args.traces, args.limit)
                    ),
                    flush=True,
                )
        finally:
            env.close()


if __name__ == "__main__":
    main()
