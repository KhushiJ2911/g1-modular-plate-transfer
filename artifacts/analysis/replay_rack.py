"""Replay Brief 21 dev action logs. State in row t precedes action[t]; last row is terminal.

Usage: replay_rack.py EVAL_ROOT OUT_DIR [--estimate-every 200]
Use 1 for RGB estimates at every near-rack step or 2 for the policy cadence.
Completed traces are reused. Use a fresh output directory for a different cadence.
One gzipped CSV per failed episode and per successful counterpart of a paired flip.
Privileged quantities are recorded only by this offline analysis.
"""

import argparse
import csv
import gzip
import os
import re
import sys
from pathlib import Path

os.environ["ORT_DISABLE_TELEMETRY"] = "1"
import numpy as np

BASE = Path(__file__).resolve().parents[1]


def load_run(directory):
    with (directory / "episodes.csv").open() as f:
        rows = list(csv.DictReader(f))
    log = directory / "policy.stderr.log"
    transitions = re.findall(
        r"^\[harness t=(\d+)\] (\w+)>(\w+) (.*)$", log.read_text() if log.exists() else "", re.M
    )
    cursor = 0
    records = {}
    for row in rows:
        ep, seed = int(row["episode"]), int(row["seed"])
        if not 1000000 <= seed < 1001000:
            raise ValueError(f"Outside Brief 21 dev seeds: {seed}")
        with np.load(directory / f"episode_{ep:03}.actions.npz") as archive:
            actions = archive["actions"]
        if len(actions) != int(row["steps"]):
            raise ValueError(f"Incomplete action log: {directory} ep {ep}")
        events = {}
        phase = "APPROACH"
        if log.exists():
            for t, grip in enumerate(actions[:, 18]):
                # Every transition changes either phase or grip. Episode resets have no
                # log marker; use the recorded open/closed commands and phase chain.
                if cursor < len(transitions):
                    when, old, new, reason = transitions[cursor]
                    if t == int(when) and old == phase:
                        phase = new
                        events[t] = (phase, reason)
                        cursor += 1
                if (grip > 0) != (phase in {"GRASP", "CARRY", "PLACE"}):
                    raise ValueError(f"Phase/grip mismatch at {directory} ep {ep} t={t}")
        records[seed] = (row, actions, events, log.exists())
    if cursor != len(transitions):
        raise ValueError(f"Unmatched log transitions in {directory}")
    return records


def replay(env, renderer, record, out, estimate_every):
    from method.teacher.diagnostics import PrivilegedSensors

    frames, states, times = [], [], []
    row, actions, events, has_log = record
    ts = env.reset(seed=int(row["seed"]))
    inner = env.inner_env
    sensors = PrivilegedSensors(inner)
    phase = "APPROACH" if has_log else "unknown"
    cameras = [inner.cameras_map[k][0] for k in env.official_config.camera_keys]
    release = next((t for t, (ph, _) in events.items() if ph == "RELEASE"), None)
    release_samples = {release + dt for dt in (-10, 0, 10)} if release else set()
    first_contact = None
    tmp = out.with_suffix(".partial")
    with gzip.open(tmp, "wt", newline="") as f:
        writer = None
        for t in range(len(actions) + 1):
            phase, reason = events.get(t, (phase, ""))
            logged = re.search(r"d=([\d.]+) tilt=(-?[\d.]+) held=([\d.]+)", reason)
            s = sensors.snapshot(t)
            delta = np.array(s["pose"][:3]) - s["site_xyz"]
            state = ts.low_dim_obs
            rec = dict(
                t=t,
                phase=phase,
                reason=reason,
                slot=s["nearest_site"],
                dx=delta[0],
                dy=delta[1],
                dz=delta[2],
                distance=s["site_distance"],
                tilt=s["tilt_deg"],
                held=int(s["held"]),
                **s["contacts"],
                grip=actions[t, 18] if t < len(actions) else "",
                arm_speed=np.abs(state[22:29]).max(),
                base_speed=np.linalg.norm(state[40:42]),
                wrist_q=state[5],
                logged_d=logged[1] if logged else "",
                logged_tilt=logged[2] if logged else "",
                est_d="",
                est_dz="",
                est_tilt="",
                est_held="",
                replay_success=int(env.episode_succeeded()),
            )
            # Render the same three task cameras, with the standard per-camera RGB path.
            # Regular samples support time-weighted error; extra event samples describe decisions.
            if first_contact is None and s["contacts"]["target_rack"]:
                first_contact = t
            regular = t % estimate_every == 0 and s["site_distance"] <= 0.10
            # The active wrist correction holds all three base velocities at zero.
            probing = (
                phase == "PLACE"
                and t < len(actions)
                and t % 2 == 0
                and actions[t, 18] > 0
                and np.all(actions[t, [0, 1, 3]] == 0)
            )
            if (
                regular
                or probing
                or t == first_contact
                or t in release_samples
                or (logged and phase == "PLACE")
            ):
                rgb = []
                for camera in cameras:
                    renderer.update_scene(inner.data, camera)
                    rgb.append(np.moveaxis(renderer.render(), -1, 0))
                frames.append(np.stack(rgb))
                states.append(state.copy())
                times.append(t)
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(rec))
                writer.writeheader()
            writer.writerow(rec)
            if ts.last() or t == len(actions):
                break
            ts = env.step(actions[t])
    if frames:
        pending = out.with_suffix(".frames.partial.npz")
        np.savez_compressed(pending, t=times, rgb=frames, state=states)
        pending.replace(out.with_suffix(".frames.npz"))
    tmp.replace(out)
    print(
        out.name,
        "steps",
        t,
        "/",
        len(actions),
        "success",
        env.episode_succeeded(),
        "/",
        row["success"],
        flush=True,
    )


def score(directory):
    sys.path.insert(0, str(BASE / "wt-harness"))
    sys.path.append(str(BASE / "mlenv/lib/python3.12/site-packages"))
    from submission.estimator import Estimator

    estimator = Estimator(BASE / "ckpts/est_v1b.pt", device="cpu")
    for cache in sorted(directory.glob("*.csv.frames.npz")):
        out = cache.with_name(cache.name.removesuffix(".frames.npz") + ".gz")
        with gzip.open(out, "rt") as f:
            rows = list(csv.DictReader(f))
        with np.load(cache) as archive:
            for t, rgb, state in zip(archive["t"], archive["rgb"], archive["state"], strict=True):
                est = estimator({"rgb": rgb, "state": state})
                delta = est["plate_rel"] - est["target_site_rel"]
                rows[t].update(
                    est_d=np.linalg.norm(delta),
                    est_dz=delta[2],
                    est_tilt=est["tilt_deg"],
                    est_held=est["held"],
                )
        tmp = out.with_suffix(".partial")
        with gzip.open(tmp, "wt", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(out)
        cache.unlink()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--estimate-every", type=int, default=200, metavar="STEPS")
    p.add_argument("--score-only", action="store_true")
    args = p.parse_args()
    if args.estimate_every < 1:
        p.error("--estimate-every must be a positive control-step interval")
    if args.score_only:
        score(args.out)
        return

    import mujoco
    import onnxruntime

    onnxruntime.disable_telemetry_events()
    from bigym.loco import make

    sys.path.insert(0, str(BASE / "wt-teacher"))
    args.out.mkdir(exist_ok=True, parents=True)
    names = ["dev_s2_st10_r1", "dev_s2_st10_r2", "dev_s2_st10_r3", "dev_t_fix_r1", "dev_t_fix_r2"]
    runs = {name: load_run(args.root / name) for name in names}
    env = make("move_plate", camera_keys=())
    renderer = mujoco.Renderer(env.inner_env.model, *env.official_config.camera_shape)
    try:
        for name, records in runs.items():
            other = (
                name.replace("t_fix", "s2_st10")
                if "t_fix" in name
                else name.replace("s2_st10", "t_fix")
            )
            for seed, record in records.items():
                if record[0]["success"] == "True" and (
                    other not in runs or runs[other][seed][0]["success"] == "True"
                ):
                    continue
                out = args.out / f"{name}_{seed}.csv.gz"
                if not out.exists():
                    replay(env, renderer, record, out, args.estimate_every)
    finally:
        renderer.close()
        env.close()
    # Release the physics/GL process before loading torch on a memory-limited laptop.
    os.execv(
        sys.executable,
        [sys.executable, "-I", __file__, str(args.root), str(args.out), "--score-only"],
    )


if __name__ == "__main__":
    main()
