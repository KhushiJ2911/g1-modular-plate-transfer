"""Extra placement-event measurements in the same pass as the lead's failure replay."""

import csv
import json
import runpy
import sys
import tempfile
from pathlib import Path

import numpy as np


def measure(folder, output, episode_index=0):
    import bigym.loco

    sys.path.insert(0, "/home/keyu/advaitic/wt-teacher")
    from method.teacher.diagnostics import PrivilegedSensors

    report = json.loads((folder / "report.json").read_text())
    official = report["episodes"][episode_index]
    if official["episode"] != episode_index or not official["completed"]:
        raise ValueError("Expected a completed episode with its original evaluator index")
    actions = folder / f"episode_{episode_index:03}.actions.npz"
    with np.load(actions, allow_pickle=False) as f:
        commands = f["actions"][:, 18]
    snapshots = []
    original = PrivilegedSensors.snapshot

    def observe(sensors, step):
        s = original(sensors, step)
        snapshots.append(
            dict(
                step=step,
                z=s["pose"][2],
                held=bool(s["held"]),
                dist=s["site_distance"],
                xy=s["site_xy_distance"],
                tilt=s["tilt_deg"],
                rack=bool(s["contacts"]["target_rack"]),
            )
        )
        return s

    envs = []
    factory = bigym.loco.make

    def capture(*args, **kwargs):
        env = factory(*args, **kwargs)
        envs.append(env)
        return env

    PrivilegedSensors.snapshot = observe
    bigym.loco.make = capture
    previous_args = sys.argv
    try:
        # The upstream analysis iterates every CSV row. Give it the original
        # row and archive for this episode so successes need no extra replay.
        with tempfile.TemporaryDirectory() as d:
            selected = Path(d)
            with (folder / "episodes.csv").open(newline="") as stream:
                rows = [r for r in csv.DictReader(stream) if int(r["episode"]) == episode_index]
            if len(rows) != 1 or any(
                int(rows[0][key]) != official[key] for key in ("seed", "source_slot", "steps")
            ):
                raise ValueError("Official report and episode CSV differ")
            with (selected / "episodes.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            (selected / actions.name).symlink_to(actions.resolve())
            sys.argv = ["/root/analysis/replay_eval.py", str(selected), str(output)]
            runpy.run_path(sys.argv[0], run_name="__main__")
        r = json.loads(output.read_text())
        env = envs[0]
        if (
            len(commands) != official["steps"]
            or bool(env.episode_succeeded()) != official["success"]
            or bool(env.episode_fell()) != official["fell"]
            or str(env.episode_termination()) != official["termination"]
        ):
            raise ValueError("Camera-free replay did not reproduce the official terminal outcome")
    finally:
        for env in envs:
            env.close()
        PrivilegedSensors.snapshot = original
        bigym.loco.make = factory
        sys.argv = previous_args
    assert snapshots[0]["step"] == 0 and len(snapshots) == len(commands) + 2
    # replay_eval repeats the final snapshot for its endpoint summary.
    frames = snapshots[:-1]
    origin = frames[0]["z"]
    near = next(
        (s["step"] for s in frames if s["held"] and s["z"] > origin + 0.03 and s["xy"] < 0.06), None
    )
    r["target_approach_step"] = near
    release = (
        None
        if near is None
        else next((i for i in range(near, len(commands)) if commands[i] < 0), None)
    )
    r["open_at_target"] = None if release is None else frames[release]
    if release is not None:
        opening_end = next(
            (i for i in range(release + 1, len(commands)) if commands[i] >= 0), len(commands)
        )
        cleared = next((s for s in frames[release : opening_end + 1] if not s["held"]), None)
        r["open_steps"] = opening_end - release
        r["true_release_delay_steps"] = None if cleared is None else cleared["step"] - release
        r["reacquired_after_open"] = bool(
            cleared and any(s["held"] for s in frames[cleared["step"] + 1 :])
        )
    output.write_text(json.dumps(r))


if __name__ == "__main__":
    measure(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 0)
