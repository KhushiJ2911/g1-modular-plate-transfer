"""Replay saved evaluator action logs (camera-free, same physics) and summarise the plate.

Analysis only. Usage: replay_eval.py RESULT_DIR OUT.jsonl
Does not modify the environment; reads privileged state after each step.
"""
import csv, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/keyu/advaitic/wt-teacher")
from method.teacher.diagnostics import PrivilegedSensors  # noqa: E402
import onnxruntime  # noqa: E402
onnxruntime.disable_telemetry_events()
from bigym.loco import make  # noqa: E402

res, out = Path(sys.argv[1]), Path(sys.argv[2])
done = set()
if out.exists():
    done = {json.loads(l)["episode"] for l in out.read_text().splitlines() if l.strip()}
env = make("move_plate", camera_keys=())
rows = list(csv.DictReader(open(res / "episodes.csv")))
for row in rows:
    ep = int(row["episode"])
    if ep in done:
        continue
    acts = np.load(res / f"episode_{ep:03d}.actions.npz")["actions"]
    ts = env.reset(seed=int(row["seed"]))
    sens = PrivilegedSensors(env.inner_env)
    s0 = sens.snapshot(0)
    z0 = s0["pose"][2]
    max_lift, held_steps, first_held, min_xy = 0.0, 0, None, s0["site_xy_distance"]
    hand_plate_min = 9.0
    side = next(s for s in env.inner_env.robot.grippers if s.name == "LEFT")
    for t, a in enumerate(acts):
        ts = env.step(a)
        s = sens.snapshot(t + 1)
        hp = np.linalg.norm(env.inner_env.robot.get_hand_pos(side) - np.asarray(s["pose"][:3]))
        hand_plate_min = min(hand_plate_min, float(hp))
        max_lift = max(max_lift, s["pose"][2] - z0)
        if s["held"]:
            held_steps += 1
            first_held = first_held if first_held is not None else t
        if s["pose"][2] - z0 > 0.03:
            min_xy = min(min_xy, s["site_xy_distance"])
        if ts.last():
            break
    end = sens.snapshot(t + 1)
    rec = dict(episode=ep, seed=int(row["seed"]), slot=int(row["source_slot"]),
               closest_hand_plate_m=round(hand_plate_min, 3), first_held=first_held,
               held_steps=held_steps, max_lift_m=round(max_lift, 3),
               min_target_xy_while_lifted=round(min_xy, 3) if max_lift > 0.03 else None,
               end_site_dist=round(end["site_distance"], 3), end_tilt=round(end["tilt_deg"], 1),
               end_contacts=end["contacts"], replay_success=bool(env.inner_env.success) if hasattr(env.inner_env, "success") else None,
               orig_success=row["success"])
    print(json.dumps(rec), flush=True)
    with out.open("a") as f:
        f.write(json.dumps(rec) + "\n")
