"""Audit recorded actions with CPU physics and no cameras; never inject poses.

This is a dataset inspection, not a teacher rollout or success measurement.
Only camera observations are omitted; physics, controllers and rules are fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ["ORT_DISABLE_TELEMETRY"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

from method.teacher.teacher import METHOD_DIR


def mesh_z_bounds(inner, prop):
    """World collision-mesh bounds, reading compiled vertices and live transforms."""
    bounds = []
    for geom in prop.colliders:
        gid = inner.model.bind(geom).id
        mid = inner.model.geom_dataid[gid]
        if mid < 0:
            raise ValueError("Expected mesh colliders")
        start = inner.model.mesh_vertadr[mid]
        count = inner.model.mesh_vertnum[mid]
        vertices = inner.model.mesh_vert[start : start + count]
        rotation = inner.data.geom_xmat[gid].reshape(3, 3)
        z = vertices @ rotation[2] + inner.data.geom_xpos[gid, 2]
        bounds.extend([float(z.min()), float(z.max())])
    return min(bounds), max(bounds)


def audit(demos, meta, output):
    import onnxruntime
    from bigym.loco import make

    onnxruntime.disable_telemetry_events()
    rows = json.loads(meta.read_text())
    result = {
        "demos_sha256": hashlib.sha256(demos.read_bytes()).hexdigest(),
        "meta_sha256": hashlib.sha256(meta.read_bytes()).hexdigest(),
        "pre_release_steps": 40,
        "lift_steps": 30,
        "carry_band_m": 0.01,
        "min_clearance_m": 0.005,
        "demos": {},
    }
    env = make("move_plate", camera_keys=())
    try:
        with np.load(demos, allow_pickle=False) as data:
            for row in rows:
                k = row["demo"]
                if k == 45:
                    continue
                ts = env.reset(seed=row["seed"])
                inner = env.inner_env
                side = next(s for s in inner.robot.grippers if s.name == "LEFT")
                nominal = row["release_step"] - 40
                samples = []
                for t in range(nominal + 1):
                    bottom, _ = mesh_z_bounds(inner, inner.plates[0])
                    _, rack_top = mesh_z_bounds(inner, inner.rack_target)
                    samples.append(
                        {
                            "step": t,
                            "hand_z": float(inner.robot.get_hand_pos(side)[2]),
                            "plate_z": float(inner.plates[0].get_position()[2]),
                            "plate_bottom_z": bottom,
                            "rack_top_z": rack_top,
                            "clearance_m": bottom - rack_top,
                            "target_contact": bool(inner.plates[0].is_colliding(inner.rack_target)),
                            "state_max_deviation": float(
                                np.abs(ts.low_dim_obs - data[f"s{k}"][t]).max()
                            ),
                        }
                    )
                    if t < nominal:
                        ts = env.step(data[f"a{k}"][t])
                        if ts.last():
                            raise ValueError(f"Demo {k} ended before alignment audit")
                carry = samples[row["grasp_step"] + 31 : nominal + 1]
                peak = max(s["hand_z"] for s in carry)
                eligible = [
                    s
                    for s in carry
                    if s["hand_z"] >= peak - 0.01
                    and s["clearance_m"] >= 0.005
                    and not s["target_contact"]
                ]
                if samples[-1]["clearance_m"] >= 0.005 and not samples[-1]["target_contact"]:
                    gate = samples[-1]
                elif eligible:
                    gate = eligible[-1]
                else:
                    raise ValueError(f"Demo {k}: no clear above-rack carry gate")
                result["demos"][str(k)] = {
                    "nominal": samples[-1],
                    "gate": gate,
                    "carry_hand_peak_z": peak,
                }
                print(
                    f"demo={k} nominal={nominal} gate={gate['step']} clearance={gate['clearance_m']:.4f}m",
                    flush=True,
                )
    finally:
        env.close()
    output.write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demos", type=Path, default=METHOD_DIR / "data/demos_lowdim.npz")
    parser.add_argument("--meta", type=Path, default=METHOD_DIR / "data/demo_meta.json")
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).with_name("alignment_audit.json")
    )
    args = parser.parse_args()
    audit(args.demos, args.meta, args.out)


if __name__ == "__main__":
    main()
