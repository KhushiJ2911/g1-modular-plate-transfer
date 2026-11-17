"""Export once in the sim environment: uv run python -m method.bc.export_human."""

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np


def export(out, metadata=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if all((out / f"ep_{i}.npz").exists() for i in range(60)):
        print("All 60 episodes already exist; nothing to load.")
        return
    records = {}
    if metadata is not None:
        records = {int(row["demo"]): row for row in json.loads(Path(metadata).read_text())}
    os.environ["ORT_DISABLE_TELEMETRY"] = "1"
    import onnxruntime

    onnxruntime.disable_telemetry_events()
    from bigym.loco import make_gym

    env = make_gym("move_plate")
    try:
        demos = env.get_demos(60)  # Once, in this process only; never in DataLoader workers.
        if len(demos) != 60:
            raise ValueError(f"Expected 60 demos, found {len(demos)}")
        for index, demo in enumerate(demos):
            path = out / f"ep_{index}.npz"
            if path.exists():
                continue
            actions = np.asarray(demo["action"], dtype=np.float32)
            length = len(actions)
            state = np.asarray(demo["obs"]["state"], dtype=np.float32)
            rgb = np.asarray(demo["obs"]["rgb"], dtype=np.uint8)
            if state.shape != (length + 1, 50) or rgb.shape != (length + 1, 3, 3, 84, 84):
                raise ValueError(f"Unexpected observation shape in demo {index}")
            # Unknown metadata stays unknown; no task-state reconstruction or rollout.
            record = records.get(index, {})
            seed = demo.get("seed", record.get("seed", -1))
            slot = demo.get("slot", record.get("slot", -1))
            with tempfile.NamedTemporaryFile(dir=out, suffix=".npz", delete=False) as stream:
                temporary = stream.name
                try:
                    np.savez_compressed(
                        stream,
                        rgb=rgb[:length:2],
                        rgb_stride=np.int64(2),
                        state=state[:length],
                        action=actions,
                        seed=np.int64(-1 if seed is None else seed),
                        slot=np.int64(-1 if slot is None else slot),
                        success=np.bool_(demo["success"]),
                    )
                    stream.flush()
                    os.fsync(stream.fileno())
                    os.replace(temporary, path)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
            print(f"{path}: {length} actions, {len(rgb[:length:2])} frames", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("/home/keyu/advaitic/data/human60"))
    sidecar = Path(__file__).resolve().parents[1] / "data" / "demo_meta.json"
    parser.add_argument(
        "--metadata",
        type=Path,
        default=sidecar if sidecar.exists() else None,
        help="Demo index/seed/slot JSON (defaults to method/data/demo_meta.json if present)",
    )
    args = parser.parse_args()
    export(args.out, args.metadata)
