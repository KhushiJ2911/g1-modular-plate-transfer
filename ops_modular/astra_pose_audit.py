"""Compare the image-derived grasp transform against privileged replay labels."""

import json
import subprocess
from pathlib import Path

import modal

from ops_modular.modular_sim import image

app = modal.App("modular-astra-pose-audit")
volume = modal.Volume.from_name("modular-data")
if modal.is_local():
    image = image.add_local_file(Path(__file__), "/root/code/pose_audit.py")


def measure(folder, split, learned, limit):
    import numpy as np

    from method.modular.kinematics import G1Kinematics
    from method.modular.loaded_wrist import LEFT_GRASP_NORMAL, register

    if learned:
        import torch

        from method.modular.plate_mask import PlateMask

        torch.set_num_threads(1)
        model = PlateMask().eval()
        model.load_state_dict(torch.load(folder / "mask_v1/weights.pt", weights_only=True))
    kin = G1Kinematics()

    def deg(a, b):
        return float(np.rad2deg(np.arccos(np.clip(np.dot(a, b), -1, 1))))

    def norm(a):
        return a / np.maximum(np.linalg.norm(a), 1e-8)

    eligible = [
        s
        for s in range(3110000, 3110060)
        if (s % 10 >= 2 if split == "fit" else s % 10 == {"selection": 1, "test": 0}[split])
    ]
    rows = []
    for s in eligible:
        path = folder / "masks" / f"{s}.npz"
        if not path.is_file() and limit:
            continue
        with np.load(path) as f:
            if not len(f["step"]):
                continue
            ph = f["phase"]
            truth = f["truth"]
            baseline = f["baseline"]
            state, normal = f["state"], f["normal"]
            target = np.flatnonzero((ph == "LOWER") & (truth[:, 7] > 0.5))
            if not len(target):
                continue
            if learned:
                rgb = f["rgb"].transpose(0, 1, 4, 2, 3).copy()
                with torch.inference_mode():
                    masks = np.concatenate(
                        [
                            (
                                model(
                                    torch.from_numpy(a.reshape(-1, 3, 84, 84)),
                                    torch.tensor([0, 1]).repeat(len(a)),
                                )
                                > 0
                            )
                            .reshape(-1, 2, 84, 84)
                            .numpy()
                            for a in np.array_split(rgb, (len(rgb) + 7) // 8)
                        ]
                    )
            else:
                masks = f["mask"]
            last = normal[target[-1]]
            record = dict(seed=s, base=deg(LEFT_GRASP_NORMAL, last), phases={})
            for phase in ("CARRY", "ALIGN", "LOWER"):
                obs = []
                for i in np.flatnonzero((ph == phase) & (truth[:, 7] > 0.5)):
                    kin.set_state(state[i])
                    p, r = kin.hand_pose("left")
                    torso = kin.data.body("torso_link")
                    m = register(
                        masks[i, 0],
                        masks[i, 1],
                        r.T @ (torso.xpos - p),
                        r.T @ torso.xmat.reshape(3, 3),
                        r.T @ (baseline[i, :3] - p),
                    )
                    if m is not None:
                        n, center, cost = m
                        obs.append(
                            dict(
                                frame_deg=deg(n, normal[i]),
                                raw_deg=deg(LEFT_GRASP_NORMAL, normal[i]),
                                normal=n.tolist(),
                                cost=cost,
                            )
                        )
                if not obs:
                    continue
                # Report confidence ablations during fit; frozen check takes the
                # best-fitting half and uses a bounded tangential correction.
                best = sorted(obs, key=lambda z: z["cost"])[: max(1, (len(obs) + 1) // 2)]
                measured = norm(np.median([z["normal"] for z in best], axis=0))
                delta = measured - (measured @ LEFT_GRASP_NORMAL) * LEFT_GRASP_NORMAL
                bounded = norm(
                    LEFT_GRASP_NORMAL + delta * min(0.9, 0.25 / max(1e-8, np.linalg.norm(delta)))
                )
                record["phases"][phase] = dict(
                    count=len(obs),
                    median_cost=float(np.median([z["cost"] for z in best])),
                    direct_deg=deg(measured, last),
                    bounded_deg=deg(bounded, last),
                    observations=obs,
                )
        rows.append(record)
        if limit and len(rows) >= limit:
            break
    summary = {}
    for phase in ("CARRY", "ALIGN", "LOWER"):
        subset = [r for r in rows if phase in r["phases"]]
        if not subset:
            continue

        def stats(values):
            return dict(
                mean=float(np.mean(values)),
                p50=float(np.median(values)),
                p95=float(np.percentile(values, 95)),
            )

        summary[phase] = dict(
            n=len(subset),
            prior=stats([r["base"] for r in subset]),
            bounded=stats([r["phases"][phase]["bounded_deg"] for r in subset]),
            direct=stats([r["phases"][phase]["direct_deg"] for r in subset]),
        )
    return dict(split=split, learned=learned, summary=summary, rows=rows)


@app.function(
    image=image, cpu=1, memory=3072, timeout=1800, volumes={"/data": volume}, max_containers=1
)
def audit(split, learned=False, limit=0):
    import inspect
    import time

    if split not in ("fit", "selection", "test") or limit not in (0, 6, 12):
        raise ValueError("Expected a predeclared data split")
    volume.reload()
    name = f"pose_{split}_{'rgb' if learned else 'oracle'}_{limit}"
    out = Path("/data/astra-integration") / (name + ".json")
    if out.exists() and not limit:
        return json.loads(out.read_text())["summary"]
    src = (
        "import json\nfrom pathlib import Path\n"
        + inspect.getsource(measure)
        + f"\nprint(json.dumps(measure(Path('/data/astra-integration'),'{split}',{learned},{limit})))"
    )
    t = time.monotonic()
    p = subprocess.run(
        ["/root/policy/bin/python", "-c", src], cwd="/root/code", capture_output=True, text=True
    )
    if p.returncode:
        raise RuntimeError(p.stderr[-2000:])
    result = json.loads(p.stdout.splitlines()[-1])
    result["seconds"] = time.monotonic() - t
    out.write_text(json.dumps(result, indent=2))
    volume.commit()
    return result["summary"]
