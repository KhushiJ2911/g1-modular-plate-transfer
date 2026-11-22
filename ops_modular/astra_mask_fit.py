"""Fit a plate mask on generation seeds, then measure two-view pose accuracy."""

import hashlib
import json
import time
from pathlib import Path

import modal

from ops_modular.astra_v2_train import ROOT, data, image

app = modal.App("modular-astra-mask-fit")
if modal.is_local():
    for filename in (
        "method/modular/plate_mask.py",
        "method/modular/loaded_wrist.py",
        "ops_modular/astra_mask_fit.py",
    ):
        image = image.add_local_file(ROOT / filename, "/root/code/" + filename)


def train(folder, smoke):
    import numpy as np
    import torch
    from torch.nn import functional as F

    from method.modular.plate_mask import PlateMask

    torch.manual_seed(141)
    seeds = list(range(3110000, 3110060))
    if smoke:
        seeds = [3110002, 3110003, 3110004, 3110005]
    train_seeds = [s for s in seeds if smoke or s % 10 >= 2]
    selection = [s for s in seeds if s % 10 == 1] if not smoke else []
    test = [s for s in seeds if s % 10 == 0] if not smoke else []

    def load(items):
        images, labels, view = [], [], []
        for seed in items:
            with np.load(folder / "masks" / f"{seed}.npz") as f:
                if not len(f["step"]):
                    continue
                rgb = f["rgb"]
                mask = f["mask"]
                images.append(rgb.reshape(-1, 84, 84, 3).transpose(0, 3, 1, 2))
                labels.append(mask.reshape(-1, 84, 84))
                view.append(np.tile([0, 1], len(rgb)))
        return [torch.as_tensor(np.concatenate(a), device="cuda") for a in (images, labels, view)]

    x, y, cam = load(train_seeds)
    val = load(selection) if selection else None
    model = PlateMask().cuda()
    optim = torch.optim.AdamW(model.parameters(), lr=0.001)
    root = folder / ("mask_smoke" if smoke else "mask_v1")
    root.mkdir(exist_ok=True)
    best = -1

    def score(split):
        xs, ys, cs = split
        intersection = [0, 0]
        union = [0, 0]
        with torch.no_grad():
            for j in range(0, len(xs), 32):
                labels = ys[j : j + 32]
                views = cs[j : j + 32]
                masks = model(xs[j : j + 32], views) > 0
                for k in (0, 1):
                    idx = views == k
                    intersection[k] += int((masks[idx] & labels[idx]).sum())
                    union[k] += int((masks[idx] | labels[idx]).sum())
        return [a / max(1, b) for a, b in zip(intersection, union, strict=True)]

    for epoch in range(1 if smoke else 25):
        order = torch.randperm(len(x), device=x.device)
        model.train()
        for ids in order.split(24):
            # Mild global exposures preserve the contour and the paired mask.
            rgb = (
                x[ids] * torch.empty((len(ids), 1, 1, 1), device=x.device).uniform_(0.9, 1.1)
            ).clamp(0, 255)
            out = model(rgb, cam[ids])
            label = y[ids].float()
            p = out.sigmoid()
            dice = 1 - (2 * (p * label).sum(dim=(1, 2)) + 1) / (
                p.sum(dim=(1, 2)) + label.sum(dim=(1, 2)) + 1
            )
            loss = F.binary_cross_entropy_with_logits(out, label) + dice.mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            if smoke:
                break
        model.eval()
        if val is None:
            stats = [0.0, 0.0]
        else:
            stats = score(val)
        if np.mean(stats) > best:
            best = np.mean(stats)
            best_epoch = epoch
            best_stats = stats
            torch.save(model.state_dict(), root / "weights.pt")
    model.load_state_dict(torch.load(root / "weights.pt", weights_only=True))
    test_stats = score(load(test)) if test else []
    return dict(
        train_seeds=train_seeds,
        selection=selection,
        test=test,
        train_images=len(x),
        best_epoch=best_epoch,
        selection_iou=best_stats,
        test_iou=test_stats,
        sha256=hashlib.sha256((root / "weights.pt").read_bytes()).hexdigest(),
    )


@app.function(
    image=image,
    gpu="T4",
    cpu=2,
    memory=8192,
    volumes={"/data": data},
    timeout=3600,
    max_containers=1,
)
def fit(smoke):
    import inspect
    import subprocess

    data.reload()
    folder = Path("/data/astra-integration")
    result = folder / ("mask_smoke" if smoke else "mask_v1") / "fit.json"
    if result.exists():
        return json.loads(result.read_text())
    src = (
        "import hashlib,json\nfrom pathlib import Path\n"
        + inspect.getsource(train)
        + f"\nprint(json.dumps(train(Path('{folder}'),{smoke})))"
    )
    t = time.monotonic()
    p = subprocess.run(["python", "-c", src], cwd="/root/code", capture_output=True, text=True)
    if p.returncode:
        raise ValueError(p.stderr[-1800:])
    r = dict(**json.loads(p.stdout.splitlines()[-1]), seconds=time.monotonic() - t)
    result.write_text(json.dumps(r, indent=2))
    data.commit()
    return r


@app.function(cpu=0.125, memory=256, timeout=86400, volumes={"/data": data})
def schedule(smoke_id):
    model = modal.FunctionCall.from_id(smoke_id).get()
    if not model["sha256"] or model["train_images"] < 50:
        raise ValueError("Unusable segmenter smoke")
    folder = Path("/data/astra-integration")
    while True:
        data.reload()
        if all((folder / "masks" / f"{s}.npz").is_file() for s in range(3110000, 3110060)):
            break
        time.sleep(45)
    c = fit.spawn(False)
    (folder / "mask_fit.call").write_text(c.object_id)
    data.commit()
    return c.get()
