"""Export the RGB segmentation and the frozen contact/visual calibration in one checkpoint."""

import hashlib
import io
import json
from pathlib import Path

import modal

from ops_modular.astra_v2_train import base, data, image

app = modal.App("modular-astra-mask-pack")
BASE_SHA = "d81acb2fe6f1cb75af5acbd0fa39cda58c17fdc14486094ce1ed1312e65b723c"
MASK_SHA = "0ee9f8d65dddb3c8a11c95652712d3c2ec42faac1a723884f667ac044ea0a70f"


@app.function(image=image, cpu=1, memory=2048, volumes={"/runs": base, "/data": data}, timeout=300)
def pack():
    import numpy as np
    import torch

    original = Path("/runs/heads") / (BASE_SHA + ".npz")
    folder = Path("/data/astra-integration/mask_v1")
    payload = (folder / "weights.pt").read_bytes()
    if (
        hashlib.sha256(original.read_bytes()).hexdigest() != BASE_SHA
        or hashlib.sha256(payload).hexdigest() != MASK_SHA
    ):
        raise ValueError("Input checkpoint digest mismatch")
    with np.load(original, allow_pickle=False) as f:
        weights = dict(f)
    trained = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    weights.update({"mask/" + k: v.numpy() for k, v in trained.items()})
    stream = io.BytesIO()
    np.savez_compressed(stream, **weights)
    encoded = stream.getvalue()
    digest = hashlib.sha256(encoded).hexdigest()
    dest = Path("/runs/heads") / (digest + ".npz")
    if not dest.exists():
        dest.write_bytes(encoded)
    if hashlib.sha256(dest.read_bytes()).hexdigest() != digest:
        raise ValueError("Invalid packed checkpoint")
    provenance = dict(sha256=digest, previous=BASE_SHA, mask=MASK_SHA, bytes=len(encoded))
    (folder / "combined.json").write_text(json.dumps(provenance))
    base.commit()
    data.commit()
    return provenance
