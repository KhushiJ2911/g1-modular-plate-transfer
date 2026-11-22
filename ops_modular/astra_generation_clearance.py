"""Prepare the fixed 3002200 camera generation comparison against the calibrated policy."""

import argparse
import hashlib
import io
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEAD = "64dfe105e7f716c0ab5b7a9c487b7a8a1cac06565070b390e75992dfbee4deff"
APP = "modular-astra-clearance-gen-bd951bc-528c854"


def prepare(dest):
    dest.mkdir(parents=True, exist_ok=False)
    archive = subprocess.check_output(["git", "archive", "bd951bc"], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as files:
        files.extractall(dest, filter="data")
    rel = "method/modular/base_clearance.py"
    source = subprocess.check_output(["git", "show", "528c854:" + rel], cwd=ROOT)
    if (
        hashlib.sha256(source).hexdigest()
        != "96699ef0c3c3e7dc6172bc9da84b25d7f559ec0fbfb59a4ad472e5579544ffb2"
    ):
        raise ValueError("Unexpected late-clearance policy")
    (dest / rel).write_bytes(source)
    sharder = dest / "ops_modular/lead_shard.py"
    text = sharder.read_text()
    edits = [
        ('modal.App("modular-shard")', f'modal.App("{APP}")'),
        ("1_000_000 <= seed < 1_100_000", "3_002_199 <= seed < 3_002_350"),
        ("Only development seeds supported", "Expected the paired 3002200 generation plan"),
        ('"MKL_NUM_THREADS": "1",', '"MKL_NUM_THREADS": "1",\n        "LP_NUM_THREADS": "2",'),
        (
            "**CPU_OPTIONS, timeout=3600, max_containers=200",
            "**CPU_OPTIONS, timeout=7200, max_containers=15",
        ),
        (
            "    if heads:\n",
            f'    if Path(heads).stem != "{HEAD}" or hashlib.sha256(Path(heads).read_bytes()).hexdigest() != "{HEAD}":\n        raise ValueError("Unexpected perception checkpoint")\n    if heads:\n',
        ),
    ]
    for old, new in edits:
        if text.count(old) != 1:
            raise ValueError(f"Unexpected frozen sharder: {old}")
        text = text.replace(old, new)
    sharder.write_text(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    prepare(parser.parse_args().destination)
