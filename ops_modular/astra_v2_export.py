"""Freeze the tested git revision; configure the lead's two sharding scripts for this arm."""

import argparse
import io
import json
import re
import subprocess
import tarfile
from pathlib import Path

V1 = "bd8ed33c87383b3b208ba251d8fa0b6f957d9f03a35cf6c8d23a81ce9cd867cd"


def replace(text, old, new, n=1):
    if text.count(old) != n:
        raise ValueError(f"Unexpected lead runner configuration: {old}")
    return text.replace(old, new)


def export(commit, arm, digest, root):
    sha = subprocess.check_output(
        ["git", "rev-parse", "--verify", commit + "^{commit}"], text=True
    ).strip()
    if not re.fullmatch(r"[0-9a-f]{64}", digest) or (arm == "v1b" and digest != V1):
        raise ValueError("Expected the arm's complete checkpoint digest")
    app = f"modular-shard-astra-{arm}-{sha[:8]}"
    dest = root / app
    dest.mkdir(parents=True, exist_ok=False)
    archive = subprocess.check_output(["git", "archive", "--format=tar", sha])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tree:
        tree.extractall(dest, filter="data")
    # The evaluator and all policy sources stay exactly as git archived them.
    # Only the private deployment, checkpoint identity/mount and call journal vary.
    module = dest / "ops_modular/lead_shard.py"
    s = replace(module.read_text(), 'modal.App("modular-shard")', f'modal.App("{app}")')
    if arm == "v2":
        if not (dest / "submission/estimator_v2.py").exists():
            raise ValueError("Frozen commit lacks v2 inference")
        s = replace(
            s, 'Path("/runs/est_v1b/last.pt")', 'Path("/models/estimator-v2/est_v2/last.pt")'
        )
        s = replace(s, V1, digest)
        s = replace(s, 'f"est_v1b/last.pt@sha256:', 'f"estimator-v2/est_v2/last.pt@sha256:')
        s = replace(
            s,
            'volumes={"/runs": volume}',
            'volumes={"/runs": volume,"/models":modal.Volume.from_name("modular-runs")}',
            2,
        )
    module.write_text(s)
    runner = dest / "ops_modular/lead_shard_run.py"
    s = replace(runner.read_text(), 'from_name("modular-shard",', f'from_name("{app}",')
    s = replace(
        s,
        "    rows = []",
        """    Path(str(out)+".calls.json").write_text(json.dumps([(b,s,k,c.object_id) for b,s,k,c in calls]))
    rows = []""",
    )
    runner.write_text(s)
    (dest / "frozen.json").write_text(
        json.dumps(dict(commit=sha, app=app, arm=arm, sha256=digest), indent=2)
    )
    return dest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("commit")
    p.add_argument("arm", choices=["v1b", "v2"])
    p.add_argument("digest")
    p.add_argument("--root", type=Path, default=Path("results/astra_v2_exports"))
    a = p.parse_args()
    print(export(a.commit, a.arm, a.digest, a.root))
