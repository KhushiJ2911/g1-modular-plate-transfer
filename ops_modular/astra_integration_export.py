"""Freeze an exact candidate and provision its private copy of the lead's dev sharder."""

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path


def export(rev, head, recovery=False, generation=False):
    digest = hashlib.sha256(head.read_bytes()).hexdigest()
    sha = subprocess.check_output(["git", "rev-parse", rev + "^{commit}"], text=True).strip()
    app = "modular-shard-astra-" + sha[:8] + "-" + digest[:8]
    if recovery:
        app += "-recovery"
    if generation:
        app += "-gen"
    root = Path("results/astra_integration") / app
    root.mkdir(exist_ok=False, parents=True)
    archive = subprocess.check_output(["git", "archive", "--format=tar", sha])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(root, filter="data")
    module = root / "ops_modular/lead_shard.py"
    script = root / "ops_modular/lead_shard_run.py"

    def update(p, old, new):
        t = p.read_text()
        if t.count(old) != 1:
            raise ValueError(f"Unexpected release infrastructure in {p}: {old}")
        p.write_text(t.replace(old, new))

    limit = 6 if generation else 23
    update(module, "max_containers=200", f"max_containers={limit}, single_use_containers=True")
    if generation:
        update(module, "1_000_000 <= seed < 1_100_000", "3_001_200 <= seed < 3_001_300")
        update(
            module, "Only development seeds supported", "Expected a 3001200-series generation seed"
        )
    update(module, "**CPU_OPTIONS, timeout=3600", "**CPU_OPTIONS, timeout=7200")
    update(module, 'modal.App("modular-shard")', f'modal.App("{app}")')
    update(
        module,
        "    if heads:\n",
        f'    if hashlib.sha256(Path(heads).read_bytes()).hexdigest() != "{digest}":\n        raise ValueError("Incorrect perception checkpoint")\n    if heads:\n',
    )
    update(
        module,
        'f"est_v1b/last.pt@sha256:{WEIGHTS_SHA}",',
        f'f"est_v1b/last.pt@sha256:{{WEIGHTS_SHA}} + perception@sha256:{digest}",',
    )
    update(script, 'from_name("modular-shard",', f'from_name("{app}",')
    # Each spawn is detached. Persist its ID before launching another episode so
    # recovery or early stopping operates only on this specific run's calls.
    update(
        script,
        "    calls = [(block, seed, slot, fn.spawn(name, seed, policy, heads)) for block, seed, slot in jobs]",
        """    calls = []
    with open(str(out)+".calls", "x") as manifest:
        for block,seed,slot in jobs:
            call = fn.spawn(name,seed,policy,heads)
            print(json.dumps([block,seed,slot,call.object_id]),file=manifest,flush=True)
            calls.append((block,seed,slot,call))""",
    )
    (root / "frozen.json").write_text(json.dumps(dict(sha=sha, digest=digest, app=app), indent=2))
    return root


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("revision")
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--recovery", action="store_true")
    p.add_argument(
        "--generation", action="store_true", help="Use only 3001200-series generation seeds"
    )
    a = p.parse_args()
    print(export(a.revision, a.checkpoint, a.recovery, a.generation))
