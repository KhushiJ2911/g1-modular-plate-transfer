"""Freeze a derivative of 414f15a's sharder and journal exact fresh-process episodes."""

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

from advaitic_eval.protocol import make_plan, source_slot_for_seed

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("/home/keyu/advaitic/plans/modular_controller/runs/astra_fixround")
HEAD = "64dfe105e7f716c0ab5b7a9c487b7a8a1cac06565070b390e75992dfbee4deff"
POLICY = "submission.policy_modular_round4:make_policy"
TARGET = (1062009, 1072036, 1092007, 1034006)
HARD = (
    1092002,
    1092014,
    1052019,
    1072036,
    1034006,
    1042003,
    1042006,
    1062001,
    1062009,
    1062026,
    1062038,
    1033005,
    1033015,
)
SANITY = tuple(ep.seed for ep in make_plan(30, 3002900))
SEEDS = {"smoke": TARGET[:1], "target": TARGET, "hard": HARD, "sanity": SANITY}


def prepare(variant, extra):
    files = [
        "submission/policy_grasp_retry.py",
        "method/modular/seated_drop.py",
        "submission/policy_modular_round4.py",
        *extra,
    ]
    hashes = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    app = f"modular-shard-astra-{variant}-{digest[:8]}"
    folder = ROOT / "results/astra_integration" / app
    folder.mkdir(exist_ok=True)
    archive = subprocess.check_output(
        [
            "git",
            "archive",
            "414f15a",
            "src",
            "submission",
            "method/modular",
            "ops_modular/lead_shard.py",
            "pyproject.toml",
            "uv.lock",
            "README.md",
        ],
        cwd=ROOT,
    )
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(folder, filter="data")
    for file in files:
        shutil.copyfile(ROOT / file, folder / file)
    shard = folder / "ops_modular/lead_shard.py"
    source = shard.read_text()
    for old, new in [
        ('modal.App("modular-shard")', f'modal.App("{app}")'),
        (
            "if not 1_000_000 <= seed < 1_100_000:",
            f"if seed not in {tuple(sorted(set(TARGET + HARD + SANITY)))}:",
        ),
        (
            "    if heads:\n",
            f'    if hashlib.sha256(Path(heads).read_bytes()).hexdigest() != "{HEAD}":\n        raise ValueError("Unexpected sidecar digest")\n    if heads:\n',
        ),
    ]:
        if source.count(old) != 1:
            raise ValueError(old)
        source = source.replace(old, new)
    shard.write_text(source)
    OUTPUT.mkdir(exist_ok=True)
    manifest = {
        "base": "414f15a",
        "app": app,
        "digest": digest,
        "files": hashes,
        "folder": str(folder),
        "factory": POLICY,
        "sidecar": HEAD,
    }
    (OUTPUT / f"{variant}.json").write_text(json.dumps(manifest, indent=2))
    print(folder)


def records(path):
    return (
        {r["seed"]: r for r in map(json.loads, path.read_text().splitlines())}
        if path.exists()
        else {}
    )


def run(variant, group, collect):
    import modal

    from ops_modular.sol_shard_collect import fetch

    manifest = json.loads((OUTPUT / f"{variant}.json").read_text())
    stem = manifest["app"]
    calls = OUTPUT / f"{stem}.calls"
    results = OUTPUT / f"{stem}.jsonl"
    submitted, completed = records(calls), records(results)
    if collect:
        for seed, entry in submitted.items():
            if seed in completed:
                continue
            row = fetch(entry["call"])
            if row is None:
                continue
            if (
                not row.get("completed")
                or row["seed"] != seed
                or row["source_slot"] != source_slot_for_seed(seed)
                or row["policy_error"]
            ):
                raise RuntimeError((entry, row))
            with results.open("a") as out:
                print(json.dumps({**row, **entry}), file=out)
            print(seed, row["success"], row["steps"], flush=True)
        return
    fn = modal.Function.from_name(stem, "episode_gpu")
    for seed in SEEDS[group]:
        if seed in submitted:
            continue
        call = fn.spawn(stem, seed, POLICY, f"/runs/heads/{HEAD}.npz")
        entry = {"seed": seed, "call": call.object_id}
        with calls.open("a") as out:
            print(json.dumps(entry), file=out)
        print(entry, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "spawn", "collect"))
    p.add_argument("variant")
    p.add_argument("group", choices=SEEDS, nargs="?", default="smoke")
    p.add_argument("--extra", action="append", default=[])
    args = p.parse_args()
    if args.command == "prepare":
        prepare(args.variant, args.extra)
    else:
        run(args.variant, args.group, args.command == "collect")
