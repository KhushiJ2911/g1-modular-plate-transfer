"""Freeze Sol's policy files and configure a dedicated export of the lead dev sharder."""

import argparse
import io
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWNED = (
    "method/modular/controller.py",
    "method/modular/observed_controller.py",
    "method/modular/placement_release.py",
    "method/modular/robust_controller.py",
)


def git(*args, root=ROOT, **kwargs):
    return subprocess.check_output(["git", *args], cwd=root, **kwargs)


def extract(archive, dest):
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        stream.extractall(dest, filter="data")


def patch(path, replacements):
    source = path.read_text()
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError(f"Expected one sharder configuration to replace in {path.name}: {old}")
        source = source.replace(old, new)
    path.write_text(source)


def prepare(dest, reference, include_release):
    base = git("rev-parse", reference, text=True).strip()
    dest.mkdir(parents=True, exist_ok=False)
    snapshot = dest / "snapshot"
    snapshot.mkdir()
    extract(git("archive", base), snapshot)
    git("init", "-q", root=snapshot)
    paths = (
        ("method/modular/placement_release.py",)
        if include_release
        else (path for path in OWNED if not path.endswith("placement_release.py"))
    )
    for path in paths:
        (snapshot / path).write_bytes((ROOT / path).read_bytes())
    git("add", "-f", ".", root=snapshot)
    git(
        "-c",
        "user.name=Sol snapshot",
        "-c",
        "user.email=sol-snapshot@localhost",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-q",
        "-m",
        f"Dev snapshot based on {base}",
        root=snapshot,
    )
    commit = git("rev-parse", "HEAD", root=snapshot, text=True).strip()
    app = f"modular-shard-sol-{commit[:10]}"
    export = dest / "export"
    export.mkdir()
    extract(git("archive", commit, root=snapshot), export)
    patch(
        export / "ops_modular/lead_shard.py",
        [
            ('modal.App("modular-shard")', f'modal.App("{app}")'),
            (
                "if not 1_000_000 <= seed < 1_100_000:",
                "if not (1_000_000 <= seed < 1_100_000 or 3_000_400 <= seed < 3_010_000):",
            ),
            (
                '"OMP_NUM_THREADS": "1",',
                '"OMP_NUM_THREADS": "1",\n        "LP_NUM_THREADS": "2",\n'
                '        "SOL_PHASE_TRACE": "1",\n'
                '        "ADV_PLACEMENT_NORMAL": "/runs/heads/normal_v1_7e99a421.npz",',
            ),
            ("timeout=3600, max_containers=200", "timeout=3600, max_containers=100"),
            (
                '    return {"seed": seed, "seconds": time.monotonic() - start, "gl": gl, **report["episodes"][0]}',
                '    lines = (dest / "eval/policy.stderr.log").read_text().splitlines()\n'
                '    trace = [json.loads(line.removeprefix("SOL_PHASE ")) for line in lines '
                'if line.startswith("SOL_PHASE ")]\n'
                '    placement = [json.loads(line.removeprefix("SOL_PLACEMENT ")) for line in lines '
                'if line.startswith("SOL_PLACEMENT ")]\n'
                f'    return {{"seed": seed, "seconds": time.monotonic() - start, "gl": gl, '
                f'"snapshot": "{commit}", "phases": trace, "placement": placement, **report["episodes"][0]}}',
            ),
        ],
    )
    patch(
        export / "ops_modular/lead_shard_run.py",
        [
            (
                "import json",
                "import json\nimport os\nfrom concurrent.futures import ThreadPoolExecutor, as_completed",
            ),
            ('modal.Function.from_name("modular-shard",', f'modal.Function.from_name("{app}",'),
            (
                "    assert all(make_plan(30, seed)[0].seed == seed for _, seed, _ in jobs)",
                """    assert all(make_plan(30, seed)[0].seed == seed for _, seed, _ in jobs)
    assert all(1_000_000 <= seed < 1_100_000 or 3_000_400 <= seed < 3_010_000 for _, seed, _ in jobs)
    if limit := os.environ.get("SOL_SHARD_PER_SLOT"):
        counts = Counter()
        selected = []
        for job in jobs:
            block, _, slot = job
            counts[block, slot] += 1
            if counts[block, slot] <= int(limit):
                selected.append(job)
        jobs = selected
    jobs = jobs[:int(os.environ.get("SOL_SHARD_EPISODES", len(jobs)))]""",
            ),
            (
                "    calls = [(block, seed, slot, fn.spawn(name, seed, policy, heads)) for block, seed, slot in jobs]",
                """    calls = []
    manifest = Path(out + ".calls")
    if manifest.exists():
        with manifest.open() as stream:
            for line in stream:
                block, seed, slot, call_id = json.loads(line)
                calls.append((block, seed, slot, modal.FunctionCall.from_id(call_id)))
        if [(b, s, j) for b, s, j, _ in calls] != jobs[:len(calls)]:
            raise ValueError("Manifest does not match this plan")
    with manifest.open("a") as stream:
        for block, seed, slot in jobs[len(calls):]:
            call = fn.spawn(name, seed, policy, heads)
            stream.write(json.dumps([block, seed, slot, call.object_id]) + "\\n")
            stream.flush()
            calls.append((block, seed, slot, call))
    if os.environ.get("SOL_SHARD_LAUNCH_ONLY") == "1":
        return""",
            ),
            (
                '    with open(out, "w") as fh:\n        for block, seed, slot, call in calls:',
                """    with open(out, "w") as fh, ThreadPoolExecutor(max_workers=len(calls)) as pool:
        pending = {pool.submit(call.get): (block, seed, slot) for block, seed, slot, call in calls}
        for result in as_completed(pending):
            block, seed, slot = pending[result]""",
            ),
            ("                r = call.get()", "                r = result.result()"),
        ],
    )
    (dest / "source.txt").write_text(f"{base}\n{commit}\n{app}\n")
    print(app, export, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dest", type=Path)
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--placement-release", action="store_true")
    args = parser.parse_args()
    prepare(args.dest.resolve(), args.ref, args.placement_release)
