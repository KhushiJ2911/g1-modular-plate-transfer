"""Frozen, paired CPU/OSMesa evaluation of bd951bc with and without FK seating."""

import argparse
import io
import json
import subprocess
import tarfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from advaitic_eval.protocol import make_plan

APP = "modular-shard-sol-fk6b02f33"
BASE = "bd951bc"
FK = "6b02f33"
HEAD = "64dfe105e7f716c0ab5b7a9c487b7a8a1cac06565070b390e75992dfbee4deff"
HEAD_PATH = f"/runs/heads/{HEAD}.npz"
START = 3_001_600
SMOKE = START - 1
POLICIES = {"base": "submission.sol_fk_pair:baseline", "fk": "submission.sol_fk_pair:candidate"}
ROOT = Path(__file__).resolve().parents[1]


def groups():
    yield "smoke", [(SMOKE, arm) for arm in POLICIES]
    yield "full", [(ep.seed, arm) for ep in make_plan(30, START) for arm in POLICIES]


def prepare(path):
    path.mkdir(parents=True, exist_ok=False)
    archive = subprocess.check_output(["git", "archive", BASE], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as content:
        content.extractall(path, filter="data")
    fk = "method/modular/fk_seating.py"
    (path / fk).write_bytes(subprocess.check_output(["git", "show", f"{FK}:{fk}"], cwd=ROOT))
    (path / "submission/sol_fk_pair.py").write_text(
        """from method.modular.robust_controller import _with_phase_trace
from method.modular.fk_seating import make_policy as make_candidate
from submission.policy_modular_calibrated import make_policy as make_baseline


def baseline():
    return _with_phase_trace(make_baseline())


def candidate():
    return _with_phase_trace(make_candidate())
"""
    )
    script = path / "ops_modular/lead_shard.py"
    text = script.read_text()
    replacements = [
        ('modal.App("modular-shard")', f'modal.App("{APP}")'),
        (
            "if not 1_000_000 <= seed < 1_100_000:",
            "if not 3001599 <= seed < 3001750:",
        ),
        (
            '"MKL_NUM_THREADS": "1",',
            '"MKL_NUM_THREADS": "1",\n        "LP_NUM_THREADS": "2",\n'
            '        "SOL_PHASE_TRACE": "1",',
        ),
        ("timeout=3600, max_containers=200", "timeout=5400, max_containers=30"),
        (
            '    if heads:\n        os.environ["ADV_HEADS"] = heads',
            f'    if heads != "{HEAD_PATH}" or hashlib.sha256(Path(heads).read_bytes()).hexdigest() != "{HEAD}":\n'
            '        raise ValueError("Unexpected calibrated checkpoint digest")\n'
            '    os.environ["ADV_HEADS"] = heads',
        ),
        (
            '    return {"seed": seed, "seconds": time.monotonic() - start, "gl": gl, **report["episodes"][0]}',
            '    logs = (dest / "eval/policy.stderr.log").read_text().splitlines()\n'
            '    phases = [json.loads(x.removeprefix("SOL_PHASE ")) for x in logs if x.startswith("SOL_PHASE ")]\n'
            '    return {"seed": seed, "seconds": time.monotonic() - start, "gl": gl,'
            ' "archive": str(dest / "eval"), "phases": phases, **report["episodes"][0]}',
        ),
    ]
    for old, new in replacements:
        if text.count(old) != 1:
            raise ValueError(f"Unexpected sharder source for {old!r}")
        text = text.replace(old, new)
    script.write_text(text)


def read_lines(path):
    return list(map(json.loads, path.read_text().splitlines())) if path.exists() else []


def collect(path, phase, once):
    import modal

    from ops_modular.sol_shard_collect import fetch

    jobs = dict(groups())[phase]
    manifest = path / f"sol_render_fk_{phase}.calls"
    output = path / f"sol_render_fk_{phase}.jsonl"
    records = read_lines(output)
    valid = {(r["seed"], r["arm"]) for r in records if r.get("completed")}
    journal = {tuple(x["key"]): x for x in read_lines(manifest)}
    function = modal.Function.from_name(APP, "episode_cpu")
    with manifest.open("a") as stream:
        for seed, arm in jobs:
            if (seed, arm) in journal:
                continue
            call = function.spawn(f"sol_fk_{phase}_{arm}", seed, POLICIES[arm], HEAD_PATH)
            row = {"key": [seed, arm], "id": call.object_id}
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            journal[(seed, arm)] = row
    prior_errors = {r["call_id"] for r in records if r.get("error")}
    pending = {key: x["id"] for key, x in journal.items() if key not in valid}
    with ThreadPoolExecutor(max_workers=8) as pool, output.open("a") as stream:
        while pending:
            work = {pool.submit(fetch, call_id): key for key, call_id in pending.items()}
            for f in as_completed(work):
                key = work[f]
                result = f.result()
                if result is None:
                    continue
                seed, arm = key
                call_id = pending.pop(key)
                result.update(arm=arm, call_id=call_id, seed=seed)
                if call_id not in prior_errors:
                    stream.write(json.dumps(result) + "\n")
                    stream.flush()
                    print(seed, arm, result.get("success"), result.get("error"), flush=True)
                if result.get("error"):
                    prior_errors.add(call_id)
            if once:
                break
            if pending:
                time.sleep(45)


def retry(path, phase):
    import modal

    plan = set(dict(groups())[phase])
    manifest = path / f"sol_render_fk_{phase}.calls"
    output = path / f"sol_render_fk_{phase}.jsonl"
    valid = {(r["seed"], r["arm"]) for r in read_lines(output) if r.get("completed")}
    entries = read_lines(manifest)
    current = {tuple(x["key"]): x for x in entries}
    attempts = Counter(tuple(row["key"]) for row in entries)
    function = modal.Function.from_name(APP, "episode_cpu")
    volume = modal.Volume.from_name("advaitic-runs")
    with manifest.open("a") as stream, output.open("a") as recovered:
        for row in read_lines(output):
            key = row["seed"], row["arm"]
            if key in valid or key not in plan or not row.get("error"):
                continue
            if row["call_id"] != current[key]["id"]:
                continue
            seed, arm = key
            name = f"sol_fk_{phase}_{arm}"
            for attempt in range(5):
                suffix = f"-r{attempt}" if attempt else ""
                for tag in (name, f"{name}_retry"):
                    archive = f"/runs/modular/{tag}-{seed}{suffix}/eval"
                    try:
                        report = json.loads(
                            b"".join(
                                volume.read_file(archive.removeprefix("/runs") + "/report.json")
                            )
                        )
                    except (FileNotFoundError, modal.exception.NotFoundError):
                        continue
                    if (
                        report["method"] != tag
                        or not report["integrity_ok"]
                        or report["infrastructure_error"]
                    ):
                        raise ValueError(f"Invalid persisted report in {archive}")
                    (episode,) = report["episodes"]
                    if episode["seed"] != seed or report["policy"] != POLICIES[arm]:
                        raise ValueError(f"Unexpected persisted episode in {archive}")
                    logs = (
                        b"".join(
                            volume.read_file(archive.removeprefix("/runs") + "/policy.stderr.log")
                        )
                        .decode()
                        .splitlines()
                    )
                    phases = [
                        json.loads(line[10:]) for line in logs if line.startswith("SOL_PHASE ")
                    ]
                    restored = dict(episode, arm=arm, phases=phases, archive=archive, seconds=None)
                    recovered.write(json.dumps(restored) + "\n")
                    recovered.flush()
                    valid.add(key)
                    break
                if key in valid:
                    break
            if key in valid:
                continue
            if attempts[key] >= 2:
                continue
            call = function.spawn(f"sol_fk_{phase}_{arm}_retry", seed, POLICIES[arm], HEAD_PATH)
            entry = {"key": list(key), "id": call.object_id}
            current[key] = entry
            stream.write(json.dumps(entry) + "\n")
            stream.flush()


def summarize(path):
    slots = {ep.seed: ep.source_slot for ep in make_plan(30, START)}
    published = read_lines(path / "sol_render_fk_full.jsonl")
    by_arm = {arm: {} for arm in POLICIES}
    for row in published:
        if row.get("error") or not row.get("completed"):
            continue
        seed, arm = row["seed"], row["arm"]
        if slots[seed] != row["source_slot"] or seed in by_arm[arm]:
            raise ValueError(f"Unexpected or duplicate report for {arm} seed {seed}")
        by_arm[arm][seed] = row
    seeds = set(by_arm["base"]) & set(by_arm["fk"])
    result = {
        "planned": len(slots),
        "matched": len(seeds),
        "missing": {arm: sorted(slots.keys() - rows.keys()) for arm, rows in by_arm.items()},
        "failed_calls": sum(bool(row.get("error")) for row in published),
    }
    outcomes = {}
    for arm, rows in by_arm.items():
        wins = {seed for seed in seeds if rows[seed]["success"]}
        losses = seeds - wins
        result[arm] = {
            "valid_episodes": len(rows),
            "matched_successes": len(wins),
            "matched_falls": sum(rows[s]["fell"] for s in seeds),
            "policy_errors": {s: rows[s]["policy_error"] for s in seeds if rows[s]["policy_error"]},
            "release_entries": sum(
                any(p["to"] == "RELEASE" for p in rows[s]["phases"]) for s in seeds
            ),
            "terminal_failure_phases": dict(
                Counter(
                    rows[s]["phases"][-1]["from"] if rows[s]["phases"] else "APPROACH"
                    for s in losses
                )
            ),
            "success_by_slot": dict(Counter(slots[s] for s in wins)),
        }
        outcomes[arm] = wins
    result["fk_rescues"] = sorted(outcomes["fk"] - outcomes["base"])
    result["fk_losses"] = sorted(outcomes["base"] - outcomes["fk"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "collect", "retry", "summary"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--phase", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.path)
    elif args.command == "collect":
        collect(args.path, args.phase, args.once)
    elif args.command == "retry":
        retry(args.path, args.phase)
    else:
        print(json.dumps(summarize(args.path), indent=2))
