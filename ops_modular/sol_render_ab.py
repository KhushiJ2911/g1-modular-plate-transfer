"""Frozen, paired CPU/OSMesa evaluation of bd951bc and an FK seating candidate."""

import argparse
import io
import json
import shutil
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
SETTLE = "24f55b7"
HEAD = "64dfe105e7f716c0ab5b7a9c487b7a8a1cac06565070b390e75992dfbee4deff"
HEAD_PATH = f"/runs/heads/{HEAD}.npz"
START = 3_001_600
RUN = "fk"
QUICK = False
CONTAINERS = 30
POLICIES = {"base": "submission.sol_fk_pair:baseline", "fk": "submission.sol_fk_pair:candidate"}
ROOT = Path(__file__).resolve().parents[1]


def groups():
    yield "smoke", [(START - 1, arm) for arm in POLICIES]
    yield "full", [(ep.seed, arm) for ep in make_plan(30, START) for arm in POLICIES]


def prepare(path):
    path.mkdir(parents=True, exist_ok=False)
    archive = subprocess.check_output(["git", "archive", BASE], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as content:
        content.extractall(path, filter="data")
    fk = "method/modular/fk_seating.py"
    if RUN == "stationary":
        for module in ("controller", "stationary_release"):
            file = Path(f"method/modular/{module}.py")
            shutil.copyfile(ROOT / file, path / file)
    elif RUN == "fallback":
        controller = "method/modular/controller.py"
        (path / controller).write_bytes(
            subprocess.check_output(["git", "show", f"{SETTLE}:{controller}"], cwd=ROOT)
        )
        for module in ("fk_seating", "site_seating", "fallback_seating"):
            file = Path(f"method/modular/{module}.py")
            shutil.copyfile(ROOT / file, path / file)
    else:
        (path / fk).write_bytes(subprocess.check_output(["git", "show", f"{FK}:{fk}"], cwd=ROOT))
    module = {
        "fallback": "fallback_seating",
        "stationary": "stationary_release",
        "fk": "fk_seating",
    }[RUN]
    (path / "submission/sol_fk_pair.py").write_text(
        f"""import json

from method.modular.robust_controller import _with_phase_trace
from method.modular.{module} import make_policy as make_candidate
from submission.policy_modular_calibrated import make_policy as make_baseline


def baseline():
    return _with_phase_trace(make_baseline())


def candidate():
    policy = make_candidate()
    return _FallbackTrace(policy) if hasattr(policy.controller, "fallback_releases") else _with_phase_trace(policy)


class _FallbackTrace:
    def __init__(self, policy):
        self.policy = policy
        self.traced = _with_phase_trace(policy)
        self.seen = 0
        self.settled = 0

    def reset(self):
        self.traced.reset()
        self.seen = 0
        self.settled = 0

    def __call__(self, obs):
        action = self.traced(obs)
        releases = self.policy.controller.fallback_releases
        for release in releases[self.seen:]:
            print("SOL_FALLBACK " + json.dumps(release), flush=True)
        self.seen = len(releases)
        settles = getattr(self.policy.controller, "fallback_settles", ())
        for settle in settles[self.settled:]:
            print("SOL_FALLBACK " + json.dumps(dict(settle, phase="SETTLE")), flush=True)
        self.settled = len(settles)
        return action
"""
    )
    script = path / "ops_modular/lead_shard.py"
    text = script.read_text()
    replacements = [
        ('modal.App("modular-shard")', f'modal.App("{APP}")'),
        (
            "if not 1_000_000 <= seed < 1_100_000:",
            f"if not {START - 1} <= seed < {START + 150}:",
        ),
        (
            '"MKL_NUM_THREADS": "1",',
            '"MKL_NUM_THREADS": "1",\n        "LP_NUM_THREADS": "2",\n'
            '        "SOL_PHASE_TRACE": "1",',
        ),
        ("timeout=3600, max_containers=200", f"timeout=5400, max_containers={CONTAINERS}"),
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
            '    releases = [json.loads(x.removeprefix("SOL_FALLBACK ")) for x in logs if x.startswith("SOL_FALLBACK ")]\n'
            '    return {"seed": seed, "seconds": time.monotonic() - start, "gl": gl,'
            ' "archive": str(dest / "eval"), "phases": phases, "fallback_releases": releases, **report["episodes"][0]}',
        ),
    ]
    if QUICK:
        replacements.append(
            (
                '    os.environ["MUJOCO_GL"] = gl',
                '    args.extend(["--max-steps", "40"])\n    os.environ["MUJOCO_GL"] = gl',
            )
        )
    for old, new in replacements:
        if text.count(old) != 1:
            raise ValueError(f"Unexpected sharder source for {old!r}")
        text = text.replace(old, new)
    script.write_text(text)


def run_tag(phase):
    if QUICK:
        suffix = "quick_v2" if RUN == "fallback" else "quick"
        return f"{RUN}_{phase}_{suffix}"
    return f"{RUN}_{phase}"


def read_lines(path):
    return list(map(json.loads, path.read_text().splitlines())) if path.exists() else []


def collect(path, phase, once, max_new=None):
    import modal

    from ops_modular.sol_shard_collect import fetch

    jobs = dict(groups())[phase]
    manifest = path / f"sol_render_{run_tag(phase)}.calls"
    output = path / f"sol_render_{run_tag(phase)}.jsonl"
    records = read_lines(output)
    valid = {(r["seed"], r["arm"]) for r in records if r.get("completed")}
    journal = {tuple(x["key"]): x for x in read_lines(manifest)}
    function = modal.Function.from_name(APP, "episode_cpu")
    unsent = [key for key in jobs if key not in journal]
    with manifest.open("a") as stream:
        for seed, arm in unsent[:max_new]:
            call = function.spawn(f"sol_{run_tag(phase)}_{arm}", seed, POLICIES[arm], HEAD_PATH)
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
    manifest = path / f"sol_render_{run_tag(phase)}.calls"
    output = path / f"sol_render_{run_tag(phase)}.jsonl"
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
            name = f"sol_{run_tag(phase)}_{arm}"
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
                    releases = [
                        json.loads(line.removeprefix("SOL_FALLBACK "))
                        for line in logs
                        if line.startswith("SOL_FALLBACK ")
                    ]
                    restored = dict(
                        episode,
                        arm=arm,
                        phases=phases,
                        archive=archive,
                        seconds=None,
                        fallback_releases=releases,
                    )
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
            call = function.spawn(
                f"sol_{run_tag(phase)}_{arm}_retry", seed, POLICIES[arm], HEAD_PATH
            )
            entry = {"key": list(key), "id": call.object_id}
            current[key] = entry
            stream.write(json.dumps(entry) + "\n")
            stream.flush()


def summarize(path):
    slots = {ep.seed: ep.source_slot for ep in make_plan(30, START)}
    published = read_lines(path / f"sol_render_{run_tag('full')}.jsonl")
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
    failure_types = {}
    for arm, rows in by_arm.items():
        wins = {seed for seed in seeds if rows[seed]["success"]}
        losses = seeds - wins
        terminal = {
            seed: (
                f"{rows[seed]['phases'][-1]['from']}/{rows[seed]['phases'][-1]['reason']}"
                if rows[seed]["phases"]
                else f"APPROACH/{rows[seed]['termination']}"
            )
            for seed in sorted(losses)
        }
        result[arm] = {
            "valid_episodes": len(rows),
            "matched_successes": len(wins),
            "matched_falls": sum(rows[s]["fell"] for s in seeds),
            "returned_worker_seconds": sum(r.get("seconds") or 0.0 for r in rows.values()),
            "matched_by_slot": {slot: sum(slots[s] == slot for s in seeds) for slot in range(6)},
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
            "terminal_failure_by_seed": terminal,
            "success_by_slot": {slot: sum(slots[s] == slot for s in wins) for slot in range(6)},
        }
        outcomes[arm] = wins
        failure_types[arm] = set(terminal.values())
    result["fk_rescues"] = sorted(outcomes["fk"] - outcomes["base"])
    result["fk_losses"] = sorted(outcomes["base"] - outcomes["fk"])
    result["new_failure_types"] = sorted(failure_types["fk"] - failure_types["base"])
    result["identical_phase_traces"] = sum(
        by_arm["base"][seed]["phases"] == by_arm["fk"][seed]["phases"] for seed in seeds
    )
    result["fallback_releases"] = {
        seed: row["fallback_releases"]
        for seed, row in by_arm["fk"].items()
        if row.get("fallback_releases")
    }
    return result


def watch(path, phase):
    output = path / f"sol_render_{run_tag(phase)}.jsonl"
    manifest = path / f"sol_render_{run_tag(phase)}.calls"
    planned = set(dict(groups())[phase])
    while True:
        collect(path, phase, once=True)
        retry(path, phase)
        rows = read_lines(output)
        valid = {(r["seed"], r["arm"]) for r in rows if r.get("completed")}
        failed = {r["call_id"] for r in rows if r.get("error")}
        calls = {tuple(row["key"]): row["id"] for row in read_lines(manifest)}
        unresolved = planned - valid
        if not unresolved or all(calls[key] in failed for key in unresolved):
            break
        time.sleep(45)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "collect", "retry", "summary", "watch"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--phase", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-new", type=int)
    variant = parser.add_mutually_exclusive_group()
    variant.add_argument("--fallback", action="store_true")
    variant.add_argument("--stationary", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--containers", type=int, choices=range(1, 31), default=30)
    args = parser.parse_args()
    if args.max_new is not None and (
        args.command != "collect" or not args.once or args.max_new < 0
    ):
        parser.error("--max-new requires collect --once and a nonnegative count")
    if args.fallback:
        APP = "modular-shard-sol-fallback-v2"
        RUN = "fallback"
        START = 3_002_200
    if args.stationary:
        APP = "modular-shard-sol-stationary-v1"
        RUN = "stationary"
        START = 3_001_100
    if args.quick:
        if not (args.fallback or args.stationary) or args.phase != "smoke":
            parser.error("--quick requires --fallback or --stationary and a smoke run")
        APP += "-quick"
        QUICK = True
    CONTAINERS = args.containers
    if args.command == "prepare":
        prepare(args.path)
    elif args.command == "collect":
        collect(args.path, args.phase, args.once, args.max_new)
    elif args.command == "retry":
        retry(args.path, args.phase)
    elif args.command == "watch":
        watch(args.path, args.phase)
    else:
        print(json.dumps(summarize(args.path), indent=2))
