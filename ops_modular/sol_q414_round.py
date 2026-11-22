"""Frozen GPU diagnostics and paired reruns for the unopened round-four cases."""

import argparse
import hashlib
import io
import json
import re
import shutil
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SEEDS = (1042006, 1062009, 1062038, 1072036, 1092014)
HOLD_SEEDS = tuple(sorted((*SEEDS, 1042003, 1062001, 1062026)))
HEAD = "64dfe105e7f716c0ab5b7a9c487b7a8a1cac06565070b390e75992dfbee4deff"
APPS = {arm: f"modular-shard-sol-q414-{arm}" for arm in ("baseline", "candidate", "measured")}
ROOT = Path(__file__).resolve().parents[1]

TRACE = """import json

from method.modular.controller import ModularController
from submission.policy_modular_combined import make_policy as make_combined


class PlacementTrace:
    def __init__(self, policy):
        self.policy = policy
        self.controller = policy.controller
        self.ready_lower = self.controller._ready_to_lower
        self.ready_release = self.controller._ready_to_release
        self.controller._ready_to_lower = self.lower
        self.controller._ready_to_release = self.release
        self.seen = 0

    def reset(self):
        self.policy.reset()
        self.seen = 0

    def capture(self, task, desired=None):
        c = self.controller
        step = c.machine.step
        if c.machine.elapsed % 4:
            return
        row = dict(
            step=step, phase=c.phase.value, elapsed=c.machine.elapsed,
            gap=(task["plate_xyz"] - c.target_xyz).tolist(),
            source_ref=c._source_site_height(),
            site=c.target_xyz.tolist(), plate=task["plate_xyz"].tolist(),
            fk_z=c.fk_plate_z, hand_z=c.measured_hand_z, desired_hand_z=c.commanded_hand_z,
            lower_start=c.lower_start_z, tilt=task["tilt_deg"], held=bool(task["held"]),
            fingers=self.state[7:9].tolist(), grip=float(self.state[44]),
            contacts={k: bool(v) for k, v in task["contacts"].items()},
            near=c.near_frames, stall_step=c.stall_step,
            supported=len(getattr(c, "supported_depths", ())),
            nominal=bool(ModularController._ready_to_release(c, task)),
            fallback=bool(c.proprio_arming), streak=c.placement_streak,
        )
        if desired is not None:
            row["align_error"] = (task["plate_xyz"] - desired).tolist()
        print("SOL_PLACE " + json.dumps(row), flush=True)

    def lower(self, task, desired):
        self.capture(task, desired)
        return self.ready_lower(task, desired)

    def release(self, task):
        ready = self.ready_release(task)
        self.capture(task)
        return ready

    def __call__(self, obs):
        self.state = obs["state"]
        action = self.policy(obs)
        c = self.controller
        for move in c.transitions[self.seen:]:
            print("SOL_MOVE " + json.dumps(move), flush=True)
        self.seen = len(c.transitions)
        return action


def make_policy():
    return PlacementTrace(make_combined())
"""


def prepare(dest, arm, integration=None):
    trace = TRACE
    allowed = HOLD_SEEDS if arm == "measured" else SEEDS
    original_app = "modular-shard"
    if arm == "measured":
        manifest = json.loads(integration.read_text())
        factory = "submission.policy_modular_round4:make_policy"
        if (manifest["base"], manifest["factory"], manifest["sidecar"]) != (
            "414f15a",
            factory,
            HEAD,
        ):
            raise ValueError("Unexpected combined integration snapshot")
        frozen = Path(manifest["folder"])
        for filename, digest in manifest["files"].items():
            if hashlib.sha256((frozen / filename).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Frozen source mismatch for {filename}")
        shutil.copytree(frozen, dest)
        original_app = manifest["app"]
        trace = trace.replace(
            "from submission.policy_modular_combined", "from submission.policy_modular_round4"
        )
        trace = trace.replace(
            "    return PlacementTrace(make_combined())",
            "    policy = make_combined()\n"
            "    policy.controller.hold_measured_release = True\n"
            "    return PlacementTrace(policy)",
        )
        trace = trace.replace(
            "        self.seen = len(c.transitions)",
            "        if c.phase.value in ('RELEASE', 'SETTLE', 'RETREAT') and (\n"
            "            c.machine.elapsed == 1 or c.open_finger_contact\n"
            "        ):\n"
            "            row = dict(step=c.machine.step - 1, phase=c.phase.value,\n"
            "                       contact=c.open_finger_contact, open_steps=c.open_steps,\n"
            "                       hold_error=float(abs(c.last_raw[4:11] - self.state[:7]).max()))\n"
            '            print("SOL_HANDOFF " + json.dumps(row), flush=True)\n'
            "        self.seen = len(c.transitions)",
        )
    else:
        dest.mkdir(parents=True, exist_ok=False)
        snapshot = subprocess.check_output(["git", "archive", "414f15a"], cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(snapshot)) as archive:
            archive.extractall(dest, filter="data")
    if arm in ("candidate", "measured"):
        path = "method/modular/fallback_seating.py"
        shutil.copyfile(ROOT / path, dest / path)
    (dest / "submission/sol_q414_trace.py").write_text(trace)
    file = dest / "ops_modular/lead_shard.py"
    source = file.read_text()
    source, count = re.subn(
        r"^    if (?:not 1_000_000 <= seed < 1_100_000|seed not in \([^\n]+\)):$",
        f"    if seed not in {allowed}:",
        source,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise ValueError("Unexpected development seed guard")
    if arm == "measured":
        # The integration export already validates the weight digest.
        source = re.sub(
            r"^    if hashlib\.sha256\(Path\(heads\).*\n        raise ValueError\([^\n]+\)\n",
            "",
            source,
            flags=re.MULTILINE,
        )
    replacements = [
        (f'modal.App("{original_app}")', f'modal.App("{APPS[arm]}")'),
        (
            '    if heads:\n        os.environ["ADV_HEADS"] = heads',
            f'    if heads != "/runs/heads/{HEAD}.npz" or hashlib.sha256(Path(heads).read_bytes()).hexdigest() != "{HEAD}":\n'
            '        raise ValueError("Unexpected calibrated head digest")\n'
            '    os.environ["ADV_HEADS"] = heads',
        ),
        (
            '    return {"seed": seed, "seconds": time.monotonic() - start, "gl": gl, **report["episodes"][0]}',
            '    lines = (dest / "eval/policy.stderr.log").read_text().splitlines()\n'
            '    phases = [json.loads(line[9:]) for line in lines if line.startswith("SOL_MOVE ")]\n'
            '    frames = [json.loads(line[10:]) for line in lines if line.startswith("SOL_PLACE ")]\n'
            '    handoff = [json.loads(line[12:]) for line in lines if line.startswith("SOL_HANDOFF ")]\n'
            '    return {"seed": seed, "seconds": time.monotonic() - start, "gl": gl,'
            ' "archive": str(dest / "eval"), "phases": phases, "frames": frames, "handoff": handoff, **report["episodes"][0]}',
        ),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError(f"Frozen sharder text changed: {old!r}")
        source = source.replace(old, new)
    file.write_text(source)


def journal(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def submit(runs, arm, seeds):
    import modal

    path = runs / f"sol_q414_{arm}.calls"
    submitted = {entry["seed"] for entry in journal(path)}
    function = modal.Function.from_name(APPS[arm], "episode_gpu")
    with path.open("a") as stream:
        for seed in seeds:
            if seed in submitted:
                continue
            result = function.spawn(
                f"sol_q414_{arm}",
                seed,
                "submission.sol_q414_trace:make_policy",
                f"/runs/heads/{HEAD}.npz",
            )
            row = {"seed": seed, "id": result.object_id}
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            print(row, flush=True)


def collect(runs, arm):
    from ops_modular.sol_shard_collect import fetch

    path = runs / f"sol_q414_{arm}.jsonl"
    saved = {entry["seed"] for entry in journal(path)}
    calls = [entry for entry in journal(path.with_suffix(".calls")) if entry["seed"] not in saved]
    with ThreadPoolExecutor(max_workers=5) as pool, path.open("a") as stream:
        for entry, row in zip(calls, pool.map(fetch, (c["id"] for c in calls)), strict=True):
            if row is None:
                continue
            row.update(seed=entry["seed"], call_id=entry["id"])
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            print(
                entry["seed"],
                row.get("success"),
                row.get("policy_error"),
                row.get("error"),
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "submit", "collect"))
    parser.add_argument("arm", choices=APPS)
    parser.add_argument("path", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, choices=HOLD_SEEDS)
    parser.add_argument("--integration", type=Path)
    args = parser.parse_args()
    allowed = HOLD_SEEDS if args.arm == "measured" else SEEDS
    if args.seeds and not set(args.seeds) <= set(allowed):
        parser.error("Seed outside the chosen experiment")
    if args.command == "prepare":
        if args.arm == "measured" and args.integration is None:
            parser.error("The measured hold export requires Astra's integration manifest")
        prepare(args.path, args.arm, args.integration)
    elif args.command == "submit":
        submit(args.path, args.arm, args.seeds or allowed)
    else:
        collect(args.path, args.arm)
