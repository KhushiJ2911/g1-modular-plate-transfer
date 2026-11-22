"""Attribute changes on the paired 3002200 generation plan using existing action archives."""

import inspect
import json
import subprocess
from pathlib import Path

import modal

from ops_modular.modular_sim import image, volume

app = modal.App("modular-astra-generation-audit-3002200")
arms = {
    "base": "sol_fallback_full_base",
    "fallback": "sol_fallback_full_fk",
    "clearance": "astra_clearance_fresh_528c854",
}


def archive(seed, arm):
    if not (3002200 <= seed < 3002350 and arm in arms):
        raise ValueError("Use a declared generation arm and seed")
    folder = Path("/runs/modular")
    names = [arms[arm], arms[arm] + "_retry"]
    paths = []
    for name in names:
        candidates = [folder / f"{name}-{seed}", *sorted(folder.glob(f"{name}-{seed}-r[0-9]*"))]
        for directory in candidates:
            path = directory / "eval"
            report_path = path / "report.json"
            if not report_path.is_file():
                continue
            report = json.loads(report_path.read_text())
            if not report["integrity_ok"] or report["infrastructure_error"]:
                continue
            (row,) = report["episodes"]
            if row["seed"] == seed and row["completed"] and not row["policy_error"]:
                paths.append((path, row))
    if not paths or len({row["success"] for _, row in paths}) != 1:
        raise ValueError(f"Missing or conflicting completed reports: {arm} {seed}")
    return paths[0]


def analyze(base, candidate, physical):
    import numpy as np
    from bigym.loco import make
    from bigym.utils.physics_utils import COLLISION_MARGIN

    from method.modular.privileged import PrivilegedTaskState

    def read(path):
        p = Path(path)
        (row,) = json.loads((p / "report.json").read_text())["episodes"]
        with np.load(p / "episode_000.actions.npz", allow_pickle=False) as f:
            actions = f["actions"]
        assert actions.shape == (row["steps"], 20)
        return row, actions

    row, actions = read(base)
    transitions = [
        json.loads(line.removeprefix("SOL_PHASE "))
        for line in (Path(base) / "policy.stderr.log").read_text().splitlines()
        if line.startswith("SOL_PHASE ")
    ]
    phases = {t["to"]: t["step"] for t in transitions}
    out = dict(seed=row["seed"], slot=row["source_slot"], success=row["success"], phases=phases)
    if candidate:
        comparison, other = read(candidate)
        assert comparison["seed"] == row["seed"] and comparison["source_slot"] == row["source_slot"]
        n = min(len(actions), len(other))
        delta = abs(actions[:n] - other[:n])
        out["pair"] = {
            "candidate_success": comparison["success"],
            "steps": [len(actions), len(other)],
            "max_error": float(delta.max()),
            "changed": {},
        }
        for tolerance in (1e-7, 1e-5, 1e-3):
            changed = np.flatnonzero(delta.max(axis=1) > tolerance)
            if len(changed):
                i = int(changed[0])
                out["pair"]["changed"][str(tolerance)] = {
                    "step": i,
                    "axes": np.flatnonzero(delta[i] > tolerance).tolist(),
                    "phase": next(
                        (t["to"] for t in reversed(transitions) if t["step"] < i), "APPROACH"
                    ),
                }
    if not physical:
        return out
    env = make("move_plate", camera_keys=())
    try:
        ts = env.reset(seed=row["seed"])
        labels = PrivilegedTaskState(env)
        inner = env.inner_env
        pads = {g.id for hand in inner.robot.grippers.values() for g in hand.pad_geoms}
        colliders = {g.id for g in labels.plate.colliders}
        opened = phases.get("RELEASE")
        times = {v: k for k, v in phases.items() if k in ("RELEASE", "SETTLE", "RETREAT", "HOLD")}
        out["geometry"] = []
        cleared = reacquired = False
        for i, command in enumerate(actions):
            if i in times or (opened is not None and i >= opened and not reacquired):
                state = ts.low_dim_obs
                truth = labels(state)
                event = times.get(i)
                if opened is not None and i >= opened:
                    if not truth["held"] and not cleared:
                        cleared = True
                        event = "first_clear"
                    if cleared and truth["held"] and not reacquired:
                        reacquired = True
                        event = "first_recontact"
                if event:
                    contacts = []
                    for ct in inner.data.contact:
                        a, b = int(ct.geom1), int(ct.geom2)
                        if ct.dist > COLLISION_MARGIN:
                            continue
                        if a in pads and b in colliders:
                            pad, normal = a, -np.asarray(ct.frame[:3])
                        elif b in pads and a in colliders:
                            pad, normal = b, np.asarray(ct.frame[:3])
                        else:
                            continue
                        contacts.append(
                            {
                                "pad": inner.model.geom(pad).name,
                                "plate_to_pad_normal": normal.tolist(),
                                "from_plate_center": (ct.pos - truth["plate_xyz"]).tolist(),
                            }
                        )
                    out["geometry"].append(
                        {
                            "step": i,
                            "event": event,
                            "held": truth["held"],
                            "supported": truth["contacts"]["target_rack"],
                            "tilt": truth["tilt_deg"],
                            "gap": (truth["plate_xyz"] - truth["site_xyz"]).tolist(),
                            "pads": contacts,
                        }
                    )
            ts = env.step(command)
        if not ts.last() or (
            env.episode_succeeded(),
            env.episode_fell(),
            str(env.episode_termination()),
        ) != (row["success"], row["fell"], row["termination"]):
            raise ValueError("Archived actions did not reproduce the original terminal outcome")
    finally:
        env.close()
    return out


@app.function(
    image=image, cpu=1, memory=1536, timeout=600, max_containers=2, volumes={"/runs": volume}
)
def inspect_pair(seed, candidate=None, physical=False):
    volume.reload()
    base, _ = archive(seed, "base")
    other = str(archive(seed, candidate)[0]) if candidate else None
    script = "import json\nfrom pathlib import Path\n" + inspect.getsource(analyze)
    script += f"\nprint(json.dumps(analyze({str(base)!r},{other!r},{physical!r})))"
    result = subprocess.run(
        ["/root/code/.venv/bin/python", "-c", script],
        cwd="/root/code",
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:])
    return json.loads(result.stdout.splitlines()[-1])
