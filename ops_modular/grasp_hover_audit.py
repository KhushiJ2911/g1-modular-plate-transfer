"""Check paired source identity, action prefixes, and commanded base recovery on fresh episodes."""

import json
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np

from method.modular.phases import TIMEOUTS, Phase


def coverage(seed):
    """Screen a fresh slot-5 seed until the frozen policy either needs the base or releases."""
    from advaitic_eval.environment import make_environment, observation, observed_source_slot
    from method.modular.controller import TARGET_STANDOFF_XY
    from submission.policy_hover import make_policy

    env = make_environment()
    try:
        ts = env.reset(seed=seed)
        assert observed_source_slot(env) == 5
        policy = make_policy()
        policy.reset()
        ctl = policy.controller
        peak = 0.0
        retry = False
        step = 0
        while True:
            obs = observation(ts)
            phase = ctl.phase
            action = policy(obs)
            if phase == Phase.ALIGN and ctl.retry_hand_z is not None:
                retry = True
                goal = ctl.target_xyz[:2] - TARGET_STANDOFF_XY
                peak = max(peak, float(np.linalg.norm(goal - obs["state"][18:20])))
            active = phase == Phase.ALIGN and ctl.stance_goal is not None
            if active or ctl.phase in (Phase.RELEASE, Phase.HOLD):
                break
            ts = env.step(action)
            if ts.last():
                break
            step += 1
        return dict(
            seed=seed,
            step=step,
            phase=ctl.phase.value,
            active=active,
            retry=retry,
            max_retry_distance=peak,
            transitions=ctl.transitions,
        )
    finally:
        env.close()


def reach(frames):
    """Static FK and position-only reach bounds under measured and restored pelvis positions."""
    from method.modular.controller import TARGET_STANDOFF_XY
    from method.modular.kinematics import G1Kinematics, pelvis_rotation

    kin = G1Kinematics()
    result = []
    for f in frames:
        state = np.zeros(50)
        state[18:22] = f["base"]
        state[:7] = f["arm_cmd"]
        kin.set_state(state)
        rot = pelvis_rotation(state)
        target = np.asarray(f["goal"])
        q0 = state[:7].copy()
        commanded = state[18:21] + rot @ kin.hand_pose("left")[0]
        row = dict(
            step=f["t"],
            phase=f["phase"],
            seed=f["seed"],
            command_residual=float(np.linalg.norm(commanded - target)),
            measured_residual=float(np.linalg.norm(np.asarray(f["fk"]) - target)),
        )
        for name, xy in (
            ("current", state[18:20].copy()),
            ("restored", np.asarray(f["target"][:2]) - TARGET_STANDOFF_XY),
        ):
            origin = np.r_[xy, state[20]]
            local = rot.T @ (target - origin)
            q = kin.ik("left", local, q_init=q0, iters=100)
            state[:7] = q
            kin.set_state(state)
            row[name] = float(np.linalg.norm(kin.hand_pose("left")[0] - local))
        result.append(row)
    return result


def inspect(folder):
    path = Path(folder)
    report = json.loads((path / "report.json").read_text())
    e = report["episodes"][0]
    assert report["integrity_ok"] and e["completed"] and not e["policy_error"]
    with np.load(path / "episode_000.actions.npz") as data:
        actions = data["actions"]
    assert len(actions) == e["steps"]
    closing = np.flatnonzero(actions[:, 18] > 0)
    walking = np.flatnonzero(np.any(np.abs(actions[:, :2]) > 1e-5, axis=1))
    movement = walking[walking > closing[0]] if len(closing) else np.array([], dtype=int)
    # A LOWER retry necessarily follows the full first 210-step insertion timeout.
    blocks = np.split(movement, np.flatnonzero(np.diff(movement) > TIMEOUTS[Phase.LOWER]) + 1)
    intervals = [[int(b[0]), int(b[-1])] for b in blocks if len(b)]
    return actions, dict(
        seed=e["seed"],
        slot=e["source_slot"],
        steps=e["steps"],
        success=e["success"],
        fell=e["fell"],
        source=report["submission_sha256"],
        policy=report["policy"],
        archive=folder,
        post_close_walks=intervals,
    )


def compare(pair):
    base, a = inspect(pair[0])
    candidate, b = inspect(pair[1])
    assert a["seed"] == b["seed"] and a["slot"] == b["slot"]
    n = min(len(base), len(candidate))
    delta = np.max(abs(base[:n] - candidate[:n]), axis=1)
    changed = np.flatnonzero(delta > 1e-5)
    return dict(
        seed=a["seed"],
        slot=a["slot"],
        base=a,
        candidate=b,
        first_delta=int(changed[0]) if len(changed) else None,
        max_delta=float(np.max(delta)),
    )


def effect(pair):
    """Check the physical approach before and after a fresh retry, using original commands."""
    from hover import replay

    from method.modular.controller import TARGET_STANDOFF_XY

    row = compare(pair)
    spans = row["candidate"]["post_close_walks"]
    start = spans[1][0] if len(spans) > 1 else None
    with np.load(Path(pair[1]) / "episode_000.actions.npz") as data:
        moving = np.flatnonzero(np.any(np.abs(data["actions"][:, :2]) > 1e-5, axis=1))
    retry_walk = moving[moving >= start] if start is not None else []
    for arm, folder in zip(("base", "candidate"), pair, strict=True):
        frames = replay(Path(folder), False)["frames"]
        site = np.asarray(frames[0]["site"])
        stance = site[:2] - TARGET_STANDOFF_XY
        samples = frames[start:] if start is not None else []
        opened = next((f["t"] for f in samples if f["open"]), None)
        distances = [np.linalg.norm(np.asarray(f["base"][:2]) - stance) for f in samples]
        walk = [frames[t] for t in retry_walk if t < len(frames)]
        row[arm].update(
            fixed_site=site.tolist(),
            start=start,
            opened=opened,
            opening_steps=[
                f["t"] for prev, f in pairwise(frames) if f["open"] and not prev["open"]
            ],
            stance_error_start=distances[0] if distances else None,
            closest_stance_after=min(distances) if distances else None,
            matched_walk_frames=len(walk),
            true_contact_walk_frames=sum(f["rack"] for f in walk),
            peak_tilt_during_walk=max((f["true_tilt"] for f in walk), default=None),
            start_gap=(np.asarray(walk[0]["true_plate"]) - site).tolist() if walk else None,
            last_walk_gap=(np.asarray(walk[-1]["true_plate"]) - site).tolist() if walk else None,
            stance_error_end=float(np.linalg.norm(np.asarray(frames[-1]["base"][:2]) - stance)),
            end_gap=(np.asarray(frames[-1]["true_plate"]) - site).tolist(),
            end_tilt=frames[-1]["true_tilt"],
            end_rack=frames[-1]["rack"],
            end_held=frames[-1]["held"],
        )
    return row


if __name__ == "__main__":
    jobs = json.loads(sys.argv[1])
    if sys.argv[2:] == ["screen"]:
        result = coverage(jobs)
    elif sys.argv[2:] == ["effect"]:
        result = [effect(pair) for pair in jobs]
    elif sys.argv[2:] == ["reach"]:
        result = reach(jobs)
    else:
        result = [compare(p) for p in jobs]
    print(json.dumps(result))
