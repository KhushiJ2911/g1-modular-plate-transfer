# Codex brief #7: DAgger-style gripper relabelling from student rollouts

Context: `method/bc/` and `submission/` (your previous work, committed at f674fe2). Rules unchanged: do not modify `src/`, `tests/`, `pyproject.toml`, `uv.lock`; no GPU, no git. Model env `/home/keyu/advaitic/mlenv/bin/python`; sim env `/home/keyu/advaitic/robot-learning-assignment/.venv/bin/python` (has bigym). Privileged sensors already exist in `/home/keyu/advaitic/wt-teacher/method/teacher/diagnostics.py` (`PrivilegedSensors.snapshot`); copy what you need into `method/dagger/`, do not import across worktrees.

## Measured
Student v1a (human60 + 73 teacher demos, 8k steps) scores **0/30 on dev**. Replaying its saved actions with privileged tracking shows:
- 18/30 **no grasp** (plate never lifts).
- **10/30 placed but never released:** the plate sits in the target rack (2.8–4.8 cm from a site, tilt 4–11°, touching the rack), the gripper stays closed until timeout, and the arm settles into a small jitter for > 1,000 steps.
- 2/30 reached the target, then failed.
- An offline probe rules out copying the gripper state: flipping state dims 7, 8, 44, 45 does not change the gripper prediction. The release decision comes from vision and arm pose, and the student's settled pose is off-distribution, so it never fires.

This is the covariate-shift problem DAgger (Ross et al., 2011) addresses: collect states the **student** visits, label them with an expert, and retrain on the union. Our teacher cannot label arbitrary mid-episode states for the arm, but the **gripper** decision has a simple privileged oracle.

## Changes
1. **`method/dagger/collect.py`:** roll out a BC checkpoint (via `submission.policy.BCPolicy`, the exact eval policy and chunking) in the evaluator environment (`advaitic_eval.environment.make_environment()`, unchanged) on given training seeds (`--seeds A:B`, never dev or final blocks; refuse any seed < 2,000,000). Run the policy in a subprocess if needed for the two Python envs (the evaluator already does this with `--policy-python`; reuse its approach or protocol rather than inventing one), or make `collect.py` itself run in the sim env and talk to a policy worker in the mlenv. Save each episode as `DIR/ep_<seed>.npz` in the **same format** as the teacher data (`rgb` every 2nd step, `rgb_stride`, `state`, `action` at every step, `seed`, `slot`, `success`), atomically, resumable, plus a JSONL summary per episode (outcome category as in `replay_eval`: no_grasp / placed_not_released / …).
   - `action` = the student's executed action, **except** dim 18 is replaced by the oracle label where defined.
   - New key `label_mask` (T, 20) bool: which dims are supervised. Arm, base and all other dims are **False** (we do not trust the student's own arm actions as labels). Dim 18 is True where the oracle is defined.
   - **Gripper oracle** (privileged, training only, documented in the README):
     - **open** (−1) when the plate is touching the target rack, the nearest target-site distance is ≤ 0.05 m, and tilt ≤ 20°;
     - **closed** (+1) when the plate is held, lifted ≥ 3 cm above its start, and not touching the target rack;
     - otherwise undefined (mask False).
     Put the thresholds in constants and log the fraction of steps per label.
   - `--relabel-execute`: optional flag that **executes** the oracle's gripper label when defined (instead of the student's), so the rollout continues past the release and we also see post-release states. Default off; record which mode in the npz.
   - `--max-steps`: stop early (default 1,700) and `--stop-after-placed N`: end the episode N steps after the plate first satisfies the "open" condition (saves sim time; default 150).
2. **Dataset:** honour an optional `label_mask` key in any episode (default all True for the existing data); chunk targets use per-step masks, and the loss uses `masked_l1` with this mask (combined with the existing active-dim mask). The `val_episodes` split works per directory as before.
3. **Docs:** a README section "DAgger gripper relabelling": why, the oracle, commands.
4. **Smoke test:** a fake episode with a partial `label_mask` contributes loss only on masked-in dims (check gradients or the loss value).

Report: what changed, design choices with one-line reasons, anything unverified, exact commands (collection on 2 OS processes over seeds 2400000–2400060 with checkpoint `/home/keyu/advaitic/ckpts/student_v1a.pt`, then training with an added `dagger` data dir).
