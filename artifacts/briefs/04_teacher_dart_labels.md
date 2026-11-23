# Codex brief #4: DART noise injection + privileged labels in teacher demos

Context: `method/teacher/` (your previous work, v2 committed at 8b4744d). Same rules: do not modify `src/`, `tests/`, `pyproject.toml`, `uv.lock`; no GPU, no git. Running generation processes already have the old code loaded; your edits will not affect them.

## Why
A BC student trained on 54 human demos predicts the held-out demos well (gripper-close timing within ~20 steps, arm L1 ≈ 0.01) but scores **0/30 in closed loop**: compounding error takes it to states never seen in training. We will (1) generate many more teacher demos and (2) use **DART** (Laskey et al., 2017): perturb the *executed* actions with noise while recording the teacher's *clean* action as the label, so the data contains off-nominal states paired with corrective labels. The teacher is closed-loop on the base and on the object (v2), and its arm targets are absolute joint positions, so the clean label after a perturbation is naturally corrective.

We also want **privileged per-step labels** for an auxiliary loss in the student (training only).

## Changes
1. `--dart-noise SCALE` (float, default 0 = off) in `run_teacher.py`. Noise is an Ornstein–Uhlenbeck process per episode (θ = 0.15 per step, so it is temporally correlated, like real drift), on normalized action dims: base vx/vy std `0.10*SCALE`, left arm dims 4–10 std `0.02*SCALE`. **No noise** on dims 2, 3, 11–19 (height, yaw, right arm, grippers). Also **no noise** from 15 steps before until 15 steps after each gripper transition, and none during the v2 release/retreat phase (we want perturbations in reach/carry/approach, not at the moment of contact). Executed action = clip(clean + noise, −1, 1). The teacher's internal state must evolve from the **real observations** (it already does), not from the noise.
   - The episode seed must determine the noise (use a separate `np.random.default_rng(seed + 10_000_019)`), so runs are reproducible.
   - DART scale must be part of `run_id`.
2. Saved npz gains: `action` = **clean teacher label** (what the student learns), `action_executed` = what was sent to the env, `dart_noise` = scale. The `state` and `rgb` are what was observed (unchanged).
3. Privileged per-step labels (saved at **every step**, float32), computed in the runner before each action:
   - `priv_plate_pos` (T,3): plate world position.
   - `priv_plate_rel` (T,3): plate position in the **pelvis frame** (rotate the world offset by −pelvis yaw; use observed pelvis x, y, z, yaw = state 18–21).
   - `priv_target_site_rel` (T,3): nearest target-rack site to the plate, in the pelvis frame.
   - `priv_plate_tilt_deg` (T,), `priv_held` (T,) bool.
   - `priv_start_rack_offset` (2,), `priv_target_rack_offset` (2,): rack offsets from nominal.
4. Same labels for the 60 **human demos**: add `method/teacher/label_human.py` that replays each human demo's actions on its own seed in the evaluator env (this reproduces 59/60 demos exactly; skip/flag demo 45 and any replay that does not succeed) and writes the same `priv_*` arrays into a sidecar `DIR/ep_<idx>.priv.npz` next to `/home/keyu/advaitic/data/human60/ep_<idx>.npz`, aligned to the human npz's `state` rows (T = len(action)). Check alignment by asserting the replayed `low_dim_obs` matches the stored `state` within 0.05 at every step; report the max deviation. Resumable (skip existing sidecars).
5. Update README: DART explained in two plain sentences; exact commands.

Report: what changed, anything unverified, exact commands.
