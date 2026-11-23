# Codex brief #6: teacher v3 = v1 + one change at a time

Context: `method/teacher/` (committed at 36101fe). Same rules: do not modify `src/`, `tests/`, `pyproject.toml`, `uv.lock`; no GPU, no git. Running generation processes already have the old code in memory.

## Measured: v2 is a regression
Same 30 seeds (2000000–2000029):
- **v1: 27/30.** All 3 failures in source slot index 0 (placement or release).
- **v2: 6/30.** v1 succeeded on 21 seeds where v2 failed; the reverse happened 0 times.
- v2 failure categories: knocked_after_release 12, grasp_miss 9, place_tilt 2, not_released 1.
  - grasp_miss: the grasp gate waited 70–100 steps. v2 dropped the demo's base **feed-forward** everywhere and used deadzone bang-bang feedback, so the base hunts around the grasp pose.
  - knocked_after_release: at release the plate was close to the site (≈1 cm vs ≈4 cm in v1) but tilted 10–13°. After opening it leaned to 23–31°. Hypothesis: the base kept moving the plate sideways while it was already partly inside the slot, levering it.
- v2 changed six things at once, so we cannot attribute anything. We now ablate one at a time.

## Changes
Replace `--mode v1|v2` with **independent boolean flags**, all on top of v1 behaviour (v1 feed-forward + P pelvis tracking everywhere unless a flag says otherwise). Keep `--mode v1` and `--mode v2` working as presets (v2 = all flags on, as now) so old results stay reproducible. Every flag value goes into `run_id` and the JSONL record.

1. `--align-then-lower` (the main candidate). At `blend_end = release_step - pre_release_steps`, pause the demo clock (arm holds its pose at that index; the plate is still above the slot). Command the base, using the demo base feed-forward = 0 while paused plus deadzone-aware P feedback, so the **plate xy** reaches the nearest target site xy within 1.0 cm (max wait 150 steps). Then **freeze the base** (vx = vy = 0) for the rest of the lowering and the release: resume the demo clock for the arm/gripper with zero base commands. No base motion while the plate is inside the slot. Check in the demo data that the plate is still above the rack at `blend_end`. If it is not for some demos, pick the gate earlier (e.g. the last step where the demo's hand z is at its carry height); explain the choice.
2. `--tilt-correct` (v2's wrist correction), off by default.
3. `--guarded-retreat` (v2's retreat logic), off by default.
4. `--deadzone-min-speed` (v2's 0.06 m/s minimum), off by default **except** inside `--align-then-lower`'s alignment wait, where it is always used.
5. `--grasp-feedback-only` (v2's no-feed-forward grasp approach), off by default.
6. DART fix: in every mode, **no noise after the release transition** (currently only guarded in v2).

Then add `method/teacher/ablate.sh`, which runs one configuration on a seed range with 2 OS processes (split seeds) into a named JSONL, and `method/teacher/compare.py`, which prints paired results between two JSONLs on the same seeds (success counts, discordant pairs, exact McNemar p-value, per-slot table, failure categories). The README gets a short "ablation" section.

Report: what changed, anything unverified, exact commands.
