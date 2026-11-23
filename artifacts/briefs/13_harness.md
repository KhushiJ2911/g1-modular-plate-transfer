# Brief #13: observation-driven harness around bce1 (plate-state estimator + release supervisor)

Worktree: `/home/keyu/advaitic/wt-harness` (branch `harness`, at 5233dc7 = the exact code of our best
policy bce1, 63% dev). Commit your work on this branch with clear messages. Read first: `README.md`
(assignment rules, esp. lines 9, 57-69, 126-136), `submission/policy.py`, `submission/model.py`,
`method/bc/` (trainer, `dataset.py` EpisodeStore cache), `method/dagger/diagnostics.py`
(PrivilegedSensors, gripper_label, constants), `/home/keyu/advaitic/analysis/relabel_dagger.py`
(deterministic replay of saved `action_executed` from the seed; verified bit-for-bit).

## Rules (hard)
- At inference the harness sees ONLY `obs["rgb"]` (3,3,84,84) uint8 and `obs["state"]` (50,) float32.
  No seed, slot, simulator, privileged or success info. Privileged data is allowed only for TRAINING the
  estimator (disclosed). Fixed open-loop scripts / trajectory replay / per-seed lookup are not allowed;
  short observation-triggered primitives inside a state machine are fine.
- Do not modify the evaluator (`src/advaitic_eval`), physics, assets, controller, success checks.
- Do not change `submission/policy.py`'s existing behaviour (bce1 must stay reproducible).
- Never use seeds in [2600000, 2700000) (confirm) or [7310000, 7320000) (test). Dev = 1000000+.
- Laptop safety: any local process heavier than unit tests must run with `ADV_THROTTLE=1` (the thermal
  watchdog pauses it). Do not run full 30-episode evals yourself; I run evals on Modal.

## Failure analysis to target (from our logs)
- bce1 failures are dominated by the release: releases 2-4 cm from the target site succeed ~80%, at
  4-5 cm only 33%; tilt > 15 deg at release is bad; "released on rack but not settled" (D2) is common.
- Secondary: pickup slips / grasp misses, drops during carry.

## Deliverables
1. **Labels.**
   - `teacher_v1_dart0` (129 eps) and `teacher_v1_gen2` (191 eps, on the Modal data volume; I will sync
     it locally to `/home/keyu/advaitic/data/teacher_v1_gen2`) already contain per-step
     `priv_plate_rel`, `priv_target_site_rel`, `priv_plate_tilt_deg`, `priv_held`.
   - Write `method/harness/label_replay.py`: adds the same priv_* arrays to episodes WITHOUT them
     (dagger3 = 120 rollouts of bce1 itself, the most valuable distribution) by deterministic replay of
     `action_executed`, exactly as relabel_dagger.py does (fail loudly if the replay diverges, e.g.
     check `state` reproduces the stored state). Output to a NEW dir (never modify source data).
     Make the priv_* definitions identical to how the teacher computes them (find that code in
     `/home/keyu/advaitic/wt-teacher/method/teacher/`), and test equality on a teacher episode.
2. **Estimator** `method/harness/estimator.py` + `train_estimator.py`: small network,
   rgb (3 cams; image index t//2 per rgb_stride) + the 19 BC proprio dims → plate_rel (3),
   target_site_rel (3) or directly distance-to-site, tilt_deg, held (logit). Reuse the BC image encoder
   design (ImageNet ResNet-18 truncated after layer3 + spatial softmax) and the EpisodeStore uint8 cache
   if practical. Split train/val by EPISODE. Report val errors (cm, deg, held accuracy), especially in
   the release region (plate within 6 cm of the site). CLI must fit our Modal launcher:
   `python -m method.harness.train_estimator --device cuda --cache-dir /cache --run-dir RUN --data-dirs D1 D2 ... [args]`
   and save `RUN/last.pt` (weights + config + normalisation). Do a tiny CPU smoke run only.
3. **Harness policy** `submission/policy_harness.py` with `make_policy()`: wraps bce1 (`ADV_CKPT`) and the
   estimator (`ADV_ESTIMATOR`). A state machine (APPROACH → GRASP → CARRY → PLACE → RELEASE → SETTLE,
   with clear transitions from estimator + proprio) that owns the gripper decision:
   - release only when est. distance-to-site ≤ D, tilt ≤ T, est. held, stable for K steps;
   - after release, hold the arms still (observation-driven, e.g. command current joint targets) for the
     settle period; never re-grasp after a correct release;
   - grasp check: if est. held is false for N steps during CARRY, reopen and hand control back to bce1
     to re-approach;
   - optional ACT-style temporal ensembling of overlapping bce1 chunks (flag, default off).
   All thresholds in one config dict with env overrides for dev sweeps (`ADV_H_*`). Same output contract
   as `submission/policy.py` ((exec_steps, 20) in [-1, 1]). Log phase transitions to stderr compactly.
4. **Offline checks** (no simulator eval): unit tests for the state machine on synthetic estimator
   streams; an offline replay script that runs estimator + state machine over recorded dagger3 / teacher
   episodes and prints, per episode, when the harness WOULD release vs when the data released, and the
   true site distance / tilt at both moments.
5. Report (`-o` file): what you built, label-equality test result, estimator val metrics (from the smoke
   run, clearly marked as smoke), open questions, and the exact commands for me to: (a) label dagger3,
   (b) train the estimator on Modal, (c) run the harness eval
   (`ADV_CKPT=... ADV_ESTIMATOR=... advaitic-eval --policy submission.policy_harness:make_policy ...`).
Keep it simple and readable; prefer fewer moving parts over cleverness.
