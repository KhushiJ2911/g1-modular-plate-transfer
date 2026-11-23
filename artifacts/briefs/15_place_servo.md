# Brief #15: estimator-driven placement servo (+ grasp-approach check) in the harness. TODAY, keep it small.

Worktree `/home/keyu/advaitic/wt-harness` (branch `harness`, HEAD 095ff8b). Builds on your brief-13 harness
(`submission/harness.py` ReleaseSupervisor, `submission/policy_harness.py`, `submission/estimator.py`).
Same hard rules as brief 13 (README 57-69, 126-136): inference sees only obs rgb/state; no evaluator /
physics / success-check changes; never seeds 2600000+ or 7310000+; `ADV_THROTTLE=1` for anything heavier
than unit tests; do not run full evals (I run them on Modal). Commit on the branch when done (if git is
read-only again, leave the files and say so).

## Problem
bce1 + release supervisor can only fix release timing. Remaining failures: the plate arrives a few cm off
the slot or tilted (D2 "on rack, not settled": end site distance 3-5 cm, tilt 24-40 deg), and short carries.
The estimator now gives plate_rel / site_rel in the pelvis frame with ~0.7 cm distance error and ~1.4 deg
tilt error near the rack (est_v1b: `/runs/est_v1b/last.pt` on Modal; I can't give you a local copy right now,
use the smoke checkpoint for plumbing tests).

## Build (each behind its own flag, default OFF so existing ablations stay valid)
1. `ADV_H_PLACE_SERVO=1`: in PLACE (estimated plate within place_distance of the site, held), override the
   BASE planar velocity command (and yaw only if clearly helpful) with a proportional controller on the
   estimated horizontal plate→site error in the pelvis frame: v = clip(kp * err_xy, ±vmax), kp and vmax in
   config (`place_kp`, default ~2.0 1/s like the teacher's residual pelvis tracking in
   `/home/keyu/advaitic/wt-teacher/method/teacher/teacher.py`; read how the teacher maps a desired pelvis
   correction to the base command and reuse the same sign/frame conventions and action normalisation). Keep
   bce1's arm actions. Only servo while not yet released; after RELEASE keep brief-13 behaviour.
   Optionally low-pass the estimate (EMA) to avoid chatter; config key.
2. `ADV_H_TILT_HOLD=1` (only if cheap and clearly correct): if estimated tilt > release_tilt while in PLACE,
   do not let the release fire and hold the base still for a few steps (bce1 often corrects itself). No new
   wrist controller unless the mapping is obvious from the action layout.
3. Unit tests for the servo math on synthetic estimates (sign/frame correctness is the main risk: test that a
   plate +x of the site produces a base command that moves the pelvis toward -x of that error, using the
   same frame definitions as the labels).
Report (`-o` file): what changed, the exact env vars, the sign-convention evidence, test results.
Keep it under ~150 new lines; simplest correct thing wins. Time budget: ~1 hour.
