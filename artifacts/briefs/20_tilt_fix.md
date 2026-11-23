# Brief #20 (FAST, ~40 min): observation-only tilt correction + stall release in the harness

Worktree `/home/keyu/advaitic/wt-harness` (branch harness, HEAD 1690d4c; includes your #13/#15/#18 work and the
merged resmlp head). Same hard rules (inference = obs rgb/state only; no evaluator/physics/success changes; no
seeds 2600000+/7310000+; `ADV_THROTTLE=1` beyond unit tests; no full evals; leave files uncommitted, I commit).
New behaviour behind default-OFF flags.

## Evidence (replay of our best valid run: bce1 + uniform temporal ensembling + release supervisor + seat cue, 24/30)
Remaining 6 failures: 2× plate ON the rack 2.9–7.9 cm from the slot but tilted 23–25° → the release tilt gate
(12°) never opens and it holds until timeout; 1× hovering at 5.7 cm / 14° without rack contact, holding forever;
1× released at 5.6 cm / 12.5° and did not settle; 1× drop in carry; 1× no grasp. Success tolerance: within the
slot's 5 cm and 20° (task checker), held 1 s after release by both grippers.

## Build
1. `ADV_H_TILT_FIX=1`: in PLACE while held and estimated tilt > `tilt_fix_deg` (default 10), run a bounded
   extremum-seeking correction on the LEFT wrist pitch/roll joint targets (action dims for left wrist roll/pitch;
   verify indices from `src/advaitic_eval/action_layout.json`): perturb one axis by ±`tilt_probe_rad` (default
   0.02) for a few control steps, compare the estimator's tilt before/after (EMA-filtered; estimator tilt MAE ≈
   1.4° near the rack), keep the direction that reduces tilt, accumulate an offset clipped to ±`tilt_limit_rad`
   (default 0.18, same as the teacher's wrist_limit_rad), alternate axes. Hold the base still while correcting.
   The teacher does this with privileged geometry (`/home/keyu/advaitic/wt-teacher/method/teacher/teacher.py`
   `_correct_tilt`, `wrist_step_rad` 0.01, `wrist_limit_rad` 0.18): reuse its step sizes/limits, not its
   privileged inputs. Offsets are added on top of bce1's (ensembled) wrist targets while in PLACE and kept
   through RELEASE/SETTLE.
2. `ADV_H_STALL_RELEASE=1`: if PLACE has lasted > `stall_steps` (default 300 control steps) with estimated
   distance ≤ `stall_distance_m` (default 0.05) and tilt ≤ `stall_tilt_deg` (default 18), release (normal
   RELEASE/SETTLE path) even if the strict gates never fired.
3. Unit tests with a synthetic tilt model (tilt = |A·offset + b|) showing the descent converges and respects
   limits; stall timer logic. Report exact env vars + defaults.
