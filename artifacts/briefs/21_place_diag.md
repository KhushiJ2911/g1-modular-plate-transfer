# Brief #21 (Astra, ~60 min): placement failure diagnosis -> one evidence-backed harness fix

Worktree `/home/keyu/advaitic/wt-harness` (branch harness, HEAD 0066f1a). Hard rules: inference = obs rgb/state only;
no evaluator/physics/success changes; never use seeds 2600000+ or 7310000+ (held-out); `ADV_THROTTLE=1` and
`nice` for anything beyond unit tests (laptop, thermal); no full evals; leave changes uncommitted (lead commits).
New behaviour behind default-OFF `ADV_H_*` flags.

## Context
Best policy (79% dev): bce1 + uniform temporal ensembling + estimator-gated release supervisor + seat cue (10 steps).
~60% of remaining failures are at the rack: plate on rack but tilted 23-25 deg (tilt gate 12 deg never opens),
hover 5-7 cm from slot without contact, released but not settled. Our tilt correction (ADV_H_TILT_FIX,
extremum seeking on left wrist with estimator tilt) HURT (64/90 vs 68/90); stall release was neutral. Success =
plate within 5 cm and 20 deg of the slot, held 1 s after release.

## Data (already downloaded, actions.npz + episodes.csv per episode)
`/home/keyu/advaitic/results/modal/evals/dev_s2_st10_r{1,2,3}` (best config, 19 failures),
`dev_t_fix_r{1,2}` (tilt fix). Replay tools: `/home/keyu/advaitic/analysis/replay_eval.py RESULT_DIR OUT.jsonl`
(camera-free, same physics, privileged state for analysis only) and `analysis/classify_failures.py`. Run with
`/home/keyu/advaitic/robot-learning-assignment/.venv/bin/python -I`, env `MUJOCO_GL=egl OMP_NUM_THREADS=1`.
Harness phase/estimator logs are in each run's policy.stderr.log if present.

## Do
1. Replay all failures; extend the replay (separate script under analysis/, don't break the old one) to log per
   step near the rack: true plate-to-slot distance + tilt, contacts, phase, the gripper command. Where possible
   also the estimator's distance/tilt (load `/home/keyu/advaitic/ckpts/est_v1b.pt` on the rendered obs if the
   replay renders, otherwise say it can't).
2. Answer with numbers: (a) in tilted-on-rack failures, what is the plate's tilt trajectory from first rack
   contact, and did the policy ever bring it under 20 deg? (b) Why did the tilt fix hurt: compare t_fix vs s2_st10
   on paired seeds (which seeds flipped, what happened). (c) Hover failures: how far above/beside the slot, is it
   a policy stall (actions ~constant) or oscillation? (d) Estimator error near the rack vs ground truth.
3. Implement the single fix best supported by (a)-(d), behind a flag, with unit tests. Candidates (choose by
   evidence, not by this list): release when tilt is within success tolerance (20 deg) rather than 12; a short
   lower-and-settle motion (descend a few mm with gripper open) when seated-but-tilted; a lateral nudge toward
   the slot during hover; anti-stall perturbation.
4. Report: findings table, the flag(s) + defaults, exact env vars to eval, and a predicted effect (number of the
   19 failures it should flip, with seeds).
