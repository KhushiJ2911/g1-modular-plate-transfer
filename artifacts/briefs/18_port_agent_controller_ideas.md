# Brief #18 (FAST, ~45 min): port seating/regrip ideas from the released BiGym2 Move Plate agent programs into our harness

Worktree `/home/keyu/advaitic/wt-harness` (branch `harness`, HEAD 9a19369). Same hard rules as briefs 13/15
(inference = obs rgb/state only; no evaluator/physics/success changes; never seeds 2600000+ / 7310000+;
`ADV_THROTTLE=1` for anything heavier than unit tests; no full evals; leave files uncommitted if git is
read-only, I'll commit). Everything new must be behind default-OFF flags (`ADV_H_*`) so existing results stay valid.

## Read first (network allowed for reading)
- Released programs (CC BY 4.0; attribute in comments + report):
  https://huggingface.co/datasets/SWIRL-Lab/bigym2-agent-rollouts/blob/main/policies/move_plate/astra-s1/policy.py
  https://huggingface.co/datasets/SWIRL-Lab/bigym2-agent-rollouts/tree/main/policies/move_plate/opus-s3
  (raw files via https://huggingface.co/datasets/SWIRL-Lab/bigym2-agent-rollouts/resolve/main/<path>).
  Extract precisely: (a) how they decide the plate is SEATED in the slot before opening (the "persistent joint
  tracking error" cue: which joints, thresholds, durations); (b) shallow-pinch detection from finger
  positions and the regrip procedure; (c) their release + retract sequence and timing.
- Our current state: bce1 + ACT-style temporal ensembling = ~71% dev. Remaining failures (replays): plate on the
  rack but never released (D1, 2-4/30), released but not settled / tilted 12-24° at 4-5 cm (D2, 2-3/30), dropped
  after reaching rack (E, 1-2/30), never grasped (A, 1/30). The estimator-gated release supervisor converts
  D1 into D2/E rather than successes, so release timing is not the issue: seating is.

## Build
1. `ADV_H_SEAT_CUE=1`: in PLACE, require the ported seating cue (from proprio: arm joint tracking error =
   commanded target (last action) vs measured joint position, persistent for N steps, i.e. the plate is
   resting on the rack and resisting) in addition to (or instead of, flag) the estimator gates, before RELEASE.
   Find the exact proprio indices of measured arm joint positions in the 50-D state (map them, cite evidence).
2. `ADV_H_REGRIP=1`: shallow-pinch detection from gripper/finger proprio during CARRY → controlled reopen,
   small advance, re-close (ported from Astra s1), at most once.
3. Unit tests on synthetic streams for both. Report: what the released programs do (with line refs), what you
   ported, exact env vars, state indices used and how you verified them. Keep it small (< 150 lines).
