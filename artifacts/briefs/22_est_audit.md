# Brief #22 (Sol, ~75 min): estimator accuracy near the rack -> better estimator if it's the bottleneck

Worktree `/home/keyu/advaitic/wt-est` (branch est2, from harness HEAD 0066f1a). Same hard rules: obs-only inference,
no evaluator/physics changes, never seeds 2600000+ / 7310000+, `ADV_THROTTLE=1` + nice locally, leave changes
uncommitted (lead commits). Modal GPU allowed for training/eval-of-estimator ONLY, app names `advaitic-est2-*`,
smoke first, total Modal spend for this brief <= $8 (T4/A10, short runs). Network: Modal CLI works.

## Context
The harness gates release/seating on a learned estimator (`est_v1b`, `/home/keyu/advaitic/ckpts/est_v1b.pt`, on
Modal `/runs/est_v1b/last.pt`; training log `/home/keyu/advaitic/ops/modal_est_v1b.log`, find its training code
via that log / ops/ / wt-harness). Its reported tilt MAE is ~1.4 deg near the rack, but that's an average; our
placement failures (plate tilted 23-25 deg on rack, hover 5-7 cm off, release before settle) and a tilt-correction
controller that HURT suggest the estimate may be wrong exactly where it matters (occlusion by the gripper, plate on
rack, late placement).

## Do
1. Audit: build a held-out set of near-rack frames with ground truth (teacher/flywheel/dagger data under
   /home/keyu/advaitic/data or regenerate from replays with privileged sensors, training seeds only). Report
   error of distance and tilt binned by true distance (0-3, 3-6, 6-10 cm), true tilt (<10, 10-20, >20 deg),
   phase (carry/place/released), and whether the gripper occludes. Key number: P(|tilt err| > 5 deg) and gate
   decision accuracy (true vs estimated "dist<=5cm & tilt<=12") near the rack.
2. If errors are large in the failure regime: fine-tune/retrain an `est_v2` on Modal emphasising those frames
   (oversample, tilt-on-rack examples, temporal smoothing input if cheap), same interface/IO as est_v1b so the
   harness can load it via ADV_ESTIMATOR. Compare on the same held-out set. Save to Modal `/runs/est_v2/last.pt`
   and locally `/home/keyu/advaitic/ckpts/est_v2.pt`.
3. If errors are small: say so plainly with numbers and stop (that rules the estimator out).
4. Report: audit table, decision, est_v2 metrics + exact ADV_ESTIMATOR path, Modal spend.
