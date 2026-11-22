# Brief 21 · Placement diagnosis

**The height guard is predicted to recover 5 of 19 baseline failures.** With a 2 cm estimated height limit on release, seed 1000024 recovered in repeats 1 and 2 and seed 1000044 recovered in all three. All five scored the physical one-second success latch in short continuations from their saved action prefixes. End-to-end success rate remains unmeasured.

## Scope and replay

The three seat-cue repeats succeeded on 23, 24 and 24 of 30 episodes (71/90). The two available tilt-fix repeats succeeded on 20 and 21 (41/60). Repeats reuse the same 30 balanced dev seeds. A run–seed pair is one observation, so the 19 baseline failures include repeated seeds.

All 38 failures and the eight successful sides of paired flips reproduced both their recorded outcomes and terminal steps, across 70,124 executed actions. The offline replay executes the saved 20-channel actions from the recorded reset seed. Row `t` describes the state **before** action `t`. Distance and signed world offsets refer to the nearest of six destination slot centres. Tilt uses the task's plate normal. The trace records pad–plate, rack, table and floor contact. A positive pad-contact predicate can persist with the gripper commanded fully open. The supervisor's logged phase was aligned against every executed left-gripper command. The final row has no outgoing action.

The estimator receives the standard three 84 × 84 RGB views in policy order and the public 50-dimensional state. The trace contains the original rounded estimates at logged transitions and local checkpoint estimates on sampled observations. Regular samples are every 200 control steps inside 10 cm. Additional samples cover first rack contact, PLACE entry, the neighbourhood of opening and each decision with an active wrist-probe base hold. The 46 traces contain 382 locally scored observations. Geometry is recorded on every step. Signed estimation errors below are estimate minus replay truth.

## What failed at the rack

| Baseline outcome | Run–seed cases | Replay evidence |
| --- | ---: | --- |
| Opened above the slot | 11 | All opened 3.48–4.23 cm above the slot centre (median 3.70 cm), with true tilt only 4.34–8.22°. Seven touched the rack at opening. At timeout, seven still touched a gripper pad, one sat outside the distance limit without pad contact, and three lay on the table. |
| Kept the gripper closed at the rack | 5 | Four ended in PLACE and one in CARRY. All five touched the rack and a pad at timeout. Four spent time inside both pose tolerances while closed and touching the rack. |
| Missed the grasp | 3 | Seed 1000003 in every repeat. The plate never touched a pad or the destination rack. |

Of the five closed cases, the four inside both pose tolerances were seed 1000000 in repeats 1 and 2, 1000010 in repeat 1 and 1000031 in repeat 3. Seed 1000000 in repeat 3 ended 7.11 cm from the slot at 67.3°. The true horizontal and vertical offsets there were 6.23 cm beside and 3.51 cm below the slot with rack contact.

### Tilt from first rack contact

Each row below is a baseline failure ending in rack contact above 20°. Columns +10, +25 and +50 are control steps after first contact. Temporary losses of contact between those samples are possible. The opening column gives the first open command in RELEASE.

| Seed | Repeat | First contact step | First tilt | +10 | +25 | +50 | Opening step / tilt | First step above 20° after contact | Final tilt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 1000041 | 1 | 537 | 7.0° | 6.7° | 6.0° | 5.0° | 588 / 5.0° | 619 | 24.4° |
| 1000041 | 2 | 538 | 6.9° | 6.7° | 5.9° | 4.8° | 586 / 5.1° | 615 | 25.3° |
| 1000041 | 3 | 539 | 7.2° | 6.9° | 6.1° | 5.1° | 594 / 4.9° | 609 | 25.3° |
| 1000044 | 1 | 495 | 4.8° | 4.1° | 18.1° | 26.9° | 504 / 4.3° | 521 | 27.0° |
| 1000044 | 2 | 499 | 4.9° | 4.4° | 9.5° | 26.8° | 508 / 4.5° | 530 | 27.0° |
| 1000044 | 3 | 499 | 5.0° | 4.1° | 9.6° | 20.3° | 506 / 4.6° | 533 | 20.3° |
| 1000000 | 3 | 785 | 10.5° | 1.4° | 6.0° | 5.6° | Stayed closed | 994 | 67.3° |

All seven were already under 20° at first contact. Six opened 15–31 steps before crossing 20°. After their first crossing, none of the seven returned below 20°. Seed 1000041 was outside 5 cm throughout closed rack contact in all three repeats. Widening the tilt gate would leave that position error unresolved.

Seven tilt-fix failures also ended tilted in rack contact. All seven first touched at 4.1–8.1°. Six received a normal opening command at 3.3–10.4°. Seed 1000001 in repeat 1 reopened following a lost-hold estimate. The individual tilt-fix trajectories are in the per-episode summary.

## Paired tilt-fix comparison

In repeats 1 and 2, the baseline scored 47/60 and the tilt-fix configuration scored 41/60. Forty pairs succeeded in both runs, twelve failed in both, seven regressed and one improved. The comparison also changed seat persistence from 10 to 5 steps and the wrist-error threshold from 0.024 to 0.032 rad. The supplied files do not establish the separate 64/90 versus 68/90 comparison.

The wrist search issues a zero-planar/zero-yaw base hold while measuring a probe. It did so in four of the 60 tilt-fix episodes. Saved actions show that hold in only one of the seven regressions. Across the 44 probed observations in the one regression, estimated tilt spanned 10.5–13.5° against 12.8–15.4° true tilt (2.20° MAE). The estimator already read 4.27 cm on the first probe when the actual distance was 6.38 cm. The other six never issued a probe in their episode. Seat settings and variation across repetitions remain possible explanations for those flips.

| Repeat / seed | Outcome with tilt-fix configuration | Opening step, baseline → tilt-fix | Tilt-fix replay at timeout or success |
| --- | --- | --- | --- |
| 1 / 1000001 | Lost | 588 → lost-hold reopen at 842 | 88 base-hold steps from 742–831. Distance worsened from 6.4 to 7.2 cm and true tilt from 13.0° to 14.7° across that interval. Ended at 10.4 cm / 100.3°. |
| 1 / 1000007 | Lost | 546 → 542 | Ended inside both tolerances at 3.2 cm / 12.6°, still touching a pad. |
| 1 / 1000008 | Lost | 536 → 538 | Ended at 3.6 cm / 21.2° with pad contact. |
| 1 / 1000042 | Lost | 810 → stayed closed | Finished at 3.9 cm / 6.4° against the rack and pad. |
| 1 / 1000044 | Gained | 504 → 510 | Settled clear of the pad at 3.9 cm / 4.9°. |
| 2 / 1000010 | Lost | 688 → 1526 | Opened at 1.5 cm / 10.4° and settled at 3.9 cm / 23.9°. |
| 2 / 1000021 | Lost | 524 → 520 | Ended at 3.7 cm / 21.1° with pad contact. |
| 2 / 1000031 | Lost | 882 → stayed closed | Finished at 4.6 cm / 3.3° against the rack and pad. |

## Hover and action motion

None of the 19 baseline failures ended hovering near the rack without contact. Their longest uninterrupted closed, unsupported interval inside 10 cm lasted 44 steps (0.88 s). The five placements still closed at timeout all touched the rack. Across their final 200 actions, the RMS step change of the 18 non-gripper channels was 0.00021–0.00059 in normalized units, and slot distance varied by 0.03–0.33 cm. On seed 1000000 in repeats 1 and 2, all 100 last decisions met the quiet gate and none met the 0.024 rad wrist-error seat cue.

Seed 1000000 genuinely hovered in both tilt-fix repeats. Neither entered PLACE or touched the rack. Over the last four seconds the following measurements distinguish the vertical miss from the sideways miss. RMS uses successive changes of the nine learned base/arm channels. Maximum span is the largest peak-to-peak range on any of those channels.

| Repeat | Median signed x / y / z from nearest slot | Median horizontal distance | Final 3-D distance | 3-D distance range over last 4 s | Action change RMS / maximum span |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | −5.15 / +1.85 / +8.44 cm | 5.48 cm | 10.10 cm | 0.30 cm | 0.00085 / 0.0118 |
| 2 | −5.14 / +1.76 / +8.39 cm | 5.43 cm | 9.95 cm | 0.23 cm | 0.00080 / 0.0078 |

Those two were stationary policy stalls. Seed 1000042 in tilt-fix repeat 1 also spent 106 closed steps without rack contact at steps 1359–1464. It swept from x = −3.2 to +4.3 cm across the nearest slot, rose from +4.8 to +8.8 cm and returned to +4.9 cm. The largest command span was 0.446. It subsequently recontacted the rack and ended held.

## Estimator error

These are per-observation errors in the selected failures. Regular samples below use true distance ≤8 cm. Persistent end poses contribute several samples per episode. Opening rows use the original policy's logged estimates aligned to the pre-action replay state.

| Population | Episodes / samples | Distance bias / MAE | Tilt bias / MAE | 90th percentile absolute error, distance / tilt |
| --- | ---: | ---: | ---: | ---: |
| Seat-cue baseline, regular | 13 / 74 | +0.15 / 1.68 cm | −3.63 / 4.45° | 2.48 cm / 8.97° |
| Baseline, regular and gripper commanded closed | 5 / 26 | +1.40 / 1.60 cm | −1.27 / 3.61° | 2.46 cm / 7.55° |
| Baseline, opening decision | 11 / 11 | −1.26 / 1.26 cm | +0.07 / 0.43° | 2.25 cm / 0.87° |
| Tilt-fix, regular | 13 / 73 | +0.37 / 1.06 cm | −2.54 / 2.65° | 1.62 cm / 4.96° |
| Tilt-fix, opening decision | 12 / 12 | −0.83 / 0.86 cm | −0.49 / 0.74° | 2.11 cm / 2.03° |

At all 11 baseline failure openings, the original estimator understated 3-D distance. It reported 2.8–3.3 cm while true distance ranged from 3.51 to 5.46 cm. Re-rendered signed height at those openings was +2.27 to +3.42 cm. All 11 exceeded +2 cm, as did one of the seven matched baseline success controls. At 28 baseline failure PLACE-entry/opening decisions, re-rendered versus original estimates differed by 0.12 cm and 0.18° on average in absolute value.

## Conditional release probes

All 11 baseline failures with a RELEASE decision and one originally successful high-opening control were tested. The [probe script](../../../analysis/probe_rack_height.py) executes saved actions until the original release decision, rebuilds the 20-step temporal ensemble from rendered RGB/state, and restores the supervisor's recorded PLACE and release-gate history. In all 12 cases the flag-off supervisor reproduced the recorded opening action exactly (maximum absolute error 0). The flag-on supervisor then used fresh RGB/state through opening and the ordinary open/hold latch afterward. Each continuation ended at task termination, 120 steps after opening, or after the decision at 200 steps beyond the original opening.

| Seed | Original failures tested | Guarded outcome from the saved prefixes |
| --- | --- | --- |
| 1000024 | r1, r2 | **2 recoveries.** Opening moved from step 524 to 534 and from 522 to 532. Success latched at 590 and 586. |
| 1000044 | r1, r2, r3 | **3 recoveries.** Opening moved from 504 to 514, 508 to 518 and 506 to 516. Success latched at 568, 574 and 572. |
| 1000004 | r1, r2, r3 | Opened 24, 14 and 16 steps later. None achieved success in the window. |
| 1000041 | r1, r2, r3 | Stayed closed throughout all three windows. The original openings already had about 4 cm lateral error. |
| 1000021 | r2, originally successful control | Opening moved from 524 to 534. Success latched at 588. |

The other eight baseline failures never requested a release. The guard has no opportunity to change them. The forecast counts recoveries among the 19 failures. Only one originally successful trajectory had a changed opening tested, so net improvement across new runs is unknown. No full evaluation was run. [Repeat 1 probes](../../../results/brief21_rack/height_probe_r1.log) · [Repeat 2 probes](../../../results/brief21_rack/height_probe_r2.log) · [Repeat 3 probes](../../../results/brief21_rack/height_probe_r3.log)

## Harness setting

`ADV_H_RELEASE_HEIGHT_GUARD` defaults to `0`. Setting it to `1` permits an otherwise approved opening only when estimated plate-centre height minus estimated slot-centre height is at most **+0.020 m**. Both positions use the same yaw-aligned pelvis frame. Plates below the slot satisfy the height check. A veto leaves the gripper closed and continues ordinary placement commands. The supervisor checks the current estimate again at each decision. This condition applies to both normal and stall-triggered openings.

From a fresh shell in the harness worktree, the lead can use the following invocation for a future dev comparison. For its control set `ADV_H_RELEASE_HEIGHT_GUARD=0` and select a separate output directory.

```bash
cd /home/keyu/advaitic/wt-harness
env ADV_THROTTLE=1 MUJOCO_GL=egl ORT_DISABLE_TELEMETRY=1 OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 UV_CACHE_DIR=/tmp/adv-harness-uv \
  ADV_CKPT=/home/keyu/advaitic/ckpts/student_v1b_bce1.pt \
  ADV_ESTIMATOR=/home/keyu/advaitic/ckpts/est_v1b.pt \
  ADV_H_SUPERVISE=1 ADV_H_EXEC_STEPS=2 ADV_H_TEMPORAL_ENSEMBLE=1 ADV_H_ENSEMBLE_DECAY=0 \
  ADV_H_SEAT_CUE=1 ADV_H_SEAT_STEPS=10 ADV_H_SEAT_ERROR_RAD=0.024 ADV_H_SEAT_CUE_ONLY=0 \
  ADV_H_RELEASE_DISTANCE_M=0.035 ADV_H_RELEASE_TILT_DEG=12 ADV_H_STABLE_STEPS=6 \
  ADV_H_TILT_FIX=0 ADV_H_STALL_RELEASE=0 ADV_H_RELEASE_HEIGHT_GUARD=1 \
  nice -n 10 uv run --locked --no-sync \
  /home/keyu/advaitic/robot-learning-assignment/.venv/bin/advaitic-eval \
  --policy submission.policy_harness:make_policy --policy-root /home/keyu/advaitic/wt-harness \
  --policy-python /home/keyu/advaitic/mlenv/bin/python --episodes 30 --seed-start 1000000 \
  --method dev_st10_height --checkpoint student_v1b_bce1.pt \
  --out /home/keyu/advaitic/results/dev_st10_height_r1
```


## Reproducing the diagnostics

The [replay script](../../../analysis/replay_rack.py) and [summary script](../../../analysis/summarize_rack.py) produce [the per-episode summary](../../../results/brief21_rack/summary.csv). Each compressed step trace is named after its run and full seed. The replay scores sampled RGB after closing the simulator to limit peak laptop memory. For dense estimates the script accepts `--estimate-every 2` at the policy cadence or `1` at each physics control step, using a fresh output directory.

```bash
cd /home/keyu/advaitic/wt-harness
export ADV_THROTTLE=1 MUJOCO_GL=egl ORT_DISABLE_TELEMETRY=1 OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export UV_CACHE_DIR=/tmp/adv-harness-uv
py=/home/keyu/advaitic/robot-learning-assignment/.venv/bin/python
nice -n 15 uv run --locked --no-sync "$py" -I ../analysis/replay_rack.py \
  ../results/modal/evals ../results/brief21_rack
nice -n 15 uv run --locked --no-sync "$py" -I ../analysis/summarize_rack.py \
  ../results/modal/evals ../results/brief21_rack
nice -n 15 uv run --locked --no-sync "$py" -I ../analysis/probe_rack_height.py \
  ../results/brief21_rack dev_s2_st10_r1 1000004 1000024 1000041 1000044
nice -n 15 uv run --locked --no-sync "$py" -I ../analysis/probe_rack_height.py \
  ../results/brief21_rack dev_s2_st10_r2 1000024 1000044 1000004 1000021 1000041
nice -n 15 uv run --locked --no-sync "$py" -I ../analysis/probe_rack_height.py \
  ../results/brief21_rack dev_s2_st10_r3 1000044 1000004 1000041
```

The 24 harness unit tests and four worker tests passed. Ruff lint passed for the evaluator, submission and tests. Every edited Python file passes the format check. The repository-wide check also flags pre-existing formatting in `submission/policy_multi.py`.
