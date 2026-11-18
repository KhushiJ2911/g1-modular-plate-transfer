# Experiment log

Every run: date/time (IST), commit, config, seeds, result, one-line interpretation. Failures included.

Seed blocks (fixed before any policy existed):
- **Dev** (all decisions): `--seed-start 1000000`
- **Final** (touched once per final candidate): `--seed-start 7310000` (public default)
- **Teacher / training data**: seeds from 2,000,000 upward, plus the 60 human-demo seeds (0–~200)

Machine: RTX 3050 4 GB laptop (15 W cap), 16 threads, 15 GB RAM. Max 2 sim workers (thermal), watchdog in `~/advaitic/ops/watchdog.sh`.

---

## Phase 0: understand and measure (Wed 18 Nov)

| Time | What | Result | Interpretation |
|---|---|---|---|
| 02:21 | `uv sync --locked`, `advaitic-check`, `advaitic-smoke` @ da5714d | integrity OK (555 files), all 6 slots reset, 7.5 s | Environment is as published |
| 02:24 | 1 full episode, HoldPose example, smoke, seed 7310000 | 28.3 s wall (1,700 steps; 2.5 s policy) | **30-trial eval ≈ 12–14 min** sequential. Evals are cheap |
| 02:25 | Upstream agent-written policies (21 runs, 7 agents × 3), from HF `SWIRL-Lab/bigym2-agent-rollouts` | best 36/100 (astra-s1); Opus 15–23; most 0–7 | Task is hard for hand-scripted pixel controllers. Reference only, not reused |
| 02:26 | Demo statistics, 60 human demos | 530–742 steps (median 620); all flagged success. Only **left arm + left gripper** used (right gripper never closes, right-arm std ≈ 0.02); height and yaw constant; base moves +0.10 m x, −0.38 m y; grasp ≈ step 229, release ≈ step 570 | Effective action space is 10-D. ~1,000 steps of slack for retries |
| 02:27 | **FAILED RUN:** 6 parallel replay workers each loading all demos via `get_demos` | ~2 GB RSS each → RAM + swap exhausted, laptop hung, 1 worker crashed | Never load full demos per worker. Extracted low-dim demos once (8 MB) |
| 02:30 | Open-loop replay of each demo on its **own** seed, in the evaluator env | **59/60 success**; demo 45 (slot 1) timed out | Demos are valid under this evaluator; sim is (nearly) deterministic |
| 03:00 | Open-loop replay of demo actions on a **new** seed with the same source slot | **1/8 success** (6 timeout, 1 terminated); stopped early for heat | ±5 cm rack shifts break blind replay → closed-loop correction is the core of the task |
| 03:14 | Thermal: CPU pkg 92 °C, GPU 88 °C with 3 sim workers (EGL rendering) | Killed workers; cooled to 74/69 °C | Cap at 2 sim workers + watchdog |

Route decision (03:10): **C** (privileged teacher → vision student), with **B** (BC on 60 human demos) as the baseline. Reasoning in `PLAN.md`.

## Phase 1: baseline and teacher (Wed 18 Nov)

| Time | What | Commit | Seeds | Result | Interpretation |
|---|---|---|---|---|---|
| 03:32 | Teacher v1 (privileged nearest-demo transfer + residual pelvis tracking), kp=2, tol=1 cm, max_wait=100 | teacher@6c9f0e6 | 2000000–2000029 | **27/30 (90%)**. All 3 failures are source slot 1 (3/6 slot-1 seeds), 24/24 on other slots | Grasp always worked (plate traces). Failures: placed 23.6° tilted (limit 20°); lowered 6.5 cm from slot and toppled; dragged off the rack by the demo's retreat. The robot-centric tracking doesn't close the loop on the plate |
| 03:32 | Same run: pelvis gates | | | 18/30 hit the 100-step release wait | 1 cm tolerance is below what the WBC can settle to (walk-entry speed ≈ 0.05 m/s, P-command too small) |
| 03:36 | BC trainer throughput test, 150 steps | bc@36337e1 | | microbatch 64: 0.43 GB peak; GPU-only step 5.7 it/s, data loader 349 it/s | GPU-bound (15 W cap). Use full batch, no accumulation |
| 03:39 | **BC baseline training** human60, H=20, exec 10, 8k steps (≈30 epochs), EMA 0.999, DrQ shift | bc@36337e1 | 54 train / 6 val demos | running | 8k instead of the 20k default: time budget |
| 03:58 | Teacher v1 data generation, successful episodes only | teacher@6c9f0e6 | 2100000–2100399 | running | v1 is good enough as a data source (failures are filtered out) |
| 04:54 | BC baseline training finished: train L1 0.0176, val L1 0.0193 (8k steps, ~1.4 it/s under sim load) | bc@36337e1 | | ckpt `bc_human60_v1` (sha 2cc397d71e8ef46a) | Val loss is low, but the loss is not the metric |
| 04:56 | **Baseline dev eval: BC human60** | bc@36337e1 | dev 1000000 (30, balanced) | **0/30** (all timeouts), 5 ms/call policy time | ~~Gripper never closes in any episode~~ **Correction (07:30):** this was from a sample of episodes; over all 30 the gripper closes in 18 (17 hold it > 300 steps). Arm moves |
| 05:15 | Diagnosis 1: policy on held-out demo observations (offline) | | 6 val demos | gripper-close predicted within 2–21 steps of human; arm/base L1 0.006–0.011 | Model is accurate in-distribution. Not a pipeline bug |
| 05:16 | Diagnosis 2: evaluator vs demo images, same seed | | demo 0 | mean abs pixel diff 5–7/255, same scene | No rendering domain gap. Failure = compounding error / covariate shift with 54 demos |
| 05:10 | Teacher v1 generation stopped (resumable) | teacher@6c9f0e6 | 2100000–2100029, 2100200–… | 46/58 success (79%), 73 saved demos incl. first 30-seed eval | Lower than the 90% eval; throttled ~45 eps/h |
| 05:15 | Teacher v2 (object-centric placement) on the **same** 30 seeds as v1 | teacher@8b4744d | 2000000–2000029 | running | Paired comparison v1 vs v2 |
| 05:34 | **Teacher v2 (six changes at once): FAILED** | teacher@8b4744d | 2000000–2000029 (paired with v1) | **6/30** vs v1 27/30; discordant 21 (v1 only) vs 0 → McNemar p ≈ 1e-6 | Clear regression. grasp_miss 9 (no base feed-forward → base hunts at grasp), knocked_after_release 12 (placed ≈1 cm from site but tilt 10–13°, leaned to 23–31° after release; likely levered by base motion while in the slot). Lesson: changed 6 things at once. Next: v1 + one flag at a time (brief #6) |
| 05:45 | Generation restarted, v1 + privileged labels: DART 0 and DART 1.0 | teacher@36101fe | 2200000+ (dart0), 2300000+ (dart1) | running | dart1 yield vs dart0 = cost of noise |
| 05:45 | **Student v1a** = baseline + 73 teacher-v1 demos (50/50 mix), same 8k steps, aux off | bc@f674fe2 | | running | Single change vs baseline: data |
| 07:00 | Student v1a training finished: train L1 0.0184, val L1 0.0181 (8k steps, 1.5–2 it/s under sim load) | bc@f674fe2 | | ckpt `student_v1a` (sha dc10b512112598e7) | Dev eval running |
| 07:05 | **DART yield check:** v1 + labels, DART 0 vs DART 1.0 (different seed blocks, unpaired) | teacher@36101fe | 2200000+ / 2300000+ | **DART 0: 34/41 (83%)**; **DART 1.0: 3/23 (13%)**: grasp_miss 11, drop_in_carry 4, place_tilt 3 | Scale 1.0 (base std 0.10, arm std 0.02) is far too strong for this teacher: base drift before the grasp is not recovered. DART 1.0 stopped. If DART is retried: scale ≤ 0.3, noise only in carry |
| 07:05 | DART 0 generation paused (resumable) to free sim slots for the teacher ablation | | | 34 new demos with privileged labels | |
| 07:06 | **Teacher ablation:** v1 + `--align-then-lower` only, same 30 seeds as v1 | teacher@2cc0622 | 2000000–2000029 | running | Paired vs v1 27/30 |
| 07:25 | **Student v1a dev eval** | bc@f674fe2 | dev 1000000 (30, balanced) | **0/30** (paired with baseline: 0 discordant pairs) | More data alone did not move the score. Diagnosed below |
| 07:35 | Diagnosis: camera-free replay of v1a's saved eval actions with privileged plate tracking (`analysis/replay_eval.py`) | | same 30 dev episodes | **no grasp 18**, **placed but never released 10** (plate in target rack 2.8–4.8 cm from a site, tilt 4–11°, touching rack, still gripped at timeout), reached target then failed 2 | 12/30 carry the plate to the target. Release is the nearest bottleneck: the arm settles into a small jitter for > 1,000 steps and the open never fires |
| 07:40 | Hypothesis test: is the gripper copying its own state ("copycat")? Flip state dims 7, 8, 44, 45 to the other phase on demo frames (`analysis/copycat_probe.py`) | | 3 human + 2 teacher demos | gripper agreement with labels 0.98–1.00 with real **and** flipped state | **Rejected.** The decision comes from vision/arm pose; the settled post-placement state is off-distribution |
| 07:20 | Teacher ablation v1 + align-then-lower: first 2 seeds | teacher@2cc0622 | 2000000–01 | 0/2, both `place_position` (11.4, 8.5 cm); v1 succeeded on 2000001 | Freezing the base at carry height lets the arm's lowering motion move the plate sideways. Running on (slowly, thermal pauses) |
| 07:45 | **Student v1b** = v1a + 34 teacher demos with labels (DART 0), same 8k steps | bc@f674fe2 | | running | Data-scale step (107 teacher demos) |
| 07:45 | Codex brief #7: DAgger-style gripper relabelling: roll out the student, label the gripper with a privileged oracle (open when placed in target rack; closed when carrying), supervise only that dim | | | in progress | Standard remedy for covariate shift; the arm is not relabelled because the teacher can't label arbitrary states |
| 07:30 | Same replay diagnosis for the **baseline** (BC human60) | | same 30 dev seeds | no grasp 15, **placed but never released 9**, reached target then failed 5, carried elsewhere 1 | Same two bottlenecks as v1a. The release failure is not caused by the teacher data |
| 07:30 | **Teacher ablation align-then-lower: ABANDONED after 2/30 seeds** | teacher@2cc0622 | 2000000–01 | 0/2 (`place_position` 11.4, 8.5 cm); paired with v1: 1 discordant pair | Not a statistical result. Mechanism seen in both: the base is frozen at carry height, and the arm's lowering motion moves the plate 8–11 cm sideways. Stopped because the teacher (v1, 90%) is not the bottleneck and sim time goes to student data. A fix would target `site − (demo plate drift during lowering)` |
| 07:40 | Teacher data generation resumed (1 worker) under teacher@2cc0622 v1 path, new JSONL `gen_v1_dart0_b` | teacher@2cc0622 | 2200041+ | paused again 08:05 for DAgger collection | |
| 08:05 | **DAgger collection 1:** student v1a rollouts with gripper-oracle labels (student keeps executing its own gripper), stop 150 steps after placement | bc@a5b5dc7 | 2400000–2400059 (2 workers) | running | Data for the release decision in the states the student actually reaches |
| 08:50 | Student v1b training finished: train L1 0.0187, val L1 0.0165 | bc@f674fe2 | | ckpt `student_v1b` (sha 3a66e26fa7d445ce) | dev eval running |
| 08:50 | DAgger collection 1 finished | bc@a5b5dc7 | 2400000–2400059 | 60 episodes: placed_not_released 13/33 at midpoint, no_grasp 13, reached-then-failed 4, other 3 (same mix as dev eval). Labels per episode ≈ 3–20% open, 33–45% closed, ~50% undefined | Collection reproduces the dev failure mix, so it covers the states we need |
| 08:55 | **Student v1a+DAgger1** = fine-tune v1a for 4k steps on human60 + teacher_v1 + dagger (1:1:1), DAgger supervises only the gripper | bc@a5b5dc7 | 175 train / 18 val episodes | running | Confound: also 4k extra training steps vs v1a |
| 08:55 | Disk: deleted 3 superseded image caches (old format, 7.2 GB, regenerable) | | | 18 GB free | |
| 09:30 | **Student v1b dev eval** | bc@f674fe2 | dev 1000000 | **0/30**. Replay: no grasp **10** (v1a 18), placed-not-released **15** (v1a 10), reached then failed 5 | More teacher data improves grasping (20/30 reach the target). Release is now the main blocker |
| 09:40 | **Student v1a+DAgger1 dev eval** | bc@a5b5dc7 | dev 1000000 | **0/30**. Gripper opens after a grasp in only 2/30. Replay: no grasp 14, placed-not-released 4, reached-then-failed 10 (hovering 2–9 cm from a site, still gripped) | DAgger fine-tune did not produce releases |
| 10:00 | Diagnosis: raw gripper prediction on the DAgger **training** frames, by oracle label (`analysis/grip_fit_probe.py`) | | 60 DAgger episodes | open-labelled (n=1,250): v1a +1.00, v1a+DAgger1 **+0.93**; 0% predicted open | Underfit, not overfit. **L1 regresses to the conditional median**: around release, near-identical frames are mostly labelled closed, so the median is +1. And the gripper-only DAgger elements were ~0.1% of the loss. → gripper as BCE classification (brief #8) |
| 09:50 | **Student v1c-long**: 30k steps (vs 8k) on human60 + teacher_v1 + teacher_v1_dart0 (112), weights 1 : 0.4 : 0.6; snapshots at 8k/15k/22k/30k | bc@a5b5dc7 | 227 train episodes | running | Tests "undertrained" (8k steps ≈ 7 epochs) |
| 10:05 | DAgger collection 2 from **v1b**, with `--relabel-execute` (oracle executes the gripper), so post-release states are recorded too | bc@a5b5dc7 | 2410000–2410059 | running | |
| 10:05 | Teacher generation stopped | teacher@2cc0622 | | teacher_v1_dart0: 112 saved demos (≈84% yield) | Enough teacher data for now |
| 10:35 | Long run v1c paused at step 3.5k (resumable). Thermal throttling held it at 1.8 it/s, i.e. ≈4.6 h for 30k | | | | GPU time goes to the BCE hypothesis first |
| 10:40 | **FAILED RUN:** BCE fine-tune crashed while building its image cache: disk full (`OSError: Not enough free space`) | bc@5233dc7 | | | Each data mix writes a full ≈5 GB uncompressed cache. Deleted the finished DAgger1 run's cache and the partial file; brief #9 = per-episode shared cache |
| 10:45 | **Student v1b-BCE1** = fine-tune v1b 6k steps, gripper as BCE (pos_weight 2, threshold 0.5), on human60 + teacher_v1 + dart0 + dagger (1 : 0.4 : 0.6 : 0.67) | bc@5233dc7 | 298 train / 24 val episodes | running | |
| 10:50 | Thermal: watchdog thresholds lowered (pause CPU ≥ 85 / GPU ≥ 82 °C, resume ≤ 72 / 70 °C); DAgger2 cut to 1 worker | | | 92 → 72 °C within a minute of pausing | |
| 10:55 | Paused at the user's request after the BCE fine-tune finishes. Dev eval of v1b-BCE1 not started yet; DAgger2 collection stopped (resumable) | | | | |
| 11:09 | v1b-BCE1 training finished: val L1 0.0118, val gripper acc 0.969, **val open-label acc 0.961** (v1a+DAgger1: 0% on open labels). DAgger2 stopped at 50/60 episodes | bc@5233dc7 | | ckpt `student_v1b_bce1` | Offline only; closed-loop dev eval pending (paused) |
| 11:35 | **Student v1b-BCE1 dev eval** | bc@5233dc7 | dev 1000000 (30, balanced) | **14/30 (47%, Wilson 95% CI 30–64%)**; gripper opens in 25/30. Paired vs v1b (0/30): discordant 14 vs 0, exact McNemar p = 1.2e-4 | First non-zero student. The L1-median diagnosis was right: the gripper loss was the release blocker |
