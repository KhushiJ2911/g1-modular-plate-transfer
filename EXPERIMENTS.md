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
| 16:55 | Replay diagnosis of v1b-BCE1's 16 dev failures | | dev 1000000 | **never released 5** (plate in slot, good pose, gripper still closed; ep 20 opened for 4 steps then re-closed), **released but bad 6** (too far 5.0–5.2 cm ×2, tilt 24–33° ×2, toppled/fell ×2), **carried elsewhere 3**, **missed grab 2** (both slot 5). By slot: 0→4, 5→4, 3→3, 2→2, 4→2, 1→1 | Release still the largest single cause; placement precision next. Grasping is mostly solved |
| Fri 18:35 | Gripper-threshold sweep on dev for v1b-BCE1: 0.4 and 0.3 (vs 0.5 = 14/30). Dev tuning; final score only on the 7310000 block | bc@5233dc7 | dev 1000000 | running | Targets the 5 never-released failures; paired comparison also shows premature drops |
| Fri 18:57 | **Threshold sweep result** (v1b-BCE1, dev) | bc@5233dc7 | dev 1000000 | 0.5: 14/30 · **0.4: 20/30 (67%, Wilson 49–81%)**, gained 8 / lost 2 vs 0.5, McNemar p = 0.11 · 0.3: 18/30, gained 6 / lost 2, p = 0.29. Slot 0 at 0.4: 4/5 (was 1/5) | Promising but **not significant** on 30 paired episodes. 0.4 chosen on dev, so it must be confirmed on unseen seeds |

### Threshold 0.4 confirmation on fresh seeds (block 2600000, never used before)
Same checkpoint `student_v1b_bce1.pt` (7535cac0), 30 episodes, paired seeds, run sequentially.

| threshold | successes | Wilson 95% | per slot 0..5 |
|---|---|---|---|
| 0.5 | 18/30 (60%) | 42–75% | 3,5,4,3,3,0 |
| 0.4 | 17/30 (57%) | 39–73% | 2,5,5,3,2,0 |

Paired vs 0.5: gained 2, lost 3, McNemar exact p = 1.0. **The dev gain (14→20) did not replicate**; it was mostly seed noise plus selection on dev. Decision: keep the checkpoint default 0.5, no threshold tuning. Combined estimate for bce1 @0.5 across 60 episodes (dev + fresh): 32/60 = 53%. Slot 5 is the weakest: 1/10 across both blocks at 0.5 (dev 1/5, fresh 0/5).
| 11-21 06:35 | **microVLA v1 (SigLIP distill 3k + flow 10k) dev eval** | vla@9eb7939, ckpt sha 107dcefe | dev 1000000 (30, balanced) | **11/30 (37%, Wilson 95% CI 22–54%)**; per slot 0–5: 2,0,3,3,1,2 of 5; all 19 failures are timeouts, 0 falls. Paired vs student-v1b-bce1 thr0.4 (20/30): 4 only-microVLA vs 13 only-baseline, exact McNemar p=0.049. Paired vs the adopted baseline bce1 @0.5 (14/30): 5 only-microVLA vs 8 only-baseline, exact McNemar p=0.58 (not significant). Final val flow loss 0.0109. |

**microVLA v1 dev failure replay (06:50).** Camera-free replay of the saved action logs reproduces 30/30 eval outcomes. Categories (heuristic: held = privileged grasp contact, lifted > 5 cm, reached = within 5 cm of rack in xy while lifted):

| Category | microVLA v1 | bce1 @0.5 |
|---|---|---|
| Success | 11 | 14 |
| A never grasped | 1 | 1 |
| B grasped, slipped before lifting (held 4–49 steps; closest hand–plate 10–14 cm vs ~8 cm in successes) | 7 | 1 |
| C lifted, stopped 5–7 cm short of rack | 3 | 3 |
| D on rack, not settled | 1 | 10 |
| E carried to rack (within 3 cm), ended on table at ~80–90° tilt | 7 | 1 |

The two models fail in different places: microVLA loses the plate at pickup (B) and at release (E), the BC student gets it onto the rack but doesn't settle it (D). Slot 1 (0/5): 3 E, 1 B, 1 D.

**microVLA v1 release-timing check (07:15).** Hypothesis: the E failures open the gripper too early or too high. Replayed the 7 E failures and 11 successes and logged the plate at the first open command after carrying. **Rejected**: height above the slot at opening is the same in both groups (about 1–3 cm below the site, already lowered), and tilt is similar (under 10° in all but one). What separates them is **horizontal offset from the slot at opening: successes 1.3–5.4 cm (9 of 11 under 4 cm), failures 4.2–8.8 cm**. In the failures the plate passed within 0.3–3.3 cm of the slot during the carry, then drifted away before release; once opened it slid off the rack onto the table (tilt 45–90° within 100 steps). Two failures (ep 13, 14) also show gripper chatter (10 and 7 open/close switches vs 2 in most successes). Pickup slips (B) are the same kind of error: the hand stops 10–14 cm from the plate vs ~8 cm. Working diagnosis: cm-level end-effector positioning in the final approach, not gripper timing. Policy executes 10 of each 20-step chunk open-loop (exec_steps=10).
| 11-21 07:13 | **microVLA v1 exec_steps 5 (vs 10) dev eval** | vla@825f841, ckpt sha 107dcefe, ADV_EXEC_STEPS=5 | dev 1000000 (30, balanced) | **12/30 (40%, Wilson 95% CI 25–58%)**; per slot 0–5: 1,0,3,4,2,2. Paired vs exec 10 (11/30): 6 only-exec5 vs 5 only-exec10, exact McNemar p=1.0, **no effect**. vs bce1 @0.5: 6 vs 8, p=0.79. 17 timeouts, 1 env termination (ep 2, step 1467, no fall, no policy error), 0 falls. Policy time 15.1 s/episode (2.2x). Hypothesis that open-loop drift causes the release offset is not supported; keep exec_steps 10. |

## microVLA track: setup and training history (backfilled 11-21 07:40 from session transcripts)

- **Source:** the user's own microVLA (github.com/KhushiJ2911/microVLA, MIT; blog khushiijaiswal.com/posts/microvla). Vendored `train.py` unmodified (vla@774d54d); adapter by Codex brief #10 (vla@9eb7939): 11.2M params, 3 cameras upsampled 84→96 px so 16-px patches tile, masked flow loss (skips dims without trusted labels), resumable trainer, `submission/policy_vla.py`. Horizon 20, exec_steps 10.
- **Data mix:** same 372 episodes as the BC student: human60 (60) + teacher_v1 + teacher_v1_dart0 (202) + DAgger (rest).
- **Env:** `transformers` 4.57 in mlenv (repo pins 5.22, which needs torch ≥ 2.6; mlenv has 2.5.1). SigLIP base-patch16-224 weights already on disk.
- **11-20 ~20:40, run `microvla_v1` WITHOUT SigLIP distillation: stopped and deleted, not evaluated.** The 40-step smoke test measured 0.75 s/step at batch 64, 1.3 GB GPU. The user pointed out that the blog shows the model does not work without distillation (round 1 = 3k distill steps), so the run was relaunched with it. *Lesson for the blog: I had initially disabled the one ingredient the original write-up identifies as essential.*
- **11-20 21:01, run `microvla_v1_distill` launched:** `ADV_THROTTLE=1 python -m method.vla.train --distill --distill-steps 3000 --siglip-path google/siglip-base-patch16-224 --microbatch-size 16 --val-every 1000 --sample-every 1000 --run-dir runs/microvla_v1_distill`.
- **11-20 21:38, laptop powered off** (clean user-initiated systemd poweroff, not thermal; the journal shows thermald "terminating on user request") at distill step ~2050. Last checkpoint was 21:21 (distill step 2000).
- **11-20 22:28, resumed** with `--resume` (same args); ~50 distill steps redone. Pre-reboot log kept at `runs/microvla_v1_distill.pre_reboot.log`.
- **Thermal incident:** the watchdog (`ops/watchdog.sh`) does not auto-start after a reboot, so the first hour after resume ran unthrottled: GPU 88 °C, CPU package 91 °C, ACPI zone 93 °C (critical trip 100 °C). Watchdog restarted 23:25; 407 pauses until the run ended. The heat is mostly CPU (one core at boost); the trainer itself uses ~1–1.4 cores. The user chose to keep the conservative thresholds (pause CPU ≥ 85 / GPU ≥ 82 °C) over speed.
- **Throughput:** flow phase ~0.96 s/step unthrottled-but-thermally-limited, ~2.9–3.5 s/step wall time with watchdog duty cycling (vs 0.75 s/step in the cool smoke test). Wall time for the 10k flow steps: ~22:35 → 03:11.
- **Final:** distill loss 0.79 → ~0.78 at 3k (plateaued slowly); flow val loss 0.548 @1k, 0.093 @2k, 0.033 @3k, 0.0178 @4k, 0.0127 @6k, 0.0115 @8k, **0.0109 @10k** (train 0.0076); peak CUDA allocated 1.28 GB. Checkpoint copied to `ckpts/microvla_v1_distill.pt` (sha256 prefix 107dcefe).
- **Low val loss did not predict task success** (11/30 dev, below bce1 14/30).

Analysis artifacts: `analysis/replay_eval.py` → `results/dev_microvla_v1_distill.replay.jsonl`; `analysis/release_timing.py` → `results/dev_microvla_v1_distill.release.jsonl`.

## bce1 failure re-read (11-21 07:40)

Re-reading `results/dev_student_v1b_bce1.replay.jsonl`: of the 10 "D on rack, not settled" failures, **5 (ep 5, 11, 14, 20, 22) are really "never released"**: plate on the rack 2.5–3.9 cm from the slot, tilt 2.6–17°, but the gripper holds it for 1,406–1,474 steps until timeout. The category heuristic above did not distinguish these. The 0.4 threshold targeted exactly these and did not replicate on fresh seeds, so the open decision itself is the weak point. Next: log per-step gripper-open probability on these seeds.

## Ops: Modal (11-21)

- Modal CLI 1.6.4 installed as an isolated uv tool; auth via `modal setup` browser flow, workspace jaiswalkhushi2911 (token in `~/.modal.toml`).
- 07:35 smoke job `ops/modal_hello.py`: T4 GPU, 15 GB, driver 620.70.06; ran and stopped cleanly.

## Modal GPU benchmark (11-21 ~07:45)

Setup: `ops/modal_train.py` (image: py3.12, torch 2.5.1+cu121, transformers 4.57.6, same pins as mlenv; code = wt-vla snapshot; data volume `advaitic-data` = human60, teacher_v1, teacher_v1_dart0, dagger, dagger2 (~3.8 GB, uploaded in ~3.5 min); cache built on a CPU container, digest 0c6a5f399d0e7038ac16af50). First image build failed: pre-downloading SigLIP via AutoProcessor needs sentencepiece; fixed by downloading only `SiglipVisionModel` (all the trainer uses).

microVLA flow phase, 300 steps, batch 64, microbatch 64, 4 loader workers, no distill, rate measured over steps 25–300:

| GPU | s/step | 10k flow steps | $/h (Modal list, GPU only) | GPU cost per 10k |
|---|---|---|---|---|
| Laptop RTX 3050 4 GB (microbatch 16, cool) | 0.75 | 2.1 h (5+ h with thermal pauses) | 0 | 0 |
| T4 | 0.97 | 2.7 h | 0.59 | $1.59 |
| L4 | 0.21 | 0.58 h | 0.80 | $0.47 |
| **A100 40 GB** | **0.090** | **0.25 h** | 2.10 | **$0.52** |

My prior (L4 best per dollar because the model is tiny) was wrong: the A100 is 2.3x faster than L4 for ~the same cost per run. T4 is slower than the laptop, likely because it has no native bf16 and the trainer autocasts to bf16. Decision: A100-40GB for training. CPU/memory charges come on top (8 vCPU, 32 GB per job).

## Finding: evaluation is not run-to-run deterministic (11-21 ~08:20)

Re-ran **the same bce1 checkpoint on the same 30 dev seeds** (through a probe wrapper that returns the policy's actions unchanged, `analysis/probe/probe_policy.py`, run from a folder whose `submission` links to wt-bc@5233dc7): **18/30 vs 14/30 originally**. 4 seeds flipped, all fail→success (1000006, 1000007, 1000011, 1000030); exact sign test on 4/4 flips p = 0.125, consistent with chance. Only 12/30 episodes are bit-identical. In the others the first difference appears at step 100–490, in all 9 arm dims at once, with magnitude < 1e-5, growing to ~1e-3 within 10 steps and eventually flipping a thresholded gripper decision. Interpretation (not yet proven): GPU numeric nondeterminism in the policy forward pass (no cudnn.deterministic / fixed algorithm selection), amplified by closed-loop physics. Saved-action replays are deterministic (30/30), so the simulator itself is not the source.

Consequences:
- A single 30-episode dev run of the same model can move by ±4 successes. Earlier "paired" McNemar comparisons treat each seed's outcome as fixed; they are not. The 0.4-threshold "gain" (14→20) that failed to replicate is consistent with this.
- Updated bce1 @0.5 estimate: dev 14 + dev rerun 18 + confirm 18 = 50/90 = 56%.
- Method change: compare candidates with repeated runs / more seeds (cheap on Modal), and report run-to-run spread alongside Wilson intervals.

Side note: the evaluator rewrites PYTHONPATH to [package root, cwd] for the policy worker, and refuses scored runs < 30 episodes (multiples of 6).

Also 08:10: Modal sim image (`ops/modal_sim.py`) passes `advaitic-check` (ok, 555 files, manifest 4fb10989…, identical to the laptop). bce1 checkpoint uploaded to volume `advaitic-ckpts`.

## bce1 release diagnosis (11-21 ~08:35)

Per-step gripper-open probability (probe run above) joined with the DAgger oracle label (open = plate touching rack, ≤ 5 cm from a site, ≤ 20° tilt) via deterministic replay (`analysis/grip_probe_join.py` → `results/probe_bce1/joined.jsonl`).

Correction: the earlier "never released" list used episode indices; seeds are not contiguous (balanced sampler). The original never-released episodes 5, 11, 14, 20, 22 are seeds 1000005, 1000011, 1000014, 1000030, 1000033. In the rerun 1000011 and 1000030 succeeded, 1000033 never reached a placed state, and **1000005 and 1000014 got stuck again**.

Open probability during oracle-open states:

| Episode (seed) | outcome | oracle-open steps | mean p(open) | max p(open) |
|---|---|---|---|---|
| successes (18) | released | 52–591 | 0.25–0.95 | 1.0 (after opening) |
| 5 (1000005) | stuck holding | 363 | **0.032** | **0.093** |
| 14 (1000014) | stuck holding | 782 | **0.071** | **0.231** |
| 15 (1000015) | stuck holding | 376 | 0.192 | 0.465 |
| 10 (1000010) | stuck holding | 53 | 0.143 | 0.401 |

**Conclusion:** in the stuck episodes the model is confidently wrong (p(open) well under 0.5 for hundreds of steps), not marginal, so no threshold can fix it without breaking the closed phase (max p(open) during oracle-closed carrying reaches 0.50–0.60 in some successes). These are placed-and-holding states the model has not learned to recognise from its own rollouts. bce1 was trained with DAgger1, collected from the older v1a student; it has never seen its own failure states. Supports the planned **DAgger round 3 collected from bce1 itself, without `--relabel-execute`**, so the student keeps holding and every stuck step becomes an open-labelled frame (stop 150 steps after placement).
