# Project research log: G1 plate transfer

Coverage: `EXPERIMENTS.md` through line 449, including “11-22 night: SFT results, failure analysis, harness” and the launched harness ablations. Times below are IST. This incremental update incorporates lines 403–449; the companion `.log_state.json` records the source position for later updates.

The strongest observation-driven student in the recorded comparisons remains **bce1: 57/90 (63%)**, from three Modal dev runs with 19, 17 and 21 successes. No recorded student meets the assignment's 90% target. Released-base pi0.5 and MolmoAct2 each scored 0/90. Added teacher data yielded bce4a at 47/90 and bce4b at 54/90; neither surpassed bce1. The harness has been implemented and its estimator trained, but its policy evaluations remain pending. All measured results and costs below come from the [master experiment log][experiments]. Briefs describe requested work; implementation reports establish what was built, without turning proposed experiments into completed results.

## 1. Task & constraints

The task is to move a plate between racks with a Unitree G1 and Dex1-1 grippers in BiGym2/MuJoCo, using the frozen Balance/Walk controller. Any destination slot is acceptable. Racks vary by up to ±5 cm in x/y. Passing requires at least 90% success across at least 30 balanced trials: 27/30, with equal coverage of the six source slots. Success requires the plate within 5 cm of a destination site, within 20° of the required orientation, touching the rack, released, clear of the table, and stable for one simulated second. Falls, floor contact, policy errors and timeouts count against success. An episode allows 1,700 control steps at 50 Hz after warmup. The evaluator, physics, controller and success checks remain fixed. [Task rules][readme]; [method contract][methods].

Inference sees only `rgb` (`uint8`, three cameras, shape `(3,3,84,84)`: head, right wrist, left wrist) and `state` (`float32`, 50 proprioceptive values). It receives no seed, source-slot identifier, task pose, simulator state, reward or success signal. Actions are normalized 20-dimensional commands; policies may return chunks. The learned policies concentrate on 10 varying action dimensions and 19 selected proprioceptive inputs. Privileged state is disclosed training/annotation information only. Fixed replay and seed lookup do not satisfy the task; observation-driven control and state machines are permitted. [README][readme]; [BC brief][brief1].

Decisions use the balanced dev sampler with `--seed-start 1000000`. Teacher/training seeds start at 2000000, alongside the human-demo seeds. Fresh block 2600000 was used for threshold confirmation and is now reserved; 7310000 is the final-test block. No learned-policy final evaluation is recorded. The initial HoldPose smoke did use seed 7310000, so describing that block as wholly untouched would be inaccurate. Later collection briefs explicitly exclude dev, confirm and test ranges. A seed-start is not a literal list of consecutive evaluated seeds: recorded balanced runs include seeds above 1000029. [Initial protocol][phase0]; [confirmation][confirmation]; [generation guard][brief12].

Local hardware was an RTX 3050 with 4 GB VRAM, a 15 W GPU cap and 15 GB RAM; simulation was capped at two workers after overheating. The watchdog pauses at CPU ≥85 °C or GPU ≥82 °C and resumes at ≤72/70 °C. Cloud work introduced separate spending limits: the model benchmark cap rose from $15 to $20 to $25 and finished at $25.99; the later zero-shot benchmark has its own $15 cap. Subsequent overnight work has a $20 cap from cumulative metered $44.94 at 03:13; no closing overnight spend is recorded. Hardware choice, thermal pauses, sample counts and loading latency all constrain the conclusions. [Local runs][phase1]; [benchmark][benchmark]; [latest work][latest]; [overnight work][night].

## 2. Timeline by phase

### Measurement and route selection — 11-18

**Question.** What makes this task difficult, and can demonstration replay provide a useful starting point?

Environment checks passed for 555 files, and all six source slots reset. A full HoldPose smoke episode took 28.3 s for 1,700 steps, suggesting roughly 12–14 minutes for a sequential 30-trial evaluation. The 60 human demonstrations lasted 530–742 steps, median 620; only the left arm, left gripper and planar base motion varied substantially. This motivated the reduced action space and action chunking. [Phase 0][phase0].

Replaying demonstrations on their own seeds succeeded **59/60**; demonstration 45 timed out. On new seeds with the same source slot, replay succeeded **1/8**, with six timeouts and one termination, before stopping for heat. The latter is a small diagnostic sample, but it showed why rack variation needs feedback. Upstream agent-written policies were consulted as reference only: the recorded best was **36/100**, under a different upstream protocol. [Phase 0][phase0].

**Conclusion and next change.** The [initial plan][plan] chose a vision/proprioception BC baseline plus a privileged demo-transfer teacher to produce training data. The early replay experiments were diagnostics, not eligible submission methods. Six replay workers loading the full dataset exhausted RAM and swap; low-dimensional demonstrations were extracted once. Three simulation workers also overheated the laptop, establishing the two-worker limit. [Phase 0][phase0].

### BC baseline and privileged teacher — 11-18

**Question.** Can a small chunked BC policy imitate the demonstrations, and can a teacher expand coverage?

The BC baseline used an ImageNet ResNet-18 truncated after layer3, spatial softmax, proprioception and an action-chunk head. It trained on 54 demonstrations with six held out, horizon 20, execute 10, EMA 0.999 and DrQ shifts, for 8k steps. Training L1 was **0.0176**, validation L1 **0.0193**, but closed-loop dev success was **0/30**, all timeouts. Held-out demonstration probes predicted gripper-close timing within **2–21 steps** and arm/base L1 **0.006–0.011**. Low imitation error did not establish control success. [Baseline runs][phase1]; [architecture request][brief1].

The initial claim that the baseline never closed its gripper was corrected: it closed in **18/30** episodes, holding for over 300 steps in 17. Replay categorized its failures as no grasp 15, placed but never released 9, reached then failed 5, and carried elsewhere 1. The recorded interpretation was compounding error/covariate shift. The early image check found a mean absolute difference of **5–7/255** between demo and evaluator images; later rendering comparisons qualify the early claim of no domain gap. [Baseline diagnosis][phase1]; [rendering comparison][dagger3].

Teacher v1 (`teacher@6c9f0e6`) selected a nearby reference demonstration, shifted its trajectory using true rack/plate information, and tracked pelvis position with demo feed-forward plus proportional feedback. It succeeded **27/30 (90%)** on training seeds 2000000–2000029. All grasps worked; placement failures involved **23.6°** tilt, lowering **6.5 cm** off-site, or dragging the plate during retreat. **18/30** hit the 100-step release wait. These were privileged, not balanced dev student trials; the teacher's 90% is not a submission pass. Subsequent collection yielded **46/58 (79%)**, producing 73 saved demonstrations when successful teacher-evaluation episodes were included. [Teacher results][phase1]; [teacher design][brief2].

Teacher v2 (`teacher@8b4744d`) changed six things together, including object-centric placement and base feedback. On the same seeds it regressed to **6/30**, versus v1's 27/30; discordant outcomes were 21 v1-only and zero v2-only, recorded McNemar p ≈ 1e-6. Failures included nine grasp misses and 12 knocked-after-release cases. Removing base feed-forward made the base hunt; moving while partly inserted was the proposed explanation for later tilt. [Paired comparison][phase1]; [requested v2][brief3].

**Conclusion and next change.** Preserve v1 as the data source and isolate changes. The align-then-lower ablation (`teacher@2cc0622`) stopped at **0/2**, with placement errors **11.4 and 8.5 cm**. This was an abandoned diagnostic, not a statistical comparison: freezing the base left uncompensated sideways plate motion during arm lowering. Student improvement became the priority. The master log calls the failing v1 slot “1”, while briefs #3/#6 call it zero-based slot 0; that discrepancy remains unresolved. [Ablation record][phase1]; [ablation brief][brief6].

### Teacher data, DART and BC students — 11-18

**Question.** Does extra coverage or noisy teacher execution solve the student's distribution shift?

Teacher collection gained clean action targets, separately stored executed actions, and privileged per-step labels. DART perturbed execution while retaining clean teacher labels. In different, unpaired training-seed blocks, DART 0 yielded **34/41 (83%)**, while DART 1.0 yielded **3/23 (13%)**: 11 grasp misses, four carry drops and three tilt failures were recorded for the noisy run. Scale 1.0 used base noise std **0.10** and arm std **0.02**; it was stopped. Smaller noise, scale ≤0.3 and restricted to carry, was proposed, without a recorded test. [Yield comparison][phase1]; [data/label specification][brief4].

Student v1a (`bc@f674fe2`) mixed the human set with 73 teacher demonstrations, 50/50, kept 8k steps and disabled the auxiliary head. Its train/val L1 was **0.0184/0.0181**, but success remained **0/30**. Replay found 18 no-grasp, 10 placed-but-holding and two reached-then-failed episodes. Flipping gripper-state inputs on three human and two teacher demos preserved label agreement **0.98–1.00**; the recorded probe rejected the simple “copy its own gripper state” hypothesis. [Student diagnosis][phase1]; [student changes requested][brief5].

Student v1b added 34 labelled DART-0 teacher demos, reaching 107 teacher episodes at that stage. Train/val L1 became **0.0187/0.0165**; dev stayed **0/30**. No-grasp failures fell to 10, 15 were placed-but-not-released, and five reached then failed: **20/30** reached the target. More data improved an intermediate behavior without improving task completion. Collection later reported 112 saved DART-0 demos at approximately 84% yield; later inventories count 129, including an evaluation-derived set. These are time-specific dataset counts, not interchangeable descriptions of a frozen dataset. [Student/data record][phase1]; [latest inventory][latest].

**Conclusion and next change.** Release became the clearest bottleneck. A 30k-step v1c-long run tested undertraining but paused at step **3.5k** under thermal throttling; no evaluation is recorded. Work shifted to relabelling the gripper in states students actually visit. [Phase 1][phase1].

### DAgger attempts and the BCE gripper — 11-18, followed by fresh-seed confirmation

**Question.** Can a privileged gripper oracle teach release without pretending to label arbitrary arm actions?

DAgger1 collected 60 v1a episodes on seeds 2400000–2400059. The student executed its own commands; only gripper labels were supervised. The original oracle labelled open when touching the destination rack, within **5 cm** and **20°**; held/lifted carry states were closed, and other states undefined. Collection stopped 150 steps after placement. The summary's “13/33 placed-not-released” is a midpoint statistic, not a final 60-episode breakdown. [Collection][phase1]; [oracle/masks][brief7].

Fine-tuning v1a for 4k steps with human, teacher and DAgger data in a 1:1:1 mix still gave **0/30**. Only **2/30** opened after grasping. On its own **1,250 open-labelled training frames**, the fine-tune predicted raw gripper **+0.93**, with **0%** predicted open. The recorded diagnosis was L1's conditional-median behavior plus dilution: gripper-only DAgger elements were approximately **0.1%** of the loss. Extra data could not help labels that the objective failed to fit. The fine-tune also had an extra-training-steps confound. [Fit probe][phase1].

bce1 (`bc@5233dc7`, checkpoint `7535cac0`) fine-tuned v1b for 6k steps, replacing gripper L1 with separately normalized BCE, open-class pos_weight 2 and threshold 0.5; other action dimensions retained L1. The logged mixture was human60/teacher_v1/dart0/dagger with weights **1:0.4:0.6:0.67**, with 298 training and 24 validation episodes. Validation L1 was **0.0118**, gripper accuracy **0.969**, open-label accuracy **0.961**. Dev improved to **14/30 (47%, Wilson 95% CI 30–64%)**, with opening in **25/30**; recorded paired discordances versus v1b were 14 versus zero, McNemar p = 1.2e-4. This was the first successful learned student. DAgger2, collected from v1b with oracle-executed gripper commands, stopped at **50/60** episodes. [BCE run][phase1]; [loss specification][brief8].

Replay of bce1's 16 failures found five never released, six released badly, three carried elsewhere and two missed grabs. Threshold 0.4 looked better on dev: **20/30**, versus 14/30 at 0.5, gained eight/lost two, p = 0.11. Threshold 0.3 gave **18/30**, p = 0.29. On fresh block 2600000, however, 0.5 achieved **18/30** and 0.4 **17/30**, gained two/lost three, p = 1.0. The dev threshold gain did not replicate. Default 0.5 was retained; the then-current combined estimate was **32/60 = 53%**, with source slot 5 at **1/10**. [Threshold tuning and confirmation][confirmation].

**Conclusion and next change.** BCE addressed a demonstrated loss-fitting failure; threshold adjustment did not establish a general improvement. Training also exposed a disk-full cache failure. Obsolete and partial caches were removed; brief #9 requested per-episode shared caching. Later bce2 reproduction deliberately exported committed `5233dc7`, because uncommitted cache changes in the BC worktree were not part of bce1's training code. [Failure/reproduction history][phase1]; [cache request][brief9]; [bce2 setup][bce2].

### microVLA: flow matching as an alternative student — 11-20 to 11-21

**Question.** Can a flow model handle multimodal gripper actions and localize objects better than the chunked BC head?

The microVLA adapter (`vla@9eb7939`) used the attributed upstream microVLA architecture, **11.2M parameters**, three cameras resized to 96 px, horizon 20/execute 10, a constant instruction and a masked flow loss for partial DAgger labels. The recorded initial mixture contained **372 episodes**. The first no-distillation run was stopped and deleted without evaluation after revisiting the source recipe. The completed run used **3k SigLIP feature-distillation steps + 10k flow steps**. It resumed after a clean shutdown; the watchdog initially failed to restart after reboot, then resumed thermal duty cycling. Final train/val flow loss was **0.0076/0.0109**, with **1.28 GB** peak allocated CUDA memory. [Training history][microhistory]; [adapter request][brief10].

Initial dev evaluation was **11/30 (37%, Wilson 22–54%)**; all 19 failures timed out. Against adopted bce1 at 0.5, the recorded paired comparison was not significant (p = 0.58). Replay reproduced **30/30** outcomes. microVLA had seven pickup slips and seven rack-then-table failures, compared with one of each for bce1 in that replay categorization. The broad “on rack, not settled” category concealed five bce1 never-release cases, so later analysis split failure types more carefully. [Initial evaluation and replay][confirmation]; [category correction][bceread].

The release-height hypothesis was rejected by the replay probe: successful and failed releases had similar height and mostly similar tilt. Horizontal opening offsets differed: **1.3–5.4 cm** in successes and **4.2–8.8 cm** in failures. The working diagnosis was final-approach positioning, not simply early opening. Reducing execute length from 10 to five produced **12/30**, versus 11/30; gained six/lost five, p = 1.0, with **2.2×** policy time. That small comparison supplied no evidence of a gain, so execute 10 remained standard. [Timing/chunk probe][confirmation].

**Conclusion and next change.** The architecture was viable but had different failure modes and no demonstrated advantage. Cloud experiments would test resolution, data and capacity while retaining the distillation recipe. An internal wording inconsistency remains: the setup history mentions 16-pixel patches, whereas the 96 px/12×12-token recipe in brief #10 and the implementation report describes patch size 8. This narrative uses the documented configuration and does not infer a hidden architecture change. [Training history][microhistory]; [adapter brief][brief10]; [Muon implementation report][muonreport].

### Modal migration and repeatability — 11-21 morning

**Question.** Can cloud throughput remove laptop thermal limits, and are single dev runs stable enough for decisions?

The Modal simulation image passed the same **555-file** integrity check. A flow-throughput benchmark used 300 steps, batch/microbatch 64 and four loader workers, with rate measured over steps 25–300. Its recorded costs were GPU-only estimates, excluding CPU and memory: [Modal benchmark][gpu].

| Device | Seconds/step | Time per 10k flow steps | GPU-only cost per 10k |
|---|---:|---|---:|
| Laptop RTX 3050, cool, microbatch 16 | 0.75 | 2.1 h; 5+ h with pauses | 0 |
| T4 | 0.97 | 2.7 h | $1.59 |
| L4 | 0.21 | 0.58 h | $0.47 |
| A100-40GB | 0.090 | 0.25 h | $0.52 |

The A100 was **2.3×** faster than L4 at roughly the same GPU cost per run, reversing the initial hardware preference. T4's poor training rate was tentatively attributed to bf16 support; that explanation was not tested. Training moved to A100 where appropriate. [GPU comparison][gpu].

An unchanged bce1 rerun scored **18/30**, versus the original 14/30 on the same dev seeds. Only **12/30** action traces were bit-identical; small initial differences amplified through feedback. Saved-action replays reproduced outcomes, implicating the policy/execution path rather than replay instability. GPU numerics were the initial hypothesis, but later CPU-policy Modal repetitions also varied. The source of all nondeterminism remains unresolved. The intermediate bce1 estimate became **50/90 = 56%**, combining original dev, dev rerun and confirmation. [Repeatability finding][nondeterminism].

**Conclusion and next change.** Comparisons switched to three 30-episode dev repeats per candidate under the same Modal conditions, with T4 NVIDIA EGL rendering and CPU policy inference. These repeats sample run variation on reused seeds; **90 trials are not 90 unique seeds**. Original paired tests and later pooled confidence intervals should be read in that light. [Protocol change][bce2].

### DAgger3, label diagnosis and strict relabelling — 11-21 morning

**Question.** Will labels from bce1's own stuck states repair release, and how should “safe to open” be defined?

A probe found confidently wrong open probabilities in stuck states: mean/max **0.032/0.093** over 363 oracle-open steps on seed 1000005 and **0.071/0.231** over 782 steps on seed 1000014. Lowering a threshold could also open during carrying, where probabilities reached **0.50–0.60**. This supported collecting bce1's own states rather than more threshold tuning. [Release diagnosis][releaseprobe].

CPU-only Modal collection initially appeared hung; it was actually spending over **12 minutes/episode** in Mesa rendering, with misleading zero load readings. T4 NVIDIA EGL reduced a full episode to **48 s**. The first-observation rendering check on seed 2500000 found **100% identical pixels** between Modal T4 and laptop NVIDIA EGL; laptop Mesa matched only **72.2%**, mean absolute difference **0.985**. Earlier teacher data had used Mesa. Any effect of that image difference on learning remains untested. [Collection setup][dagger3].

DAgger3 collected **120 episodes**, seeds 2500000–2500119, in 20 T4 shards and **251 s** wall time. Outcomes were success 47, reached-then-failed 29, placed-not-released 22, no-grasp 14, other six and fell two. Labels totalled **8,257 open / 30,633 closed / 63,340 undefined**; stuck episodes contributed **2,662** open labels. This collector yield, **47/120 (39%)**, is not a dev score: it uses training seeds and early stopping. [DAgger3 result][dagger3].

bce2a restarted the bce1 recipe from v1b for 6k steps with DAgger3; bce2b fine-tuned bce1 for 3k steps. Repeated dev results were **24/90 (27%)** and **42/90 (47%)**, versus bce1 **57/90 (63%)**. In first-run replay, bce2a had 10 released-but-unsettled failures, versus four for bce1. The diagnosis was that matching the open-label boundary to the task's outer success tolerance taught marginal releases that later settled out of tolerance. The no-release problem also persisted. [bce2 results][bce2result].

To choose a stricter rule, 270 dev episodes were replayed. Among releases, success fractions by distance were **0.65 (n=23)** at 0–2 cm, **0.82 (n=49)** at 2–3 cm, **0.79 (n=67)** at 3–4 cm, **0.33 (n=39)** at 4–5 cm, **0.08 (n=26)** at 5–7 cm and **0.00 (n=21)** beyond 7 cm. The 45 never-release episodes all failed. This observational, policy-confounded analysis contradicted the initial 2 cm guess. The selected rule was rack contact, distance **≤4 cm**, tilt **≤10°**. Excluded open states became unlabelled, not closed, to avoid teaching indefinite holding where the arm was not corrected. [Boundary analysis][relabel].

Offline replay generated `dagger3s`. Every one of **120** episodes first reproduced the original labels/masks bit-for-bit. **2,444** open labels became undefined; no closed labels changed and no labels were added. **5,813** open labels remained, including **1,966/2,662** from stuck episodes. Independent verification found non-gripper arrays unchanged and zero problems. [Relabelling result][relabel].

bce3a/bce3b changed only loose to strict labels within the same two initialization/step recipes. bce3a reached **44/90 (49%)**, improving on bce2a's 24/90; bce3b reached **43/90 (48%)**, close to bce2b's 42/90. Neither beat bce1. The source reports approximate pooled z-test p ≈ 0.002 for bce3a versus bce2a, but reuse of seeds limits the independence assumption. [Strict-label results][strictresults].

In parallel, microVLA v2a increased images to 128 px and scored **14/90 (16%)**; v2b added loose DAgger3 and scored **44/90 (49%)**, versus v1 **32/90 (36%)** under the repeated Modal protocol. The source reports p ≈ 0.001 for v2a versus v1 and p ≈ 0.07 for v2b versus v1, with the same pooled-test caveat. Resolution hurt; DAgger data looked useful for the flow student, without establishing a definitive gain on fresh seeds. [v2 training][v2training]; [repeated results][strictresults].

**Conclusion and next change.** Oracle design matters, and its thresholds are training choices. Strict labels repaired much of one regression but did not surpass bce1; microVLA's response to the same data differed. The next phase compared capacity/optimization with low-budget post-training of pretrained VLAs.

### Benchmark: larger microVLA/Muon versus pretrained VLA SFT — 11-21 afternoon/evening

**Question.** Within a small cloud budget, is more from-scratch capacity or adaptation of a pretrained VLA the better next investment?

The benchmark began at **$9.77** cumulative metered cost. The initial $15 budget rose to $20 (hard stop $29.77) and then $25 (hard stop $34.77). The common fully labelled dataset for the optimizer sweep and pretrained fine-tunes was **262 episodes / 178,845 frames**, human60 + teacher_v1 + teacher_v1_dart0. DAgger was excluded from the LeRobot models because its gripper-only masks were unsupported by those losses. Conversion preserved episode/frame counts. The final large microVLA instead used the v2b mixture including DAgger, so this was not a fully data-matched four-model comparison. [Benchmark setup][benchmark].

The adapter mapped three images, 19 proprioceptive values and 10 active action dimensions to the common interface, horizon 20/execute 10. pi0.5 and SmolVLA froze their pretrained VLMs and trained action experts/projections. MolmoAct2 ultimately trained VLM LoRA plus its expert, with checkpointing. These are different adaptation modes and sample budgets. Normalizers were checked against source statistics, with maximum difference approximately **1e-8**. Smoke checkpoints returned valid action chunks; that verified interface wiring, not task learning. [Pipeline and smoke checks][benchmark].

The larger microVLA had **35.33M parameters**, versus 11.2M. Brief #11's hybrid Muon implementation applied orthogonalized Nesterov updates to transformer matrices, split fused QKV projections, used Moonlight scale matching, and retained AdamW for embeddings, heads, norms, biases and modulation. The implementation report records CPU validation; the experiment log records nine tests passing and the later actual cloud sweep. Synthetic optimizer checks do not establish robot-policy improvement. [Experiment record][benchmark]; [implementation report][muonreport]; [brief #11][brief11].

The actual L4 sweep used the large model, no distillation, 3,000 flow steps and seed 42. Muon and its AdamW fallback shared the listed learning rate. Recorded held-out results were: [LR sweep][benchmark].

| Optimizer | LR | Validation flow loss at 3k | Training flow loss at 3k | Seconds/step |
|---|---:|---:|---:|---:|
| AdamW | 1.5e-4 | 0.0280 | 0.0095 | 0.44 |
| AdamW | 3e-4 | 0.0234 | 0.0084 | 0.44 |
| AdamW | 6e-4 | 0.0202 | 0.0081 | 0.44 |
| Muon | 3e-4 | 0.0345 | 0.0064 | 0.55 |
| Muon | 6e-4 | 0.0267 | 0.0062 | 0.55 |
| Muon | 9e-4 | 0.0255 | 0.0061 | 0.55 |

Muon had lower training loss but worse validation loss and was recorded as 25% slower per step. The chosen final optimizer was **AdamW 6e-4**. This is a one-seed, short, no-distillation proxy comparison; no Muon checkpoint received a recorded task evaluation. The suggestion that a still higher Muon rate would not close the gap is an inference in the source, not a tested result. [Sweep interpretation][benchmark].

`mvla_large_final` used 3k SigLIP distillation steps and 20k flow steps, batch 64, with the v2b mix and weights **1/0.68/0.32/0.5/0.5/0.5**. Training took **3108 s**, approximately 52 minutes on A100. Validation loss improved to **0.0161**, versus v2b's 0.0191, yet dev fell to **11 + 9 + 10 = 30/90 (33%)**, versus 44/90. The recorded pooled comparison gives z = 2.12, p ≈ 0.034, subject to the repeated-seed caveat. Model size and training length changed together. Overfitting and learning-rate interactions were hypotheses; they were not established causes. [Large-model outcome][benchmark].

SmolVLA trained **5k × batch 32**, about 30 minutes on A100, then scored **14 + 15 + 15 = 44/90 (49%)**. pi0.5 trained **1k × 32**, 68 minutes on A100, after a longer plan was cut for budget; its valid rerun scored **1 + 1 + 2 = 4/90 (4%)**. MolmoAct2 trained **1.5k × 8**, 76 minutes on A100, and scored **1 + 2 + 1 = 4/90 (4%)**. The source estimates roughly **0.9 / 0.18 / 0.07 epoch** for SmolVLA/pi0.5/MolmoAct2. These are limited post-training budgets, not estimates of the models' ceilings. [Fine-tune outcomes][benchmark].

pi0.5's first **0/90** evaluation was invalid as a learning comparison: every episode had a policy error because loading took approximately **180 s**, exceeding the default **120 s** policy timeout. The valid rerun used the evaluator's documented `--policy-timeout 600`, without code changes. Readiness under the default timeout remains an unresolved submission constraint. MolmoAct2's reported gradient norm of 0.000 was also corrected: LeRobot logs zero when global clipping is disabled, while MolmoAct2 clips within optimizer groups; the initial “no gradients” hypothesis was wrong. Memory failures in several MolmoAct2 modes were real, but the zero log did not demonstrate broken training. [Timeout and gradient corrections][benchmark].

**Conclusion and next change.** bce1 remained strongest. SmolVLA matched v2b within this budget; scaling microVLA and the tested Muon settings did not help. The source's benchmark total was **$25.99**, $0.99 above the $25 cap, at cumulative metered **$35.76**. Slow evaluation/container waiting contributed to the overrun. Next work shifted toward zero-shot floors, additional teacher data and an observation-driven release harness. [Benchmark summary][benchmarksummary]; [spend][benchmark].

### Post-training follow-up: zero-shot floors, new data and harness — 11-22

**Question.** How much adaptation did the pretrained bases need, and can targeted data plus explicit release supervision improve bce1?

Released pi0.5 and MolmoAct2 bases were wrapped through the real LeRobot training/checkpoint pipeline with weight updates replaced by a no-op. They retained pretrained action heads, used dataset MEAN_STD normalization, padded to 32 action dimensions and mapped the first 10 to this task. Because action semantics differ across embodiments, the source expected approximately zero success; the completed evaluations subsequently measured that floor. These are “no weight adaptation” floors under the stated mapping, not universal zero-shot capability tests. [Zero-shot setup][latest].

A missing `__main__` guard caused spawned DataLoader workers to rerun the wrapper; adding the guard fixed it, with approximately **$0.3** recorded for the failed attempt. For pi0.5, **812/812** base tensors were bit-identical after normalized key mapping, maximum difference zero. MolmoAct2's different key layout prevented the same comparison; preservation relies on the no-op update guarantee. Three 30-episode dev repeats per base were launched at **02:05** on L4 with timeout 600. Both released bases finished at **0/90**, with all **30/30** episodes of each run timing out at **1,700 steps**, without crashes. The corresponding tiny fine-tunes each achieved 4/90, so those observed successes followed weight adaptation. This comparison establishes a floor only for this embodiment mapping. One L4 input was preempted and automatically restarted, adding evaluation time without a separately recorded charge. The separate zero-shot budget is $15 from baseline cumulative spend $35.76; final zero-shot spending is not itemized. [Setup][latest]; [completed results][night].

Brief #12 produced `ops/modal_teacher.py`. Its report established CLI/file behavior and static/mocked checks; subsequent master-log entries establish actual cloud execution. A separate three-seed dry run succeeded **1/3**. The full run, seeds **2800000–2800239**, used **24 T4 shards × 10** and yielded **191/240 = 79.6%**, saving **191** success-only `teacher_v1_gen2` episodes in approximately **3.5 minutes** wall time. Rendering used NVIDIA EGL. File keys/shapes/dtypes matched the prior teacher format. The latest teacher inventory is recorded as **73 + 129 + 191**; cumulative metered spend after this entry was **$37.32**. This meter includes activity after the benchmark baseline and is not a separately itemized teacher-job charge. This data was subsequently used in bce4a/bce4b and SmolVLA v2, described below. [Generation results][latest]; [launcher report][report12]; [SFT results][night].

The harness wraps bce1 with an RGB/proprioception plate-state estimator, release supervisor, grasp checks/retries and optional temporal ensembling. Brief #13 specifies replay labels in a new directory, episode-level train/validation splits, release-region estimator errors, synthetic state-machine tests and offline release comparisons. Its implementation report records the estimator, replay labeller, supervisor and offline tooling, initially handed off without policy evaluation. The later master entry supersedes the earlier “Not started” status with reviewed implementation and cloud training, described below. [Harness brief][brief13]; [implementation handoff][report13]; [subsequent evidence][night].

**Conclusion and next change.** The zero-shot comparison is complete. Work moved to measuring whether the new data improves students and whether estimated plate pose can support safer release decisions.

### SFT on gen2, threshold sweeps and harness training — 11-22 night/morning

**Question.** Does continuing the best student on more teacher data improve closed-loop control, and can release supervision repair the remaining failures?

bce4a continued from bce1 for **4,000 steps**; bce4b repeated the bce1 recipe from v1b for **6,000 steps**. Both used the committed bce1 code export `bc_5233dc7`, A100-40GB, BCE pos_weight 2, warmup 200, microbatch 64, two workers and six validation episodes. The human60 / teacher_v1 / teacher_v1_dart0 / dagger / teacher_v1_gen2 weights were **1 / 0.4 / 0.6 / 0.67 / 0.9**. bce4a finished in **250 s**, val **0.10677**, gripper accuracy **0.970**; bce4b finished in **299 s**, val **0.11060**. The local detached-client output stopped at step 3000, but volume logs established that both remote runs exited successfully. [SFT setup][latest].

After repairing invalid checkpoint launches, bce4a scored **16 + 17 + 14 = 47/90 (52%)** and bce4b **19 + 18 + 17 = 54/90 (60%)**, versus bce1's 57/90. The source reports z ≈ 1.5, p ≈ 0.13 for bce4a versus bce1, and calls bce4b's difference nonsignificant. These are repeated dev seeds, not independent fresh-seed confirmation. The two recipes compare continuing from best with restarting from v1b using more data; they do not isolate every effect of data, initialization and training duration. [SFT outcomes][night].

First-run replay used the established failure classifier. Each row covers only **30 episodes**, so the apparent change in failure mix is preliminary. [Failure analysis][night].

| Run 1 | Success | A: no grasp | B: no lift | C: short | D1: on rack, never released | D2: on rack, unsettled | E: drop |
|---|---:|---:|---:|---:|---:|---:|---:|
| bce1 | 19 | 3 | 0 | 2 | 1 | 4 | 1 |
| bce4a | 16 | 0 | 1 | 1 | 7 | 1 | 4 |
| bce4b | 19 | 1 | 1 | 1 | 5 | 3 | 0 |

Grasp misses fell from **3 to 0–1** in these runs, while never-release failures rose from **1 to 5–7**. The source interprets this as better grasping but worse holding behavior. Its proposed mechanism is unproven: the teacher's release wait, up to **100 steps**, may add many closed-gripper examples while the plate is on the rack. This motivated a dev-only threshold sweep rather than an immediate claim that more data helped overall. [Interpretation][night].

Lower global thresholds did not improve pooled success. bce4b at **0.3** scored **19 + 14 + 19 = 52/90**, and at **0.2**, **14 + 14 + 15 = 43/90**; bce1 at **0.3** scored **22 + 12 + 18 = 52/90**. Defaults at 0.5 remained bce4b 54/90 and bce1 57/90. In bce4b run 1, threshold 0.3 versus 0.5 reduced never-release cases to **1 versus 5**, but drops rose to **3 versus 0** and no-lift cases to **2 versus 1**; both runs had **19** successes. This supports conditioning release on pose rather than adopting a lower global threshold. The **12–22/30** spread for bce1 at 0.3 also reinforces the need to report every repeat. No fresh confirm/test result is recorded for these sweeps. [Threshold results][night].

SmolVLA v2 continued the 5k checkpoint for a planned **15k additional steps**, total **20k**, batch 32 on A100. Its `plate_transfer_v2` dataset contains **453 episodes / 313,860 frames**, the original **262** plus **191** gen2 episodes; conversion used six CPU shards taking **121–184 s**. LeRobot recomputes normalization statistics on load, so the data, steps and normalization changed together. Additional checkpoints at +5k/+10k/+15k permit later step comparisons but do not remove the data confound. An initial attached run died at **2357/15000** after approximately **14 min**, losing approximately **$0.5**. The detached relaunch later stopped during the power outage at **11186/15000**, with checkpoint **010000** saved. It resumed from that checkpoint using the saved training config and `--resume=true`, through server-side spawning. No completed v2 policy evaluation is recorded. [SFT launch history][latest]; [interruption and resume][night].

The harness implementation was reviewed and committed as **2a27e6c + 095ff8b**, superseding the handoff report's read-only-git block. `ADV_H_SUPERVISE=0` permits a cadence-only control, and **10/10** harness tests passed. Replay labelled **120/120** DAgger3 episodes into `dagger3_priv`, with every pre-action state bit-for-bit equal. The estimator used dart0, gen2 and these own-policy labels; human60, teacher_v1 and the earlier DAgger sets lacked the required privileged arrays. The **50-step L4 smoke** reported held accuracy **0.979** and release-region distance MAE **2.52 cm**, taking **259 s** including cache construction. Those are smoke metrics, not the full model's accuracy. [Harness record][night].

The first full estimator launch, `est_v1`, failed after **5 s** because four workers exceeded the supported 0..2 range. `est_v1b` used two workers on A100, planned 10k steps, batch 64/microbatch 32 and validation every 1000 steps; throughput was approximately **40 s/1000 steps**. At **step 7000**, validation on six held-out episodes per directory, **15,256 rows**, gave plate error **1.65 cm**, site error **1.14 cm**, distance MAE **1.27 cm**, tilt MAE **2.16°** and held accuracy **0.984**. Within true distance **≤6 cm**, **3,354 rows**, distance MAE was **0.68 cm**, tilt MAE **1.37°**, held accuracy **0.951**. These are episode-held-out offline estimates; average errors do not establish safe release decisions. The outage stopped training at **8000/10000**; the saved `last.pt` is used for evaluation, while the quoted metrics are explicitly from step 7000. [Estimator metrics and interruption][night].

The real-environment harness smoke passed with supervision on/off, and later with bce1, bce4b, execute 10 and temporal ensembling. These checks establish valid loading/actions, not task success. **15 dev evaluations**, three 30-episode repeats for each configuration, were spawned on T4: `h_pass` (bce1, execute 2, supervision off), `h_full` (bce1 plus supervisor, execute 2), `h_exec10` (full supervisor at execute 10), `h_bce4b` (bce4b plus supervisor), and `h_ens` (full harness plus temporal ensembling). Default supervisor settings are distance **≤3.5 cm**, tilt **≤12°**, stability **6 steps**, held probability **≥0.7**, settle **75 steps**. **Results are pending**; no harness success rate is available. [Launch record][night].

**Conclusion and next change.** New teacher data and lower gripper thresholds have not surpassed bce1. Offline estimator accuracy makes a pose-gated release test feasible, but only the pending closed-loop ablations can establish its value. Operationally, checkpoint validation and a real policy-process smoke now precede full launches; deployed functions with server-side spawning replaced client-dependent chains after the outage. [Overnight record][night].

## 3. Results table

`D` means the balanced dev sampler starting at 1000000; `C` means confirmation starting at 2600000. Repeated runs reuse the same seed set. Costs are quoted only where the master log supplies them; “unitemized” means no policy-specific charge was recorded. Laptop cost is not measured as a monetary total. Except where noted, learned policies use horizon 20/execute 10. Table entries for the same checkpoint describe distinct evaluations and must not be summed into unique-seed estimates.

### Observation-driven student evaluations

| Policy/configuration | Dev success k/n; repeat spread | Seeds/protocol | Key config and identity | Recorded cost/time | Evidence |
|---|---|---|---|---|---|
| BC human60 | 0/30 | D, laptop | `36337e1`; human 54 train/6 val; 8k steps; L1; ckpt `2cc397d71e8ef46a` | Laptop; spend unitemized | [Runs][phase1] |
| Student v1a | 0/30 | D, laptop | `f674fe2`; +73 teacher demos; 8k; aux off; `dc10b512112598e7` | Laptop; spend unitemized | [Runs][phase1] |
| Student v1b | 0/30 | D, laptop | `f674fe2`; +34 DART-0 demos; 8k; `3a66e26fa7d445ce` | Laptop; spend unitemized | [Runs][phase1] |
| v1a + DAgger1 | 0/30 | D, laptop | `a5b5dc7`; +4k fine-tune; L1; gripper-only DAgger labels | Laptop; spend unitemized | [Runs][phase1] |
| bce1 threshold 0.5, original | 14/30 | D, laptop | `5233dc7`; v1b +6k; BCE pos_weight 2; `7535cac0` | Laptop; ~1 h+ training | [Runs][phase1], [recipe][bce2] |
| bce1 threshold 0.4 | 20/30 | D, threshold selection | Same checkpoint; dev-tuned threshold | Laptop; spend unitemized | [Sweep][confirmation] |
| bce1 threshold 0.3 | 18/30 | D, threshold selection | Same checkpoint; dev-tuned threshold | Laptop; spend unitemized | [Sweep][confirmation] |
| bce1 threshold 0.5, probe rerun | 18/30 | D, laptop; unchanged returned actions | Same checkpoint; repeatability probe | Unitemized | [Rerun][nondeterminism] |
| bce1 threshold 0.5, Modal reference | **57/90 (63%); 19,17,21** | D ×3; T4 EGL, CPU policy | Same adopted checkpoint | Unitemized; policy ~1 s/episode | [Repeated results][strictresults] |
| bce4a, gen2 continuation | 47/90 (52%); 16,17,14 | D ×3; Modal | bce1 +4k; `bc_5233dc7`; gen2 mix; threshold 0.5 | 250 s A100 training; spend unitemized | [SFT][night] |
| bce4b, gen2 recipe repeat | 54/90 (60%); 19,18,17 | D ×3; Modal | v1b +6k; same code/mix; threshold 0.5 | 299 s A100 training; spend unitemized | [SFT][night] |
| bce4b threshold 0.3 | 52/90; 19,14,19 | D ×3; T4 | Same bce4b checkpoint; dev-only override | Unitemized | [Sweep][night] |
| bce4b threshold 0.2 | 43/90; 14,14,15 | D ×3; T4 | Same bce4b checkpoint; dev-only override | Unitemized | [Sweep][night] |
| bce1 threshold 0.3, Modal control | 52/90; 22,12,18 | D ×3; T4 | Same adopted bce1 checkpoint; separate from laptop sweep | Unitemized | [Sweep][night] |
| bce2a, loose labels | 24/90 (27%); 10,7,7 | D ×3; T4 EGL, CPU policy | v1b +6k; DAgger3; `6066cb33` | 305 s A100 training; spend unitemized | [Recipe][bce2], [results][strictresults] |
| bce2b, loose labels | 42/90 (47%); 13,12,17 | D ×3; T4 EGL, CPU policy | bce1 +3k; DAgger3; `6c24751d` | 217 s A100 training; spend unitemized | [Recipe][bce2], [results][strictresults] |
| bce3a, strict labels | 44/90 (49%); 15,15,14 | D ×3; T4 EGL, CPU policy | v1b +6k; DAgger3s; `dc97b409` | Unitemized | [Strict results][strictresults] |
| bce3b, strict labels | 43/90 (48%); 16,15,12 | D ×3; T4 EGL, CPU policy | bce1 +3k; DAgger3s; `4216de3f` | Unitemized | [Strict results][strictresults] |
| microVLA v1, initial | 11/30 (37%) | D, laptop | 11.2M; 3k distill +10k flow; 96 px; `107dcefe` | Laptop; flow ~22:35–03:11 with pauses | [Eval][confirmation], [training][microhistory] |
| microVLA v1, execute 5 | 12/30 (40%) | D, laptop | Same checkpoint; execute 5 | Policy 15.1 s/episode, 2.2× original | [Chunk probe][confirmation] |
| microVLA v1, repeated | 32/90 (36%); 14,8,10 | D ×3; T4 EGL, CPU policy | Same v1 checkpoint; execute 10 | Unitemized; policy 17–27 s/episode | [Repeated results][strictresults] |
| microVLA v2a | 14/90 (16%); 3,6,5 | D ×3; T4 EGL, CPU policy | 128 px; 3k distill +10k flow; `6149b38f` | 1429 s A100 training; spend unitemized | [Training][v2training], [results][strictresults] |
| microVLA v2b | 44/90 (49%); 17,13,14 | D ×3; T4 EGL, CPU policy | 96 px; +loose DAgger3; `425b4edd` | Training entry: 1075 s A100; summary: ~18 min L4; unresolved | [Training][v2training], [summary][benchmarksummary] |
| microVLA large final | 30/90 (33%); 11,9,10 | D ×3; T4 EGL, CPU policy | 35.33M; 3k distill +20k flow; AdamW 6e-4; DAgger mix; `b065543a` | 3108 s /52 min A100; spend unitemized | [Outcome][benchmark] |
| SmolVLA fine-tuned | 44/90 (49%); 14,15,15 | D ×3; L4 inference | Frozen VLM, expert SFT; 5k ×32; 262 fully labelled episodes | ~30 min A100; ~21 min/30 eval; spend unitemized | [Benchmark][benchmark] |
| pi0.5 fine-tuned, valid rerun | 4/90 (4%); 1,1,2 | D ×3; L4; timeout 600 | Frozen VLM, expert SFT; 1k ×32 | 68 min A100; 35–49 min/30 eval; spend unitemized | [Benchmark][benchmark] |
| MolmoAct2 fine-tuned | 4/90 (4%); 1,2,1 | D ×3; L4; timeout 600 | VLM LoRA + expert; 1.5k ×8; 737M learnable | 76 min A100; 44–47 min/30 eval, map 70 min incl. wait; spend unitemized | [Benchmark][benchmark] |
| pi0.5 released base | 0/90; every run 0/30 | D ×3; L4; timeout 600 | No weight updates; dataset normalization and embodiment mapping | Separate $15 zero-shot cap shared with MolmoAct2; actual charge unitemized | [Final results][night], [mapping][latest] |
| MolmoAct2 released base | 0/90; every run 0/30 | D ×3; L4; timeout 600 | No-op weight updates; same mapping; tensor equality not independently established | Same shared cap; preempted input added time, no separate charge | [Final results][night], [mapping][latest] |

For the main repeated comparison, the master log reports Wilson 95% intervals: bce1 **53–73%**; bce2a **19–37%**; bce2b **37–57%**; bce3a **39–59%**; bce3b **38–58%**; microVLA v1 **26–46%**; v2a **9–24%**; v2b **39–59%**. These are the source's pooled intervals and do not account fully for repeated seeds. Its pooled bce1 history is **107/180 = 59%**, combining laptop dev 14,18, Modal dev 19,17,21 and confirm 18; it mixes conditions and is not an independent held-out estimate. [Intervals and cumulative history][strictresults].

### Confirmation, teachers and diagnostic/reference policies

| Policy/run | Dev success | Other result k/n | Seeds and key config | Recorded cost/time | Evidence |
|---|---|---|---|---|---|
| bce1 threshold 0.5 confirmation | — | 18/30 (60%) | C; same `7535cac0`; slots 0–5: 3,5,4,3,3,0 | Unitemized | [Confirmation][confirmation] |
| bce1 threshold 0.4 confirmation | — | 17/30 (57%) | C; slots 0–5: 2,5,5,3,2,0 | Unitemized | [Confirmation][confirmation] |
| Teacher v1 initial | — | 27/30 (90%) | 2000000–2000029; privileged; `6c9f0e6`; kp=2, tolerance 1 cm, wait 100 | Laptop; unitemized | [Teacher][phase1] |
| Teacher v2 | — | 6/30 | Same training seeds; privileged; `8b4744d`; six bundled changes | Laptop; unitemized | [Teacher][phase1] |
| v1 align-then-lower ablation | — | 0/2; abandoned | 2000000–2000001; `2cc0622`; intended 30 not completed | Laptop; unitemized | [Ablation][phase1] |
| Teacher v1 early collection | — | 46/58 (79%) yield | Attempted block 2100000+, partial ranges; success-only save | ~45 episodes/h thermally throttled; unitemized | [Collection][phase1] |
| Teacher v1 DART 0 yield check | — | 34/41 (83%) | 2200000+; `36101fe`; privileged labels | Laptop; unitemized | [Yield][phase1] |
| Teacher v1 DART 1.0 yield check | — | 3/23 (13%) | 2300000+; `36101fe`; noisy execution, clean targets | Laptop; stopped; unitemized | [Yield][phase1] |
| bce1 DAgger3 collection | — | 47/120 (39%) yield | 2500000–2500119; early-stop collector, own gripper execution | 251 s wall, 20 T4 shards; unitemized | [DAgger3][dagger3] |
| Teacher v1 gen2 dry run | — | 1/3 | 2800000–2800002; separate check dataset | T4; unitemized | [Latest][latest] |
| Teacher v1 gen2 full collection | — | 191/240 (79.6%) yield | 2800000–2800239; `2cc0622` v1, DART 0, NVIDIA EGL | ~3.5 min wall, 24 T4 shards; no separate charge | [Latest][latest] |
| Own-seed demo replay | — | 59/60 | Human demo seeds, diagnostic replay | Laptop; unitemized | [Phase 0][phase0] |
| New-seed same-slot replay | — | 1/8; heat stop | Exact seed list unrecorded; diagnostic replay | Laptop; unitemized | [Phase 0][phase0] |
| HoldPose example | — | Success count unrecorded; smoke only | 7310000, one full episode, non-qualifying | 28.3 s wall; 2.5 s policy | [Phase 0][phase0] |
| Upstream agent-written policies | — | Best 36/100; other exact k/n unavailable here | External reference, upstream protocol; 21 runs, seven agents ×3 | Not recorded | [Phase 0][phase0] |

The teacher's later “112 saved, approximately 84% yield” entry has no exact attempts denominator. It is an inventory update, not an additional reconstructable k/n evaluation. DAgger1/2 are likewise data collections without complete policy-success counts in the source; their sizes must not be turned into success rates.

### Invalid, pending and unevaluated tracks

| Track | Recorded status | Config/seeds | Cost/time or limitation | Evidence |
|---|---|---|---|---|
| pi0.5 first fine-tuned eval | 0/90 recorded, policy errors throughout; invalid capability comparison | D ×3, L4, default timeout 120 | Checkpoint load exceeded timeout; valid timeout-600 rerun listed above | [Timeout][benchmark] |
| Student v1c-long | Paused at 3.5k; no eval | Planned 30k, human+teacher+DART-0 | ~4.6 h projected under throttling, not completed runtime | [Paused run][phase1] |
| microVLA without distillation | Stopped/deleted; no eval | Initial v1 attempt | Throughput smoke only | [Training history][microhistory] |
| Large AdamW/Muon sweep checkpoints | Offline flow losses only; no policy success | Six LR points; seed 42; 3k steps, no distill | L4; rates in sweep table; charges unitemized | [Sweep][benchmark] |
| bce4a_r2, bad checkpoint launch | 0/30, all policy_error; invalid capability comparison | D; nonexistent container path; other four stopped, one never launched | 52 s for failed run; charges unitemized; corrected results listed above | [Launch repair][night] |
| SmolVLA v2 | Training resumed; no completed policy evaluation | 453 episodes; +15k planned after original 5k; batch 32, A100 | Attached attempt lost ~14 min / ~$0.5; later interruption at 11186/15000 | [Setup][latest], [resume][night] |
| est_v1 / est_v1b | First launch failed; replacement reached 8000/10000; offline metrics only | A100; 6 val episodes/dir; step-7000 metrics above | First failure after 5 s; ~40 s/1000 steps on replacement; charges unitemized | [Estimator][night] |
| h_pass / h_full / h_exec10 / h_bce4b / h_ens | Spawned; no policy success results | D ×3 each, 15 evals total, T4; saved est_v1b | Cadence-only, full, exec10, bce4b and ensemble controls; charges unitemized | [Harness ablations][night] |

### Cost accounting and comparability

The throughput table's $0.47/$0.52 per 10k steps are **GPU-only projections** for a small-model throughput benchmark. They are not measured complete training/evaluation charges for individual models. The source does not itemize a dollar total per evaluated policy, so none is imputed here.

The model benchmark moved cumulative metered spending from **$9.77** to **$35.76**, explicitly recorded as **$25.99** benchmark spend. Intermediate recorded checkpoints were **$23.79** ($14.02 benchmark) and **$31.96** ($22.19 benchmark). It exceeded the final $25 cap by **$0.99**. A subsequent entry gives cumulative **$37.32** after the additional-data/zero-shot setup period. The zero-shot evaluation cap is a separate **$15**, not a completed spend figure. Overnight SFT/harness work records another **$20 cap** against cumulative metered **$44.94 at 03:13**, with no closing meter or completed overnight spend. The attached SmolVLA v2 failure lost approximately **$0.5**; this is an incident estimate, not its total training charge. No per-policy charges or allocation of intervening spending are inferred. [Budget ledger][benchmark]; [latest meter][latest]; [overnight baseline][night].

The benchmark summary calls v2b training “~18 min L4”, while its detailed training entry says **1075 s on A100-40GB**. Both are preserved; dollar attribution and hardware should be reconciled from the primary log before a final report claims a precise cost comparison. The summary also abbreviates dev seeds to 1000000–1000029, whereas balanced traces include 1000030 and 1000033. Seed-start and sampler identity are the reliable descriptions available here. [v2 detail][v2training]; [summary][benchmarksummary]; [seed correction][releaseprobe].

## 4. What worked / what didn't

### What worked, with limits

- **Demonstration audit and reduced action space.** The 59/60 own-seed replay established useful training behavior, and action statistics identified the 10 varying dimensions. The 1/8 new-seed replay exposed feedback as essential. Replay itself remains ineligible as a solution. [Measurements][phase0].
- **Teacher v1 as a data generator.** Its 27/30 initial run supported success-filtered collection, eventually including 191 new same-renderer demonstrations. Its later 46/58 and 191/240 yields give a more cautious view of robustness than the initial 90%. [Early teacher][phase1]; [new data][latest].
- **Failure replay and objective diagnosis.** Saved-action replay repeatedly reproduced 30/30 outcomes, revealing placed-but-holding states and showing that L1 failed even on open-labelled training frames. BCE changed a zero-success student into 14/30 and later 57/90 under repeated Modal evaluation. [BCE evidence][phase1]; [repeated results][strictresults].
- **Verification before relabelling.** Original oracle reproduction and unchanged non-gripper arrays made the strict-label ablation auditable. Its 44/90 versus loose-label 24/90 recovered performance in one recipe, while still falling short of bce1. [Relabel checks][relabel]; [results][strictresults].
- **Repeated runs and fresh-seed confirmation.** These exposed a dev-selected threshold gain that did not transfer and sizeable variation for the same policy. They improve interpretation, although repeats do not replace independent seeds. [Confirmation][confirmation]; [repeatability][nondeterminism].
- **Cloud hardware measurement.** Actual throughput favored A100 training; NVIDIA EGL on T4 made collection practical and matched the laptop's NVIDIA rendering in the checked observation. CPU image caching and conversion were separated from GPU training. [GPU benchmark][gpu]; [collector][dagger3].
- **SmolVLA under the tested SFT budget.** Its 44/90 matched microVLA v2b with roughly 30 minutes of expert-only SFT. It remains below bce1; this is not evidence of meeting the target. [Benchmark][benchmarksummary].

- **Harness label alignment and offline validation.** Replay reproduced every pre-action state across 120/120 DAgger3 episodes, and 10/10 harness tests passed. At estimator step 7000, release-region distance/tilt MAE was 0.68 cm/1.37° on 3,354 held-out rows. These checks support running ablations; they do not yet show an improvement in task success. [Harness evidence][night].

### What did not work, and the response

- **More teacher data alone and low offline loss.** v1a/v1b remained 0/30 despite better intermediate behavior. Larger microVLA lowered validation loss to 0.0161 but fell to 30/90. gen2 SFT reached 47/90 and 54/90, below bce1; first-run replay suggested fewer grasp misses but more never-release cases. Response: select through closed-loop success and replay, and test pose-gated release. The scaling cause and teacher-wait explanation remain unproven. [Students][phase1]; [large run][benchmark]; [new SFT][night].
- **Bundled teacher changes and base freezing.** v2 fell from 27/30 to 6/30; align-then-lower stopped at 0/2. Response: independent flags and preserving the v1 path. Compensation for arm-induced lateral drift was suggested but is untested. [Teacher trials][phase1]; [brief #6][brief6].
- **DART at scale 1.0.** Yield fell to 3/23 versus 34/41 for DART 0 in different seed blocks. Response: stop strong noise; smaller carry-only perturbations remain a proposal. [Yield check][phase1].
- **Gripper L1 and early DAgger.** Open-label training accuracy remained zero, and v1a+DAgger1 stayed 0/30. Response: BCE with a separately normalized gripper loss. [Probe and fix][phase1].
- **Dev threshold selection and global release thresholds.** Threshold 0.4's 20/30 dev score became 17/30 on confirmation, versus 18/30 at default 0.5. Later bce4b thresholds 0.3/0.2 reached 52/90 and 43/90, versus default 54/90; bce1 at 0.3 reached 52/90 versus default 57/90. First-run replay showed fewer never-release cases offset by more drops. Response: retain the reference thresholds and test release conditioned on estimated pose. [Confirmation][confirmation]; [later sweeps][night].
- **Loose DAgger3 release labels.** Extra own-policy data hurt BC; strict labels repaired part of the damage without beating bce1. Response: mask borderline states and retain the unchanged-label control. The arm remains uncorrected by gripper-only labels. [Diagnosis][bce2result]; [strict results][strictresults].
- **Higher image resolution, more parameters and tested Muon settings.** 128 px scored 14/90, large microVLA 30/90, and Muon lost the short held-out-loss comparison. Response: retain 96 px and choose AdamW for the large final run. No broad optimizer or architecture ceiling follows. [Resolution][strictresults]; [benchmark][benchmark].
- **RAM, disk and thermal overload.** Full-demo loads per worker exhausted memory; whole-mixture caches filled disk; simulation and post-reboot training overheated the laptop. Responses were one-time low-dimensional extraction, fewer workers, removal of obsolete/partial caches, a per-episode-cache implementation request, and restarting/lowering watchdog thresholds. Watchdog auto-start after reboot is not recorded as fixed. [Initial failures][phase0]; [cache failure][phase1]; [reboot][microhistory].
- **Cloud collection misdiagnosis.** Zero reported CPU load was mistaken for a lock hang. Moving output to local disk did not address the real rendering bottleneck. Process CPU inspection and T4 NVIDIA EGL resolved collection speed; the harmless local-output copy pattern was retained. [Ops sequence][dagger3].
- **Bad launches and dependency mismatches.** SigLIP image building failed through an unnecessary processor dependency; selecting only the vision model fixed it. bce2 rejected four loader workers and relaunched with two. v2 JSON shell quoting and a missing prebuilt cache caused failed launches; quoting was repaired and cache built on CPU. LeRobot conversion needed its dataset extra, the removed W&B flag was dropped, and pi0.5 needed its saved preprocessor redirected to provisioned tokenizer files. [Modal setup][gpu]; [bce2 launch][bce2]; [v2 launch][v2training]; [LeRobot][benchmark].
- **MolmoAct2 memory and misleading gradient logs.** Several configurations OOMed on L4/A100; the completed recipe used LoRA, checkpointing, bf16 and batch eight on A100. Reading LeRobot's logging/clipping path corrected the false zero-gradient diagnosis. [Benchmark corrections][benchmark].
- **Idle pipeline and duplicate jobs.** Completed SmolVLA/pi0.5 jobs waited about 2.5 hours for evaluation to be queued; the log records no GPU charge for that gap. A detached-launch misunderstanding then produced duplicate live evaluation apps, and stopping them lost both evaluations, costing approximately $0.4 for about 15 minutes on two L4s. Response: completion watchers/chained evaluations and one attached relaunch. [Ops failure][benchmark].
- **Cold-load timeouts and budget overrun.** pi0.5's first evaluation produced no actions within 120 s. Timeout 600 made the experiment valid, while default-timeout readiness remains unresolved. The MolmoAct2 evaluation map outlasted estimates, and the benchmark exceeded its cap by $0.99; better completion watching does not by itself enforce a spending stop. [Timeout/spend][benchmark].
- **Zero-shot wrapper spawn failure and preemption.** DataLoader workers reran the wrapper's main function; a `__main__` guard fixed it, at approximately $0.3 for the failed attempt. Modal automatically retried a preempted L4 input. Both bases completed at 0/90, all timeouts; deployment plumbing alone supplied no task success under this mapping. [Wrapper][latest]; [final runs][night].
- **Broken checkpoint paths and transfers.** A `/ckpts/../runs/<run>/last.pt` shortcut did not exist in the container, yielding 30 policy errors in 52 s; a separate download was truncated, and one launch hit `RetryTimeoutError`. Responses: check `torch.load` before upload, re-download, verify hashes/sizes, upload into ckpts, and verify each replacement report's checkpoint field. The failed launch is kept separate from corrected bce4 results. [Repairs][night].
- **Full launches before a useful smoke.** Thirty short episodes still spent over 15 minutes resetting on T4. A dedicated smoke now uses the evaluator's `PolicyProcess`, its 120 s first-exchange timeout, one reset at dev seed 1000000 and 20 validated real-environment action calls. bce1 passed with load 14.7 s and 0.015 s/call; a missing checkpoint failed with its traceback. This approximately one-minute check validates startup, not task success. [Smoke procedure][night].
- **Client-dependent jobs and outage.** The attached SmolVLA run died after a client `ConnectionError: Deadline exceeded`; a detached run plus watcher initially replaced it. During the approximately 04:05–10:25 power cut, watchers and detached jobs also died: estimator training stopped at 8000/10000 and SmolVLA at 11186/15000, followed by roughly six idle hours. Response: deploy the sim/harness/LeRobot apps and use server-side `Function.spawn` through `ops/spawn.py`; resume from saved checkpoints. The intended independence from the laptop has not yet been demonstrated through another outage. [Initial disconnect][latest]; [outage and deployment][night].
- **Worker-limit and watchdog checks.** The estimator repeated the unsupported four-worker launch error and relaunched with two. After the outage, `pgrep -f watchdog.sh` self-matched and falsely indicated protection; checking the exact command line established the need to restart the watchdog. These repairs address the recorded failures; reboot auto-start is still unverified. [Recovery][night].

## 5. Open questions & next steps

These are proposed follow-ups unless explicitly described as already launched. They are not additional experiment results.

1. **Close the launched harness ablations.** Collect all h_pass, h_full, h_exec10, h_bce4b and h_ens reports, errors, exact seeds, per-slot counts and repeat spread. Compare the execute-2 passthrough with bce1 before attributing changes to supervision; compare exec10 and ensemble controls separately. Inspect true pose at release through offline replay, including premature release and disturbance during settling. Smoke passes and offline estimator MAEs are not success claims. [Launched ablations][night].
2. **Finish and evaluate SmolVLA v2.** Record the resumed endpoint, checkpoint identity, total training/cost history and three dev repeats. Compare against the 5k reference, and use intermediate checkpoints to inspect the step curve while disclosing the changed dataset and normalization. Neither the unfinished run nor the planned 20k endpoint has a recorded success rate. [SFT setup][latest]; [resume][night].
3. **Test the new-data mechanism.** gen2 did not improve overall bce4 success. Check whether teacher release-wait frames actually shift open probabilities, using more than one replay run; distinguish that hypothesis from different mixture, continuation and step choices. Preserve the negative global-threshold results. Separating data quantity from Mesa/NVIDIA rendering and training duration still requires controlled comparisons. [SFT and sweep][night]; [rendering question][dagger3].
4. **Validate estimator decisions and recovery.** The evaluated estimator is the saved step-8000 checkpoint, while published offline metrics are from step 7000. Reconcile that endpoint, inspect release-region tails and held misclassifications, and run the offline timing comparison before asserting safer release. Grasp retries and temporal ensembling need their own measured outcomes. [Estimator and harness][night]; [requested checks][brief13].
5. **Resolve repeatability and seed reporting.** Investigate variation with CPU inference, retain per-run/per-slot counts and exact balanced seed lists, and use additional dev seeds when possible. Existing pooled tests reuse seeds and mix some conditions; repeated-run Wilson intervals alone do not describe all uncertainty. Keep confirm/test out of training and further tuning. [Repeatability][nondeterminism]; [repeated runs][strictresults]; [later spread][night].
6. **Keep benchmark claims and costs proportional to evidence.** The zero-shot floors are now measured, but precise final zero-shot/overnight spending is missing. A stronger pi0.5/MolmoAct2 adaptation comparison needs an explicit budget and sample controls. SmolVLA's initial result, Muon's proxy-loss comparison and the large-model regression do not settle model ceilings. Reconcile v2b hardware and dataset inventories before precise efficiency claims. [Benchmark][benchmarksummary]; [latest accounting][night].
7. **Complete operational and submission readiness.** Verify server-side spawning survives client loss, restore/check watchdog protection after every reboot, retain checkpoint validation and policy smoke before launches, and enforce spending stops including waiting containers. Validate cold loading under default submission constraints. After dev selection, freeze a candidate and run the reserved final evaluation under the assignment, reporting every requested trial and the earlier smoke use of that block. No student final-test pass is recorded. [Recovery][night]; [rules][readme].

### Evidence and maintenance notes

The master experiment log is authoritative for run outcomes, numbers, costs and corrections. Earlier brief claims such as “gripper never closes” are superseded by its later full-run diagnoses. Conflicting slot numbering, seed-range shorthand, patch wording and v2b hardware are disclosed above rather than silently repaired. Counts from partial collections, repeated seeds, privileged teachers, smoke checks and invalid policy-error runs retain their original scope.

For later updates, start at the incorporated line count in `.log_state.json`, read only newly added experiment-log content, and integrate it into the relevant narrative, policy table, failure analysis and next steps. Pending rows should be replaced with sourced outcomes when available. Advance the saved line count and last heading only after the matching narrative is written. The recorded prefix hash can detect an upstream edit to already incorporated text; a mismatch needs explicit reconciliation rather than silently treating a changed prefix as unseen appended material.

[experiments]: ../robot-learning-assignment/EXPERIMENTS.md
[readme]: ../robot-learning-assignment/README.md
[methods]: ../robot-learning-assignment/METHODS.md
[plan]: ../robot-learning-assignment/PLAN.md
[phase0]: ../robot-learning-assignment/EXPERIMENTS.md#phase-0-understand-and-measure-wed-18-nov
[phase1]: ../robot-learning-assignment/EXPERIMENTS.md#phase-1-baseline-and-teacher-wed-18-nov
[confirmation]: ../robot-learning-assignment/EXPERIMENTS.md#threshold-04-confirmation-on-fresh-seeds-block-2600000-never-used-before
[microhistory]: ../robot-learning-assignment/EXPERIMENTS.md#microvla-track-setup-and-training-history-backfilled-11-21-0740-from-session-transcripts
[bceread]: ../robot-learning-assignment/EXPERIMENTS.md#bce1-failure-re-read-11-21-0740
[gpu]: ../robot-learning-assignment/EXPERIMENTS.md#modal-gpu-benchmark-11-21-0745
[nondeterminism]: ../robot-learning-assignment/EXPERIMENTS.md#finding-evaluation-is-not-run-to-run-deterministic-11-21-0820
[releaseprobe]: ../robot-learning-assignment/EXPERIMENTS.md#bce1-release-diagnosis-11-21-0835
[dagger3]: ../robot-learning-assignment/EXPERIMENTS.md#dagger-round-3-setup-on-modal-11-21-08400915
[bce2]: ../robot-learning-assignment/EXPERIMENTS.md#bce2-bce1-recipe--dagger3-11-21-0945
[bce2result]: ../robot-learning-assignment/EXPERIMENTS.md#bce2-eval-result-dagger3-made-it-worse-11-21-1015
[v2training]: ../robot-learning-assignment/EXPERIMENTS.md#microvla-v2-training-11-21-modal-a100-40gb-code-wt-vla825f841
[relabel]: ../robot-learning-assignment/EXPERIMENTS.md#dagger-relabel-choosing-the-open-rule-from-data-11-21-1050
[strictresults]: ../robot-learning-assignment/EXPERIMENTS.md#results-bce3-strict-labels-and-microvla-v2-11-21-1150
[benchmark]: ../robot-learning-assignment/EXPERIMENTS.md#benchmark-from-scratch-larger-microvla--muon-vs-post-training-pretrained-vlas-11-21-from-1200
[benchmarksummary]: ../robot-learning-assignment/EXPERIMENTS.md#benchmark-summary-dev-seeds-10000001000029-3--30-each
[latest]: ../robot-learning-assignment/EXPERIMENTS.md#11-22-zero-shot-base-vlas-more-teacher-data
[brief1]: ../briefs/01_bc_baseline.md
[brief2]: ../briefs/02_teacher.md
[brief3]: ../briefs/03_teacher_v2.md
[brief4]: ../briefs/04_teacher_dart_labels.md
[brief5]: ../briefs/05_student_v1.md
[brief6]: ../briefs/06_teacher_ablation_flags.md
[brief7]: ../briefs/07_dagger_gripper.md
[brief8]: ../briefs/08_gripper_bce.md
[brief9]: ../briefs/09_shared_cache.md
[brief10]: ../briefs/10_microvla.md
[brief11]: ../briefs/11_muon.md
[muonreport]: ../wt-muon/method/vla/11_muon.md
[brief12]: ../briefs/12_teacher_gen_modal.md
[report12]: ../briefs/12_teacher_gen_modal.report.md
[brief13]: ../briefs/13_harness.md

[night]: ../robot-learning-assignment/EXPERIMENTS.md#11-22-night-sft-results-failure-analysis-harness-user-cap-20-overnight-baseline-metered-4494-at-0313
[report13]: ../briefs/13_harness.report.md
