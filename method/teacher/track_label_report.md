# Brief 16: measured tracking-label results

Run date: 2026-11-22. Starting branch `teacher`, HEAD `2cc062224c64510496cc73378d3d9103ae26e4bc`.

Design, limitations, and exact run/check commands are in [track_label.md](track_label.md). The evaluator, physics, task success rules, Teacher implementation, lockfile, and source data were not changed. These are offline labels, with no generated-label rollout or training result.

## Three teacher episodes

All 2,258 stored pre-action states passed byte-for-byte replay. Absolute action differences below are in normalized [-1,1] units. Arm summary covers dimensions 4:18 (both arms).

| Seed | Demo | Steps | Arm mean abs | Arm max abs | Worst step / dim | Dropped at cost 4 |
|---|---:|---:|---:|---:|---|---:|
| 2200000 | 30 | 708 | 0.00194472 | 0.03149915 | 44 / 9 | 0.00% |
| 2200001 | 38 | 877 | 0.00196044 | 0.05509174 | 329 / 8 | 0.00% |
| 2200003 | 24 | 673 | 0.00236726 | 0.06084526 | 286 / 8 | 0.00% |

The arm labels are close, but not identical. For each episode, its worst difference occurs where the source executed demo action `t*` and the requested label uses `t*+1`: seed 2200000, step 44, dim 9: 0.14987528 -> 0.18137443; seed 2200001, step 329, dim 8: 0.15890813 -> 0.10381639; seed 2200003, step 286, dim 8: 0.17376328 -> 0.11291802. These comparisons were checked directly against the source executed actions and selected demo actions. Larger base and gripper differences are expected because tracking disables waits and applies the strict opening rule. The test does not promise exact teacher reproduction.

Each cell below is **mean / max @ worst physical step**. All 20 dimensions are shown; arm targets are dimensions 4 through 17.

| Dim / name | 2200000 | 2200001 | 2200003 |
|---|---|---|---|
| 0 base_x | 0.0000050 / 0.0003263 @ 541 | 0.0015888 / 0.0443736 @ 838 | 0.0000918 / 0.0159620 @ 671 |
| 1 base_y | 0.0000071 / 0.0005949 @ 541 | 0.0021101 / 0.0652081 @ 867 | 0.0000923 / 0.0219264 @ 671 |
| 2 base_height | 0.0000000 / 0.0000000 @ 0 | 0.0000000 / 0.0000000 @ 0 | 0.0000000 / 0.0000000 @ 0 |
| 3 base_yaw | 0.0000000 / 0.0000000 @ 0 | 0.0000000 / 0.0000000 @ 0 | 0.0000000 / 0.0000000 @ 0 |
| 4 left_shoulder_pitch_joint | 0.0015453 / 0.0059421 @ 135 | 0.0013645 / 0.0104381 @ 329 | 0.0016633 / 0.0129192 @ 288 |
| 5 left_shoulder_roll_joint | 0.0007743 / 0.0060259 @ 367 | 0.0014887 / 0.0087310 @ 491 | 0.0014805 / 0.0068541 @ 191 |
| 6 left_shoulder_yaw_joint | 0.0011741 / 0.0116596 @ 74 | 0.0015935 / 0.0393633 @ 329 | 0.0018702 / 0.0458366 @ 286 |
| 7 left_elbow_joint | 0.0048861 / 0.0259023 @ 44 | 0.0053002 / 0.0443885 @ 329 | 0.0069900 / 0.0558082 @ 288 |
| 8 left_wrist_roll_joint | 0.0029206 / 0.0189483 @ 72 | 0.0022818 / 0.0550917 @ 329 | 0.0028828 / 0.0608453 @ 286 |
| 9 left_wrist_pitch_joint | 0.0039914 / 0.0314991 @ 44 | 0.0045020 / 0.0265740 @ 329 | 0.0047768 / 0.0350272 @ 288 |
| 10 left_wrist_yaw_joint | 0.0024811 / 0.0184655 @ 74 | 0.0023857 / 0.0209578 @ 154 | 0.0029477 / 0.0206326 @ 136 |
| 11 right_shoulder_pitch_joint | 0.0005382 / 0.0038999 @ 291 | 0.0005703 / 0.0045325 @ 89 | 0.0006470 / 0.0038494 @ 92 |
| 12 right_shoulder_roll_joint | 0.0005275 / 0.0052732 @ 49 | 0.0004858 / 0.0044860 @ 36 | 0.0006325 / 0.0054846 @ 34 |
| 13 right_shoulder_yaw_joint | 0.0006694 / 0.0073493 @ 347 | 0.0006523 / 0.0066146 @ 77 | 0.0007378 / 0.0085132 @ 79 |
| 14 right_elbow_joint | 0.0024479 / 0.0251201 @ 29 | 0.0022270 / 0.0225897 @ 34 | 0.0026867 / 0.0231733 @ 29 |
| 15 right_wrist_roll_joint | 0.0014664 / 0.0097511 @ 85 | 0.0012229 / 0.0095044 @ 463 | 0.0015914 / 0.0101041 @ 407 |
| 16 right_wrist_pitch_joint | 0.0027305 / 0.0182898 @ 373 | 0.0023325 / 0.0167770 @ 463 | 0.0030175 / 0.0186836 @ 407 |
| 17 right_wrist_yaw_joint | 0.0010731 / 0.0113786 @ 347 | 0.0010389 / 0.0101458 @ 77 | 0.0012176 / 0.0131788 @ 79 |
| 18 left_gripper | 0.3898305 / 2.0000000 @ 204 | 0.2941847 / 2.0000000 @ 297 | 0.1337296 / 2.0000000 @ 261 |
| 19 right_gripper | 0.0000000 / 0.0000000 @ 0 | 0.0000000 / 0.0000000 @ 0 | 0.0000000 / 0.0000000 @ 0 |

## Three bce1 / dagger3 episodes

All 4,014 saved states passed byte-for-byte replay. At cost threshold 4, 367/4,014 steps (9.1430%) are masked across all dimensions; no rows are deleted.

| Seed | Demo | Steps | Final phase / last demo phase | Stalled transitions | Arm mean abs | Arm max abs | Masked steps |
|---|---:|---:|---|---:|---:|---:|---:|
| 2500000 | 27 | 1700 | 286 / 607 | 1589 | 0.06491004 | 0.56111515 | 42 (2.4706%) |
| 2500001 | 31 | 614 | 55 / 650 | 581 | 0.10768444 | 1.25347114 | 245 (39.9023%) |
| 2500002 | 52 | 1700 | 473 / 576 | 1531 | 0.04963005 | 0.66024244 | 80 (4.7059%) |

The largest arm deviations are all in dim 7, left elbow: seed 2500000 at physical step 308 / phase 207 (0.56111515, cost 6.82865), seed 2500001 at 219 / 55 (1.25347114, cost 22.15999), and seed 2500002 at 249 / 208 (0.66024244, cost 8.40894). All three worst steps are masked. Differences are against the teacher transfer chosen at reset, not evidence that the generated correction would solve the task.

Phase behavior:

- 2500000 advances to about phase 200 by physical step 300, then reaches 286 around step 550 and stays there to step 1699.
- 2500001 reaches phase 55 by physical step 50 and remains there to step 613. This is a clear forward-window lock. Cost filtering masks 39.90%, but the remaining low-cost states do not prove that this early phase remains appropriate.
- 2500002 reaches phase 208 by physical step 250, phase 404 around step 650, then advances again around step 1050 and reaches 473 by step 1400.

Exact sampled trajectories (`physical_step:phase`, every 50 steps, plus final):

Seed 2500000:

```text
0:0 50:55 100:102 150:132 200:158 250:196 300:200 350:207 400:242 450:251 500:285 550:286 600:286 650:286 700:286 750:286 800:286 850:286 900:286 950:286 1000:286 1050:286 1100:286 1150:286 1200:286 1250:286 1300:286 1350:286 1400:286 1450:286 1500:286 1550:286 1600:286 1650:286 1699:286
```

Seed 2500001:

```text
0:0 50:55 100:55 150:55 200:55 250:55 300:55 350:55 400:55 450:55 500:55 550:55 600:55 613:55
```

Seed 2500002:

```text
0:0 50:98 100:122 150:158 200:203 250:208 300:209 350:226 400:260 450:373 500:398 550:400 600:403 650:404 700:404 750:404 800:404 850:404 900:404 950:404 1000:406 1050:453 1100:454 1150:466 1200:466 1250:470 1300:472 1350:472 1400:473 1450:473 1500:473 1550:473 1600:473 1650:473 1699:473
```

All steps, rather than only the sampled trajectory, are stored in `track_phase` and the JSON `phase` report field.

Per-dimension differences, including masked steps:

| Dim / name | 2500000 | 2500001 | 2500002 |
|---|---|---|---|
| 0 base_x | 0.0184754 / 0.1371703 @ 63 | 0.1604137 / 0.2106031 @ 46 | 0.0575310 / 0.2154175 @ 409 |
| 1 base_y | 0.2683375 / 0.6065442 @ 459 | 0.3246675 / 0.8632609 @ 399 | 0.0548060 / 0.6621823 @ 436 |
| 2 base_height | 0.0000000 / 0.0000000 @ 0 | 0.0000000 / 0.0000000 @ 0 | 0.0000000 / 0.0000000 @ 0 |
| 3 base_yaw | 0.0000000 / 0.0000000 @ 0 | 0.0000000 / 0.0000000 @ 0 | 0.0000000 / 0.0000000 @ 0 |
| 4 left_shoulder_pitch_joint | 0.0365331 / 0.1369677 @ 308 | 0.1896419 / 0.3850531 @ 220 | 0.0545118 / 0.2271419 @ 249 |
| 5 left_shoulder_roll_joint | 0.0320438 / 0.1594687 @ 369 | 0.0385561 / 0.0931255 @ 289 | 0.0232722 / 0.1015741 @ 229 |
| 6 left_shoulder_yaw_joint | 0.1128759 / 0.2317961 @ 309 | 0.1298764 / 0.2766127 @ 220 | 0.0315492 / 0.2367402 @ 280 |
| 7 left_elbow_joint | 0.2060211 / 0.5611151 @ 308 | 0.3930379 / 1.2534711 @ 219 | 0.0796518 / 0.6602424 @ 249 |
| 8 left_wrist_roll_joint | 0.0301966 / 0.2503221 @ 311 | 0.0849205 / 0.3065034 @ 223 | 0.0609381 / 0.4282675 @ 285 |
| 9 left_wrist_pitch_joint | 0.1231731 / 0.2944013 @ 308 | 0.1216022 / 0.4001987 @ 219 | 0.2011546 / 0.3673245 @ 1015 |
| 10 left_wrist_yaw_joint | 0.1802415 / 0.2371880 @ 1199 | 0.1076855 / 0.3171291 @ 199 | 0.0435326 / 0.0836475 @ 969 |
| 11 right_shoulder_pitch_joint | 0.0070573 / 0.0211403 @ 14 | 0.0174198 / 0.0204212 @ 10 | 0.0375248 / 0.0556301 @ 364 |
| 12 right_shoulder_roll_joint | 0.0135025 / 0.1232197 @ 0 | 0.0294656 / 0.1232261 @ 0 | 0.0101705 / 0.1232332 @ 0 |
| 13 right_shoulder_yaw_joint | 0.0085082 / 0.0749812 @ 0 | 0.0170862 / 0.0749989 @ 0 | 0.0162824 / 0.0750081 @ 0 |
| 14 right_elbow_joint | 0.0691284 / 0.2911067 @ 11 | 0.1552493 / 0.2939773 @ 10 | 0.0131424 / 0.2912142 @ 10 |
| 15 right_wrist_roll_joint | 0.0141087 / 0.1119812 @ 0 | 0.0135547 / 0.1120206 @ 0 | 0.0148152 / 0.1120139 @ 0 |
| 16 right_wrist_pitch_joint | 0.0603614 / 0.1238003 @ 2 | 0.1774221 / 0.1865421 @ 47 | 0.0871544 / 0.2041919 @ 430 |
| 17 right_wrist_yaw_joint | 0.0149890 / 0.1187255 @ 0 | 0.0320641 / 0.1187808 @ 0 | 0.0211208 / 0.1187732 @ 0 |
| 18 left_gripper | 1.2517647 / 2.0000000 @ 239 | 0.2084691 / 2.0000000 @ 215 | 1.4576471 / 2.0000000 @ 191 |
| 19 right_gripper | 0.0000000 / 0.0000000 @ 0 | 0.0000000 / 0.0000000 @ 0 | 0.0000000 / 0.0000000 @ 0 |

## Full dataset

All **120 episodes / 102,230 saved pre-action states** passed deterministic,
camera-free, byte-for-byte replay. The two shards each processed 60 episodes.
Both recorded labeler SHA256s match the delivered implementation.

With `--drop-cost` (threshold 4), **16,483 / 102,230 steps = 16.1234%** are
masked across all 20 dimensions; **85,747 steps remain labeled in every
dimension**. No rows or RGB frames are deleted. The first-three sample rate
(9.1430%) understates the full-dataset rate. Cost filtering remains optional;
the unfiltered CLI default labels every dimension at every step.

Full-dataset arm mean absolute label-vs-executed difference is 0.06568168;
maximum is 1.61537707 (normalized units, before masking). The strict gripper
oracle provides 5,813 open and 30,633 closed labels; 65,784 states use the
demo gripper fallback. These are label statistics, not rollout performance.

Staged output: `results/dagger3_track` (120 NPZ episodes, shard configs and
JSON reports). `summary.json` and `verification.json` contain combined totals
and the independent archive audit. Source action execution and original
success metadata remain unchanged.

## Validation and delivery

- 48 teacher unit tests passed, including 19 tracking tests: monotonicity, inclusive/zero/clipped windows, tie breaking, gripper phase cost, config/seed guards, strict gripper boundaries, exact signed-zero comparison, original-action-only replay, whole-row filtering, end clamping, original base feedback reuse, and exclusive publication.
- Independent saved-file audit passed for all 120 episodes: 3,277,108,396 bytes of preserved `.npy` content (including RGB) compared byte for byte; every phase is monotone and within the configured window; every arm target matches the selected demo at `t*+1`; every 20-dimensional mask agrees with the cost filter. SHA256s of preserved members are recorded in `results/dagger3_track/verification.json`. Exact audit command: `ADV_THROTTLE=1 PYTHONPATH=. uv run --active --no-sync python results/track_label/verify_outputs.py`, with the environment in `track_label.md`.
- Ruff check, format check, and `git diff --check` passed for the changes.
- Git staging/commit is blocked by read-only metadata: `index.lock` cannot be created under `/home/keyu/advaitic/robot-learning-assignment/.git/worktrees/wt-teacher`.
- The requested `/home/keyu/advaitic/data/dagger3_track` destination is outside this session's writable roots. Generated data is staged in `results/dagger3_track` inside the worktree. No external destination was written.
- Machine-readable reports: `results/track_label/teacher_sanity.json`, `results/track_label/student_sanity.json`, and `results/dagger3_track/track_report.shard{0,1}.json`. Source arrays, including success metadata, are preserved; success is about the source rollout only.
