# Experiments

## Modular controller

### 2026-11-22 — Sol privileged development

- `sol_teacher_00` (HEAD `9a19369872f80b7683707ef9e9d769844a728f96` with local changes, seeds 3000000–3000029) — 25/30 official success, 30/30 held and lifted ≥5 cm.
  Command `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl uv run --locked --offline python -m method.modular.run_privileged --policy teacher --seed-start 3000000 --episodes 30 --workers 6 --out /home/keyu/advaitic/plans/modular_controller/runs/sol_teacher_00.jsonl`
- `sol_teacher_30` (HEAD `9a19369872f80b7683707ef9e9d769844a728f96` with local changes, seeds 3000030–3000059) — 26/30 official success, 30/30 held and lifted ≥5 cm. Combined paired baseline 51/60.
  Command `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl uv run --locked --offline python -m method.modular.run_privileged --policy teacher --seed-start 3000030 --episodes 30 --workers 3 --out /home/keyu/advaitic/plans/modular_controller/runs/sol_teacher_30.jsonl`
- `sol_freeze_00` (HEAD `9a19369872f80b7683707ef9e9d769844a728f96` with local changes, seeds 3000000–3000059) — 51/60 official success. The reset aperture reference was invalid and the retreat path never activated. Superseded by the corrected run.
  Command `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl uv run --locked --offline python -m method.modular.run_privileged --policy teacher-freeze --seed-start 3000000 --episodes 60 --workers 6 --out /home/keyu/advaitic/plans/modular_controller/runs/sol_freeze_00.jsonl`
- `sol_freeze_01` (HEAD `9a19369872f80b7683707ef9e9d769844a728f96` with local changes, seeds 3000000–3000059) — 51/60 official success with corrected aperture threshold. This version also required absence of held contact to count open steps.
  Command `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl uv run --locked --offline python -m method.modular.run_privileged --policy teacher-freeze --seed-start 3000000 --episodes 60 --workers 6 --out /home/keyu/advaitic/plans/modular_controller/runs/sol_freeze_01.jsonl`
- `sol_mod_smoke_a` (HEAD `9a19369872f80b7683707ef9e9d769844a728f96` with local changes, seeds 3000000–3000002) — 0/3 official success, 3/3 held and lifted ≥5 cm. Lift-phase exit was overly strict.
  Command `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl uv run --locked --offline python -m method.modular.run_privileged --policy modular --seed-start 3000000 --episodes 3 --workers 3 --out /home/keyu/advaitic/plans/modular_controller/runs/sol_mod_smoke_a.jsonl`
- `sol_mod_smoke_b` (HEAD `9a19369872f80b7683707ef9e9d769844a728f96` with local changes, seeds 3000000–3000005) — 0/6 official success, 6/6 held and lifted ≥5 cm. All reached ALIGN.
  Command `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl uv run --locked --offline python -m method.modular.run_privileged --policy modular --seed-start 3000000 --episodes 6 --workers 3 --out /home/keyu/advaitic/plans/modular_controller/runs/sol_mod_smoke_b.jsonl`
- `sol_mod_smoke_c` (HEAD `9a19369872f80b7683707ef9e9d769844a728f96` with local changes, seeds 3000000–3000005) — 3/6 official success, 6/6 held and lifted ≥5 cm with bounded placement correction.
  Command `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl uv run --locked --offline python -m method.modular.run_privileged --policy modular --seed-start 3000000 --episodes 6 --workers 3 --out /home/keyu/advaitic/plans/modular_controller/runs/sol_mod_smoke_c.jsonl`
- `sol_mod_tune_d` (HEAD `dc760fb9f3a26fa4436552764f94ddb93c063d04` with local changes, seeds 3000000–3000011) — 7/12 official success, 12/12 held and lifted ≥5 cm with more carry clearance and a seated release gate.
  Command `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl uv run --locked --offline python -m method.modular.run_privileged --policy modular --seed-start 3000000 --episodes 12 --workers 6 --out /home/keyu/advaitic/plans/modular_controller/runs/sol_mod_tune_d.jsonl`
- `sol_mod_tune_e` (HEAD `dc760fb9f3a26fa4436552764f94ddb93c063d04` with local changes, seeds 3000000–3000011) — 9/12 official success, 12/12 held and lifted ≥5 cm. Physically open gripper timing enabled guarded withdrawal from incidental pad contact.
  Command `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl uv run --locked --offline python -m method.modular.run_privileged --policy modular --seed-start 3000000 --episodes 12 --workers 6 --out /home/keyu/advaitic/plans/modular_controller/runs/sol_mod_tune_e.jsonl`
- `sol_freeze_02` (HEAD `dc760fb9f3a26fa4436552764f94ddb93c063d04` with local changes, seeds 3000000–3000059) — 51/60 official success. Pair rescued 2 of 4 teacher knocked-after-release failures and lost 2 baseline successes. Aperture-defined 75-step hold was reached in 9 episodes. The other 51 ended on the official success latch during the freeze.
  Command `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl uv run --locked --offline python -m method.modular.run_privileged --policy teacher-freeze --seed-start 3000000 --episodes 60 --workers 6 --out /home/keyu/advaitic/plans/modular_controller/runs/sol_freeze_02.jsonl`
- Calibration probe (HEAD `9a19369872f80b7683707ef9e9d769844a728f96`, teacher `961a6d3d015d97e68a62c13d8a2e184bd2a0a56f`, seeds 3000000–3000002) — pooled a single left-hand grasp transform in the plate frame.
  Command `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl uv run --locked --offline python -` with the calibration probe in the Sol report.
- Snapshot cross-check (HEAD `9a19369872f80b7683707ef9e9d769844a728f96`, seeds 3000000–3000005) — 18 of 18 samples matched teacher diagnostics and estimator labels.
  Command `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl uv run --locked --offline python -` with the adapter cross-check in the Sol report.
- Alignment trace (HEAD `9a19369872f80b7683707ef9e9d769844a728f96`, seed 3000000) — before compensation, the loaded plate plateaued at z ≈ 0.893 m for the 0.975 m target.
  Command `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl uv run --locked --offline python -` with the phase trace in the Sol report.
- Source-contact trace (HEAD `dc760fb9f3a26fa4436552764f94ddb93c063d04` with local changes, seeds 3000000, 3000014, 3000015, 3000021) — on failing seeds the hand touched and moved the source plate before closing.
  Command `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl uv run --locked --offline python -` with the phase trace in the Sol report.
### Astra · M0 and M1 · 2026-11-22

All episode runs used camera-free CPU simulation. For the table, `K` expands to `ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 uv run --locked python -m method.modular.ik_reach_eval`, and `S` expands to `modal run ops_modular/astra_smoke.py::smoke --seed 3000100`. Counts default to one. Position and angular entries for M1 are maxima over the final ten control samples. Local runs used zero Modal containers. The [Astra report](../plans/modular_controller/runs/astra_report.md) includes the inline probes and raw output locations.

| Date | Git HEAD at run | Command | Seeds | Result |
|---|---|---|---|---|
| 2026-11-22 | `9a19369` | `uv run --locked python -`, reset/layout probe | 3000000 | 50 state values, 20 actions, both hand sites 148 mm along wrist x. |
| 2026-11-22 | `9a19369` | `K reach --start 3000100 --trace`, upright torso | 3000100 | 15.842 mm, 2.263°, zero falls. |
| 2026-11-22 | `9a19369` | `K fk --start 3000100`, 0.045 rad pitch | 3000100 | Random walking endpoint, left/right 15.908/17.172 mm, zero falls. |
| 2026-11-22 | `9a19369` | `K reach --start 3000100`, 0.045 rad pitch | 3000100 | 2.642 mm, 0.524°, zero falls. |
| 2026-11-22 | `9a19369` | `S`, first build | 3000100 requested | `modular-astra-reach`, build rejected lock refresh, zero episode containers, 40.7 s client wall. |
| 2026-11-22 | `9a19369` | `S`, frozen lockfile | 3000100 | `modular-astra-reach`, one CPU container, 4.45 s simulation job, 45.0 s client wall. 2.642 mm, 0.524°, zero falls. |
| 2026-11-22 | `9a19369` | `K fk --start 3000000 --count 20` | 3000000–3000019 | Both hands below 2 mm on 0/20. Worst 54.517 mm, torso-frame control 7.20 nm, zero falls, 27.4 s. |
| 2026-11-22 | `9a19369` | `K reach --start 3000000 --count 15` | 3000000–3000014 | Initial approach 13/15, zero falls, 42.2 s. |
| 2026-11-22 | `9a19369` | `K reach --start 3000015 --count 15` | 3000015–3000029 | Initial approach 15/15, zero falls, 41.0 s. |
| 2026-11-22 | `9a19369` | `K reach --start 3000030 --count 15` | 3000030–3000044 | Initial approach 13/15, zero falls, 33.2 s. |
| 2026-11-22 | `9a19369` | `K reach --start 3000045 --count 15` | 3000045–3000059 | Initial approach 14/15, zero falls, 31.6 s. Combined 55/60. |
| 2026-11-22 | `9a19369` | `K reach --start 3000009 --trace` | 3000009 | Failure reproduced, 192.899 mm and 10.874°, zero falls. |
| 2026-11-22 | `9a19369` | `K reach --start 3000040 --trace` | 3000040 | Failure reproduced, 178.357 mm and 26.682°, zero falls. |
| 2026-11-22 | `dc760fb` | `uv run --locked python -`, plate/IK diagnostic | 3000009 | Plate x moved from 0.524 m at step 50 to 0.760 m at step 90. IK residual grew as the plate left reach. |
| 2026-11-22 | `dc760fb` | `K reach --start 3000009` | 3000009 | Retracted approach 2.231 mm, 0.332°, zero falls. |
| 2026-11-22 | `dc760fb` | `K reach --start 3000040` | 3000040 | Retracted approach 2.270 mm, 0.315°, zero falls. |
| 2026-11-22 | `dc760fb` | `K reach --start 3000100` | 3000100 | Retracted approach 2.735 mm, 0.725°, zero falls. |
| 2026-11-22 | `dc760fb` | `S`, retracted approach | 3000100 | `modular-astra-reach`, one CPU container, 5.03 s simulation job, 37.0 s client wall, 2.735 mm, 0.725°, zero falls. |
| 2026-11-22 | `dc760fb` | `K reach --start 3000000 --count 15` | 3000000–3000014 | Retracted approach 15/15, zero falls, 40.2 s. |
| 2026-11-22 | `dc760fb` | `K reach --start 3000015 --count 15` | 3000015–3000029 | Retracted approach 15/15, zero falls, 63.2 s. |
| 2026-11-22 | `dc760fb` | `K reach --start 3000030 --count 15` | 3000030–3000044 | Retracted approach 15/15, zero falls, 36.8 s. |
| 2026-11-22 | `dc760fb` | `K reach --start 3000045 --count 15` | 3000045–3000059 | Retracted approach 15/15, zero falls, 58.6 s. Combined 60/60. Worst 2.669 mm, 0.678°. |
| 2026-11-22 | `dc760fb` | `K fk --trace --start 3000200 --count 20` | 3000200–3000219 | 1,051 privileged torso-label samples, zero falls, 57.2 s. |
| 2026-11-22 | `dc760fb` | `K fk --trace --start 3000220 --count 20` | 3000220–3000239 | 1,045 torso-label samples, zero falls, 66.2 s. |
| 2026-11-22 | `dc760fb` | `K fk --trace --start 3000000 --count 20` | 3000000–3000019 | 1,052 samples, 20 endpoint cross-checks, zero falls, 52.0 s. |
| 2026-11-22 | `dc760fb` | `uv run --locked python -`, torso regression in report | Fit 3000200–3000239, score 3000000–3000019 | 16.150 mm median, 43.052 mm max, 0/20 below 2 mm for both hands. Regression retained as a diagnostic only. |

The recorded SHAs identify the base of the working-tree candidates. The lead's `dc760fb` checkpoint contains the initial approach. The final evaluation module has Git blob `ac27774d2e16fcacd1662c862c3b7b2993d10f10`.

- 2026-11-22 `sol_mod_f_00` (HEAD `dc760fb9f3a26fa4436552764f94ddb93c063d04` with local controller edits, seeds 3000000–3000059) — 32/60 official success, 49/60 M2, 163.9 s elapsed.
  Command `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl uv run --locked --offline python -m method.modular.run_privileged --policy modular --seed-start 3000000 --episodes 60 --workers 6 --out /home/keyu/advaitic/plans/modular_controller/runs/sol_mod_f_00.jsonl`
- 2026-11-22 `sol_mod_f_60` (HEAD `dc760fb9f3a26fa4436552764f94ddb93c063d04` with local controller edits, seeds 3000060–3000119) — 28/60 official success, 41/60 M2, 225.9 s elapsed. Combined M3 60/120.
  Command `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl uv run --locked --offline python -m method.modular.run_privileged --policy modular --seed-start 3000060 --episodes 60 --workers 4 --out /home/keyu/advaitic/plans/modular_controller/runs/sol_mod_f_60.jsonl`
- 2026-11-22 `sol_mod_g_cross` (HEAD `dc760fb9f3a26fa4436552764f94ddb93c063d04` with local controller edits, seeds 3000012–3000026) — 0/15 official success and M2. Tucked approach and staged reach stalled when the open hand missed its pose gate.
  Command `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl uv run --locked --offline python -m method.modular.run_privileged --policy modular --seed-start 3000012 --episodes 15 --workers 5 --out /home/keyu/advaitic/plans/modular_controller/runs/sol_mod_g_cross.jsonl`

### 2026-11-22 — Sol grasp and placement sweeps

- `sol_mod_h_cross` (HEAD `dc760fb9f3a26fa4436552764f94ddb93c063d04` with local edits, seeds 3000000–3000023) — 0/24 official success, 0/24 M2. Bias alone preserved the poor IK branch. 79.0 s.
  Command `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl uv run --locked --offline python -m method.modular.run_privileged --policy modular --seed-start 3000000 --episodes 24 --workers 6 --out /home/keyu/advaitic/plans/modular_controller/runs/sol_mod_h_cross.jsonl`
- `sol_mod_i_cross` (HEAD `dc760fb9f3a26fa4436552764f94ddb93c063d04` with local edits, seeds 3000012–3000023) — 2/12 official success, 12/12 M2. Fixed positive-elbow initial guess restored grasp reach.
  Command `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl uv run --locked --offline python -m method.modular.run_privileged --policy modular --seed-start 3000012 --episodes 12 --workers 6 --out /home/keyu/advaitic/plans/modular_controller/runs/sol_mod_i_cross.jsonl`
- `sol_mod_i_00` (HEAD `dc760fb9f3a26fa4436552764f94ddb93c063d04` with local edits, seeds 3000000–3000059) — 24/60 official success, 60/60 M2. 179.4 s.
  Command `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl uv run --locked --offline python -m method.modular.run_privileged --policy modular --seed-start 3000000 --episodes 60 --workers 6 --out /home/keyu/advaitic/plans/modular_controller/runs/sol_mod_i_00.jsonl`
- `sol_mod_j_cross` (HEAD `01f9572254d2433211cdf549773c622c65746e6f` with local edits, seeds 3000012–3000023) — 12/12 official success and M2 after a closer standoff, slower lowering, and plate-normal feedback.
  Command `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl uv run --locked --offline python -m method.modular.run_privileged --policy modular --seed-start 3000012 --episodes 12 --workers 6 --out /home/keyu/advaitic/plans/modular_controller/runs/sol_mod_j_cross.jsonl`
- `sol_mod_j_00` (HEAD `01f9572254d2433211cdf549773c622c65746e6f` with local edits, seeds 3000000–3000059) — 0 completed, BrokenProcessPool during overlapping sweeps. Retried with four workers.
  Command `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl uv run --locked --offline python -m method.modular.run_privileged --policy modular --seed-start 3000000 --episodes 60 --workers 6 --out /home/keyu/advaitic/plans/modular_controller/runs/sol_mod_j_00.jsonl`
- `sol_mod_j_60` (HEAD `01f9572254d2433211cdf549773c622c65746e6f` with local edits, seeds 3000060–3000119) — 60/60 official success and M2, 132.6 s.
  Command `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl uv run --locked --offline python -m method.modular.run_privileged --policy modular --seed-start 3000060 --episodes 60 --workers 5 --out /home/keyu/advaitic/plans/modular_controller/runs/sol_mod_j_60.jsonl`
- Additional one-process diagnostic probes at steps in the Sol report. Seeds 3000012/3000014 isolated a 20–25 mm pregrasp position error, seeds 3000000/3000014 exposed the incorrect IK branch, and seeds 3000014/3000015/3000017 isolated lower/seat position and plate-normal errors.
  Command `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl uv run --locked --offline python -` with probe details in the Sol report.

- 2026-11-22 `sol_mod_j_00_retry` (HEAD `59805e64c274126428961e1c3edcc4226b751d89`, requested seeds 3000000–3000059) — partial local retry, 26/26 official success and M2 logged on 3000000–3000025 before the log stopped advancing.
  Command `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl uv run --locked --offline python -m method.modular.run_privileged --policy modular --seed-start 3000000 --episodes 60 --workers 4 --out /home/keyu/advaitic/plans/modular_controller/runs/sol_mod_j_00_retry.jsonl`
- 2026-11-22 Modal smoke (HEAD `59805e64c274126428961e1c3edcc4226b751d89` with local runner edits, seed 3000399, app `modular-sol-privileged`) — first attempt failed at uv lock freshness check before starting a rollout, and the second hit an offline project rebuild in one CPU container. With the lock frozen and the image environment reused, one CPU container completed the third attempt in 7.6 s, 512 steps, successful with M2.
  Command `modal run ops_modular/sol_rollout.py::smoke --seed 3000399 --out /home/keyu/advaitic/plans/modular_controller/runs/sol_modal_j_smoke.json`

### Astra · M4 observation-only wiring and phase audit · 2026-11-22

The deployed source snapshot was created on `59805e6` with Astra's candidate files present in the worktree. Lead checkpoint `e588ae9` contains those files and the measured controller. The estimator is `advaitic-runs:est_v1b/last.pt`, SHA-256 `bd8ed33c87383b3b208ba251d8fa0b6f957d9f03a35cf6c8d23a81ce9cd867cd`. The [round-2 report](../plans/modular_controller/runs/astra_report2.md) contains the error tables and reproduction commands.

| Date | Command | Seeds | Outcome |
|---|---|---|---|
| 2026-11-22 | `modal volume get advaitic-runs est_v1b/last.pt checkpoints/est_v1b/last.pt` | — | Local checkpoint downloaded and verified. |
| 2026-11-22 | `uv run --locked python -`, one camera-free read of the site and plate frames | 3000320 | Site pose matched the pinned dish-drainer MJCF's +90° world-x rotation. Slot site height 0.85 m. |
| 2026-11-22 | `uv run --locked python -m pytest -q tests/modular/test_policy_modular.py` with the policy interpreter from the report | 1000000 | Two local one-episode/four-step checks, 30.53 s and 9.00 s. Both passed. The latter checked reset determinism. |
| 2026-11-22 | `modal deploy ops_modular/modular_sim.py` | — | New app `modular-sim` v1 deployed in 92.1 s client wall. Image integrity check verified 555 upstream files. |
| 2026-11-22 | `uv run --locked /home/keyu/.local/share/uv/tools/modal/bin/python /home/keyu/advaitic/ops/spawn.py modular-sim eval_run '[["astra-m4-smoke-v1",1000000,1,true]]'` | 1000000 | One T4, 14.92 s remote command wall. 10 steps, zero policy errors/falls, integrity OK, smoke cutoff. Full report in `modular/astra-m4-smoke-v1/eval` on `advaitic-runs`. |

### 2026-11-22 — lead: first official dev evals of the observation-only modular policy (e588ae9)
- Code: commit `e588ae9` (controller variant j + `EstimatedTaskState` + `submission/policy_modular.py`), frozen export
  `/home/keyu/advaitic/export/modular_e588ae9`, estimator `est_v1b/last.pt` sha256 `bd8ed33c…cd867cd`.
- Modal: dedicated account (profile `chinmaykarkar7`, `MODAL_CONFIG_PATH=~/.modal_modular.toml`), app `modular-sim`, T4, volume `advaitic-runs`.
- Deploy: `modal deploy ops_modular/modular_sim.py` from the export (115 s).
- Smoke: `eval_run("lead_e588ae9_smoke_1000000", 1000000, 1, smoke=True)` → 10 steps, no policy error, no fall,
  submission sha256 `4848e1aa…`, NON_QUALIFYING (smoke).
- Scored (predeclared, disjoint dev blocks, spawned detached): seeds 1000000 (`fc-01M80XKEBMT38XE55Y1VZ48FDS`),
  1010000 (`fc-01M80XKF7P1VGHNJT47MTQS8D1`), 1020000 (`fc-01M80XKG3V1J96MCTQ43EPHGV5`), 30 episodes each. Results: pending.

### Astra · Rendered estimator measurements · 2026-11-22

All Modal jobs below used `modular-sim` v1 on the primary profile, pinned estimator `bd8ed33c…cd867cd` and the controller in `59805e6`. Lead checkpoint `e588ae9` preserves the policy and audit code. The 20 episodes contributed 11,938 paired pre-action samples. Under the privileged driver 19 reached official success, one timed out and zero fell. Pre-grasp plate error was 1.84 cm mean / 4.06 cm p95. During release the target-rack contact proxy missed 142 of 216 live contacts. See the [phase report](../plans/modular_controller/runs/astra_report2.md) for all metrics.

| Date | Command | Seed | Frames | Privileged result | Wall, s |
|---|---|---:|---:|---|---:|
| 2026-11-22 | Local `uv run --locked results/astra-policy-env/bin/python -m method.modular.phase_errors run 3000300 OUT.jsonl.gz` (software EGL) | 3000300 | 591 | success, excluded from pooled stats | 591.28 |
| 2026-11-22 | `modular-sim phase_episode("astra2-phase", 3000300)` | 3000300 | 591 | success | 31.27 |
| 2026-11-22 | `modular-sim phase_episode("astra2-phase", 3000301)` | 3000301 | 527 | success | 29.22 |
| 2026-11-22 | `modular-sim phase_episode("astra2-phase", 3000302)` | 3000302 | 554 | success | 27.54 |
| 2026-11-22 | `modular-sim phase_episode("astra2-phase", 3000303)` | 3000303 | 566 | success | 28.70 |
| 2026-11-22 | `modular-sim phase_episode("astra2-phase", 3000304)` | 3000304 | 592 | success | 32.01 |
| 2026-11-22 | `modular-sim phase_episode("astra2-phase", 3000305)` | 3000305 | 536 | success | 28.61 |
| 2026-11-22 | `modular-sim phase_episode("astra2-phase", 3000306)` | 3000306 | 572 | success | 29.08 |
| 2026-11-22 | `modular-sim phase_episode("astra2-phase", 3000307)` | 3000307 | 529 | success | 29.09 |
| 2026-11-22 | `modular-sim phase_episode("astra2-phase", 3000308)` | 3000308 | 603 | success | 29.32 |
| 2026-11-22 | `modular-sim phase_episode("astra2-phase", 3000309)` | 3000309 | 526 | success | 26.45 |
| 2026-11-22 | `modular-sim phase_episode("astra2-phase", 3000310)` | 3000310 | 518 | success | 25.61 |
| 2026-11-22 | `modular-sim phase_episode("astra2-phase", 3000311)` | 3000311 | 1700 | timeout | 68.31 |
| 2026-11-22 | `modular-sim phase_episode("astra2-phase", 3000312)` | 3000312 | 549 | success | 27.57 |
| 2026-11-22 | `modular-sim phase_episode("astra2-phase", 3000313)` | 3000313 | 577 | success | 28.73 |
| 2026-11-22 | `modular-sim phase_episode("astra2-phase", 3000314)` | 3000314 | 471 | success | 24.29 |
| 2026-11-22 | `modular-sim phase_episode("astra2-phase", 3000315)` | 3000315 | 512 | success | 27.78 |
| 2026-11-22 | `modular-sim phase_episode("astra2-phase", 3000316)` | 3000316 | 477 | success | 25.62 |
| 2026-11-22 | `modular-sim phase_episode("astra2-phase", 3000317)` | 3000317 | 493 | success | 26.92 |
| 2026-11-22 | `modular-sim phase_episode("astra2-phase", 3000318)` | 3000318 | 533 | success | 28.95 |
| 2026-11-22 | `modular-sim phase_episode("astra2-phase", 3000319)` | 3000319 | 512 | success | 26.72 |

20 T4 invocations, pool capped at five concurrent containers. Total remote command time 601.79 s. The official smoke used one additional T4 invocation (14.92 s). Estimated execution cost across both was $0.12 before startup and billing overhead.

### 2026-11-22 — Sol dedicated-account completion of J

- Dedicated CPU smoke, app `modular-sol-privileged`, HEAD `e588ae9b85325cf4b8c6abe28f9a42c16373ed7c`, seed 3000399. One CPU container, 6.9 s remote wall, 512 steps, successful with M2. Command `MODAL_CONFIG_PATH=/home/keyu/.modal_modular.toml modal run ops_modular/sol_rollout.py::smoke --seed 3000399 --out /home/keyu/advaitic/plans/modular_controller/runs/sol_dedicated_cpu_smoke.json`.
- Dedicated CPU completion, same app and SHA, seeds 3000026–3000059. Seven single-worker CPU containers, 59.6 s launch-to-collection, 34/34 official-task successes and M2. Combined with the 26 and 60 completed local episodes, J passes both privileged milestones at 120/120 on 3000000–3000119.
  Commands `MODAL_CONFIG_PATH=/home/keyu/.modal_modular.toml modal deploy ops_modular/sol_rollout.py`, `MODAL_CONFIG_PATH=/home/keyu/.modal_modular.toml modal run ops_modular/sol_rollout.py::launch --seed-start 3000026 --episodes 34 --per-shard 5 --out /home/keyu/advaitic/plans/modular_controller/runs/sol_dedicated_j_missing_calls.jsonl`, and `MODAL_CONFIG_PATH=/home/keyu/.modal_modular.toml modal run ops_modular/sol_rollout.py::collect --calls /home/keyu/advaitic/plans/modular_controller/runs/sol_dedicated_j_missing_calls.jsonl --out /home/keyu/advaitic/plans/modular_controller/runs/sol_dedicated_j_missing.jsonl`.

### 2026-11-22 — Sol rendered input substitutions

- Rendered smoke, HEAD `2818804ae9bea7a0655c0a0dfc90840f455026c2` plus local ablation harness, app `modular-sol-inputs`, seed 3000399, one T4, 27.4 s measured remote job wall, 512 steps, success and M2. Command `MODAL_CONFIG_PATH=/home/keyu/.modal_modular.toml modal run ops_modular/sol_ablation.py::smoke --seed 3000399 --out /home/keyu/advaitic/plans/modular_controller/runs/sol_ablation_smoke.json`.
- Eleven-condition paired experiment on seeds 3000400–3000411 and eight-condition extension on 3000412–3000459, same app and HEAD. 60 seed-shards, each two CPU cores, 4 GiB and a T4, app concurrency limit 16. Launch manifests are `sol_ablation_pilot_calls.jsonl` and `sol_ablation_main_calls.jsonl`. Commands use the dedicated profile and explicit language below.
  - `MODAL_CONFIG_PATH=/home/keyu/.modal_modular.toml modal deploy ops_modular/sol_ablation.py`
  - `MODAL_CONFIG_PATH=/home/keyu/.modal_modular.toml modal run ops_modular/sol_ablation.py::launch --start 3000400 --count 12 --out /home/keyu/advaitic/plans/modular_controller/runs/sol_ablation_pilot_calls.jsonl`
  - `MODAL_CONFIG_PATH=/home/keyu/.modal_modular.toml modal run ops_modular/sol_ablation.py::launch --start 3000412 --count 48 --variants truth,source_xyz,placement_xyz,destination,orientation,held,contacts,all --out /home/keyu/advaitic/plans/modular_controller/runs/sol_ablation_main_calls.jsonl`
  - `MODAL_CONFIG_PATH=/home/keyu/.modal_modular.toml modal run ops_modular/sol_ablation.py::collect --calls /home/keyu/advaitic/plans/modular_controller/runs/sol_ablation_pilot_calls.jsonl --out /home/keyu/advaitic/plans/modular_controller/runs/sol_ablation_pilot.jsonl`
  - `MODAL_CONFIG_PATH=/home/keyu/.modal_modular.toml modal run ops_modular/sol_ablation.py::collect --calls /home/keyu/advaitic/plans/modular_controller/runs/sol_ablation_main_calls.jsonl --out /home/keyu/advaitic/plans/modular_controller/runs/sol_ablation_main.jsonl`

### Modular controller · 2026-11-22 · Astra, frozen-J perception diagnosis

- Frozen controller and policy from e588ae9, est_v1b digest bd8ed33c87383b3b208ba251d8fa0b6f957d9f03a35cf6c8d23a81ce9cd867cd. The completed rendered privileged-driven audit (3000300–3000319, 11,938 frames, 19/20 successes, no falls) is tabulated in `plans/modular_controller/runs/astra_findings.md`.
- Dedicated Modal account, app `modular-astra-perception`, T4/2 CPU/4 GiB per episode. `modal deploy -m ops_modular.astra_perception`; `python ops_modular/astra_jobs.py spawn smoke estimate 3020000 1` then `collect`, followed by `spawn diagnostic estimate 3020001 23` then `collect`. Prefix all Modal commands with `MODAL_CONFIG_PATH=/home/keyu/.modal_modular.toml` and run Python via the locked uv environment with the Modal SDK interpreter. Smoke returned a physical timeout at 1700 steps, 68.9 s. Frozen observation-only J completed 1/24 on fresh generation seeds. First failed guards: LOWER 13, LIFT 4, SETTLE 4, RETREAT 2.
- `python -m ops_modular.astra_actions` copied the three interrupted official archives. `MUJOCO_GL=egl ORT_DISABLE_TELEMETRY=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked python -m method.modular.replay_archives $RUNS/astra_r3/actions $RUNS/astra_r3/camera_free.jsonl --workers 3` verified 17/17 failed action logs with matching outcomes and 1700 steps each. Of 21 archived episodes, 4 completed the task and 17 failed. Among failures, 14/17 lifted ≥5 cm and contacted the target rack, 5/17 opened after lifting. The incomplete blocks do not qualify for an official score. No repeat official evaluation was launched.
