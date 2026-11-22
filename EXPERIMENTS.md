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
