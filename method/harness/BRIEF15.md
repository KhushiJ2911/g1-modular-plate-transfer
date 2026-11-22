# Brief 15: placement servo and grasp-approach regression check

Changes: `submission/harness.py` adds two independent, default-OFF flags.
In PLACE, while estimated held probability >= 0.7 and plate/site distance <=
`place_distance_m` (0.08 m), the servo replaces only action slots 0:2 with
`clip(place_kp * (target_site_rel - plate_rel)[:2], -place_vmax, place_vmax)`.
bce1 retains height, yaw and arms; RELEASE/SETTLE keep brief-13's latched hold.
No EMA or wrist controller was added. This cannot fix short carries outside PLACE.

`tilt_hold` independently zeros planar and yaw velocities for each returned chunk
while PLACE/held/in-range and estimated tilt > `release_tilt_deg` (12 degrees).
The default chunk is two steps (40 ms); each fresh observation reevaluates the
hold. It persists while tilted, yields to servo/bce1 when level, and preserves
BC arms/height. The existing tilt gate blocks release throughout the hold.

Exact settings (booleans accept 0/1 or false/true):

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `ADV_H_PLACE_SERVO` | `0` | Enable planar placement feedback |
| `ADV_H_PLACE_KP` | `2.0` | Proportional gain, 1/s |
| `ADV_H_PLACE_VMAX` | `0.10` | Per-axis speed cap, m/s |
| `ADV_H_TILT_HOLD` | `0` | Enable base stop while tilted |

Use `ADV_H_SUPERVISE=1 ADV_H_PLACE_SERVO=1 ADV_H_PLACE_KP=2.0 ADV_H_PLACE_VMAX=0.10`
for servo alone; add `ADV_H_TILT_HOLD=1` for both. `ADV_H_SUPERVISE=0` bypasses
both. Retain `ADV_CKPT` for bce1 and set `ADV_ESTIMATOR=/runs/est_v1b/last.pt`
on Modal. Keep `ADV_THROTTLE=1 ORT_DISABLE_TELEMETRY=1` for heavier local work.

Sign evidence: `method/harness/data.py:pelvis_relative` labels both points as
`R(-yaw) * (world_xyz - pelvis_xyz)`. Their difference cancels translation.
Teacher `wt-teacher/method/teacher/teacher.py:_pelvis_velocity` uses the same
rotation for desired pelvis correction, then `_normalize_xy` applies
`2*(v-low)/(high-low)-1`. Fixed `action_layout.json` planar bounds are [-1,1]
m/s, so normalization is identity. No second yaw rotation is needed. A plate
0.03 m +x of the site commands -0.06 m/s at kp=2 before speed clipping.

Verification: all 13 `method.harness.test_harness` unittest cases passed.
New checks cover world-to-pelvis signs at three yaws, clipping, zero error,
BC preservation, default-off behavior, held/region gating, tilt priority,
release/settle latching, config parsing and invalid values. The grasp-approach
check confirms no new base override during APPROACH/GRASP/CARRY; it adds no
grasp heuristic. Ruff check and format --check passed for src/submission/tests
and method/harness (28 Python files). All commands used `uv run --locked --no-sync`.

The old scratch checkpoint was absent; the existing two-update smoke script
recreated `/tmp/adv-harness-smoke-b15/run/last.pt` using only episodes 2200000/1.
Real bce1 + smoke-estimator wrapper calls passed all four flag combinations,
reset repeatability, finite float32 (2,20) outputs and supervise=0 passthrough.
Logs: `/tmp/adv-harness-b15-unit.log`, `/tmp/adv-harness-smoke-b15.log`,
`/tmp/adv-harness-b15-flags.json`. Smoke inference ran on CPU with
`ADV_THROTTLE=1 ORT_DISABLE_TELEMETRY=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`.
These are synthetic/offline plumbing checks, not policy rollout evidence.
No full evals, reserved seeds, evaluator/physics edits or success-rate claims.

Commit attempt failed: shared Git metadata/index.lock is read-only. Files remain on `harness` at 095ff8b.
