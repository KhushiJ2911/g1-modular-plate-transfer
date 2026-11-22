# Brief 20 — wrist tilt correction and stall release

Both options are independent and default off. To add both to an existing
supervised harness run, set `ADV_H_TILT_FIX=1 ADV_H_STALL_RELEASE=1`.

| Environment variable | Default | Meaning |
| --- | ---: | --- |
| `ADV_H_TILT_FIX` | `0` | Enable wrist correction while the plate is held in PLACE |
| `ADV_H_TILT_FIX_DEG` | `10.0` | Start probing above this filtered tilt in degrees |
| `ADV_H_TILT_PROBE_RAD` | `0.02` | Trial displacement from the accepted offset on one axis |
| `ADV_H_TILT_PROBE_STEPS` | `6` | Minimum elapsed control steps per baseline or trial measurement |
| `ADV_H_TILT_STEP_RAD` | `0.01` | Maximum accepted increment after an improving trial |
| `ADV_H_TILT_LIMIT_RAD` | `0.18` | Symmetric bound on each axis, including trial displacements |
| `ADV_H_STALL_RELEASE` | `0` | Allow release after a prolonged stay in PLACE |
| `ADV_H_STALL_STEPS` | `300` | PLACE age in control steps that must be strictly exceeded |
| `ADV_H_STALL_DISTANCE_M` | `0.05` | Maximum estimated 3-D plate-to-site distance for stall release |
| `ADV_H_STALL_TILT_DEG` | `18.0` | Maximum estimated tilt for stall release, inclusive |

The supervisor adds the radian corrections to each ensembled bce1 target at
action indices 8 (left wrist roll) and 9 (left wrist pitch), scaling by each
joint's range. During a measurement it holds the observed base height and
commands zero planar and yaw velocity. Each fresh tilt estimate updates an
EMA with prior sample weights decaying by 0.75 per observation.
Baseline and trial measurements have separate windows. An improving trial
advances the accepted offset toward the probe. A failed trial reverses the
direction for that axis on its next turn. The axes alternate. [Action layout](../../src/advaitic_eval/action_layout.json)

The wrist targets retain the accepted offset once the plate is level.
A lost-held observation cancels a pending trial. Leaving PLACE clears the
search. When releasing with an offset applied, the supervisor latches the last
commanded wrist pair from the preceding chunk through RELEASE and SETTLE.
The remaining joints hold the observed release pose and both grippers open.

Stall release uses the current held, distance, and tilt estimates. It can fire
without satisfying the quiet, stability, or seating gates. The timer starts
on each entry to PLACE. With two control steps per observation, age 302 is the
first eligible decision at the defaults.

The synthetic closed-loop test feeds back `abs([65, -55] @ offset + b)` in
degrees, using the offsets recovered from emitted actions. Both reachable
cases started at 24°. The clean case ended at 9.60°, and the opposite-sign
case with alternating ±1.4° observation error ended at 9.50°. Both stayed
within 10° for the last 50 observations. A third case with an unreachable
goal exercised the wrist limits. Rollout performance is unmeasured.
