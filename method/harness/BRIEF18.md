# Brief 18 — Move Plate seating cue and shallow-pinch retry

The seating and regrip adaptations come from SWIRL-Lab’s released BiGym2
Move Plate Astra s1 and Opus s3 programs, licensed under CC BY 4.0.
[Dataset card](https://huggingface.co/datasets/SWIRL-Lab/bigym2-agent-rollouts/blob/d996b62535e008cf7a5f77d3899082ae652eed02/README.md) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Released programs

- **Astra s1 seating.** In the lowering segment (`1210 <= t < 1390`), starting
  at `t > 1270`, it counts consecutive observations with measured left elbow
  `q[3] < 0.017 rad` and measured left wrist pitch minus **current raw** wrist
  command `q[5] - a[9] < 0.024 rad`. A failed predicate clears the count. The
  fifth qualifying observation latches shoulder pitch and time. Zero wrist
  error passes the signed inequality. [Astra s1 policy, lines 144–155](https://huggingface.co/datasets/SWIRL-Lab/bigym2-agent-rollouts/blob/d996b62535e008cf7a5f77d3899082ae652eed02/policies/move_plate/astra-s1/policy.py#L144-L155)
- **Astra s1 pinch retry.** At local `t=525`, 55 steps after closure starts
  and 15 after the full-close command, read the left fingers `q[7:9]`. Retry
  once if the maximum is `< 0.004 rad` or their sum is `< -0.023 rad`. Hold
  the arm command, open immediately, and drive toward the current pelvis
  position plus 3.5 cm in world +x during elapsed steps 10–59. The base
  controller uses a 0.09 m/s per-axis cap before its speed-floor adjustment
  and a 1.2 cm per-axis deadband. Ramp closed from step 60 through 85. At 100,
  refresh the rack reference and add 80 steps to the nominal timeline delay.
  [Astra s1 policy, lines 65–75](https://huggingface.co/datasets/SWIRL-Lab/bigym2-agent-rollouts/blob/d996b62535e008cf7a5f77d3899082ae652eed02/policies/move_plate/astra-s1/policy.py#L65-L75) · [Lines 112–125](https://huggingface.co/datasets/SWIRL-Lab/bigym2-agent-rollouts/blob/d996b62535e008cf7a5f77d3899082ae652eed02/policies/move_plate/astra-s1/policy.py#L112-L125) · [Lines 168–208](https://huggingface.co/datasets/SWIRL-Lab/bigym2-agent-rollouts/blob/d996b62535e008cf7a5f77d3899082ae652eed02/policies/move_plate/astra-s1/policy.py#L168-L208)
- **Astra s1 release.** After contact, hold closed through elapsed step 10,
  ramp fully open by step 50, and retract the shoulder by 0.4 rad over steps
  75–185 with wrist orientation compensation. The timed fallback starts
  opening at `t=1390`, reaches fully open at `1430`, and retracts over
  `1460–1580`. [Astra s1 policy, lines 9–15](https://huggingface.co/datasets/SWIRL-Lab/bigym2-agent-rollouts/blob/d996b62535e008cf7a5f77d3899082ae652eed02/policies/move_plate/astra-s1/policy.py#L9-L15) · [Lines 156–165](https://huggingface.co/datasets/SWIRL-Lab/bigym2-agent-rollouts/blob/d996b62535e008cf7a5f77d3899082ae652eed02/policies/move_plate/astra-s1/policy.py#L156-L165)
- **Opus s3 seating and release.** Lower the wrist target at up to 2 mm per
  step. A single observation with target wrist z more than 2 cm below the
  measured FK wrist z triggers release. The fallback is `k > 200` in the
  lowering phase. The next phase commands fully open immediately and holds
  the arm target until `k > 20`. Retract for `k=1–29` toward placement x
  minus 2 cm and measured release z plus 8 cm at up to 3 mm per step, then
  toward placement x minus 12 cm and lift z (1.22 m) at 4 mm per step until
  `k > 60`. For pickup it commands full close through `k > 30`, then lifts.
  [Opus s3 policy, lines 58–62](https://huggingface.co/datasets/SWIRL-Lab/bigym2-agent-rollouts/blob/d996b62535e008cf7a5f77d3899082ae652eed02/policies/move_plate/opus-s3/policy.py#L58-L62) · [Lines 238–249](https://huggingface.co/datasets/SWIRL-Lab/bigym2-agent-rollouts/blob/d996b62535e008cf7a5f77d3899082ae652eed02/policies/move_plate/opus-s3/policy.py#L238-L249) · [Lines 298–319](https://huggingface.co/datasets/SWIRL-Lab/bigym2-agent-rollouts/blob/d996b62535e008cf7a5f77d3899082ae652eed02/policies/move_plate/opus-s3/policy.py#L298-L319)

## Harness port and flags

| Environment variable | Default | Behavior |
| --- | --- | --- |
| `ADV_H_SEAT_CUE` | `0` | Require persistent wrist tracking error in `PLACE` alongside the release gates. |
| `ADV_H_SEAT_CUE_ONLY` | `0` | With `ADV_H_SEAT_CUE=1`, replace the strict estimator distance and tilt checks with the seat cue inside the coarse `PLACE` region (8 cm default). Retains estimated held, quiet, and stability checks. |
| `ADV_H_SEAT_ERROR_RAD` | `0.024` | Minimum absolute tracking error, positive radians. |
| `ADV_H_SEAT_STEPS` | `5` | Minimum elapsed control steps between qualifying fresh observations. |
| `ADV_H_REGRIP` | `0` | Enable one shallow-pinch recovery per episode. |

Set the flags on `submission.policy_harness:make_policy` with `ADV_H_SUPERVISE=1`
(default). Seating compares `state[5]` with the denormalized, clipped `action[9]`
from the final row of the preceding `PLACE` chunk. The port uses Astra’s wrist
joint and threshold magnitude as a minimum absolute lag. Astra’s elbow angle
corresponds to its nominal zero-elbow pose. The harness checks elbow velocity as
part of the left-arm quiet gate. Both endpoints of an elapsed interval must
qualify. With two actions per observation, the five-step setting
requires four qualifying observations spanning six steps (120 ms). A matching or
unsafe observation resets the streak. On release, the harness opens and latches
the measured pose for `RELEASE` and `SETTLE`. Wrist lag is a contact proxy.

The retry checks Astra’s two finger predicates on the first eligible `CARRY`
observation at least 15 control steps after the initial full-close command.
The check requires the plate estimate to be held, observed closure, and the
carry to still be outside the `PLACE` region. Recovery holds the measured arm
and height, and uses Astra’s open, advance, and re-close schedule. At the first
observation at or after 100 recovery steps, the harness returns to `GRASP` to
confirm the hold. The model and supervisor keep their original chunk cadence.

## State mapping and verification

All indices are zero-based. The arm order on each side is shoulder pitch,
shoulder roll, shoulder yaw, elbow, wrist roll, wrist pitch, wrist yaw.

| Public state | Meaning |
| --- | --- |
| `0:7`, `9:16` | Left and right measured arm angles in radians, matching action target slots `4:11` and `11:18`. |
| `7:9`, `16:18` | Left and right Dex1 finger joint angles, fingers 1 and 2. |
| `18:22` | Measured pelvis x, y, z and yaw used for the recovery goal, pose hold and velocity transform. |
| `22:29`, `40:42` | Left arm velocities and pelvis planar velocities used by the existing quiet gate. |
| `44:46` | Left and right scalar gripper actuator observations. Index 44 confirms left closure. |

The evaluator forwards `low_dim_obs` as `state`. Pinned BiGym2 forms `qpos`
followed by `qvel` and preserves joint order when cropping. A constructor-only
check confirmed the compiled joint names, component slices `0:44`, `44:46`,
`46:50`, one unnormalized 50-D frame, 50 Hz, and absolute action bounds matching
the committed layout. [Evaluator observation](../../src/advaitic_eval/environment.py#L44-L49) · [Action layout](../../src/advaitic_eval/action_layout.json#L1-L20) · [BiGym2 qpos/qvel assembly](https://github.com/swirl-uk/BiGym2/blob/d71501359b215b9d78e27dcb70b220412d474dfa/bigym/bigym_env.py#L475-L486) · [BiGym2 crop](https://github.com/swirl-uk/BiGym2/blob/d71501359b215b9d78e27dcb70b220412d474dfa/bigym/loco/env.py#L1124-L1140)

All 17 harness unit tests and the Ruff lint and format checks passed with
`uv run --locked --no-sync`. Synthetic streams cover command alignment,
persistence and reset, both seating modes, both pinch thresholds, recovery
timing across chunks, frame and deadband, and the once-per-episode latch.
Rollout success is unmeasured.
