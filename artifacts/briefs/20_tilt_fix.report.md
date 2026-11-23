Implemented both observation-only controls on `harness` and left the changes uncommitted. Both flags default off.

Tilt correction alternates probes on left wrist roll (`action[8]`) and pitch (`action[9]`) while holding the base still. It compares baseline and probe tilt over six-control-step windows using an EMA with 0.75 decay per observation. On release, it latches the last corrected wrist targets through SETTLE.

Stall release opens both grippers on the first observation **after** 300 PLACE steps when the plate is estimated held, within 5 cm, and at or below 18°. At the default two-step cadence, the first eligible PLACE age is 302. Re-entering PLACE restarts the timer.

| Environment variable | Default |
|---|---:|
| `ADV_H_TILT_FIX` | `0` |
| `ADV_H_TILT_FIX_DEG` | `10.0` |
| `ADV_H_TILT_PROBE_RAD` | `0.02` |
| `ADV_H_TILT_PROBE_STEPS` | `6` |
| `ADV_H_TILT_STEP_RAD` | `0.01` |
| `ADV_H_TILT_LIMIT_RAD` | `0.18` |
| `ADV_H_STALL_RELEASE` | `0` |
| `ADV_H_STALL_STEPS` | `300` |
| `ADV_H_STALL_DISTANCE_M` | `0.05` |
| `ADV_H_STALL_TILT_DEG` | `18.0` |

**All 22 harness unit tests passed.** Synthetic tilt fell from 24° to 9.60° with clean observations and to 9.50° with alternating ±1.4° error. Both stayed within 10° for the final 50 observations. The unreachable-target case respected the ±0.18 rad limits. Rollout performance remains unmeasured. [Brief 20 report](/home/keyu/advaitic/wt-harness/method/harness/BRIEF20.md)

Ruff lint and changed-file formatting passed. The broader format check flagged pre-existing formatting in [policy_multi.py](/home/keyu/advaitic/wt-harness/submission/policy_multi.py).