Implemented both default-off features in the [harness](/home/keyu/advaitic/wt-harness/submission/harness.py).

- `ADV_H_SEAT_CUE=1` requires at least `0.024 rad` of left wrist-pitch tracking error for five elapsed control steps in `PLACE`. It compares measured `state[5]` with the denormalized `action[9]` from the last executed action. `ADV_H_SEAT_CUE_ONLY=1` replaces the strict estimator distance and tilt checks while retaining the coarse placement region, held, and quiet gates.
- `ADV_H_REGRIP=1` checks Astra’s shallow-pinch thresholds on `state[7:9]` early in `CARRY`. It can reopen, advance toward a point 3.5 cm ahead, and ramp closed once per episode.

The [Brief 18 report](/home/keyu/advaitic/wt-harness/method/harness/BRIEF18.md) records the exact released predicates, release and retract timings, source line references, and verified 50-D state mapping. It also explains the adaptation from Astra’s signed pose check to an absolute tracking-error threshold.

All 17 harness unit tests and Ruff checks passed. Rollout success is unmeasured. Changes are uncommitted.