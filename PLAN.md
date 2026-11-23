# Plan (written Wed 18 Nov 03:10 IST, deadline Thu 19 Nov 23:00 IST)

## Measured facts that drive the plan
- Full episode ≈ 28 s → 30-trial eval ≈ 13 min. Max 2 sim workers (thermals) → ≈150 generated demos/hour.
- Open-loop demo replay: 59/60 on own seed, ~1/8 on a new seed with the same slot. Closed-loop correction is the task.
- Best agent-written pixel controller upstream: 36%.
- Only left arm (7) + left gripper + base vx/vy matter: 10-D effective action.

## Route
- **Baseline (B):** vision+proprio BC with action chunking, trained on the 60 human demos only.
- **Teacher (privileged, training-time only):** demo-transfer controller that knows the true plate/rack poses. It picks a human demo and tracks that demo's base/arm trajectory shifted by the true rack/slot offset, with closed-loop base position tracking. Used to generate many demos on fresh seeds. Disclosed.
- **Improvement (C):** same student architecture trained on human + teacher demos; DAgger round if failures cluster.
- **Ablations:** data scale (60 / ~300 / all), chunk length H, augmentation on/off, EMA.

## Checkpoints
| When | Gate |
|---|---|
| Wed 12:00 | Baseline dev score; teacher dev score on 30 seeds |
| Wed 18:00 | Teacher ≥ 90%, else fall back to vision state machine built on teacher structure |
| Thu 08:00 | Student dev score; DAgger if needed |
| Thu 13:00 | Freeze policy. Final eval once on 7310000 block |
| Thu 22:00 | Send (after user approves all writing) |
