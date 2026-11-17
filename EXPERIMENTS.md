# Experiment log

Every run: date/time (IST), commit, config, seeds, result, one-line interpretation. Failures included.

Seed blocks (fixed before any policy existed):
- **Dev** (all decisions): `--seed-start 1000000`
- **Final** (touched once per final candidate): `--seed-start 7310000` (public default)
- **Teacher / training data**: seeds from 2,000,000 upward, plus the 60 human-demo seeds (0–~200)

Machine: RTX 3050 4 GB laptop (15 W cap), 16 threads, 15 GB RAM. Max 2 sim workers (thermal), watchdog in `~/advaitic/ops/watchdog.sh`.

---

## Phase 0: understand and measure (Wed 18 Nov)

| Time | What | Result | Interpretation |
|---|---|---|---|
| 02:21 | `uv sync --locked`, `advaitic-check`, `advaitic-smoke` @ da5714d | integrity OK (555 files), all 6 slots reset, 7.5 s | Environment is as published |
| 02:24 | 1 full episode, HoldPose example, smoke, seed 7310000 | 28.3 s wall (1,700 steps; 2.5 s policy) | **30-trial eval ≈ 12–14 min** sequential. Evals are cheap |
| 02:25 | Upstream agent-written policies (21 runs, 7 agents × 3), from HF `SWIRL-Lab/bigym2-agent-rollouts` | best 36/100 (astra-s1); Opus 15–23; most 0–7 | Task is hard for hand-scripted pixel controllers. Reference only, not reused |
| 02:26 | Demo statistics, 60 human demos | 530–742 steps (median 620); all flagged success. Only **left arm + left gripper** used (right gripper never closes, right-arm std ≈ 0.02); height and yaw constant; base moves +0.10 m x, −0.38 m y; grasp ≈ step 229, release ≈ step 570 | Effective action space is 10-D. ~1,000 steps of slack for retries |
| 02:27 | **FAILED RUN:** 6 parallel replay workers each loading all demos via `get_demos` | ~2 GB RSS each → RAM + swap exhausted, laptop hung, 1 worker crashed | Never load full demos per worker. Extracted low-dim demos once (8 MB) |
| 02:30 | Open-loop replay of each demo on its **own** seed, in the evaluator env | **59/60 success**; demo 45 (slot 1) timed out | Demos are valid under this evaluator; sim is (nearly) deterministic |
| 03:00 | Open-loop replay of demo actions on a **new** seed with the same source slot | **1/8 success** (6 timeout, 1 terminated); stopped early for heat | ±5 cm rack shifts break blind replay → closed-loop correction is the core of the task |
| 03:14 | Thermal: CPU pkg 92 °C, GPU 88 °C with 3 sim workers (EGL rendering) | Killed workers; cooled to 74/69 °C | Cap at 2 sim workers + watchdog |

Route decision (03:10): **C** (privileged teacher → vision student), with **B** (BC on 60 human demos) as the baseline. Reasoning in `PLAN.md`.
