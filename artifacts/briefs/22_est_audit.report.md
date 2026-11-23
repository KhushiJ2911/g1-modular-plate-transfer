**Trained and saved `est_v2`.** The audit implicates distance estimation in false release decisions. On Flywheel2, `est_v1b` falsely passed 37 of 149 hover frames at the 5 cm gate. `est_v2` cut that to 13 of 149. Tilt estimation improved on Flywheel2, while steeply tilted held plates remained difficult on the independent DAgger2 set.

### Held-out Flywheel2 audit

Both checkpoints saw the same 1,804 frames from 40 of 48 replayed episodes. Ground truth came from replaying recorded actions and verifying the saved proprioception bit for bit. Inference used only saved RGB and state. The gate in the table is `distance ≤ 5 cm and tilt ≤ 12°`.

| Slice | Frames | Distance MAE, cm | Tilt MAE, ° | Tilt error >5° | Gate accuracy |
|---|---:|---:|---:|---:|---:|
| **All within 10 cm** | **1,804** | **0.72 → 0.39** | **2.78 → 1.24** | **3.9% → 3.2%** | **92.5% → 94.0%** |
| Distance 0–3 cm | 368 | 0.53 → 0.30 | 1.07 → 0.63 | 0.5% → 0.0% | 99.5% → 99.5% |
| Distance 3–6 cm | 1,167 | 0.52 → 0.30 | 0.99 → 0.59 | 1.2% → 0.3% | 88.9% → 90.9% |
| Distance 6–10 cm | 269 | 1.82 → 0.90 | 12.88 → 4.90 | 20.4% → 20.1% | 98.1% → 99.6% |
| Tilt <10° | 1,570 | 0.49 → 0.25 | 0.88 → 0.51 | 0.3% → 0.1% | 93.9% → 96.6% |
| Tilt 10–20° | 149 | 0.59 → 0.58 | 1.62 → 1.13 | 2.0% → 0.0% | 72.5% → 63.1% |
| Tilt >20° | 85 | 5.14 → 2.55 | 39.87 → 14.95 | 75.3% → 64.7% | 100% → 100% |
| Carry | 96 | 0.49 → 0.18 | 0.83 → 0.49 | 0.0% → 0.0% | 100% → 100% |
| Place | 1,204 | 0.52 → 0.26 | 0.82 → 0.45 | 0.3% → 0.2% | 92.8% → 95.6% |
| Released | 449 | 0.53 → 0.40 | 1.47 → 0.96 | 2.9% → 0.2% | 89.1% → 87.5% |
| Gripper occludes every measurable view | 705 | 0.49 → 0.26 | 0.83 → 0.41 | 0.1% → 0.0% | 96.2% → 97.7% |
| At least one view clear of the gripper | 1,099 | 0.86 → 0.47 | 4.03 → 1.77 | 6.4% → 5.2% | 90.1% → 91.5% |

“Released” means the recorded gripper command was open after a detected hold. A measurable view had at least 10 plate pixels with the gripper hidden in the diagnostic render. Occlusion required the gripper to cover at least 10% of those pixels.

- **Release gate.** Flywheel2 false opens fell from **116 to 94 of 534** true negatives, and false blocks fell from **20 to 15 of 1,270** true positives. At the harness’s default **3.5 cm** geometry threshold, accuracy rose from **84.0% to 89.7%**. With rack contact and true tilt of 23–25°, both checkpoints got all 16 gate decisions right. Those frames came from one episode.
- **DAgger2 challenge.** Across 1,245 frames from 25 episodes, distance MAE improved **0.98 → 0.78 cm**, tilt MAE changed **3.15 → 3.22°**, the >5° error rate stayed **11.5%**, and 5 cm gate accuracy improved **63.2% → 65.1%**. For held plates above 20°, the >5° error rate rose **45.2% → 54.8%** across 42 frames from three episodes. Closed-loop seating and tilt-correction performance remain unmeasured.

### Checkpoint and spend

- **Local** `ADV_ESTIMATOR=/home/keyu/advaitic/ckpts/est_v2.pt`. The harness loaded it and produced a valid action chunk. [Checkpoint](/home/keyu/advaitic/ckpts/est_v2.pt)
- **Modal** `ADV_ESTIMATOR=/runs/est_v2/last.pt`. The 2,000-step A10 fine-tune emphasized hover and tilted frames, retained the existing estimator interface, and selected the checkpoint on a separate development split. [Training run](https://modal.com/apps/jaiswalkhushi2911/main/ap-BkQrV1J6TbnjnQsqHslj2U)
- **Modal spend** Approximately **$0.05** for the successful T4 smoke and A10 run at list rates. Allowing for the full app lifetimes, including retries from the initial smoke startup error, gives a conservative estimate of **about $0.22** before credits. [Modal pricing](https://modal.com/pricing)

Changes are uncommitted on `est2`. All 42 evaluator tests and 22 harness tests passed. The changed files passed lint and formatting checks. The repository-wide formatting check flagged the existing `submission/policy_multi.py`.

Full bins and paired estimates are available in the [Flywheel2 baseline](/home/keyu/advaitic/analysis/est2/v1b-test.json), [Flywheel2 v2](/home/keyu/advaitic/analysis/est2/v2-test.json), [DAgger2 baseline](/home/keyu/advaitic/analysis/est2/v1b-challenge.json), [DAgger2 v2](/home/keyu/advaitic/analysis/est2/v2-challenge.json), and [paired comparison](/home/keyu/advaitic/analysis/est2/paired-comparison.json).