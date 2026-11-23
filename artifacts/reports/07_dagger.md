Implemented in [collect.py](/home/keyu/advaitic/wt-bc/method/dagger/collect.py), the BC dataset/trainer, and `masked_l1`.

- Reuses the evaluator worker and exact BCPolicy chunks to preserve evaluation behavior.
- Saves gripper-only masks, executed actions, label fractions, outcomes, and execution mode.
- Uses atomic files and locks for resumable concurrent collection.
- Counts only labelled, nonpadding active elements in loss and microbatch weighting.
- Adds a CPU config combining human60, teacher_v1, and DAgger.

**Exact two-process collection and training commands:** [README](/home/keyu/advaitic/wt-bc/method/bc/README.md:288). They cover seeds **2400000–2400060 inclusive (61 episodes)** and the requested checkpoint.

**Verified:** Ruff check/format, partial-mask gradient checks, exact CPU training resume, both collector modes synthetically, two concurrent short real rollouts, saved-data loading, and collection resume without duplicate summaries.

**Unverified:** full collection, union retraining, physical release behavior, and policy improvement. Two-worker training hit sandbox socket restrictions; the documented command uses `--num-workers 0`.

No protected source/configuration files were changed. No GPU or git commands were used.