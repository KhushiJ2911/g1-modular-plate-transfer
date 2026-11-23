Implemented in `method/bc/` and `submission/`. No GPU, git, or changes to the forbidden files.

- **Loss:** dim 18 supports BCE open logits; other dims retain L1. Independent masked denominators prevent gripper-label dilution, including across microbatches.
- **Inference:** saved threshold applies at every executed step, after averaging logits. Legacy checkpoints retain their mapping.
- **Compatibility:** unchanged head shape allows L1→BCE `--init-from` directly.
- **Logging:** train/val L1, BCE, overall gripper accuracy, and open-label accuracy.
- **Documentation:** [README](/home/keyu/advaitic/wt-bc/method/bc/README.md:143) covers defaults, initialization semantics, and dev-only `ADV_GRIPPER_THRESHOLD`. Final thresholds must be baked into the checkpoint.

Ruff, formatting, and CPU smoke passed. The real `student_v1b.pt` loaded and initialized exactly. On the ambiguous rare-open toy, weighted BCE fell **4.24→1.85**, with open recall **0%→100%**; L1 stayed at **0%**. That toy predicts open everywhere after weighting—it verifies optimization, not selective release.

Reproduce checks:

```bash
export CUDA_VISIBLE_DEVICES='' ORT_DISABLE_TELEMETRY=1
export UV_CACHE_DIR=/tmp/adv-bc-uv-cache
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

uv run --locked --no-sync \
  /home/keyu/advaitic/robot-learning-assignment/.venv/bin/ruff check method/bc submission
uv run --locked --no-sync \
  /home/keyu/advaitic/robot-learning-assignment/.venv/bin/ruff format --check method/bc submission
uv run --locked --no-sync \
  /home/keyu/advaitic/mlenv/bin/python -m method.bc.smoke_test --num-workers 0
```

Five-source CPU fine-tune, **not executed**:

```bash
CUDA_VISIBLE_DEVICES='' ORT_DISABLE_TELEMETRY=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
UV_CACHE_DIR=/tmp/adv-bc-uv-cache uv run --locked --no-sync \
  /home/keyu/advaitic/mlenv/bin/python -m method.bc.train \
  --config method/bc/student_v1b_bce.yaml \
  --init-from /home/keyu/advaitic/ckpts/student_v1b.pt
```

The [config](/home/keyu/advaitic/wt-bc/method/bc/student_v1b_bce.yaml) includes all five requested sources, BCE weight 1, experimental positive weight 5, and threshold 0.5. It holds out two episodes per source because `dagger2` was still growing; finish collection before launching.

Full training, threshold sweeps, and real task success remain unverified.