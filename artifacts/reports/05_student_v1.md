Implemented in `method/bc/` and `submission/`.

- **Dataset:** embedded/sidecar privileged labels, per-dimension masks, train-only normalization; held stays binary. Clean `action` always supplies targets.
- **Sampling:** `WeightedRandomSampler` assigns directory-level probability; default is proportional to training frames. Validation holds out episodes per directory.
- **Aux head:** linear fused-feature → 9 outputs, masked MSE/BCE, separate logging. Independent microbatch weighting preserves the labeled-data objective.
- **Speed:** cuDNN benchmarking, channels-last backbone input, two workers by default, CUDA pinned memory.
- **Initialization:** `--init-from` loads EMA weights with fresh optimizer/step. Normalization rebasing preserves the source predictions. Horizon and ordered proprio dimensions must match.
- **Compatibility:** configurable checkpoint-stored proprio indices; inference omits the aux head and loads old/new checkpoints. EMA and resumability remain supported.

[student_v1.yaml](/home/keyu/advaitic/wt-bc/method/bc/student_v1.yaml) includes human60/teacher_v1/teacher_v2, weights `[2,1,1]`, H=20, exec_steps=10, 15,000 steps and aux_weight=0.1. This reserves 50% of draws for humans.

CPU smoke **passed**, including masks/gradients, weighted sampling, bitwise model/EMA resume, initialization, old/new policy loading and custom proprio. Ruff lint/format checks and config validation passed.

Exact smoke and static-check commands:

```bash
CUDA_VISIBLE_DEVICES='' ORT_DISABLE_TELEMETRY=1 /home/keyu/advaitic/mlenv/bin/python -m method.bc.smoke_test

UV_CACHE_DIR=/tmp/advaitic-uv-cache uv run --locked --no-sync /home/keyu/.cache/uv/archive-v0/ZcJum6eSe-tIuzaS/ruff-0.17.4.data/scripts/ruff check method/bc submission

UV_CACHE_DIR=/tmp/advaitic-uv-cache uv run --locked --no-sync /home/keyu/.cache/uv/archive-v0/ZcJum6eSe-tIuzaS/ruff-0.17.4.data/scripts/ruff format --check method/bc submission
```

Operator fine-tuning command:

```bash
ORT_DISABLE_TELEMETRY=1 /home/keyu/advaitic/mlenv/bin/python -m method.bc.train \
  --config method/bc/student_v1.yaml \
  --init-from method/bc/checkpoints/human60/last.pt
```

Omit `--init-from` when resuming. Legacy training checkpoints require initialization into a new run directory.

Unverified: CUDA performance/numerics, real-data training and student rollouts. Two-worker smoke was blocked by sandbox socket permissions; zero-worker smoke passed. Locked dependency sync was unavailable offline, hence cached Ruff. No GPU work or git commands were run; protected files were unchanged.

Details and additional commands are in [README.md](/home/keyu/advaitic/wt-bc/method/bc/README.md). Smoke results establish wiring, not task success.