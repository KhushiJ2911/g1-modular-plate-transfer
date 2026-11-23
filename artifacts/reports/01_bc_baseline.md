Implemented the exporter, memmap dataset, BC model, resumable EMA trainer, evaluated policy, YAML config and CPU smoke test. Commands and design details are in [method/bc/README.md](/home/keyu/advaitic/wt-bc/method/bc/README.md).

- Frozen BatchNorm preserves pretrained statistics.
- Microbatches of 8 retain logical batch 64 while reducing memory use.
- Training-only statistics avoid validation leakage.
- Model code lives in `submission/` so the evaluator digest covers inference.

CPU smoke passed, including bitwise-equivalent resumed training and policy output checks. Cached ImageNet loading, YAML configuration, Ruff and formatting checks passed. Protected files were unchanged.

Real-demo export, GPU fp16/memory use and simulator performance remain unverified. No task-trained weights or success rate were produced.

Run from the repository root:

```bash
ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl uv run --locked python -m method.bc.export_human \
  --out /home/keyu/advaitic/data/human60 \
  --metadata method/data/demo_meta.json

/home/keyu/advaitic/mlenv/bin/python -m method.bc.train \
  --config method/bc/human60.yaml

MUJOCO_GL=egl ADV_CKPT=method/bc/checkpoints/human60/last.pt uv run advaitic-eval \
  --policy submission.policy:make_policy \
  --policy-python /home/keyu/advaitic/mlenv/bin/python \
  --episodes 30 --seed-start 1000000 --method bc-human60 \
  --out results/bc-human60-dev-001
```

Repeat the training command to resume automatically; use a new evaluation output directory each time.