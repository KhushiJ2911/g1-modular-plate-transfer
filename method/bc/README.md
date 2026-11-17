# BC baseline: human60

Run all commands from the repository root. The simulator/exporter uses the
locked repo environment; training and inference use the separate model Python.
No evaluator, dependency pin, physics, controller, or evaluation rule is changed.

## Export (sim environment, once)

```bash
uv sync --locked
ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl uv run --locked python -m method.bc.export_human \
  --out /home/keyu/advaitic/data/human60 \
  --metadata method/data/demo_meta.json
```

`get_demos(60)` runs once in one process (about 2.5 GB RAM). Existing episodes
are skipped, and a completed export does not reload demos. Writes use a temporary
file and `os.replace`. Metadata is optional: the pinned gym adapter discards
seed/slot, so the exporter uses `method/data/demo_meta.json` when present;
`--metadata` can select another demo-index/seed/slot sidecar. Without available
metadata, these fields are `-1`; they never enter the model.

The human and future teacher format is `ep_<index>.npz` with:

- `rgb`: uint8 `(ceil(T/2),3,3,84,84)`, cameras head/right wrist/left wrist;
  `rgb_stride`: scalar `2`. Frame `j` is observation step `2*j`.
- `state`: float32 `(T,50)` and `action`: float32 `(T,20)`, every control step.
- `seed`, `slot`: scalar integers; `success`: scalar bool. All episodes are
  retained, including unsuccessful ones.

The terminal observation at step `T` is omitted. Actions are already normalized;
there is no transition shift in the exporter.

## Train (operator GPU job)

The supplied model environment needs torch, torchvision and NumPy. The first
real training run uses torchvision's ImageNet ResNet-18 weights and caches them
in the usual torch hub cache (`TORCH_HOME` can override it). Later runs reuse
them, and loading a policy checkpoint never downloads weights.

```bash
/home/keyu/advaitic/mlenv/bin/python -m method.bc.train \
  --config method/bc/human60.yaml
```

Run the same command to resume `method/bc/checkpoints/human60/last.pt` automatically.
It restores online weights, EMA, optimizer, GradScaler, step, and Python/NumPy/
torch/CUDA RNG. Batches depend on the absolute training step, so prefetch cannot
shift samples after resume. The LR schedule is derived from the saved step and
unchanged config. Training/data changes require a new run directory; relocating
the cache or changing device/worker count is allowed. Completed runs exit.

All settings are CLI flags (`--help`). CLI overrides YAML. The dependency-free
YAML reader supports flat keys, scalar values and inline lists; it also accepts
JSON. Boolean CLI flags have `--no-...` counterparts. For multiple data sources:

```bash
/home/keyu/advaitic/mlenv/bin/python -m method.bc.train \
  --config method/bc/human60.yaml \
  --data-dirs /home/keyu/advaitic/data/human60 /home/keyu/advaitic/data/teacher \
  --run-dir method/bc/checkpoints/human-plus-teacher
```

Directories are processed in argument order, with episodes sorted numerically
within each. Six validation episodes are chosen by
`np.linspace(0, N-1, 6, dtype=int)` on that ordered list. For human60 these are
`0, 11, 23, 35, 47, 59`; the other 54 train. Train/validation episode paths and
the data identity are saved in each checkpoint. Do not change source ordering
when comparing methods. Validation is an episode split, not an unseen-seed
simulator success estimate.

On first use, the trainer streams compressed episodes into a single uint8
`<hash>.u8` file, with a small `<hash>.npz` state/action/offset index. Both splits
and up to two workers use the same read-only memmap. Only small state/action
arrays and individual episodes during cache construction need resident RAM.
The cache key includes resolved episode paths, sizes, modification timestamps
and a format version. Do not edit source episodes while a job is running. A
per-cache lock and atomic index publication protect concurrent cache creation.
Expect roughly 1.2 GB of image cache for 60 demos. Delete old hashes to reclaim
disk space after replacing source data.

## Model and training choices

- Shared ImageNet ResNet-18 through layer3, learned 1x1 projection to 32 keypoints
  and spatial softmax per camera: spatial features with a small prediction head.
  A learned 16-dimensional camera embedding accompanies each camera's features.
- Frozen BatchNorm statistics preserve pretrained estimates with correlated
  robot batches; convolution and BN affine parameters still train.
- Proprio indices `0:9,18:22,40:44,44:46` (19 dims), standardized using **training
  episodes only**, including every control step. Standard deviations floor at
  `1e-3`. A 128-dimensional GELU embedding joins camera features.
- Two 512-wide GELU layers with dropout 0.1 predict `H x 10` normalized actions.
  Active indices are `0,1,4,5,6,7,8,9,10,18`. Other actions use training means,
  saved in the checkpoint. Left gripper is thresholded to binary at inference.
- Samples begin at stored frames (even steps), but target actions remain at
  **every** control step. End padding repeats the last action; a true padding
  mask excludes it from L1 and its denominator. Validation also weights by
  valid action elements.
- Training-only DrQ shifts (replication pad 4, crop to 84) and ImageNet
  normalization. No augmentation during validation/inference.
- Default `H=20`, `exec_steps=10`: predict 0.4 seconds, refresh feedback every
  0.2 seconds. Optional ACT-style temporal ensembling is off by default;
  `--temporal-ensemble` combines overlapping predictions with exponential
  weights (decay 0.01, older first), still returning 10 actions per call.
- AdamW: head `3e-4`, backbone `1e-4`, weight decay `1e-4`; 500-step linear
  warmup, then cosine over 20k steps. Batch 64 is accumulated as microbatches
  of 8, with valid-element weighting. fp16 autocast/GradScaler on CUDA,
  gradient norm clip 1.0, and EMA decay 0.999. EMA validates and serves inference.
  CPU smoke runs use fp32. No GPU work was run while implementing this baseline.

`last.pt` is replaced atomically every 1k steps and at the end; each save includes
full EMA validation. `log.csv` is append-only with
`step,train_loss,val_loss,lr,it/s` (head LR, latest validation loss; NaN until the
first validation). Train loss averages batches since the previous log. Throughput
includes loading/training/validation and prior checkpoint overhead. Logs after a
crash can contain entries beyond the last durable checkpoint; resumed work
appends rather than deleting those entries.

Microbatching targets the <3 GB GPU limit. Training prints peak allocated and
reserved CUDA memory in GB and rejects a completed run whose reserved peak is
at least 3 GB. CUDA driver/context overhead is outside torch's counters. The GPU
memory bound and GPU fp16 numerics still need operator verification; lower
`--microbatch-size` in a new run if necessary.

## CPU smoke and static checks

```bash
CUDA_VISIBLE_DEVICES='' /home/keyu/advaitic/mlenv/bin/python -m method.bc.smoke_test
uv run --locked ruff check method submission
uv run --locked ruff format --check method submission
```

The smoke uses two temporary synthetic episodes and random initial weights,
with no downloads or simulator. It trains 20 CPU steps across an interruption,
checks bitwise equivalence to 20 uninterrupted steps (model and EMA), checks
cache reuse/frame alignment/padding/train-only statistics/masked L1, then loads
`make_policy()` and checks output shape/dtype/range/constants/binary gripper,
reset and optional ensembling. Temporary checkpoints are deleted. It establishes
software wiring only, not robot performance or the 90% assignment target.

## Dev evaluation and final weight location

```bash
MUJOCO_GL=egl ADV_CKPT=method/bc/checkpoints/human60/last.pt uv run advaitic-eval \
  --policy submission.policy:make_policy \
  --policy-python /home/keyu/advaitic/mlenv/bin/python \
  --episodes 30 --seed-start 1000000 --method bc-human60 \
  --out results/bc-human60-dev-001
```

Choose a new output directory for every run. Equivalent requested command with
placeholders: `MUJOCO_GL=egl ADV_CKPT=<ckpt> uv run advaitic-eval --policy
submission.policy:make_policy --policy-python /home/keyu/advaitic/mlenv/bin/python
--episodes 30 --seed-start 1000000 --method bc-human60 --out results/<new_dir>`.
For traceability also pass `--checkpoint` with your checkpoint hash/identifier.
Keep final reviewer seeds separate from this dev block.

After training, the operator can place the chosen checkpoint at the default
inference path (no git operation is performed by the scripts):

```bash
mkdir -p submission/weights
cp method/bc/checkpoints/human60/last.pt submission/weights/policy.pt
```

`ADV_CKPT` overrides that path. The policy loads EMA on CUDA if available,
otherwise CPU, and reads only `obs['rgb']`/`obs['state']`. The canonical model
is in `submission/model.py` so the evaluator's submission digest covers it;
`method/bc/model.py` re-exports that same implementation for training.

Attribution: BiGym2 human demonstrations from the pinned assignment dependency;
torchvision ImageNet-pretrained ResNet-18; spatial softmax in the robomimic
style, DrQ random shifts, ACT-style action chunks/L1/optional temporal ensemble.
Implementation was prepared with Codex assistance. No task success rate has
been measured for this baseline yet.
