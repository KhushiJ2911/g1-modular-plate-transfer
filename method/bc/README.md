# BC and student v1

Run commands from the repository root. Export/evaluation use the committed uv
lock; training/inference use `/home/keyu/advaitic/mlenv/bin/python`. Candidate
implementation is in `method/bc/` and `submission/`.

Measured human-only baseline: validation L1 0.019 and closed-loop dev score
**0/30**, all timeouts with no gripper closure. Offline held-out gripper timing
was within about 20 steps. These observations motivate teacher data and auxiliary
supervision; student v1 has no measured task-success result yet.

## Data and export

```bash
uv sync --locked
ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl uv run --locked python -m method.bc.export_human \
  --out /home/keyu/advaitic/data/human60 --metadata method/data/demo_meta.json
```

`get_demos(60)` runs once (about 2.5 GB RAM); existing episodes are skipped.
Export uses atomic replacement. Optional metadata supplies seed/slot, otherwise
both are `-1`. Seed, slot and success never enter the network. Unsuccessful
human/teacher episodes are retained.

Each `ep_<index>.npz` contains `rgb` uint8 `(ceil(T/2),3,3,84,84)`,
`rgb_stride=2`, `state` float32 `(T,50)` and clean `action` float32 `(T,20)`.
Camera order is head/right wrist/left wrist. Frame `j` corresponds to control
step `2*j`; actions are supervised at every control step, without a transition
shift. The terminal observation at `T` is omitted. `action_executed` is ignored:
controller corrections must not replace clean teacher targets.

Optional labels can be embedded in the episode or stored in
`ep_<index>.priv.npz`. Embedded values take precedence **per key**. Sidecars are
never enumerated as episodes. Known optional fields cached in the index are:

| Field | Shape |
| --- | --- |
| `priv_plate_rel`, `priv_target_site_rel` | `(T,3)` |
| `priv_plate_tilt_deg`, `priv_held` | `(T,)` |
| `priv_target_rack_offset`, `priv_start_rack_offset` | `(2,)` |

Present arrays must have the exact shape and finite values; held must be 0/1.
Missing labels remain masked. The nine-dimensional auxiliary target at sample
step `t` is `[plate_rel[t], target_site_rel[t], target_rack_offset, held[t]]`.
The mask is boolean `(9,)`: missing keys disable only their own dimensions;
an unlabeled episode has an entirely zero mask. Tilt/start-offset labels are
preserved in the index for future use but do not enter this target.

Normalization uses **training episodes only**, at every control step, ignoring
missing dimensions. The first eight auxiliary dimensions and configured proprio
indices get mean/std normalization (std floor `1e-3`). Held remains binary, with
mean 0/std 1. An auxiliary dimension absent throughout training uses mean 0/std 1.
Masked normalized targets are zero. Stats are stored in the checkpoint.

The cache streams one compressed episode at a time into one uint8 `<hash>.u8`
image file and a small `<hash>.npz` state/action/privileged-label/offset index.
Splits/workers share a read-only memmap; they do not hold the full RGB dataset
in RAM. The versioned key includes resolved episode and sidecar paths, sizes,
and modification times, including sidecar additions/removals. Sidecars arriving
after training starts require a new run, initialized from the previous weights.
A file lock and atomic index publication protect concurrent cache construction.
Do not change input files during a job. Old cache hashes can be removed later.

## Student configuration and sampling

`student_v1.yaml` supplies human60, teacher_v1 and teacher_v2; `H=20`,
`exec_steps=10`, 15,000 steps, batch 64/microbatch 8, auxiliary weight 0.1,
2 workers and default 19 proprio indices. `data_weights: [2, 1, 1]` assigns
50% of training draws to humans and 25% to each teacher directory, so teacher
growth cannot drown out human data. These weights specify **directory mass**,
not per-episode or per-frame weights. Sampling is uniform over stored frames
within each source, with replacement via `WeightedRandomSampler`. Omitted or
empty `data_weights` defaults to source training-frame counts, reproducing a
uniform mixture over all training frames. Zero weight disables a training source;
weights must be finite/nonnegative, one per directory, with positive total.

`val_episodes` is held out **per directory** using
`np.linspace(0, N-1, val_episodes, dtype=int)` within its numerically sorted files.
Every source must contain more than `val_episodes` episodes (at least 7 for the
example). For human60 alone, the six held-out indices remain
`0, 11, 23, 35, 47, 59`. Validation enumerates all held-out frames without weighted
sampling and aggregates action loss by valid action elements and auxiliary loss
by labeled dimensions. It measures an episode holdout, not simulator success.

## Training, initialization and resume

Operator training command (requires a GPU; not executed during this change):

```bash
ORT_DISABLE_TELEMETRY=1 /home/keyu/advaitic/mlenv/bin/python -m method.bc.train \
  --config method/bc/student_v1.yaml \
  --init-from method/bc/checkpoints/human60/last.pt
```

Omit `--init-from` to train from ImageNet initialization. Fine-tuning loads the
source **EMA model weights only**, with a fresh optimizer, scaler, EMA copy,
step 0 and RNG seed. It never downloads ImageNet weights when initializing from
a checkpoint. Horizon and ordered proprio dimensions must match the source;
architecture mismatches fail explicitly. Missing auxiliary weights initialize a
new head; a saved auxiliary head can also be discarded with `--aux-weight 0`.
Training-split normalization is recomputed for the new mixture. The first
proprio linear layer and existing auxiliary output layer are algebraically
rebased to preserve the source function in physical coordinates under those
new statistics. The initialization source path is saved for traceability.

Resume with the same config **without** `--init-from`:

```bash
ORT_DISABLE_TELEMETRY=1 /home/keyu/advaitic/mlenv/bin/python -m method.bc.train \
  --config method/bc/student_v1.yaml
```

`last.pt` automatically restores online/EMA weights, AdamW state, GradScaler,
step and Python/NumPy/torch/CUDA RNG. Each weighted batch uses an independent
seed derived from its absolute step, preserving the sample sequence across
prefetch/restarts. Run/cache paths, worker count and device may change on resume;
training settings and data identity must match. CUDA benchmarking can select
nondeterministic kernels; bitwise resume is checked on CPU only. Version-1
checkpoints remain valid for inference and `--init-from`; resuming their old
training state is rejected because cache/sampling semantics changed.

All settings have CLI flags; CLI overrides flat YAML/JSON. Inline lists support
`data_dirs`, `data_weights` and `proprio_dims`. For example,
`--proprio-dims 44 0 18` changes both the selected state indices and their order.
The default remains `0:9,18:22,40:44,44:46` (19 dims). The ordered indices are
stored in config/checkpoints and read by the policy. A different proprio
architecture requires a fresh run without an incompatible `--init-from`.

## Model, losses and speed

- Shared ResNet-18 through layer3, ImageNet initialization, 32 spatial-softmax
  keypoints per camera, and a learned 16-dimensional camera embedding retain
  the existing visual architecture. BatchNorm statistics stay frozen.
- The 128-dimensional GELU proprio embedding joins all camera features. The
  existing two-layer 512-wide GELU/dropout action head predicts `H x 10` actions.
  A separate **linear 368-to-9 head** reads the same fused feature, encouraging
  observation features to encode privileged geometry without using it as input.
- Total optimization loss is action L1 plus `aux_weight * auxiliary_loss`.
  Auxiliary loss averages eight MSE terms and one held BCE-with-logits term over
  the **available dimensions**. Each microbatch is scaled by its valid action
  count for L1 and independently by its labeled-dimension count for auxiliary
  loss, preserving the logical batch objective with partial/missing labels.
  `aux_weight=0` omits the head and its loss entirely.
- Action end-padding repeats the final clean action and is excluded from L1.
  Samples begin at stored even steps. Aux labels refer to the current sample
  step, not the predicted future action chunk.
- Training-only DrQ shifts and ImageNet normalization are unchanged. Backbone
  input is contiguous `channels_last`; cuDNN benchmarking is enabled in training.
  Default workers are 2; CUDA loaders pin memory and transfers are nonblocking.
- AdamW head/backbone LR `3e-4`/`1e-4`, weight decay `1e-4`, 500-step warmup then
  cosine, fp16 autocast/scaling on CUDA, gradient clipping 1.0, EMA decay 0.999.
  CPU uses fp32. EMA validates and serves inference.

`last.pt` is atomically replaced every 1,000 steps and at completion, with full
validation on each save. `log.csv` columns are
`step,train_loss,val_loss,train_aux_loss,val_aux_loss,lr,it/s`. Action and auxiliary
losses are logged separately, before multiplying by `aux_weight`; auxiliary
loss is zero when disabled or no labels are available. Train action loss averages
batches since the last log; train auxiliary loss averages labeled dimensions.
Validation fields retain the latest result (NaN before first validation).
Throughput includes validation/checkpoint overhead. Logs append across resume;
entries after the last durable checkpoint can repeat following a crash.

The trainer reports allocated/reserved CUDA peaks and rejects reserved peak
>=3 GB. Existing human-only measurements were about 5 it/s, 0.43 GB peak; these
are **not student-v1 measurements**. Student GPU throughput/memory, CUDA numerics,
cuDNN benchmarking and pinned-memory behavior remain unverified here.

## CPU verification

```bash
CUDA_VISIBLE_DEVICES='' ORT_DISABLE_TELEMETRY=1 /home/keyu/advaitic/mlenv/bin/python \
  -m method.bc.smoke_test
# Also exercise worker prefetch on a host permitting local multiprocessing sockets:
CUDA_VISIBLE_DEVICES='' ORT_DISABLE_TELEMETRY=1 /home/keyu/advaitic/mlenv/bin/python \
  -m method.bc.smoke_test --num-workers 2
uv run --locked ruff check method/bc submission
uv run --locked ruff format --check method/bc submission
```

Synthetic coverage includes three source directories; embedded/sidecar/partial/
missing labels; sidecar cache invalidation and embedded precedence; clean actions
versus differing executed actions; per-directory splits; train-only stats;
weighted source probabilities and deterministic batches; masks and loss
gradients; padding/frame alignment; 20-step exact CPU resume against uninterrupted
training (online and EMA); initialization/rebased normalization/fresh optimizer;
old/new checkpoint inference; disabled auxiliary head; reordered proprio subset;
policy output contract, reset and temporal ensembling. No simulator, downloads,
fixture injection or task success evidence is involved.

In the restricted implementation sandbox, the zero-worker smoke passes. The
2-worker attempt is blocked by socket permissions. Locked uv dependency sync is
unavailable offline; static checks instead use the already-cached Ruff binary
through uv without syncing:

```bash
UV_CACHE_DIR=/tmp/advaitic-uv-cache uv run --locked --no-sync \
  /home/keyu/.cache/uv/archive-v0/ZcJum6eSe-tIuzaS/ruff-0.17.4.data/scripts/ruff \
  check method/bc submission
UV_CACHE_DIR=/tmp/advaitic-uv-cache uv run --locked --no-sync \
  /home/keyu/.cache/uv/archive-v0/ZcJum6eSe-tIuzaS/ruff-0.17.4.data/scripts/ruff \
  format --check method/bc submission
```

## Inference and dev evaluation

`submission/model.py` is canonical and `method/bc/model.py` re-exports it, keeping
inference code inside the submission digest. The policy accepts only observed
RGB/state; it constructs no auxiliary head and discards saved auxiliary weights.
It loads either the old checkpoint schema (default proprio indices) or the new
schema (checkpoint indices). Inactive actions use saved training means. Active
indices remain `0,1,4,5,6,7,8,9,10,18`; left gripper is thresholded to +/-1 and
outputs clipped to [-1,1]. Optional ACT-style temporal ensembling is unchanged.

```bash
ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl \
ADV_CKPT=method/bc/checkpoints/student_v1/last.pt uv run --locked advaitic-eval \
  --policy submission.policy:make_policy \
  --policy-python /home/keyu/advaitic/mlenv/bin/python \
  --episodes 30 --seed-start 1000000 --method bc-student-v1 \
  --out results/bc-student-v1-dev-001
```

Choose a new output directory each run and add `--checkpoint` with the checkpoint
hash/identifier. Keep reviewer seeds separate. Student evaluation and real-data
training have not been run in this implementation. A synthetic smoke does not
demonstrate the assignment's 90% policy target.

The operator can copy chosen weights to `submission/weights/policy.pt`, or set
`ADV_CKPT` to override that default. The policy uses CUDA when available,
otherwise CPU. No GPU or git commands were used for this implementation.

Attribution: pinned BiGym2 human demonstrations; torchvision/ImageNet ResNet-18;
robomimic-style spatial softmax; DrQ shifts; ACT-style action chunks/L1/optional
ensembling. Implementation prepared with Codex assistance.
