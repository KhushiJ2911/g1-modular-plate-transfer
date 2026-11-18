# BC and student v1

Run commands from the repository root. Export/evaluation use the committed uv
lock; training/inference use `/home/keyu/advaitic/mlenv/bin/python`. Candidate
implementation is in `method/bc/` and `submission/`.

Measured human-only baseline: validation L1 0.019 and closed-loop dev score
**0/30**, all timeouts with no gripper closure. Offline held-out gripper timing
was within about 20 steps. These observations motivate teacher data and auxiliary
supervision. Student v1a (human60 + 73 teacher episodes, 8k steps) also measured
**0/30** on dev: 18 no-grasp, 10 placed but never released, and 2 reached the
target then failed. See the DAgger section for the next training experiment.

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

Optional `label_mask` is boolean `(T,20)`, with True for each trusted action
element. Missing masks default to all True, preserving existing human/teacher
supervision. Chunk targets select the per-step masks at `t:t+H`, intersect them
with `ACTIVE_DIMS`, and exclude end-padding. `masked_l1` divides by the number
of supervised action elements, with zero loss/gradients for an empty mask.
Validation aggregation and microbatch accumulation use that same element count.
Inactive-action defaults use training labels only (zero if a dimension has no
labels), so untrusted student arm actions cannot change those defaults. Proprio
statistics still include all training observations. Cache schema 3 stores masks
and invalidates older cache identities; use `--init-from` in a new run directory
to migrate older training runs. Old checkpoints remain valid for inference.

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
image file and a small `<hash>.npz` state/action/mask/privileged-label/offset index.
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

## BCE gripper fine-tune

`gripper_loss: l1` is the default and retains the old action loss and inference
sign convention. In `bce` mode the unchanged `H x 10` head predicts an **open
logit** at active dim 18 for each chunk step. The target is `action[:,18] < 0`.
The action loss becomes
`mean_valid_L1(other active dims) + gripper_weight * mean_valid_gripper_BCE`.
Both losses intersect label masks with nonpadding steps and use independent
denominators across the entire logical batch, including microbatches. Thus a
gripper-only sample contributes no arm loss and its gripper labels carry the
same weight as each fully labeled sample's gripper labels. An empty component
has zero loss/gradient. Auxiliary loss is still added separately when enabled.

The four saved config fields (also CLI flags) are:

| Setting | Default | Meaning |
| --- | --- | --- |
| `gripper_loss` | `l1` | `l1` regression or `bce` open logits |
| `gripper_weight` | `1.0` | Multiplier on the separately averaged BCE, nonnegative |
| `gripper_pos_weight` | `1.0` | BCE positive/open term multiplier, strictly positive |
| `gripper_threshold` | `0.5` | Open iff sigmoid(logit) is strictly greater, in [0,1] |

The weights/threshold must be finite. In L1 mode the three numeric fields do
not change the objective or action mapping. Old checkpoints missing all four
fields use L1, and `ADV_GRIPPER_THRESHOLD` is ignored for them.

`--init-from` **supports L1-to-BCE directly**: head dimensions and saved tensors
are identical. The dim-18 output is simply reinterpreted as a logit; its sign
meaning therefore changes (an inherited +0.93 becomes sigmoid(+0.93) ~0.72 open
probability). There is no sign flip, head reset or probability calibration.
The existing normalization rebasing, fresh optimizer/EMA and horizon/proprio
compatibility checks still apply. Use a fresh run directory to change loss
mode; resuming an L1 run as BCE is rejected. Older version-2 L1 training runs
can resume with the new defaults; their CSV headers are extended atomically,
leaving unavailable historical metrics blank.

With ambiguous frames, unweighted BCE learns the conditional open frequency;
at threshold 0.5 it can still predict closed for a rare open class. Positive
weighting and/or a dev-selected threshold can change that decision. Increasing
`gripper_pos_weight` favors open recall and may also cause premature releases;
neither the example weight nor threshold has been validated on real rollouts.

CPU fine-tune command, **provided but not executed**:

```bash
CUDA_VISIBLE_DEVICES='' ORT_DISABLE_TELEMETRY=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
UV_CACHE_DIR=/tmp/adv-bc-uv-cache uv run --locked --no-sync \
  /home/keyu/advaitic/mlenv/bin/python -m method.bc.train \
  --config method/bc/student_v1b_bce.yaml \
  --init-from /home/keyu/advaitic/ckpts/student_v1b.pt
```

`uv run --locked --no-sync` uses the committed project lock without changing it;
the explicit Python command uses the existing torch/torchvision model environment
(these packages are not in the evaluator lock). The config includes **human60 +
teacher_v1 + teacher_v1_dart0 + dagger + dagger2**, H=20/exec=10, 8k fresh steps,
batch 64/microbatch 8, zero auxiliary weight and zero workers. Source weights
`[1,0.68,0.32,0.5,0.5]` preserve v1b's original human/teacher ratio and allocate
one third of draws to the two DAgger directories together. Two episodes per
directory are held out: `dagger2` was still being collected (five complete
episodes on first check), so the usual six-episode holdout would reject that
snapshot. This changes the offline
validation split from v1b; losses are not directly comparable. Once every
source has at least seven episodes, a fresh run can use `--val-episodes 6`.
Finish collection and keep input files fixed before launching. Cache and run
output stay under `method/bc/`. Its
`gripper_weight=1`, `gripper_pos_weight=5`, `gripper_threshold=0.5` are an
experimental starting point; the default positive weight remains 1 elsewhere.
Omit `--init-from` with the same config to resume.

For BCE inference, temporal ensembling averages **raw logits**, then each
executed step opens to -1 iff its sigmoid exceeds the threshold; all other
steps close to +1. Classification happens before action clipping. Set
`ADV_GRIPPER_THRESHOLD` only for **dev-set sweeps**, for example:

```bash
CUDA_VISIBLE_DEVICES='' ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl \
ADV_GRIPPER_THRESHOLD=0.3 ADV_CKPT=method/bc/checkpoints/student_v1b_bce/last.pt \
UV_CACHE_DIR=/tmp/adv-bc-uv-cache uv run --locked --no-sync advaitic-eval \
  --policy submission.policy:make_policy \
  --policy-python /home/keyu/advaitic/mlenv/bin/python \
  --episodes 30 --seed-start 1000000 --method bc-student-v1b-bce-t03 \
  --out results/bc-student-v1b-bce-t03-dev-001
```

Use a new result directory per threshold/run and record the checkpoint identifier.
**The final submission must bake the chosen threshold into the checkpoint's
`config.gripper_threshold` (and matching saved config), unset the override, and
verify that artifact. It must not depend on `ADV_GRIPPER_THRESHOLD`.** Changing
only the YAML after training does not change an existing checkpoint. Threshold
sweeps, full union training and real release/success performance remain unverified.

## Model, losses and speed

- Shared ResNet-18 through layer3, ImageNet initialization, 32 spatial-softmax
  keypoints per camera, and a learned 16-dimensional camera embedding retain
  the existing visual architecture. BatchNorm statistics stay frozen.
- The 128-dimensional GELU proprio embedding joins all camera features. The
  existing two-layer 512-wide GELU/dropout action head predicts `H x 10` actions.
  A separate **linear 368-to-9 head** reads the same fused feature, encouraging
  observation features to encode privileged geometry without using it as input.
- Total optimization loss is the configured action loss (L1 or L1+BCE as above)
  plus `aux_weight * auxiliary_loss`.
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
`step,train_loss,val_loss,train_aux_loss,val_aux_loss,lr,it/s`, followed by
`train_l1,train_gripper_bce,train_gripper_accuracy,train_gripper_open_accuracy`
and the corresponding four `val_` columns. Action and auxiliary
losses are logged separately, before multiplying by `aux_weight`; auxiliary
loss is zero when disabled or no labels are available. Train action loss averages
batches since the last log; train auxiliary loss averages labeled dimensions.
Validation fields retain the latest result (NaN before first validation).
Throughput includes validation/checkpoint overhead. Logs append across resume;
entries after the last durable checkpoint can repeat following a crash.
L1/BCE metrics aggregate valid elements before division; BCE includes
`gripper_pos_weight` but excludes the outer `gripper_weight` multiplier. BCE is
zero when disabled or unlabeled. Gripper accuracy measures all valid gripper
labels; open accuracy measures only open-labeled elements (open recall). Both
use the saved threshold in BCE mode and the legacy zero/sign rule in L1 mode;
accuracy is NaN when its denominator is empty. Console pairs show train/val.
Checkpoints also retain the latest `val_metrics` dictionary.

The trainer reports allocated/reserved CUDA peaks and rejects reserved peak
>=3 GB. Existing human-only measurements were about 5 it/s, 0.43 GB peak; these
are **not student-v1 measurements**. Student GPU throughput/memory, CUDA numerics,
cuDNN benchmarking and pinned-memory behavior remain unverified here.

## CPU verification

```bash
CUDA_VISIBLE_DEVICES='' ORT_DISABLE_TELEMETRY=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
UV_CACHE_DIR=/tmp/adv-bc-uv-cache uv run --locked --no-sync \
  /home/keyu/advaitic/mlenv/bin/python -m method.bc.smoke_test --num-workers 0
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
The BCE additions check mixed full/gripper-only/arm-only/empty masks, separate
microbatch denominators and gradients, positive weighting, per-step strict
threshold boundaries, overrides (including invalid values and legacy ignoring),
classification before clipping, logit ensembling, BCE logging and empty BCE
batches. The 100-update synthetic output-head test has identical inputs with
1/32 open labels: weighted BCE (`pos_weight=62`) falls from 4.24 to 1.85 and
open recall rises from 0% to 100%, while L1 stays at 0% open recall. This
deliberately ambiguous toy predicts open for all inputs once weighted; it
demonstrates a working objective/gradient, not selective real release behavior.
The real `student_v1b.pt` also passed CPU policy loading/forward on a synthetic
observation and exact EMA initialization with unchanged normalization stats.

In the restricted implementation sandbox, the zero-worker smoke passes. The
2-worker attempt is blocked by socket permissions. Locked uv dependency sync is
unavailable offline; static checks instead use the already-cached Ruff binary
through uv without syncing:

```bash
UV_CACHE_DIR=/tmp/adv-bc-uv-cache uv run --locked --no-sync \
  /home/keyu/advaitic/robot-learning-assignment/.venv/bin/ruff \
  check method/bc submission
UV_CACHE_DIR=/tmp/adv-bc-uv-cache uv run --locked --no-sync \
  /home/keyu/advaitic/robot-learning-assignment/.venv/bin/ruff \
  format --check method/bc submission
```

## DAgger gripper relabelling

The student visits arm/vision states absent from demonstration data. In the
measured v1a run, ten plates reached the rack but the student stayed closed in a
settled pose for over 1,000 steps. Flipping state dimensions 7, 8, 44 and 45 did
not change its gripper prediction. Following the DAgger idea (Ross et al., 2011),
we label observations from student rollouts and train on the union. The teacher
cannot provide reliable arbitrary-state arm targets, so these new episodes
supervise **only gripper action dimension 18**. This experiment does not supply
new grasp-approach or arm corrections.

`method/dagger/collect.py` runs in the sim environment, calls the unchanged
`advaitic_eval.environment.make_environment()`, and uses the evaluator's
`PolicyProcess`/worker with `submission.policy:make_policy` in the model env.
That factory constructs the exact `BCPolicy` with the requested checkpoint.
Only RGB/state cross the worker boundary. Returned chunks, reset behavior,
thresholding and temporal ensembling use the evaluation path unchanged. Oracle
checks happen at **every pre-action control step**, including inside chunks;
labels belong to that observation, not the resulting next state. Sensors are a
local read-only copy of the teacher diagnostics; no cross-worktree imports,
controller overrides or pose injection are used.

Training-only oracle constants live in `method/dagger/diagnostics.py`:

| Label | Conditions at the observation being labelled |
| --- | --- |
| Open, `-1` | Touching the target rack, nearest of its six sites at distance ≤ 0.05 m, tilt ≤ 20° |
| Closed, `+1` | Held by a gripper, plate height ≥ reset height + 0.03 m, no target-rack contact |
| Undefined | Everything else; no action supervision |

Tilt uses the fixed task convention: angle of the rotated plate local +z to
world -y. Open takes priority. These three open predicates are exactly the
oracle; there are no extra table/floor exclusions. Undefined labels leave the
student action in the array but its mask is False. `label_mask` is bool `(T,20)`;
all dimensions other than 18 are always False. Each summary logs counts and
fractions of open/closed/undefined steps, plus initial/final geometry, maximum
lift, held-step count, first held/placed step, slot and termination.

Every rollout, including failure and early cutoff, is saved as
`DIR/ep_<seed>.npz`, with the teacher-compatible RGB/stride/state/action/seed/
slot/success keys and the new mask. `action` is the executed student action with
only defined dim-18 labels replaced. `action_executed` records what actually
entered the environment. Default execution is pure student; `--relabel-execute`
instead executes defined gripper labels, letting collection see post-release
states while continuing the student's arm chunks. The boolean mode is saved
in both NPZ and JSONL. Keep modes in separate output directories.

`--max-steps` defaults to 1,700. `--stop-after-placed N` defaults to 150 and
stops after N additional actions from the first state meeting the open condition.
Both must be positive. Natural environment termination takes precedence. A
cutoff is recorded as a training rollout cutoff; it cannot count as scored
success. Outcome precedence is `success`, `fell` (robot fall or plate touching
floor), `no_grasp` (maximum lift < 3 cm), `placed_not_released` (final open
geometry and still held), `reached_target_then_failed` (open geometry visited
earlier), then `other`. These are broad replay-style diagnostic categories;
raw geometry is retained to make them auditable. A cutoff shortly after
placement can be categorized as reached-target failure even after release.

NPZ publication is atomic and precedes the shared `summary.jsonl` append.
Completed files embed their summary so resume repairs an interrupted log append.
Per-seed locks prevent duplicate collection and a short JSONL lock serializes
appends across OS processes. Rerunning the same command skips completed files;
an interrupted episode restarts from reset. Checkpoint/source hashes and rollout
settings in `collector_config.json` prevent mixing configurations. Incompatible
files or missing previously logged episodes fail explicitly. The seed parser
uses Python's stop-exclusive `A:B` convention, refuses all seeds below
2,000,000, and conservatively limits collection to `[2000000,3000000)` so dev
and reviewer/public evaluation blocks cannot be collected accidentally.

Exact two-process command for **2400000 through 2400060 inclusive (61 seeds)**,
from the repository root, with CPU software rendering and CPU inference:

```bash
mkdir -p /home/keyu/advaitic/data/dagger
CUDA_VISIBLE_DEVICES='' ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl \
LIBGL_ALWAYS_SOFTWARE=1 EGL_PLATFORM=surfaceless OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/keyu/advaitic/robot-learning-assignment/.venv/bin/python -m method.dagger.collect \
  --checkpoint /home/keyu/advaitic/ckpts/student_v1a.pt \
  --policy-python /home/keyu/advaitic/mlenv/bin/python \
  --seeds 2400000:2400030 --out /home/keyu/advaitic/data/dagger \
  --max-steps 1700 --stop-after-placed 150 \
  > /home/keyu/advaitic/data/dagger/process_a.log 2>&1 &
dagger_pid_a=$!
CUDA_VISIBLE_DEVICES='' ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl \
LIBGL_ALWAYS_SOFTWARE=1 EGL_PLATFORM=surfaceless OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/keyu/advaitic/robot-learning-assignment/.venv/bin/python -m method.dagger.collect \
  --checkpoint /home/keyu/advaitic/ckpts/student_v1a.pt \
  --policy-python /home/keyu/advaitic/mlenv/bin/python \
  --seeds 2400030:2400061 --out /home/keyu/advaitic/data/dagger \
  --max-steps 1700 --stop-after-placed 150 \
  > /home/keyu/advaitic/data/dagger/process_b.log 2>&1 &
dagger_pid_b=$!
wait "$dagger_pid_a"
wait "$dagger_pid_b"
```

For 60 episodes with exclusive endpoint 2400060, change only the second stop to
`2400060`. To collect post-release interventions instead, add
`--relabel-execute` to **both** commands and choose a fresh output directory;
set the training config's DAgger directory to that directory.

After both collectors finish successfully, fine-tune the union with:

```bash
CUDA_VISIBLE_DEVICES='' ORT_DISABLE_TELEMETRY=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/keyu/advaitic/mlenv/bin/python -m method.bc.train \
  --config method/dagger/student_v1a_dagger.yaml \
  --num-workers 0 \
  --init-from /home/keyu/advaitic/ckpts/student_v1a.pt
```

The config retains v1a's human60/teacher_v1 directories and adds `data/dagger`.
Equal `[1,1,1]` source sampling preserves the original human:teacher ratio while
giving student states one third of draws. Loss still averages **labelled
elements**, so sampling mass is not identical to gripper loss mass. It uses
8k fresh steps, H=20/exec=10, aux weight 0 (as v1a), CPU and a fresh run directory.
Six validation episodes are held out **per directory**, including DAgger.
Entirely undefined batches have zero action loss and skip optimizer/EMA updates
when auxiliary labels are also absent, avoiding unsupervised Adam/decay changes.
The command uses zero data-loader workers because this implementation sandbox
blocks the local sockets needed by PyTorch's two-worker loader. The config
retains two workers for an unrestricted host; omit the override there.
To resume training, omit `--init-from`. No new privileged input enters inference.

Smoke checks (no task-solving claim):

```bash
ORT_DISABLE_TELEMETRY=1 /home/keyu/advaitic/robot-learning-assignment/.venv/bin/python \
  -m method.dagger.smoke_test
CUDA_VISIBLE_DEVICES='' ORT_DISABLE_TELEMETRY=1 /home/keyu/advaitic/mlenv/bin/python \
  -m method.bc.smoke_test
```

The collector smoke uses fake observations/sensors to verify threshold
boundaries, masks, action modes, chunk cadence, pre-action frame alignment,
cutoffs, atomic saves and interrupted JSONL recovery. The BC smoke includes a
fake episode with per-step partial masks, zero gradients outside its labels,
inactive-dimension exclusion, padding, empty-mask loss, microbatch equivalence,
masked action defaults and an entirely unlabelled training batch.

Implementation verification passed Ruff check/format, both synthetic smokes,
and real sim/model-worker collection with student_v1a on CPU software EGL:
one 2-step default-mode rollout and two concurrent 12-step intervention-mode
rollouts into one directory. Resume skipped both saved files without duplicate
summaries; their NPZ shapes/modes and BC dataset loading were checked. These
short real rollouts had only undefined oracle labels. Physical release behavior,
full 61-episode collection, union retraining and subsequent dev performance are
unverified. The two-worker BC smoke was interrupted after sandbox socket
`PermissionError`; the full zero-worker smoke, including exact resume, passed.

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
