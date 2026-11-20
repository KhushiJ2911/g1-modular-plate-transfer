# microVLA flow-matching student v1

This is a second student architecture alongside the BC student; the existing
BC policy remains available. The starting BC result supplied with the brief is
32/60 (53%) on dev plus fresh seeds. Its observed release, placement and slot-5
grasp failures motivate this experiment; they are not microVLA results.

Flow matching learns a conditional action distribution and samples from it.
A deterministic L1/L2 head can regress to a median/mean when the same observed
situation has multiple plausible actions. A flow model can represent both modes,
so a rare gripper-open versus gripper-closed release decision need not be averaged
away. This is a hypothesis to test: imbalance can still suppress a rare mode,
and sampling can select the wrong one. A transformer over patch tokens from all
three cameras may also localize the plate and rack more precisely than spatial
softmax. No real-data training or closed-loop success has been demonstrated here.

## Source and architecture

`vendor/train.py` and `vendor/LICENSE` are the unchanged MIT source from
[KhushiJ2911/microVLA](https://github.com/KhushiJ2911/microVLA), copyright 2026
KhushiJ2911. Top-level imports work in `/home/keyu/advaitic/mlenv`; `model.py`
imports `Config`, `MicroVLA` (and thus RMSNorm, GeGLU, Vision, attention_mask,
rope, Expert and JointBlock), tokenizer and augmentations directly. No classes
were copied. The adapter uses its own masked objective instead of vendor `loss()`.

| Choice | Reason |
| --- | --- |
| Resize 84 → 96 with antialiased bilinear interpolation | Preserve the entire rack/plate view; patch8 gives a 12×12 grid without border loss. |
| Three cameras, shared ViT and learned camera embeddings | Fuse head/right-wrist/left-wrist observations with camera identity. |
| Vendor default widths/depths | Keep this an architecture test with an unchanged upstream backbone. |
| Logical batch 64, microbatch 2 | Limit transformer activation memory on the 4 GB RTX 3050. |
| H=20, execute 10 | Match the BC action chunk and observation cadence. |
| Continuous gripper in the flow | Test distribution learning instead of a separate BCE head. |
| EMA 0.999 | Smooth the model used for validation and inference. |
| SigLIP off by default | Avoid downloading weights or consuming disk; local weights are optional. |

Default model: **11,198,026 parameters** (vision 1,893,952; prefix backbone
5,704,704; action expert 2,265,600; embeddings/heads 1,333,770).
Per sample: **432 image + 11 language + 19 state = 462 prefix tokens**, plus
20 bidirectional action tokens = **482 total**. An 80 px center crop would
use 300 image/350 total tokens and about 53% of the dense joint-attention score
storage, but removes two pixels at every border; preserving the view is the v1
choice. The 11-word instruction is always
`move the plate from the left rack to the right rack`. The learned language
path remains, but the constant text carries no information for this single task.

## Data, normalization and masks

The v1 YAML uses exactly the five BC BCE source directories and source masses
`[1, 0.68, 0.32, 0.5, 0.5]`; sampling, numeric episode ordering, two held-out
episodes per source, RGB stride alignment and clean action targets come from
`method/bc/dataset.py`. Its schema-3 cache is opened **read-only**, without even
opening a writable lock. A small optional `read_only` argument was added to that
loader; the default BC cache-building behavior is unchanged.

The loader requires its existing `rgb_stride=2` format (RGB at `2*j`, all action
steps included). Other strides fail explicitly. NPZ input RGB is uint8
`(ceil(T/2),3,3,84,84)`, state `(T,50)`, action `(T,20)`, optional bool
`label_mask (T,20)`. No real image cache is built, updated or resized on disk.

Active actions are `(0,1,4,5,6,7,8,9,10,18)`. As in BC they retain the dataset's
normalized controller values in `[-1,1]`: action center **0**, scale **1**, no
extra whitening. Gripper targets remain continuous ±1. Inference executes open
`-1` if the sampled gripper is `<0`, otherwise closed `+1`. The other ten
controller dimensions use BC's trusted-label training action means.

Proprio indices are
`[0,1,2,3,4,5,6,7,8,18,19,20,21,40,41,42,43,44,45]`.
BC computes the per-dimension mean μ and std σ from **all control steps in
training episodes**, without source-weight reweighting; σ is floored at 0.001.
The adapter converts BC z scores to `clip((state-μ)/(3σ), -1, 1)`, preserving
three standard deviations before saturating. Vendor binning is
`min(floor((clipped+1)*128),255)` with a separate 256-bin token range for each
state dimension. Checkpoints store μ, σ, the factor 3 and the tokenizer.
The report includes the current train-split numeric normalization preview.

For mask `M = label_mask & ~pad_mask[...,None]`, training draws Gaussian ε and
`t ~ Beta(1.5,1)` scaled to `[0.001,1]`. It interpolates
`x_t = t*ε + (1-t)*action` on labeled elements; unlabeled inputs are pure noise,
so untrusted DAgger targets never enter the action context. Target velocity is
`ε-action`; loss is `sum(M*(v-target_velocity)^2)/sum(M)`. Implementation uses
`where` to discard masked values, even masked NaNs. End-padding and DAgger
arm/base targets contribute zero. Microbatches share the **logical batch's**
valid-element denominator. An entirely unlabeled batch skips Adam/decay/EMA.

Random shifts (replicate padding 4) and small brightness/contrast jitter are
imported from the vendor; jitter is shared across a sample's cameras.
Training/validation/inference use the same image interpolation and state bins.

## Training and evaluation commands

Run from `/home/keyu/advaitic/wt-vla`. Keep the root committed lock and the
existing model environment; no dependency installation is required for v1.
`requirements.txt` records model dependencies, including optional transformers.
`uv --active --frozen --no-sync` checks project lock compatibility while using
that already provisioned environment without writing to it.

```bash
export VIRTUAL_ENV=/home/keyu/advaitic/mlenv
export UV_CACHE_DIR=/tmp/microvla-uv
export PYTHONDONTWRITEBYTECODE=1
export ORT_DISABLE_TELEMETRY=1

# CPU smoke; all data/checkpoints below are temporary synthetic fixtures.
CUDA_VISIBLE_DEVICES='' uv run --active --frozen --no-sync python -B -m method.vla.smoke_test

# CPU real-data recipe, only after an exact matching BC cache is available.
CUDA_VISIBLE_DEVICES='' uv run --active --frozen --no-sync python -B -m method.vla.train \
  --config method/vla/microvla_v1.yaml --device cpu

# Same recipe after interruption; config and data identity must match.
CUDA_VISIBLE_DEVICES='' uv run --active --frozen --no-sync python -B -m method.vla.train \
  --config method/vla/microvla_v1.yaml --device cpu --resume
```

The default YAML device is CUDA for later GPU experiments; remove the
`CUDA_VISIBLE_DEVICES=''` prefix and use `--device cuda` in a GPU session.
No GPU was accessed during implementation. CUDA uses bf16 autocast, fp32
parameters/AdamW, gradient clipping 1, EMA 0.999, a 500-step warmup and cosine LR.
CPU uses fp32/two threads. Default 10k action steps, batch 64/microbatch 2.

`last.pt` is fsynced and atomically replaced every 250 steps and at completion.
Only one checkpoint is retained to conserve disk. It includes optimizer, EMA,
phase, step, RNG, data fingerprint, normalization, model config and split paths.
`--resume` also supports interruption during distillation. Changing training
settings/data on resume fails; run directory, device, cache path and worker count
may change. JSONL logs hold train flow loss, full held-out flow loss every 250
steps, and sampled L1 by controller dimension plus gripper sign accuracy overall
and on open labels every 500 steps. Sample diagnostics use two evenly spaced
held-out batches; counts and null metrics expose absent labels. `--val-batches N`
can cap flow validation for a development run (default 0 means all held-out
frames). Validation preserves training RNG, and all reported losses aggregate
sums/counts rather than batch averages.

Optional local-only vendor-style SigLIP distillation (new run directory):

```bash
CUDA_VISIBLE_DEVICES='' uv run --active --frozen --no-sync python -B -m method.vla.train \
  --config method/vla/microvla_v1.yaml --device cpu --distill --distill-steps 1000 \
  --siglip-path /path/to/provisioned/siglip-base-patch16-224 \
  --run-dir method/vla/checkpoints/microvla_v1_distilled
```

This phase distills one random camera using normalized teacher patch features,
cosine plus MSE loss and a trainable projection. Teacher patch16 at 192 px and
student patch8 at 96 px share the 12×12 patch grid; position interpolation is
explicit. Loading uses `local_files_only=True`; no weights are downloaded.
The real teacher path and its CUDA memory use remain unverified.

Evaluation uses the frozen evaluator environment and a separate model process:

```bash
ADV_CKPT="$PWD/method/vla/checkpoints/microvla_v1/last.pt" \
ADV_FLOW_STEPS=10 CUDA_VISIBLE_DEVICES='' ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl \
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/microvla-uv \
uv run --frozen --no-sync advaitic-eval \
  --policy submission.policy_vla:make_policy --policy-root "$PWD" \
  --policy-python /home/keyu/advaitic/mlenv/bin/python \
  --method microvla_v1 --checkpoint microvla_v1-last \
  --episodes 60 --seed-start 8000000 --out method/vla/results/dev60
```

Use an evaluator environment already provisioned from `uv.lock` for that command
(unset the model `VIRTUAL_ENV` if needed). This worktree has no provisioned
evaluator environment; the command is documented and was **not run**. The output
directory must be new. `ADV_FLOW_STEPS` is a development sweep override; default
10 Euler steps is stored in the config. Policy reset reinstates a fixed CPU RNG
seed each episode; each chunk draws new noise from that generator. Each chunk's
image/state/language prefix KV cache is built once and reused for all Euler steps.
CUDA is selected only when available, otherwise inference runs on CPU.

## CPU smoke evidence and resource budget

The smoke test covers a rare open mode (one of nine labels), partial DAgger
masks, padding, masked NaN invariance, zero masked-target gradients, exact
microbatch-gradient weighting, empty-batch optimizer/EMA behavior, tiny overfit,
cached/uncached Euler agreement, saved-policy loading/action mapping/reset,
changed-resume rejection, bitwise model/EMA/Adam/RNG resume equivalence, and an
optional-phase restart with a synthetic teacher. No pose injection or simulator
rollout occurs. Fake NPZ/cache/checkpoints live only in a new temporary directory
and are removed afterward.

Measured tiny overfit loss: **1.623354 → 0.002293** (45 fixed-noise/time updates).
This is a numerical wiring check, not evidence of rare-mode learning, gripper
release success or the 90% policy target.

The full model's CPU microbatch-2 forward/backward measured approximately
0.38–0.48 s on two threads, implying approximately 0.065–0.083 logical batch-64 steps/s
(~12–15 s/step) before Adam, augmentation, I/O and validation. This is a short CPU
extrapolation, not CUDA throughput. Static fp32 model+EMA+gradient+Adam storage
is **0.209 GiB**; a dense joint-attention score tensor across six layers at
microbatch 2 is **42.54 MiB**, excluding other activations/backward/workspaces.
A conservative planning range is **0.6–1.8 GiB training VRAM**, leaving a nominal
2.2+ GiB on a 4 GiB card for evaluation. CUDA allocator/context and evaluator
peaks are unmeasured; this range is not a fit guarantee. Lower microbatch size to
1 in a new run if needed; accumulation retains batch 64. Peak allocated/reserved
CUDA memory is printed by real training at completion. SigLIP adds a substantial
teacher allocation and is excluded from this budget.

## Current data prerequisite

Read-only audit found 372 current episodes, 362 train/10 validation and 306,696
training control rows. Existing cache `1946b0d61101559f44a9df77` has the first
four sources (322 episodes); `fc0f42f086d17f18facd685b` has three older sources.
Neither includes dagger2. The exact current five-source cache key is
`52bdf4d0bc2540b9f4050a6a`, absent in `/home/keyu/advaitic/data/cache`.
The default training loader therefore fails with a precise missing-cache error.
Point `--cache-dir` at an already matching schema-3 cache if one is provisioned;
creating one would require a separate authorized data/cache-preparation task
and additional disk. No current data/cache directory was changed here.
