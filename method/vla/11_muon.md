# Brief 11: Muon and a larger microVLA

Implemented and CPU-validated on 2026-11-21 in `/home/keyu/advaitic/wt-muon`.
Requested destination: `/home/keyu/advaitic/reports/11_muon.md`. That directory
is outside this session's writable roots (worktree and `/tmp`), so the complete
report is saved as `method/vla/11_muon.md` instead. No approval is available in
this session. No attempt was made to bypass the filesystem restriction.

## Files changed

- `method/vla/muon.py`: new Newton–Schulz update, explicit parameter split,
  serialisable hybrid optimizer, group counts and startup table.
- `method/vla/train.py`: optimizer config/validation, both-phase integration,
  group scheduling and backward-compatible resume validation.
- `method/vla/microvla_large.yaml`: 35.33M model recipe, 96 px, shared LR 6e-4,
  Muon, batch/microbatch 64, 20,000 flow steps and 3,000 optional distill steps.
- `method/vla/test_muon.py`: nine CPU tests using fake tensors and an in-memory
  store; temporary checkpoints, no image cache.
- `method/vla/smoke_large.py`: reproducible CPU count/forward/backward/update
  and memory-accounting smoke.
- `method/vla/README.md`: optimizer semantics, large recipe and commands.
- `method/vla/11_muon.md`: this report.

No changes to `src/`, `tests/`, `pyproject.toml`, `uv.lock`, or
`method/vla/vendor/train.py`; pre/post SHA-256 comparison confirmed all 19 protected files
unchanged. No git commands, downloads, GPU work, real-data training, evaluator
changes or policy rollouts. The real data directories were not written.
`uv --no-cache` avoids the persistent uv cache, `-B` disables Python bytecode,
and all synthetic checkpoints were under a temporary `/tmp` directory.

## Design choices

Credit: [Keller Jordan's Muon](https://github.com/KellerJordan/Muon) and
[Moonlight: Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982).
Implementation follows the formulas specified in the brief, without adding
packages or requiring `torch.optim.Muon` (environment: torch 2.5.1+cu121).

| Choice | Reason |
| --- | --- |
| Nesterov: `b = .95*b + .05*g`, then `u = .05*g + .95*b` | Apply orthogonalisation to the lookahead direction; leave accumulated gradients intact. |
| Quintic coefficients `(3.4445, -4.7750, 2.0315)`, 5 iterations | Use the requested standard approximate polar update. |
| Frobenius normalization with epsilon 1e-7, transpose to rows <= cols | Stabilize the polynomial and form the smaller Gram matrix. |
| fp32 CPU, bf16 CUDA, local autocast disabled during iteration | Make arithmetic dtype explicit, even under an ambient autocast context. |
| `0.2 * sqrt(max(rows, cols))` per projection | Moonlight update-RMS matching permits a shared AdamW-scale LR and weight-decay schedule. |
| Decoupled `p *= 1 - lr*wd` before each Muon update | Apply decay consistently to Muon parameters too. |
| Explicit block/projection whitelist | Keep embeddings, heads, norms and other 2-D tensors off Muon. |
| Split ViT fused Q/K/V into equal thirds | Avoid orthogonalising independent projections together. |
| Split vendor grouped-query Q/K/V as `(heads*head_dim, head_dim, head_dim)` | Respect its unequal projection sizes and apply matching per projection. |
| Adaptive RMSNorm modulation uses AdamW | Its zero-initialised scale/shift/gate projections control conditioning rather than feature mixing. |
| All AdamW fallback parameters retain uniform configured weight decay | Preserve the existing recipe, including biases/norms. |
| `optimizer: adamw` remains the default, unchanged parameter order/AdamW state format | Keep default behavior and legacy checkpoint loading intact. |
| One hybrid state dict, with Muon momentum and scalar step per matrix | Restore both optimizer algorithms through ordinary optimizer checkpoint loading. |
| Muon uses warmup/cosine in each phase with that phase's length | Give both Muon and fallback groups identical multipliers, including an explicit `muon_lr` override. |
| Default AdamW distillation keeps its previous flat LR | Preserve existing AdamW behavior; its flow schedule is unchanged. |
| Optimizers reset at distill-to-flow transition | Preserve existing phase semantics; phase-local resume still restores buffers. |
| Keep images at 96 px | Follow the supplied evaluation finding that 128 px was worse. |

Config defaults: `optimizer: adamw`, `muon_lr: null`, `muon_momentum: 0.95`,
`muon_ns_steps: 5`. `muon_lr` is accepted only with Muon and overrides Muon
matrices only; `lr` still controls AdamW fallback. Both rates are multiplied
by the same warmup/cosine factor. The large recipe sets shared `lr: 0.0006`;
this is a starting recommendation, not a completed real-data tuning result.

For every phase, startup prints group name, optimizer, tensor/parameter count,
and per-optimizer totals. Checkpoints store all config fields. A legacy
checkpoint missing just the four new fields receives their original AdamW
defaults. Any optimizer, LR, momentum, iteration-count or other training-config
change on resume is refused before data loading. Device/run/cache/worker
exceptions remain as before.

The standard five-step quintic is approximate: its singular values need not
be exactly one, and no 0.6 lower bound is possible for rank-deficient or
sufficiently ill-conditioned input. The tests verify random seeded square
and rectangular examples and explicitly check the rank-deficient exception.

## Exact architecture and optimizer counts

| Component | v1 | Large |
| --- | ---: | ---: |
| Vision | 1,893,952 | 4,924,032 |
| Prefix transformer | 5,704,704 | 16,914,432 |
| Action expert | 2,265,600 | 11,284,480 |
| Embeddings and heads | 1,333,770 | 2,207,114 |
| Total | 11,198,026 | 35,330,058 |

Large overrides: ViT width/depth/heads 256/6/4; prefix width/depth/heads/head_dim
384/8/6/64; prefix MLP 1536; expert width/MLP 256/1024. No adjustment from the
suggested dimensions was needed. ViT width divides by 4 and prefix width by 6;
the vendor expert projects its 256-wide features to the joint 6x64 attention
space, so expert_width need not divide by heads.

Both models use 432 image + 11 language + 19 state = **462 prefix tokens**,
plus 20 actions = **482 total tokens per sample** (three cameras, patch8).

Flow-phase hybrid split (tensor counts refer to actual stored parameters;
fused QKV is one tensor with three independently orthogonalised row blocks):

| Optimizer | Group | v1 tensors | v1 parameters | Large tensors | Large parameters |
| --- | --- | ---: | ---: | ---: | ---: |
| adamw | embeddings_heads/fallback | 12 | 1,333,770 | 12 | 2,207,114 |
| adamw | vision/fallback | 39 | 124,480 | 55 | 205,440 |
| muon | vision/qkv | 4 | 442,368 | 6 | 1,179,648 |
| muon | vision/matrices | 12 | 1,327,104 | 18 | 3,538,944 |
| adamw | prefix/fallback | 12 | 3,072 | 16 | 6,144 |
| muon | prefix/qkv | 6 | 589,824 | 8 | 1,572,864 |
| muon | prefix/matrices | 18 | 5,111,808 | 24 | 15,335,424 |
| adamw | expert/fallback | 24 | 594,432 | 32 | 3,158,016 |
| muon | expert/qkv | 6 | 294,912 | 8 | 1,048,576 |
| muon | expert/matrices | 18 | 1,376,256 | 24 | 7,077,888 |
| **muon** | **TOTAL** | **64** | **9,142,272** | **88** | **29,753,344** |
| **adamw** | **TOTAL** | **87** | **2,055,754** | **115** | **5,576,714** |

Everything else uses AdamW: patch convolution, vision project, positional and
camera embeddings, language/state tokens, time MLP, action input/output
projections, final norm, all block norms/biases and adaptive modulation.
With `optimizer: adamw`, the sole legacy group contains all 11,198,026 / 35,330,058
flow parameters respectively.

Distillation split, assuming the provisioned SigLIP base teacher has hidden
width 768 (the actual head is sized from the loaded teacher config):

| Optimizer | Group | v1 parameters | Large parameters |
| --- | --- | ---: | ---: |
| adamw | vision/fallback | 124,480 | 205,440 |
| muon | vision/qkv | 442,368 | 1,179,648 |
| muon | vision/matrices | 1,327,104 | 3,538,944 |
| adamw | distill_head/fallback | 197,376 | 295,680 |
| **muon** | **TOTAL** | **1,769,472** | **4,718,592** |
| **adamw** | **TOTAL** | **321,856** | **501,120** |
| All | Total optimized | 2,091,328 | 5,219,712 |

Only vision and its distillation head are optimized in this phase; prefix,
expert and teacher are outside the optimizer. Ordinary AdamW uses one group
containing those same 2,091,328 / 5,219,712 parameters.

## CPU verification

All nine unit tests passed in approximately 2.1 seconds on two CPU threads:

1. Seeded random square/rectangular Newton–Schulz singular values, explicit
   fp32 under CPU bf16 autocast, zero input and input immutability.
2. Scaled-update entrywise RMS against an actual first bias-corrected AdamW
   step, excluding LR and weight decay.
3. Fused equal and unequal Q/K/V splits exactly equal separate updates.
4. Every phase's optimized tensor appears exactly once; no missing tensors;
   embeddings, heads, norms, biases and modulation excluded from Muon.
5. Two-step manual Nesterov/decay agreement, gradients untouched, serialized
   momentum restore, exact next-step equality, missing/zero-gradient behavior.
6. Warmup/cosine endpoints and relative `muon_lr` in both phases; unchanged
   default AdamW state format and legacy-config resume acceptance.
7. Invalid optimizer/LR/momentum/iteration settings rejected.
8. Fixed 30-step tiny flow overfit, including a matched-LR control.
9. Actual `train()` interruption/resume using an in-memory fake store and
   synthetic teacher: AdamW and Muon; mid-distillation, exactly at transition,
   mid-flow; bitwise model/EMA/optimizer/torch-RNG agreement; optimizer changes
   rejected before touching data. AdamW resumes also remove the four new config
   keys on disk to exercise legacy checkpoint compatibility.

Newton–Schulz singular-value measurements:

| Random shape (seed 42) | Min | Max |
| --- | ---: | ---: |
| 8 x 8 | 0.695844 | 1.133886 |
| 16 x 16 | 0.682252 | 1.098630 |
| 32 x 32 | 0.684737 | 1.134333 |
| 8 x 32 | 0.695264 | 1.133025 |
| 32 x 8 | 0.725067 | 1.134274 |

RMS measurement: entrywise RMS of `delta_parameter/lr` on the **first AdamW
step** with iid Gaussian gradients and zero weight decay. Bias correction
makes this direction approximately `sign(g)`, RMS 1. This is the reproducible
unit-test proxy for the AdamW update scale; later AdamW RMS is data-dependent.
Muon is measured after orthogonalisation and Moonlight scaling, before LR or
decay; tolerance is 0.2 +/- 0.04 relative to this AdamW reference.

| Shape | Muon RMS | AdamW RMS |
| --- | ---: | ---: |
| 8 x 8 | 0.192587 | 1.000000 |
| 8 x 32 | 0.195072 | 1.000000 |
| 32 x 8 | 0.199030 | 1.000000 |

Tiny overfit: same initialization, four fake samples, three 16 px cameras,
19 state dimensions, four-action horizon, fixed noise/time, full supervision,
no augmentation or decay scheduling, clip norm 1, 30 updates. AdamW LR is the
v1 default **3e-4**; the candidate Muon shared LR is **6e-4**. The latter stays
at or below the AdamW curve at every recorded step and finishes at **0.896538**
versus **1.191233** from the same initial **1.536023**. A matched-LR Muon control
at **3e-4** finishes at **1.211227**, slightly worse than AdamW. The faster
candidate therefore demonstrates numerical wiring with a different LR, not
an isolated optimizer advantage or real task learning.

Full curves (loss after the indicated number of updates):

| Update | AdamW 3e-4 | Muon 3e-4 control | Muon 6e-4 candidate |
| ---: | ---: | ---: | ---: |
| 0 | 1.536023 | 1.536023 | 1.536023 |
| 1 | 1.525096 | 1.525096 | 1.514247 |
| 2 | 1.514191 | 1.514205 | 1.492617 |
| 3 | 1.503297 | 1.503343 | 1.471105 |
| 4 | 1.492409 | 1.492508 | 1.449700 |
| 5 | 1.481522 | 1.481695 | 1.428392 |
| 6 | 1.470632 | 1.470903 | 1.407171 |
| 7 | 1.459733 | 1.460129 | 1.386028 |
| 8 | 1.448820 | 1.449371 | 1.364952 |
| 9 | 1.437889 | 1.438627 | 1.343934 |
| 10 | 1.426934 | 1.427893 | 1.322963 |
| 11 | 1.415948 | 1.417167 | 1.302030 |
| 12 | 1.404925 | 1.406447 | 1.281123 |
| 13 | 1.393860 | 1.395730 | 1.260230 |
| 14 | 1.382744 | 1.385013 | 1.239340 |
| 15 | 1.371572 | 1.374293 | 1.218443 |
| 16 | 1.360336 | 1.363568 | 1.197525 |
| 17 | 1.349027 | 1.352835 | 1.176577 |
| 18 | 1.337638 | 1.342092 | 1.155587 |
| 19 | 1.326161 | 1.331334 | 1.134544 |
| 20 | 1.314586 | 1.320559 | 1.113435 |
| 21 | 1.302905 | 1.309765 | 1.092249 |
| 22 | 1.291107 | 1.298948 | 1.070975 |
| 23 | 1.279183 | 1.288105 | 1.049602 |
| 24 | 1.267122 | 1.277233 | 1.028121 |
| 25 | 1.254912 | 1.266329 | 1.006522 |
| 26 | 1.242543 | 1.255391 | 0.984798 |
| 27 | 1.230003 | 1.244414 | 0.962942 |
| 28 | 1.217280 | 1.233397 | 0.940949 |
| 29 | 1.204361 | 1.222335 | 0.918815 |
| 30 | 1.191233 | 1.211227 | 0.896538 |

Large smoke passed: exact counts/tokens above; batch-1 CPU fp32
forward/backward **0.535 s**, finite
loss **1.340710**, all present gradients finite, followed by a finite
hybrid optimizer update on the actual large matrices. Four final-prefix
parameters have no gradient because the vendor's last prefix finish output is
unused by the action path: `layers.7.prefix.ffn_norm.scale`, `out.weight`,
`ffn.up.weight`, `ffn.down.weight`. They remain correctly assigned to optimizer
groups; parameters with no gradient receive no decay/update, matching AdamW.
The vendor architecture was left unchanged.

Ruff lint and format checks were run on the four changed/new Python files,
using the already installed evaluator-environment Ruff binary, with
`--no-cache`. No evaluator tests were needed because evaluator files were not
changed. The pre-existing full smoke creates a synthetic image cache, so it was
not run under this brief's prohibition on writing any cache directory; the
new tests exercise optimizer training/resume entirely in RAM instead.

Reproduce CPU checks from the worktree:

```bash
export VIRTUAL_ENV=/home/keyu/advaitic/mlenv
export PYTHONDONTWRITEBYTECODE=1 ORT_DISABLE_TELEMETRY=1
CUDA_VISIBLE_DEVICES='' uv run --active --frozen --no-sync --offline --no-cache \
  --no-python-downloads python -B -m unittest method.vla.test_muon -v
CUDA_VISIBLE_DEVICES='' uv run --active --frozen --no-sync --offline --no-cache \
  --no-python-downloads python -B -m method.vla.smoke_large
/home/keyu/advaitic/robot-learning-assignment/.venv/bin/ruff check --no-cache \
  method/vla/muon.py method/vla/train.py method/vla/test_muon.py method/vla/smoke_large.py
/home/keyu/advaitic/robot-learning-assignment/.venv/bin/ruff format --check --no-cache \
  method/vla/muon.py method/vla/train.py method/vla/test_muon.py method/vla/smoke_large.py
```

## L4 memory estimate: batch 64, microbatch 64, bf16 autocast

Parameters, EMA, gradients and optimizer state remain fp32; bf16 is used for
autocast activations/matmuls and CUDA Newton–Schulz. Full-state accounting is
`12*P + 4*Muon_params + 8*AdamW_params` bytes = **0.547 GiB**,
where 12 covers model/EMA/gradient, and moment buffers cost 4 versus 8 bytes.
This is conservative for parameters with no gradient.

Saved-tensor hooks on the batch-1 CPU fp32 forward counted **204,006,472 bytes**
of distinct saved nonparameter storage. Multiplying by 64 and halving for bf16
gives a rough **6.08 GiB** activation proxy. It is not a CUDA measurement:
explicit fp32 RMSNorm/rope operations, autocast weight copies, kernel choices,
attention masks, workspaces, allocator fragmentation and backward temporaries
change the peak. Budget **9–14 GiB** for flow with memory-efficient attention.

A single dense fp32 joint attention-score tensor across all eight layers at
batch 64 costs **2.66 GiB** (`64*6*482^2*8*4`). Retaining multiple such
tensors under a math-attention fallback could move the peak toward **15–22 GiB**.
An L4 advertised as 24 GB has about 22.35 GiB, so the high end has little
headroom. Distillation uses one camera and a frozen teacher, but actual SigLIP
memory has not been measured and is excluded from the flow estimate. No
fit/throughput guarantee is claimed. If needed, start a new run with
`--microbatch-size 16` and keep `--batch-size 64` for exact accumulation.
Real training prints peak allocated/reserved CUDA memory on completion.

## Recommended three-point sweeps and final run

**Muon shared LR sweep: 3e-4, 6e-4, 9e-4. AdamW baseline: 1.5e-4, 3e-4, 6e-4.**

Moonlight matching removes the need for the much larger LR often used by
unscaled Muon. Include the proven recipe's 3e-4 anchor, test a modest 2x
increase supported by the tiny overfit, and cap the first Muon sweep at 3x:
its fallback still trains embeddings, gates and heads with AdamW, so raising
the shared LR to several e-3 would also raise those sensitive components.
The baseline brackets its existing 3e-4 setting by a factor of two and overlaps
two Muon settings. No fixed 5x LR multiplier follows from the first-step RMS
ratio because mature AdamW updates, preconditioning and momentum differ.
Keep `muon_lr` unset for this first comparison. The 30-step fake-data result
only motivates the candidate; it does not select a real-data optimum.

Screen v1 for 5,000 flow updates per point, same seed/data/batch/schedule, no
distillation, full held-out flow loss. Use held-out loss and sampled action/
gripper diagnostics to shortlist, then the unchanged evaluator for real policy
performance; lower flow loss alone does not establish success. All commands
below are for the later Modal L4 and were **not run here**. They require the
existing matching schema-3 cache and provisioned model environment. Do not
build a cache as part of training. Use fresh run directories.

```bash
cd /home/keyu/advaitic/wt-muon
export VIRTUAL_ENV=/home/keyu/advaitic/mlenv
export PYTHONDONTWRITEBYTECODE=1 ORT_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
unset CUDA_VISIBLE_DEVICES

# Three Muon points, warmup 500 + cosine, shared LR for both algorithms.
for rate in 0.0003 0.0006 0.0009; do
  uv run --active --frozen --no-sync --offline --no-cache --no-python-downloads \
    python -B -m method.vla.train --config method/vla/microvla_v1.yaml \
    --optimizer muon --muon-momentum 0.95 --muon-ns-steps 5 --lr "$rate" \
    --device cuda --steps 5000 --batch-size 64 --microbatch-size 64 \
    --no-distill --seed 42 --run-dir "method/vla/checkpoints/lr11_muon_${rate}"
done

# AdamW baseline: identical screen budget and schedule.
for rate in 0.00015 0.0003 0.0006; do
  uv run --active --frozen --no-sync --offline --no-cache --no-python-downloads \
    python -B -m method.vla.train --config method/vla/microvla_v1.yaml \
    --optimizer adamw --lr "$rate" --device cuda --steps 5000 \
    --batch-size 64 --microbatch-size 64 --no-distill --seed 42 \
    --run-dir "method/vla/checkpoints/lr11_adamw_${rate}"
done

# Final large run: 20k flow steps, 96 px, shared 6e-4, no teacher prerequisite.
# Replace 6e-4 if held-out screening favors another rate; it is provisional.
uv run --active --frozen --no-sync --offline --no-cache --no-python-downloads \
  python -B -m method.vla.train --config method/vla/microvla_large.yaml \
  --device cuda --lr 0.0006 --no-distill \
  --run-dir method/vla/checkpoints/microvla_large_muon11

# Optional final run WITH 3k SigLIP distill steps followed by 20k flow steps.
# SIGLIP_PATH must name an already provisioned local model directory.
uv run --active --frozen --no-sync --offline --no-cache --no-python-downloads \
  python -B -m method.vla.train --config method/vla/microvla_large.yaml \
  --device cuda --lr 0.0006 --distill --distill-steps 3000 \
  --siglip-path "${SIGLIP_PATH:?Set SIGLIP_PATH to the provisioned local SigLIP directory}" \
  --run-dir method/vla/checkpoints/microvla_large_muon11_distilled
```

Append `--resume` to the exact corresponding command after interruption.
Do not change optimizer, rates, batch size, schedule, phase settings or model
size when resuming. The large YAML preserves v1's `distill: false`, so the
teacher phase remains explicit. The distilled run has a different name to
avoid accidentally overwriting the flow-only run.

## Unverified and limits

- No CUDA/bf16 execution, L4 throughput, attention-backend choice, or measured
  GPU peak; only CPU fp32 was executed.
- No real SigLIP loading, teacher feature parity, or real teacher memory;
  distillation/resume tests use a deliberately synthetic teacher.
- No real-data LR sweep or training; no validation-based optimum, final
  checkpoint or rollout success rate. No 90% policy-target claim.
- Current data/cache availability was not audited; all new training tests
  replace EpisodeStore with an in-memory fake store.
- The RMS test uses an explicit first-step AdamW reference, not a universal
  bound on later AdamW update sizes. Five NS iterations do not guarantee the
  tested spectral interval for every matrix.
- No cross-device bitwise resume claim: the exact resume test is CPU-to-CPU.
- No write to the requested external report directory was possible; this
  worktree copy is complete and ready to copy when filesystem access permits.
