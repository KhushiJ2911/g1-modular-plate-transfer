# Brief 10 — microVLA second student

Implemented 2026-11-20 in `/home/keyu/advaitic/wt-vla`; no git commands and no
GPU access. Requested destination: `/home/keyu/advaitic/reports/10_microvla.md`.
The session permits writes only inside the worktree and `/tmp`, so this report
is delivered at `method/vla/10_microvla.md` instead.

## Changes

- `method/vla/model.py`: imports the unchanged MIT microVLA architecture and
  adds task preprocessing, token counts and mask-aware flow loss sums/counts.
- `method/vla/train.py`, `microvla_v1.yaml`, `requirements.txt`: 10k-step recipe,
  same five-source BC mixture, batch 64/microbatch 2, EMA 0.999, CUDA bf16, atomic
  250-step checkpoints, explicit resume, masked validation/sample diagnostics,
  optional local-only SigLIP phase off by default.
- `submission/policy_vla.py`: `make_policy`, existing observation contract,
  20-step horizon/10 executed actions, fixed per-episode RNG, cached prefix KV,
  10 Euler steps (development `ADV_FLOW_STEPS` override), CPU fallback.
- `method/vla/smoke_test.py`: temporary fake episodes with rare open gripper
  labels, masked-loss/gradient/overfit/policy/resume and fake-teacher phase checks.
- `method/bc/dataset.py`: optional read-only cache opening, including missing
  cache refusal without mkdir/lock/write; original BC default behavior retained.
- `method/vla/README.md`: rationale, normalization, recipes, budget and caveats.

`src/`, `tests/`, existing submission files, `pyproject.toml`, `uv.lock`, vendor
source/license, manifests, fixtures, physics, controller and success checks were
not modified. Existing data/cache directories were read only. The temporary
synthetic fixture cache is created under `/tmp` and deleted after the smoke run.

## Design choices

| Choice | One-line reason |
| --- | --- |
| Import upstream classes | Top-level imports work in mlenv; no fork/copy required. |
| Distributional flow, continuous ±1 gripper | Sampling can retain an open/closed mode that regression may average away; improvement remains a hypothesis. |
| Patch-token transformer over all 3 cameras | May preserve plate/rack localization better than the BC spatial-softmax bottleneck. |
| Resize 84→96 | Patch8 divides 96 and no image border is discarded; accept more tokens than an 80 crop. |
| Vendor default model, microbatch 2 | Keep the architecture intact while bounding activation memory on 4 GB hardware. |
| Fixed 11-word language prefix | Keep the cheap language pathway; it carries no task-discriminating information. |
| BC z-score divided by 3, clipped to [-1,1] | Fit vendor 256-bin state tokens without clipping every value beyond one std. |
| Exact mask sum/count loss | DAgger supervises only trusted dimensions and episode-end padding contributes zero. |
| Masked input targets replaced by noise | Prevent untrusted DAgger arms leaking into the supervised gripper's action context. |
| Read-only BC cache | Reuse format/splits/mixture with no new real image cache and no existing cache writes. |
| SigLIP disabled/local-only | No weight download or disk growth without a future explicit preparation task. |
| One atomic rolling checkpoint | Resume every 250 steps while avoiding 40 retained large snapshots. |

Active actions `(0,1,4,5,6,7,8,9,10,18)` use **identity action normalization**
(center 0 / scale 1), as in BC; raw controller targets are already in [-1,1]. Open
is sampled gripper<0, executed as -1; otherwise +1. Inactive actions use trusted
training-label means. H=20 / execute 10 is configurable. Full held-out flow loss is
logged every 250; sampled per-dimension L1 and gripper sign accuracy (all/open)
every 500 on two evenly spaced held-out batches. Logs include element counts and
null when no open label exists. Samples used for L1 are continuous before
controller clipping/sign conversion.

## Measured resources and checks

Default **11,198,026 params**: vision1,893,952; prefix5,704,704;
action expert2,265,600; embeddings/heads1,333,770. **482 tokens/sample**:
432 image +11 language +19 state +20 action. Prefix 462 tokens. An 80px crop
would use 350 total tokens and ~53% dense joint-attention score storage, at the
cost of losing the image border.

CPU smoke passed. Tiny fixed-noise/time overfit loss was **1.623354 → 0.002293** in
45 updates. Resume matched uninterrupted **model, EMA, Adam state and torch RNG
bit for bit**. Tests also check sample/cache equivalence, valid float32 `(10,20)`
actions, open sign mapping, noise-reset reproducibility, no masked-target
influence/gradient (including masked NaNs), logical-batch gradient normalization,
no optimizer/EMA change on empty supervision, read-only cache immutability and
missing-cache refusal. SigLIP phase/restart was exercised with a synthetic
teacher, without transformers weights. These are software/numerical checks;
there is no simulator rollout or policy success result.

Full-model CPU microbatch 2 forward/backward: approximately 0.477 seconds on
2 threads, extrapolated **0.0655 batch 64 steps/s** (~15.3 s/step) before Adam,
augmentation, I/O and validation. Static fp32 model+EMA+grad+Adam state:
**0.209 GiB**. Dense joint-attention scores across six layers at microbatch 2:
**42.54 MiB**, excluding other activations and backward. CUDA planning estimate:
**0.6–1.8 GiB**, excluding evaluator and optional SigLIP; nominally leave 2.2+ GiB
on a 4 GiB card. This is an analytical budget, not measured VRAM or a fit guarantee.
CUDA throughput, allocator peaks and simultaneous evaluation headroom are
unverified. Real training prints CUDA peak allocated/reserved memory.

## Current data/cache audit and normalization

The five sources currently contain 372 episodes: 362 train / 10 validation;
306,696 training control rows. Held-out episodes are listed below. Numeric
normalization was audited by reading state columns only; it was not a training
run and did not create a checkpoint. BC train-row means/stds are unweighted by
source mixture. State input is `clip((raw-mean)/(3*max(std,0.001)),-1,1)`;
vendor quantization is `min(floor((input+1)*128),255)`.

| Source | Episodes | Held out |
| --- | ---: | --- |
| human60 | 60 | ep_0.npz, ep_59.npz |
| teacher_v1 | 73 | ep_2000001.npz, ep_2100227.npz |
| teacher_v1_dart0 | 129 | ep_2200000.npz, ep_2200150.npz |
| dagger | 60 | ep_2400000.npz, ep_2400059.npz |
| dagger2 | 50 | ep_2410000.npz, ep_2410055.npz |

| State index | Mean | Std (floor 0.001) | Saturated beyond ±3 std |
| ---: | ---: | ---: | ---: |
| 0 | -0.37691936 | 0.50848967 | 0.000% |
| 1 | 0.32001212 | 0.13550612 | 0.480% |
| 2 | 0.42257491 | 0.59213173 | 5.794% |
| 3 | 0.28811237 | 0.77514225 | 0.000% |
| 4 | 0.043548658 | 0.55587286 | 5.948% |
| 5 | 0.026485492 | 0.41902414 | 3.316% |
| 6 | -0.080940597 | 0.26112968 | 0.191% |
| 7 | 0.006330092 | 0.019894619 | 0.000% |
| 8 | 0.0060946248 | 0.019976355 | 0.000% |
| 18 | 0.031644795 | 0.04134253 | 0.371% |
| 19 | -0.13842626 | 0.17166573 | 0.000% |
| 20 | 0.7467289 | 0.0021937487 | 1.181% |
| 21 | -0.016616391 | 0.057355773 | 0.379% |
| 40 | 0.0061366903 | 0.039266575 | 3.020% |
| 41 | -0.01909532 | 0.075916372 | 3.653% |
| 42 | 4.8005324e-05 | 0.014536088 | 2.686% |
| 43 | -0.0036555766 | 0.049413644 | 3.113% |
| 44 | 0.40837702 | 0.44473577 | 0.000% |
| 45 | 0.0029508048 | 0.035415214 | 0.708% |

The existing schema 3 cache `1946b0d61101559f44a9df77` covers only the first four
sources / 322 episodes; `fc0f42f086d17f18facd685b` covers an older three-source
mixture. Neither contains dagger2. Exact current five-source cache key
`52bdf4d0bc2540b9f4050a6a` is missing. The read-only lookup failed clearly and
**wrote nothing**. A real training invocation needs an existing matching cache
via `--cache-dir`; provisioning a new cache would be a separate task with more
disk and permission to write it. No training on real data was attempted.

## Exact commands

From the worktree root, using the pre-existing model environment and the
committed root lock without syncing/installing anything:

```bash
export VIRTUAL_ENV=/home/keyu/advaitic/mlenv
export UV_CACHE_DIR=/tmp/microvla-uv
export PYTHONDONTWRITEBYTECODE=1
export ORT_DISABLE_TELEMETRY=1

# Executed CPU smoke (UV_CACHE_DIR used in this session: /tmp/adv-vla-uv).
CUDA_VISIBLE_DEVICES='' uv run --active --frozen --no-sync python -B -m method.vla.smoke_test

# Real-data CPU training recipe; blocked until a matching cache is provisioned.
CUDA_VISIBLE_DEVICES='' uv run --active --frozen --no-sync python -B -m method.vla.train \
  --config method/vla/microvla_v1.yaml --device cpu

# Resume the same run (configuration/data identity checked).
CUDA_VISIBLE_DEVICES='' uv run --active --frozen --no-sync python -B -m method.vla.train \
  --config method/vla/microvla_v1.yaml --device cpu --resume

# Optional phase, new run, weights must already exist locally. Not executed.
CUDA_VISIBLE_DEVICES='' uv run --active --frozen --no-sync python -B -m method.vla.train \
  --config method/vla/microvla_v1.yaml --device cpu --distill --distill-steps 1000 \
  --siglip-path /path/to/provisioned/siglip-base-patch16-224 \
  --run-dir method/vla/checkpoints/microvla_v1_distilled
```

A later GPU session can use `--device cuda` and omit `CUDA_VISIBLE_DEVICES=''`;
the YAML default CUDA recipe uses bf16. None was run in this session. The
checkpoint is `method/vla/checkpoints/microvla_v1/last.pt`.

For evaluation, use an environment already installed from the committed
`uv.lock` (no evaluator environment is provisioned in this worktree). The
model environment is supplied separately with `--policy-python`; unset the
model `VIRTUAL_ENV` before this command so uv uses the evaluator project env:

```bash
unset VIRTUAL_ENV
ADV_CKPT="$PWD/method/vla/checkpoints/microvla_v1/last.pt" \
ADV_FLOW_STEPS=10 CUDA_VISIBLE_DEVICES='' ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl \
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/microvla-uv \
uv run --frozen --no-sync advaitic-eval \
  --policy submission.policy_vla:make_policy --policy-root "$PWD" \
  --policy-python /home/keyu/advaitic/mlenv/bin/python \
  --method microvla_v1 --checkpoint microvla_v1-last \
  --episodes 60 --seed-start 8000000 --out method/vla/results/dev60
```

The evaluation command was **not executed**. Use a new output directory and a
real trained checkpoint. `ADV_FLOW_STEPS` is for development sweeps only; 10 is
the default. Prefix KV caching is on. The policy resets the CPU noise generator
to the checkpoint seed each episode and draws a fresh deterministic noise chunk
per call; it runs on CPU when CUDA is absent.

## Unverified / limits

- Real-data training and sampled-action quality: not run; current cache incomplete.
- Closed-loop release, placement, slot 5 grasp and 90% success target: not measured.
- CUDA bf16 correctness, measured memory/throughput and eval co-residency: not tested.
- Actual SigLIP/transformers execution and teacher memory: not tested or downloaded.
- Full evaluator tests: not needed for candidate-only changes and not run.
- External report destination: outside writable sandbox; this local report is the deliverable.

Focused Ruff lint/format and the CPU smoke were run. SHA256 checks confirm vendor,
root dependency files and all protected existing source/test/submission files
unchanged; the only existing-file change is the optional BC read-only cache mode.
