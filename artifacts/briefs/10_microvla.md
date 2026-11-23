# Codex brief #10: microVLA (pi0.5-style, flow matching) as a second student architecture

Worktree: `/home/keyu/advaitic/wt-vla` (branch `vla`, based on 5233dc7 = BC track with BCE gripper). Rules unchanged: do not modify `src/`, `tests/`, `pyproject.toml`, `uv.lock`, the evaluator, physics or success checks; no GPU, no git. Do not touch `/home/keyu/advaitic/data/` or any existing cache directory except to read.

## Context
- `method/vla/vendor/train.py` + `LICENSE` are an **unmodified** copy of https://github.com/KhushiJ2911/microVLA (MIT). Keep that file unmodified; import from it or copy the needed classes into `method/vla/model.py` with a header comment crediting the source. Prefer importing; if the vendor file's top-level imports (e.g. LIBERO, datasets, wandb) make import impossible in our env (`/home/keyu/advaitic/mlenv`), copy the classes instead (Config, RMSNorm, GeGLU, Vision, attention_mask, rope, Expert, JointBlock, MicroVLA) and say so.
- Current best student: `submission/` ResNet-18 + spatial softmax + chunk head (H=20, exec 10), L1 on arm/base dims + BCE gripper. 53% (32/60) on dev+fresh seeds. Its main failures: gripper never released, plate placed slightly outside the 5 cm sphere, missed grasps at slot 5.
- Why this architecture is worth testing (first principles, put this in the README): flow matching learns the full action distribution and **samples** from it instead of regressing to the median/mean, so a rare bimodal decision (gripper open vs closed at release) should not be averaged away; plus a transformer over patch tokens of all 3 cameras may localise the plate/rack better than spatial softmax.

## Data / interface (must match ours)
- Episodes: `.npz` with `rgb (T/stride, 3 cams, 3, 84, 84) uint8`, `state (T, 50)`, `action (T, 20)`, optional `label_mask (T, 20)` (DAgger: gripper-only labels), `rgb_stride`. Reuse `method/bc/dataset.py` (and its existing cache format and mixture weights) for loading/sampling, so no new image cache is written; disk has ~2 GB free.
- Actions: predict only `ACTIVE_DIMS = (0,1,4,5,6,7,8,9,10,18)` (10-D) with the same normalisation as the BC track. State: the same `proprio_dims` as `method/bc/student_v1b_bce.yaml` (19 dims) → `state_dim=19` discretised tokens (normalise to [-1,1] with clipping as the vendor code expects; document the stats).
- Images: 3 cameras at 84 px; patch size 8 does not divide 84. Pick one: resize to 96 (12×12 patches) or 80 crop. Recommend and justify; consider token count/VRAM (RTX 3050 4 GB, must also leave room for the eval).
- Instruction: constant task string ("move the plate from the left rack to the right rack"); keep the language path (it is cheap) but note it carries no information for one task.
- **Masked flow-matching loss:** pad mask (chunks past episode end) and `label_mask` must be respected; DAgger gripper-only samples supervise only the gripper dim. Elements with mask 0 contribute nothing; normalise by valid elements. Gripper dim: keep it as a continuous ±1 dim inside the flow (that is the point of the test); at inference execute open if the sampled value < 0.
- Horizon 20, execute 10 (same as BC) by default; configurable.

## Training recipe
- Config `method/vla/microvla_v1.yaml`, data = same 5 dirs and weights as `student_v1b_bce.yaml`.
- Optional SigLIP distillation phase (vendor recipe) behind a flag, default **off** for v1: weights for `google/siglip-base-patch16-224` are not downloaded and disk is tight. Write the code path, I will decide.
- Random-shift augmentation and colour jitter as in vendor; EMA 0.999; bf16 autocast on CUDA; resumable checkpoints every 250 steps (`--resume`), atomic writes. Default 10k steps, batch 64 (with microbatching if needed). Log train/val flow loss and, every N steps on a few val batches, sampled-action L1 per dim and gripper sign accuracy (overall and on open-labelled elements).
- Print a VRAM/throughput estimate in the smoke test on CPU (params, tokens per sample).

## Inference
- `submission/policy_vla.py` exposing `make_policy` with the same `obs` contract as `submission/policy.py` (`rgb (3,3,84,84)`, `state (50,)`), checkpoint from `ADV_CKPT`, 10 Euler steps by default (`ADV_FLOW_STEPS` env override for dev sweeps only), fixed-seed noise per episode for reproducibility, prefix KV cache. Must run on CPU if CUDA is absent.

## Smoke test (`method/vla/smoke_test.py`, CPU)
Fake data with a rare bimodal gripper label: masked loss ignores masked elements; tiny overfit run reduces loss; policy loads a saved checkpoint and returns a valid 20-D action; resume works.

Report to `/home/keyu/advaitic/reports/10_microvla.md`: what changed, design choices with one-line reasons, param count, tokens per sample, exact train/eval commands (eval with `advaitic-eval --policy submission.policy_vla:make_policy`), anything unverified.
