Implemented the `resmlp` action head and wired `head_type`, `head_blocks`, and `fuse_skip` through YAML, CLI, checkpoints, and policy loading. The residual head defaults to three pre-norm blocks with zero-initialized output projections. `fuse_skip` defaults to `False` and optionally feeds the proprio embedding into each block’s output projection. Changes are uncommitted on `bcres`.

- **BCE1 compatibility:** Before and after the change, `student_v1b_bce1.pt` produced bit-identical raw predictions and policy actions on the same seeded CPU inputs. Both comparisons passed `torch.equal`.
- **Residual head:** CPU forward and backward checks passed with and without `fuse_skip`. The blocks started as identities, and gradients reached the head input and each block’s output projection.
- **Training and loading:** Both CPU optimizer updates completed. `submission.policy.make_policy()` loaded the EMA checkpoint and returned finite `(10, 20)` actions on a held-out frame. Ruff lint and format checks passed. [Checkpoint](/tmp/bcres-19a/run-supervised/last.pt)

`/tmp/bcres-19a/episodes` contains symlinks to `dagger3/ep_2500004.npz` for training and `dagger3/ep_2500001.npz` for validation. Seed `43` gave labeled samples in both training batches. The exact training command was:

```bash
ADV_THROTTLE=1 CUDA_VISIBLE_DEVICES='' ORT_DISABLE_TELEMETRY=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 UV_CACHE_DIR=/tmp/bcres-uv-cache \
uv run --locked --offline --no-sync /home/keyu/advaitic/mlenv/bin/python \
  -m method.bc.train \
  --data-dirs /tmp/bcres-19a/episodes --data-weights 1 \
  --cache-dir /tmp/bcres-19a/cache --run-dir /tmp/bcres-19a/run-supervised \
  --head-type resmlp --head-blocks 2 --device cpu --steps 2 \
  --batch-size 4 --microbatch-size 4 --val-episodes 1 \
  --val-every 2 --save-every 2 --log-every 1 --num-workers 0 \
  --no-pretrained --aux-weight 0 --gripper-loss bce --warmup-steps 0 --seed 43
```