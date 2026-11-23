# Codex brief #1: BC baseline (vision + proprio, action chunking)

You are working in a git worktree of a robot-learning assignment repo. Read `README.md`, `METHODS.md`, `submission/policy.py`, `src/advaitic_eval/environment.py`, `src/advaitic_eval/runner.py`, `src/advaitic_eval/action_layout.json` first. **Do not modify anything under `src/`, `tests/`, `pyproject.toml` or `uv.lock`.** Training code goes under `method/bc/`; the evaluated policy goes in `submission/`. You cannot use the GPU or write git metadata; the operator runs GPU jobs and commits.

## Two Python environments
- Sim/data env: the repo's `.venv` (`uv run ...`): has bigym, mujoco, numpy. No torch.
- Model env: `/home/keyu/advaitic/mlenv/bin/python`: torch 2.5.1+cu121, torchvision, numpy, pillow, pyarrow, matplotlib. No bigym. Training and the evaluated policy run here; the evaluator is launched with `--policy-python /home/keyu/advaitic/mlenv/bin/python`.

## Facts already measured (trust these)
- Observations: `obs["rgb"]` uint8 (3 cams [head, right wrist, left wrist], 3, 84, 84); `obs["state"]` float32 (50,). State layout: 0–6 left arm q, 7–8 left fingers, 9–15 right arm q, 16–17 right fingers, 18–21 pelvis x,y,z,yaw (world), 22–43 velocities of 0–21 (pelvis velocities = 40–43), 44–45 gripper states, 46–49 base-controller pelvis.
- Actions: normalized (20,) in [-1,1]: 0 vx, 1 vy, 2 pelvis height, 3 yaw rate, 4–10 left arm targets, 11–17 right arm targets, 18 left gripper, 19 right gripper.
- In all 60 human demos only **10 dims vary**: 0, 1, 4–10, 18. Dims 2 (≈0.13), 3 (0), 11–17 (right arm, std ≈ 0.02) and 19 (−1) are essentially constant. Dim 18 (left gripper) is binary ±1: closes ≈ step 229, opens ≈ step 570. Demos are 530–742 steps; the episode budget is 1,700 steps.
- Demos: `from bigym.loco import make_gym; env = make_gym("move_plate"); demos = env.get_demos(60)` (repo `.venv`). Each is a dict: `obs.rgb` (T+1,3,3,84,84) uint8, `obs.state` (T+1,50), `action` (T,20), `success`. Loading all 60 with images takes ~2.5 GB RAM. **Load once in a single process; never per worker.**

## Deliverables
1. `method/bc/export_human.py` (runs in the repo `.venv`): export the 60 demos to `DIR/ep_<idx>.npz` (default `/home/keyu/advaitic/data/human60/`) with `rgb` (T,3,3,84,84) uint8 **every 2nd step** (key `rgb_stride=2`; frame j = step 2j), `state` (T,50) and `action` (T,20) at **every** step, `seed`, `slot` (-1 if unknown), `success`. Same format the privileged teacher will write later, so the trainer handles both. `np.savez_compressed`, atomic write (temp + `os.replace`), skip existing files.
2. `method/bc/dataset.py`: loads a list of episode directories. On first use, builds a **single uint8 memmap cache** of all frames on disk (path configurable, e.g. `/home/keyu/advaitic/data/cache/<hash>.u8`) plus an index, so datasets of hundreds of episodes don't need to fit in RAM. A sample = time t at a stored frame (even t): 3 images at t, proprio at t, target chunk `action[t:t+H]` over the **10 active dims**, padded with the last action past the end, plus a pad mask.
   - Proprio input: dims 0–8 (left arm + fingers), 18–21 (pelvis pose), 40–43 (pelvis velocity), 44–45 (grippers). Normalize with dataset mean/std saved in the checkpoint.
   - Hold out 6 episodes as validation (deterministic split by index).
3. `method/bc/model.py`: **ImageNet-pretrained ResNet-18** (torchvision, weights downloaded once and cached) truncated after `layer3`, shared across the 3 cameras, followed by a **spatial softmax** (robomimic-style keypoints, 32 keypoints per camera) + a learned camera embedding. Concatenate with an MLP embedding of proprio → MLP (2×512, GELU, dropout 0.1) → `H×10` outputs. Images: random-shift augmentation (pad 4, random crop back to 84; DrQ) during training only; ImageNet normalization. Replace BatchNorm with GroupNorm **or** freeze BN statistics (choose one, explain in a comment).
   Loss: L1 on the 10 dims (ACT uses L1), masked over padded steps. Make H a config value (default 20).
4. `method/bc/train.py` (model env): config via CLI/YAML. AdamW (lr 3e-4 head, 1e-4 backbone, wd 1e-4), linear warmup 500 steps then cosine, batch 64, fp16 autocast with GradScaler, grad clip 1.0, **EMA of weights (decay 0.999)** used for validation and saved checkpoints. Default 20k steps. Checkpoint every 1k steps to `RUN_DIR/last.pt` (atomic) and **resume automatically** from it if present (model, EMA, optimizer, scaler, step, RNG). Append-only `RUN_DIR/log.csv` (step, train loss, val loss, lr, it/s). Keep GPU memory < 3 GB; print peak memory. `num_workers` ≤ 2.
5. `submission/policy.py`: replace the example with the student policy. `make_policy()` loads the checkpoint from env var `ADV_CKPT` if set, else `submission/weights/policy.pt` (relative to this file). Uses CUDA if available, else CPU. `__call__(obs)` returns an `(E, 20)` chunk, where `E = exec_steps` (config, default 10 ≤ H): expand the 10 predicted dims to 20 by filling the constant dims with their **dataset means** stored in the checkpoint; set dim 18 to +1/−1 by thresholding at 0; clip to [-1,1]; float32. `reset()` clears state. Optional temporal ensembling (ACT) behind a flag, default off. No access to anything except `obs["rgb"]` and `obs["state"]`. Keep imports inside the policy minimal (torch, torchvision, numpy).
6. `method/bc/README.md`: exact commands for export, training and dev evaluation:
   `MUJOCO_GL=egl ADV_CKPT=<ckpt> uv run advaitic-eval --policy submission.policy:make_policy --policy-python /home/keyu/advaitic/mlenv/bin/python --episodes 30 --seed-start 1000000 --method bc-human60 --out results/<new_dir>`
7. A CPU smoke test `method/bc/smoke_test.py` (model env): builds a fake 2-episode dataset in a temp dir, trains 20 steps on CPU, saves a checkpoint, loads it through `submission.policy.make_policy()`, calls it on a fake obs and checks the output shape/dtype/range. Run it.
8. A short report at the end: what you built, design choices with one-line reasons, anything unverified, and the exact commands for me.

## Constraints
- Keep code small and readable. No new dependencies beyond what is listed.
- `.gitignore` ignores any `checkpoints/` directory, so final weights go in `submission/weights/` (do not add weights to git yourself).
- Run `uv run ruff check method submission && uv run ruff format --check method submission` if available.
