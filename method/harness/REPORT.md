# Brief 13: observation-driven bce1 harness

Implemented on `harness`, based on `5233dc7`. This is an implementation and
plumbing-check report. No harness success rate has been measured, and the smoke
estimator checkpoint must not be treated as a trained deployment model.

Commit status: the session could edit this worktree but could not commit.
`git add` failed because the shared worktree metadata at
`/home/keyu/advaitic/robot-learning-assignment/.git/worktrees/wt-harness` is
mounted read-only (`index.lock`: Read-only file system). No Git metadata was
relocated or changed. From an ordinary writable terminal on branch `harness`:

```bash
git add method/harness submission/estimator.py submission/harness.py submission/policy_harness.py
git commit -m 'harness: add plate estimator, release supervisor and verified offline tooling'
```

## Implementation

- `label_replay.py` replays saved **action_executed**, camera-free, from the
  training seed. Every pre-action float32 state must match bit-for-bit. Existing
  labels must also match exactly. Writes go to a separate new directory; every
  original array is preserved. No fixture injection or trajectory policy is used.
- Label definitions match the teacher's `method/teacher/labels.py` and
  `diagnostics.py`: nearest of six destination sites; pelvis translation and
  negative-yaw rotation; angle of plate local +z to world -y; either-gripper
  contact predicate for held. Plate xyz is rounded to float32 before its frame
  transform, exactly as in the teacher; site xyz retains its original precision.
- `train_estimator.py` uses the existing EpisodeStore uint8 memmap cache. Every
  control-step state uses RGB index `t//2`, including odd steps. Inputs are only
  three RGB cameras and the **19 BC proprio dimensions**. A shared ImageNet
  ResNet-18 through layer3, spatial softmax, camera embeddings, and proprio MLP
  feed a 256-wide head predicting plate xyz, nearest-site xyz, tilt, and held logit.
  Regression uses normalized smooth L1; held uses BCE. Privileged quantities are
  **training targets only**. Inference implementation lives under `submission/`
  so the evaluator's submission digest covers it.
- Whole episodes are held out, six per source by default, using the existing
  deterministic EpisodeStore split. All normalization uses training episodes
  only. Duplicate seeds across input directories are refused to avoid split
  leakage. Missing labels and incomplete archives fail explicitly.
- `RUN/last.pt` contains model weights, config, normalization, data digest,
  train/validation episode lists, and validation metrics. `metrics.jsonl` logs
  validation; `config.json` records settings. This deliberately simple trainer
  requires a **fresh run directory**, with no resume or EMA implementation.
- `submission.policy_harness:make_policy` wraps the unchanged bce1 loader and
  the estimator. It reads only RGB/state; no simulator imports or metadata enter
  inference. `submission/policy.py`, `submission/model.py`, the evaluator,
  dependencies/lock, assets, controller, and success checks are unchanged.

The supervisor follows APPROACH → GRASP → CARRY → PLACE → RELEASE → SETTLE.
APPROACH waits for bce1's close proposal. GRASP closes until estimated held and
observed left-gripper closure persist; a missed-grasp timeout reopens. CARRY
blocks bce1's early open proposals. Losing estimated held in CARRY or PLACE
reopens and returns to APPROACH. PLACE requires distance, tilt, held and quiet
observed arm/base velocities over successive observations before opening.

At RELEASE, both grippers open and both arms latch absolute targets computed
from the current observed joints, using the fixed action calibration. Base/yaw
velocity commands become zero and pelvis height is held. SETTLE keeps these
targets for at least 75 steps (1.5 s), then continues holding/open forever until
episode reset. There is no success signal to justify a subsequent re-grasp.
The unused right gripper stays open in every phase; bce1 uses the left hand.

All settings are in `submission/harness.py:DEFAULT_CONFIG`. Every key has an
`ADV_H_<UPPERCASE_KEY>` override. Defaults: 2 executed steps per call (25 Hz
observation decisions), release distance 3.5 cm, tilt 12°, stable duration 6
control steps, held probability 0.7, held confirmation 4 steps, loss detection
10 steps, placement region 8 cm, grasp timeout 100 steps, reopen 10 steps.
Streak duration measures intervals between actual qualifying observations;
predicted future actions do not count as evidence. Arm/base speed limits are
0.4 rad/s and 0.04 m/s. Phase transitions and complete config go to stderr.
Changing execution cadence affects bce1's closed-loop behavior even with fixed
weights. `ADV_H_TEMPORAL_ENSEMBLE=1` enables the existing BC raw-chunk/logit
ensemble; default is off, independently of the checkpoint's ensemble setting.

## Verification and smoke metrics

1. Real teacher replay: `teacher_v1_dart0/ep_2200000.npz`, **708/708 states
   bit-for-bit equal**, and all four requested priv_* arrays bit-for-bit equal.
2. Real dagger3 replay: `ep_2500000.npz`, **1700/1700 states equal**. New output
   `/tmp/adv-harness-dagger-label-smoke/` contains all four labels and preserves
   every original array exactly. These are annotation replays, not policy runs.
3. Ten harness unit tests passed: phase progression, early-open blocking,
   distance/tilt/held/speed gates, stability reset, grasp miss/drop recovery,
   observed closure, permanent post-release hold/open, reset, action contract,
   seed guards, divergence rejection, and offline release timing/geometry.
4. Tiny CPU smoke: two teacher episodes (2200000 validation, 2200001 training),
   eight sampled training rows, sixteen validation rows, **two updates**, random
   initialization (`--no-pretrained`). Checked train-only statistics, episode
   split, odd/even RGB alignment, checkpoint load, real bce1 wrapper/reset with
   ensemble off/on, and a six-step offline prefix. The smoke checkpoint is
   `/tmp/adv-harness-smoke-v1/run/last.pt`; it is not committed.
5. Offline CLI checked with recorded proposals and recomputed bce1 proposals on
   that six-step prefix. Neither prefix contains a release. A synthetic full
   stream separately checks predicted release at step 16 versus recorded step
   20, with true 2.5 cm / 7° geometry reported at the predicted release.
6. Ruff check/format passed for `src submission tests method/harness`.
   Repository non-integration tests: **34 passed, 8 deselected**. Pytest's first
   attempt failed importing an unrelated ROS auto-plugin (`lark` missing);
   disabling third-party plugin autoload resolved it without repository changes.

**Smoke only**, not an estimate of trained-model quality:

| Validation subset | Rows | Plate position error | Site position error | Distance MAE | Tilt MAE | Held accuracy (p ≥ .5) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All sampled rows | 16 | 16.60 cm | 18.55 cm | 14.37 cm | 1.87° | 56.25% |
| True plate-to-site distance ≤ 6 cm | 6 | 11.49 cm | 15.38 cm | 22.05 cm | 2.64° | 66.67% |

Position errors are mean Euclidean vector errors; distance/tilt errors are mean
absolute scalar errors. Release-region selection uses **true training labels**
only for validation reporting, including held and released states. Empty release
subsets report null metrics/count zero. Full precision evidence is in `checks/`.

All local replay/model/smoke commands ran with `ADV_THROTTLE=1`; CPU model work
used two threads and no visible GPU. No full 30-episode evaluation was run. No
90% policy claim follows from these tests.

## Seed and data boundaries

Confirmed: **[2600000,2700000) and [7310000,7320000) are excluded** from replay,
training and offline analysis by explicit guards. No held-out block was used.
Boundary guard tests use synthetic integers, never simulator episodes.
The completed dart0+dagger3 audit covered 249 unique, non-reserved seeds.
Gen2 was actively syncing during implementation (transient empty files and a
growing count); a complete local gen2 audit was therefore not possible. Finish
the sync to the expected 191 episodes before training; the trainer refuses
partial/unreadable archives rather than silently dropping them. Input files
must remain fixed during cache construction/training. Dev evaluation is 1000000+.

## Operator commands

Run from `/home/keyu/advaitic/wt-harness`. `uv run --locked --no-sync` uses the
committed lock without modifying it and invokes the existing sim/model
environments, as in the BC workflow. Torch/torchvision remain in the model env.

### (a) Label dagger3 into a new directory

```bash
ADV_THROTTLE=1 ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
UV_CACHE_DIR=/tmp/adv-harness-uv uv run --locked --no-sync \
  /home/keyu/advaitic/robot-learning-assignment/.venv/bin/python \
  -m method.harness.label_replay /home/keyu/advaitic/data/dagger3 \
  --out /home/keyu/advaitic/data/dagger3_priv
```

The destination must not already exist for the single-shard command. Optional
`--shard 0/2` and `--shard 1/2` may share a new directory, without overwriting
any episode. There is no automatic resume; choose a fresh output for a new run.
Repeat the teacher equality check independently with:

```bash
ADV_THROTTLE=1 ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
UV_CACHE_DIR=/tmp/adv-harness-uv uv run --locked --no-sync \
  /home/keyu/advaitic/robot-learning-assignment/.venv/bin/python \
  -m method.harness.label_replay /home/keyu/advaitic/data/teacher_v1_dart0 --verify --limit 1
```

### (b) Train on Modal

Upload the completed new dataset (operator action; not executed here):

```bash
ADV_THROTTLE=1 UV_CACHE_DIR=/tmp/adv-harness-uv uv run --locked --no-sync \
  modal volume put advaitic-data /home/keyu/advaitic/data/dagger3_priv /dagger3_priv
```

Use the existing launcher with **ADV_CODE pointing to this worktree**, since
the launcher's default code worktree is wt-vla:

```bash
ADV_THROTTLE=1 ADV_CODE=/home/keyu/advaitic/wt-harness \
UV_CACHE_DIR=/tmp/adv-harness-uv uv run --locked --no-sync \
  modal run --detach /home/keyu/advaitic/ops/modal_train.py::run \
  --gpu L4 --module method.harness.train_estimator --run-name plate-estimator-v1 \
  --data 'teacher_v1_dart0 teacher_v1_gen2 dagger3_priv' \
  --args '--steps 20000 --batch-size 64 --microbatch-size 8 --val-episodes 6 --val-every 1000 --num-workers 2'
```

The equivalent command inside a container with data/cache/runs mounted is:

```bash
python -m method.harness.train_estimator --device cuda --cache-dir /cache \
  --run-dir /runs/plate-estimator-v1 \
  --data-dirs /data/teacher_v1_dart0 /data/teacher_v1_gen2 /data/dagger3_priv \
  --steps 20000 --batch-size 64 --microbatch-size 8 --val-episodes 6 --val-every 1000
```

This writes `/runs/plate-estimator-v1/last.pt` on `advaitic-runs`. Use a fresh
run name for every training launch. The full job/GPU path has not been run here.

### Offline timing analysis before evaluation

After downloading the trained estimator to the path below:

```bash
ADV_THROTTLE=1 CUDA_VISIBLE_DEVICES='' ORT_DISABLE_TELEMETRY=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
UV_CACHE_DIR=/tmp/adv-harness-uv uv run --locked --no-sync \
  /home/keyu/advaitic/mlenv/bin/python -m method.harness.offline_replay \
  --data-dirs /home/keyu/advaitic/data/dagger3_priv /home/keyu/advaitic/data/teacher_v1_dart0 \
  --estimator /home/keyu/advaitic/ckpts/plate-estimator-v1.pt --device cpu \
  --bc-checkpoint /home/keyu/advaitic/ckpts/student_v1b_bce1.pt \
  -o results/harness-offline-v1.jsonl
```

Create `results/` first and choose a new output filename. Omit `--bc-checkpoint`
to use recorded executed actions as the BC proposals. Each line reports when
the harness **would** release, all recorded close→open command edges after
actual held contact, true site distance/tilt/held at each event, and the timing
delta to the first data edge. Seed/true labels never enter the decision code.
This follows the recorded states even after an intervention would have changed
physics, so timing comparisons are counterfactual diagnostics, never success
evidence. `--limit 1 --max-steps 6` is a small CPU plumbing check.

### (c) Harness evaluation (operator only; do not run this full job locally)

Inside the Modal evaluator container, with this worktree's code installed and
the estimator available at the chosen path:

```bash
ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl \
ADV_CKPT=/ckpts/student_v1b_bce1.pt \
ADV_ESTIMATOR=/runs/plate-estimator-v1/last.pt \
advaitic-eval --policy submission.policy_harness:make_policy \
  --policy-python /root/mlenv/bin/python \
  --episodes 30 --seed-start 1000000 --method bce1-harness-v1 \
  --checkpoint bce1-plus-plate-estimator-v1 --out /tmp/eval/bce1-harness-v1-dev-001
```

The sim launcher currently snapshots wt-vla directly; ensure it snapshots this
worktree and mounts the estimator before launching. No external launcher was
changed by this implementation. Log both checkpoint hashes for a real result.
For dev sweeps prepend, for example,
`ADV_H_RELEASE_DISTANCE_M=0.03 ADV_H_RELEASE_TILT_DEG=10 ADV_H_STABLE_STEPS=8`.
All durations are control steps, not policy-call counts. Use a fresh result
directory/run name for every sweep. Temporal ensembling is a separate ablation.

### Repeat local unit/smoke checks

```bash
ADV_THROTTLE=1 CUDA_VISIBLE_DEVICES='' ORT_DISABLE_TELEMETRY=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
UV_CACHE_DIR=/tmp/adv-harness-uv uv run --locked --no-sync \
  /home/keyu/advaitic/mlenv/bin/python -m unittest method.harness.test_harness -v

ADV_THROTTLE=1 CUDA_VISIBLE_DEVICES='' ORT_DISABLE_TELEMETRY=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
UV_CACHE_DIR=/tmp/adv-harness-uv uv run --locked --no-sync \
  /home/keyu/advaitic/mlenv/bin/python -m method.harness.smoke_test \
  --data-dir /home/keyu/advaitic/data/teacher_v1_dart0 --out /tmp/adv-harness-smoke-repeat \
  --bc-checkpoint /home/keyu/advaitic/ckpts/student_v1b_bce1.pt
```

## Open questions

Full trained release-region errors and held calibration must be measured on the
Modal run before selecting D/T/K. Regressing two xyz vectors can compound error
in their difference; direct distance supervision is a possible follow-up if
release-region errors remain large. The nearest-site target can switch between
slots. Occlusion and carry/drop states make labelled dagger3 particularly useful.

The supervisor improves gating/settling only: bce1 still supplies pre-release
arm motion, including any poor insertion trajectory. It cannot infer actual
rack contact or success from an explicit input; a geometrically plausible
release may still be unsupported. False-positive geometry/held estimates can
cause an irreversible release latch; false negatives can block release. A
reopen may not recover an actual floor drop. Holding the observed joint pose
may change insertion forces and needs real dev evaluation, especially for D2.
The cadence change (10→2 steps for the default bce1 checkpoint) also needs an
ablation against `ADV_H_EXEC_STEPS=10`. No claim is made that this reaches 90%.

Attribution: existing bce1/BC encoder and EpisodeStore; torchvision ImageNet
ResNet-18; teacher privileged label definitions; deterministic replay workflow
from `analysis/relabel_dagger.py`; existing ACT-style ensembling; Codex assistance.
