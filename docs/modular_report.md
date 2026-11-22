# Modular G1 plate transfer

The controller converts camera and robot observations into a continuous estimate of the plate and destination. It uses that estimate to pick up, carry, seat and release the plate with geometric feedback. The lead froze commit `54080e586a9c892d7a903b99be316564328a4f77` and factory `submission.policy_modular_combined:make_policy`. All four fresh development blocks had already reached 27 successes apiece at freeze. The uninterrupted qualifying confirmation is in progress.

## Observations, actions and control

The policy receives three CHW `uint8` RGB views (head, right wrist and left wrist, each 84×84) and a 50-value proprioceptive vector with pelvis pose and motion, joints and finger state. A shared ResNet-18 encoder and selected proprioception feed the frozen `est_v1b` heads for pelvis-frame plate and nearest-site XYZ, tilt and held probability. The calibrated sidecar adds target-rack contact from public features and a common loaded-grasp normal. The task adapter transforms positions to world coordinates, locks the destination from an initial stationary median, and uses measured arm orientation to propagate the plate normal after pickup.

`RGB + proprioception` → `est_v1b + contact head + calibrated FK` → `task estimate` → `phase machine` → `geometric targets and normalized action`

- **APPROACH → PREGRASP → GRASP → LIFT.** Drive the pelvis to a geometric standoff. Sample the source while stationary at reset and pregrasp. Reach relative to the estimated plate with damped least-squares IK and verify the pinch and lift with finger measurements and FK. Phase timeouts and grasp retries are bounded.
- **CARRY → ALIGN → LOWER.** Track the destination standoff, center over the locked site, then descend with bounded position and orientation feedback. The combined candidate tries ordinary release confirmation first. A late fallback uses FK height, descent stall or rack contact, and the stationary source support height when the locked site is low.
- **RELEASE → SETTLE → RETREAT → HOLD.** Open both grippers and hold joint targets during settling. The ordinary retreat gate requires 75 measured-open steps, rack support and pose checks. If the open hand stalls after settling or guarded retreat, the clearance wrapper can withdraw the base a bounded distance while monitoring plate displacement, tilt and support.

The output is one 20-value action in `[-1,1]`: pelvis planar velocity, height target and yaw rate; seven absolute joint targets per arm; and two gripper commands. FK and IK use a separate fixed-robot MuJoCo model driven by measured joints. Inference uses public observations and fixed geometry. Simulator truth supplied training labels and guided controller development on generation seeds. Archived action sequences supported retrospective diagnostics and matched-prefix intervention tests. [Layout and controller](../method/modular/controller.py) · [Runtime and interface provenance](../EXPERIMENTS.md#report-provenance-and-exact-seed-joins--2026-11-23)

## Data and training

| Artifact | Data and use |
| --- | --- |
| **Frozen `est_v1b`** | Supervised harness estimator initialized from Torchvision ImageNet ResNet-18. Trained on `teacher_v1_dart0` (123 training episodes within seeds 2200000–2200150), `teacher_v1_gen2` (185 within 2800001–2800239), and `dagger3_priv` (114 within 2500000–2500119), with six held-out episodes per source. These are seed envelopes with gaps. The checkpoint records exact membership and the saved step (8,000). Privileged targets are plate XYZ, destination XYZ, tilt and held. The DAgger3 labels were recovered by replaying recorded student actions and checking pre-action states. |
| **Contact head and loaded normal** | Fit the contact head from frozen public features and simulator contact labels on 3021000–3021039 (privileged-driven rollouts) and 3022000–3022023 (observation-driven rollouts). Whole-episode split was 51 fit and 13 validation. Fit one loaded hand-frame normal on 12 generation grasps in the 3056500 series and checked it on six independent grasps. Package the normal with the contact coefficients. Both artifacts are used by the combined candidate. |
| **Separate modular DAgger experiment** | The observation-only collector ran on 3110000–3110299. Saved actions were replayed in a separate camera-free environment to create step-aligned pose, site, held and contact labels. Heads for pickup XYZ residual, held and target contact used a 240/30/30 train/selection/test split by seed plus 60 historical teacher episodes, with missing teacher contact labels masked. This sidecar and the experimental v2 encoder are outside the round-three checkpoint. |

A single nominal pre-close grasp transform was pooled from teacher-v1 rollouts on generation seeds 3000000–3000002. The controller applies this geometric constant relative to the current estimated plate. [Estimator provenance](../EXPERIMENTS.md#report-provenance-and-exact-seed-joins--2026-11-23) · [Contact-head fits](../EXPERIMENTS.md#modular-controller--2026-11-22--astra-frozen-j-perception-diagnosis) · [Loaded calibration and DAgger](../EXPERIMENTS.md#modular-controller--2026-11-22--integrated-visual-release) · [Grasp-transform calibration](../EXPERIMENTS.md#2026-11-22--sol-privileged-development)

## Ablations and evaluation

Each row identifies its cohort. Gains on identical reset seeds or on an identical archived prefix support a component comparison. The fresh signal totals from different blocks measure generalization separately.

| Frozen policy or change | Same-seed comparison | Reported cohort result |
| --- | --- | --- |
| `e588ae9`, estimated-state feedback baseline | **12/72** on the common, locally auditable development set | **18/88** returned on planned starts 1000000, 1010000, 1020000. |
| `9de5c51`, grasp-calibrated placement orientation, site lock and learned contact | **42/72** on the same seeds | **52/88** on the same planned blocks. |
| `bd951bc`, median pickup, proprioceptive grasp/lift checks and loaded-normal calibration | **65/72** on the same three-way intersection. On the full 88-seed two-way join, **52/88 → 80/88** against `9de5c51`. | **81/90** dev (28/30, 28/30, 25/30). Fresh signal **158/180 (87.8%)** on starts 1030000, 1040000, 1050000, 1060000, 1070000, 1080000. |
| `528c854`, guarded open-hand base withdrawal | Generation plan 3002200, same original prefix and rendered feedback after the recovery point: **24/30 → 26/30**. The stopped round-two baseline and clearance both scored **34/34** among their returned pairs. | Fresh starts 1031000, 1041000, 1051000, 1061000: **112/119** returned, or **112–113/120** with one unresolved. |
| `54080e5`, late FK/proprio release and low-site settling plus the withdrawal | On 3002200, standalone fallback from reset was **25/30** versus **24/30** baseline, with zero activations. Adding clearance in the archived comparison raised **25/30 → 26/30**. Clearance alone also reached **26/30** in its matched-prefix test. | At 03:15 IST, **114/114 completed** of 120 on fresh starts 1071000, 1081000, 1091000, 1032000. The four blocks had 27, 29, 29 and 29 successes. These completed outcomes guarantee **at least 95.0%** across the full plan. |

The early sharded totals had outstanding calls. The 72-seed intersection includes recovered failure archives and is a comparison on available reports. Full-time FK seating (`6b02f33`) improved selected dev **81/90 → 83/90**, then fell to **153/180** against `bd951bc`’s **158/180** on the same fresh seeds. In a separate uninterrupted official run, `6b02f33` passed development block 1000000 at **27/30**, with source slots 0–5 scoring 4, 5, 5, 4, 5 and 4 out of five. Integrity passed, with zero falls or policy errors. The other two uninterrupted FK runs stopped at freeze. The late release fallback recovered each controlled height-bias pair **0/2 → 2/2** at +20 mm plate height and −20 mm locked-site height. These camera-free continuations test the failure mechanism. The single full-reset gain on 3002200 diverged before the fallback could act.

[Common-seed joins](../EXPERIMENTS.md#report-provenance-and-exact-seed-joins--2026-11-23) · [Round-one fresh signal](../EXPERIMENTS.md#2026-11-23-0051--paired-180-episode-fresh-block-signal-eval-blocks-10300001080000-retired-after-this-use) · [Round-two fresh signal](../EXPERIMENTS.md#2026-11-23-023x--round-2-fresh-block-signal-528c854-methodmodularbase_clearance--112113120-9394)

## Failure modes and evaluation limits

Of 28 audited baseline `e588ae9` failures, 19 reached the destination area and stayed held to timeout, seven lost their grasp, and two fell into other categories. Original-action replay of all 22 fresh `bd951bc` failures found **15 pad recontacts after opening and initially clearing, five reached the destination without opening, one incomplete transfer, and one pickup failure**. For the seven audited `528c854` timeouts, **six recontacted after opening and one remained clear but tilted**. At all six pad returns the estimator reported held=false. In six episodes that did withdraw, action-matched replays found that the motion check stopped the base after an apparent 1.41–1.52 cm of plate travel. Physical travel was 0.000–0.831 cm. Extending withdrawal on three archived generation failures changed the commands and recovered zero episodes. [Baseline and event replay log](../EXPERIMENTS.md)

Sharded runs used the evaluator’s full episode horizon with a one-episode `--smoke` invocation per seed. They are non-qualifying. Development blocks were reused for selection, and fresh signal blocks were retired after one use. Early returns favor short successes. Generation continuations start from an archived prefix and serve as diagnostic evidence. Simulation success and separately reported policy wall time are different measures.

| Final frozen submission | Seed start | Successes / episodes | Source slots 0–5 | Original report, integrity and video |
| --- | ---: | --- | --- | --- |
| `54080e5` combined confirmation | 2600000 | In progress | Pending | Pending |
| Same frozen candidate, final reviewer run once | 7310000 | TBD | TBD | TBD |

Each final block uses one uninterrupted run of the unmodified evaluator. A 30-episode pass requires at least 27 successes, five episodes per source slot and valid integrity checks. In the original `report.json`, `per_source_slot` keys 1–6 correspond to zero-based source slots 0–5 in this report. The lead's confirmation uses EGL on a T4, CPU Torch inference and the default zero recorded videos. The separate clip at `runs/astra_video/placement_comparison.mp4` shows an FK timeout and a withdrawal recovery on matched generation seed 3001616. [Protocol and deliverables](../README.md#deliverables) · [Experiment log](../EXPERIMENTS.md)

## Reproduce

Use a clean checkout of the frozen SHA. The commands below reproduce a development block with OSMesa and record three videos. The lead's actual confirmation uses the same factory and weights in the EGL container defined by the qualifying runner. [Qualifying invocation](../ops_modular/lead_qualify.py)

```bash
git checkout --detach 54080e5
uv sync --locked --python 3.12 --no-dev --extra video
uv run --locked --no-dev --extra video advaitic-check

export MODAL_CONFIG_PATH=/home/keyu/.modal_modular.toml
est_sha=bd8ed33c87383b3b208ba251d8fa0b6f957d9f03a35cf6c8d23a81ce9cd867cd
head_sha=64dfe105e7f716c0ab5b7a9c487b7a8a1cac06565070b390e75992dfbee4deff
mkdir -p checkpoints/{est_v1b,astra}
uv run --locked --with modal modal volume get advaitic-runs est_v1b/last.pt checkpoints/est_v1b/last.pt
uv run --locked --with modal modal volume get advaitic-runs "heads/$head_sha.npz" "checkpoints/astra/$head_sha.npz"
printf '%s  %s\n' "$est_sha" checkpoints/est_v1b/last.pt \
  "$head_sha" "checkpoints/astra/$head_sha.npz" | sha256sum --check

uv venv --python 3.12 .policy
uv pip install --python .policy/bin/python --index-url https://download.pytorch.org/whl/cpu \
  torch==2.5.1 torchvision==0.20.1
eval_site=$(uv run --locked --no-dev --extra video python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
policy_site=$(.policy/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
printf '%s\n' "$eval_site" > "$policy_site/advaitic_eval_env.pth"

export ADV_ESTIMATOR="$PWD/checkpoints/est_v1b/last.pt" ADV_HEADS="$PWD/checkpoints/astra/$head_sha.npz"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa ORT_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=''
factory=submission.policy_modular_combined:make_policy

# Development reproduction with the frozen policy.
seed=1000000
uv run --locked --no-dev --extra video advaitic-eval \
  --policy "$factory" --policy-python "$PWD/.policy/bin/python" \
  --method modular-combined --checkpoint "est_v1b@$est_sha + heads@$head_sha" \
  --episodes 30 --seed-start "$seed" --out "results/modular-$seed" --video-episodes 3
```

For historical comparisons, select the SHA in the table, keep `est_v1b` fixed, and use `submission.policy_modular:make_policy` for `e588ae9`, `submission.policy_modular_heads:make_policy` for `9de5c51` with `heads/contact_v1_51257d33.npz`, `submission.policy_modular_calibrated:make_policy` for `bd951bc`, or `method.modular.base_clearance:make_policy` for `528c854`. The archived sharded protocol is in [lead_shard.py](../ops_modular/lead_shard.py) and [lead_shard_run.py](../ops_modular/lead_shard_run.py). Reproduction of those cohorts is separate from selecting a candidate on them.

To rebuild the contact head and calibration from the original labeled archives, point `runs` at the project’s saved results directory:

```bash
runs=/home/keyu/advaitic/plans/modular_controller/runs
out=$(mktemp -d)
OPENBLAS_NUM_THREADS=1 uv run --locked python -m method.modular.fit_refinement \
  contact "$out/contact_v1.npz" "$runs/astra_r3/training" "$runs/astra_r3/training_obs"
uv run --locked python ops_modular/astra_calibrate_normal.py "$runs/grasp_hand_frame.fit.json" \
  --contact "$out/contact_v1.npz" --output "$out"
sha256sum "$out/"*.npz
```

The historical CUDA training invocation below assumes the original archives and cache are mounted. The pinned checkpoint is the exact reproducibility target for evaluation. The launch budget was 10,000 steps, and the archived checkpoint was saved at 8,000.

```bash
# Recreate privileged labels from the original DAgger3 action archives if needed.
MUJOCO_GL=egl ORT_DISABLE_TELEMETRY=1 uv run --locked python -m method.harness.label_replay \
  /data/dagger3 --out /data/dagger3_priv

# Run in the CUDA training environment with the original data snapshot.
python -m method.harness.train_estimator --device cuda --cache-dir /cache \
  --run-dir /runs/est_v1b \
  --data-dirs /data/teacher_v1_dart0 /data/teacher_v1_gen2 /data/dagger3_priv \
  --steps 10000 --batch-size 64 --microbatch-size 32 --val-episodes 6 \
  --val-every 1000 --num-workers 2 --seed 42
```

The separate DAgger collector and seed dispatch are in [astra_dagger.py](../ops_modular/astra_dagger.py) and [astra_dagger_dispatch.py](../ops_modular/astra_dagger_dispatch.py). With saved embeddings at `/data/astra-dagger/sol` and the backbone at `/base/est_v1b/last.pt`, prepare the teacher samples and fit the sidecar in the CUDA training environment:

```bash
uv run --locked python -m method.modular.v2_teacher /data/teacher_v1_dart0 /data/est_v2/teacher --count 60
python -m method.modular.train_dagger_heads /data/astra-dagger/heads.npz
```

[Training and data-generation log](../EXPERIMENTS.md)

## Attribution

SWIRL BiGym2 supplies the task, fixed robot geometry, simulation and human demonstrations, and the distributed package retains the upstream NVIDIA controller and asset terms. The Advaitic assignment supplies the evaluator, scoring and observation/action interfaces. The harness project supplies `est_v1b`, teacher/DAgger archives and the label-replay workflow. Torchvision supplies ImageNet-pretrained ResNet-18 initialization. [Upstream and third-party notices](../THIRD_PARTY_NOTICES.md)

AI assistance included Codex workers **Astra** (kinematics, perception heads, integration and clearance), **Sol** (geometric phase control, release/settle fallback and paired tests), and **Astra-grasp** (pickup and loaded-grasp calibration), with **Claude** as lead for coordination, selection and evaluation. The implementation and experimental decisions are documented in the source and [experiment log](../EXPERIMENTS.md).
