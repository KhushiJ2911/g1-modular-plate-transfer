# Modular G1 plate transfer

**Final policy:** commit `aa1e2731e0bdfa2a9f315f8cc9c758e3270474dc`, factory `method.modular.vertical_clearance:make_policy`.

| Run | Result |
| --- | --- |
| Reviewer block `7310000` (run once) | **27/30, PASS**, exactly at the 27/30 bar |
| Last dev round (8 new blocks) | 231/240 (96.25%), all 8 blocks ≥ 27/30 |
| Confirmation block `2600000` (run once, earlier commit `54080e5`) | 25/30, FAIL |

The reviewer result has no margin. One more failure would have failed the block. Slot 3 is the weakest slot (3/5 on `7310000`).

Full log: [EXPERIMENTS.md](../EXPERIMENTS.md). Evaluator output: [report.json](final_7310000/report.json), [episodes.csv](final_7310000/episodes.csv).

## How it works

The policy uses only the evaluator's observations: three 84×84 RGB cameras (head, right wrist, left wrist) and the 50-value proprioception vector.

```
RGB + proprio → estimator (est_v1b) + contact head → task state → phase machine → IK → action
```

**Perception.** `est_v1b` is a ResNet-18 trained earlier in this project. It predicts plate position, nearest rack-site position, plate tilt and whether the plate is held. A small linear head on the same features predicts rack contact. The controller locks the destination site from a median over the first stationary frames. After pickup it tracks plate orientation through arm forward kinematics (FK) instead of vision.

**Control.** A fixed sequence of phases:

1. **Approach, pregrasp, grasp, lift.** Walk to a standoff, reach the plate with damped least-squares IK, close, and check the pinch and lift with finger joints and FK. Retry the grasp a bounded number of times.
2. **Carry, align, lower.** Walk to the destination, center over the locked site, and lower. If release is not confirmed normally, a fallback triggers on FK height, a stalled descent or rack contact.
3. **Release, settle, retreat.** Open both hands and hold the arm pose measured at opening. Wait for the plate to settle, then back the base away while watching plate motion. The final change adds a short vertical lift of the open hand so it clears the plate before leaving ([vertical_clearance.py](../method/modular/vertical_clearance.py)).

The action is the standard 20-value vector: base velocity, height and yaw rate, seven joint targets per arm, and two gripper commands. IK and FK use a separate fixed-base MuJoCo model driven by measured joint angles.

Simulator state was used only to make training labels and to debug on development seeds. It is never read at inference. There is no per-seed lookup or replay.

## Training data

| Model | Data |
| --- | --- |
| `est_v1b` estimator | ImageNet ResNet-18, fine-tuned for 8,000 steps on 422 episodes from three sources: `teacher_v1_dart0` (seeds 2200000–2200150), `teacher_v1_gen2` (2800001–2800239) and `dagger3_priv` (2500000–2500119). 6 held-out episodes per source. Labels come from simulator state. |
| Contact head | Fit on frozen estimator features with simulator contact labels, seeds 3021000–3021039 and 3022000–3022023 (51 train / 13 validation episodes). |
| Loaded-grasp normal | One constant fit on 12 grasps (3056500 series) and checked on 6 others. |
| Grasp offset | One constant pre-close hand-to-plate transform, averaged from teacher rollouts on seeds 3000000–3000002. |

A separate DAgger run (seeds 3110000–3110299) trained extra heads, but the final policy does not use them.

## Results

### Early rounds (sharded, not qualifying)

These runs used one process per episode. They later turned out to be about 3 points too optimistic compared with the real evaluator format, so they are only useful for comparing versions with each other.

| Commit | Change | Result |
| --- | --- | --- |
| `e588ae9` | Baseline estimated-state controller | 18/88 |
| `9de5c51` | Grasp-calibrated place orientation, site lock, contact head | 52/88 |
| `bd951bc` | Median pickup estimate, proprio grasp checks, loaded-normal calibration | 81/90 dev; 158/180 on fresh blocks |
| `528c854` | Guarded base withdrawal after release | 112/119 on fresh blocks |
| `54080e5` | Late release fallback + withdrawal | 114/114 completed on fresh blocks |

### Qualifying format

From round 4 on, every dev block ran exactly like the reviewer run: one unmodified `advaitic-eval` process for 30 balanced episodes. Each round used blocks that had not been used before.

| Round | Commit | Change | Result | Blocks ≥ 27/30 |
| --- | --- | --- | ---: | ---: |
| 4 | `414f15a` | `54080e5` + clear internal state on reset | 220/240 (91.7%) | 6/8 |
| 4 (same blocks) | `528c854` | for comparison | 214/240 (89.2%) | 4/8 |
| 5 | `99e72f6` | grasp-loss guard, rack-contact release, seating-drop check | 220/240 (91.7%) | 6/8 |
| 6 | `876a00f` | hold measured arm pose at opening | 222/240 (92.5%) | 8/8 |
| 7 | `aa1e273` | vertical hand lift before retreat | **231/240 (96.25%)** | **8/8** |

Round 7 per block (start seed: successes): 1039000: 29, 1045000: 30, 1055000: 27, 1065000: 29, 1075000: 30, 1085000: 29, 1095000: 30, 1046000: 27.

Rounds 6 and 7 ran on different blocks, so part of the 222 → 231 gain may be block-to-block noise. On one shared block (`3004800`) both scored 27/30 with the same successful seeds.

### Protected blocks

| Block | Commit | Result | Per slot (0–5) |
| --- | --- | --- | --- |
| 2600000 | `54080e5` | 25/30, FAIL | 3, 5, 5, 2, 5, 5 |
| 7310000 | `aa1e273` | 27/30, PASS | 5, 5, 5, 3, 4, 5 |

Each block ran once, on a T4 with EGL rendering, after the policy was frozen. Neither was used for tuning. Integrity checks passed and there were no infrastructure errors. Submission SHA-256: `1b05190d2013d5d08f230e9253615c13dd8fc5b009686e72a55bfa7fa5586f25`. In `report.json` the slots are numbered 1–6.

## Failures

Most failures happen after the plate is placed. The hand opens, clears the plate, then touches it again as the robot moves away, and the episode times out.

- `bd951bc`, 22 failures replayed: 15 touched the plate again after opening, 5 never opened at the rack, 1 did not finish the carry, 1 failed pickup.
- Round 5, 20 failures: 11 touched again after opening, 3 tilted after opening, 5 stayed closed (2 on the rack, 3 off it), 1 opened during lift.
- Round 6, 18 failures: 7 touched again after opening, 7 stayed closed (3 on the rack, 4 off it), 2 tilted on the rack, 2 opened during lift.
- `7310000`: 3 timeouts, no falls (`7310000` slot 4, `7310003` slot 3, `7310008` slot 3).

The estimator reported held=false at every re-touch, so the held signal cannot catch this. That is why the last two fixes hold the arm still at opening and lift the hand straight up before the base moves.

## Next steps

- **Re-touch after release.** This is still the biggest failure type. Check finger-to-plate clearance with FK before the base moves, instead of relying on held=false.
- **Closed grasps that stall at the rack.** Add a timeout-based release when the plate is seated but not confirmed.
- **Slot 3.** It is the weakest slot on both protected blocks. Run a slot-3-only failure sweep on new dev seeds.
- **Margin.** Measure the pass rate on more fresh dev blocks before trusting the 27/30 result, since it has no slack.

## Reset audit

Resetting the policy state between episodes did not explain the gap between sharded and single-process runs. The reset cleanup changed none of the 81 actions in the tested comparison, and only 7 of 20 round-4 failures passed when rerun in a fresh process.

## Reproduce

From a clean checkout of the final commit. This runs one round-7 dev block on CPU with OSMesa. The reported numbers used T4/EGL with the same pinned weights ([lead_qualify.py](../ops_modular/lead_qualify.py)).

```bash
git checkout --detach aa1e2731e0bdfa2a9f315f8cc9c758e3270474dc
uv sync --locked --python 3.12 --no-dev
uv run --locked --no-dev advaitic-check

est_sha=bd8ed33c87383b3b208ba251d8fa0b6f957d9f03a35cf6c8d23a81ce9cd867cd
head_sha=64dfe105e7f716c0ab5b7a9c487b7a8a1cac06565070b390e75992dfbee4deff
# Both weight files are committed under checkpoints/.
printf '%s  %s\n' "$est_sha" checkpoints/est_v1b/last.pt \
  "$head_sha" "checkpoints/astra/$head_sha.npz" | sha256sum --check

uv venv --python 3.12 .policy
uv pip install --python .policy/bin/python --index-url https://download.pytorch.org/whl/cpu \
  torch==2.5.1 torchvision==0.20.1
eval_site=$(uv run --locked --no-dev python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
policy_site=$(.policy/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
printf '%s\n' "$eval_site" > "$policy_site/advaitic_eval_env.pth"

export ADV_ESTIMATOR="$PWD/checkpoints/est_v1b/last.pt" ADV_HEADS="$PWD/checkpoints/astra/$head_sha.npz"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa ORT_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=''

seed=1039000
uv run --locked --no-dev advaitic-eval \
  --policy method.modular.vertical_clearance:make_policy --policy-python "$PWD/.policy/bin/python" \
  --method modular-controller --checkpoint "est_v1b@sha256:$est_sha+heads:$head_sha.npz" \
  --episodes 30 --seed-start "$seed" --out "results/modular-$seed"
```

Factories for older commits:

| Commit | Factory |
| --- | --- |
| `e588ae9` | `submission.policy_modular:make_policy` |
| `9de5c51` | `submission.policy_modular_heads:make_policy` (heads `contact_v1_51257d33.npz`) |
| `bd951bc` | `submission.policy_modular_calibrated:make_policy` |
| `528c854` | `method.modular.base_clearance:make_policy` |
| `54080e5`, `414f15a` | `submission.policy_modular_combined:make_policy` |
| `99e72f6` | `submission.policy_modular_round4:make_policy` |
| `876a00f` | `submission.policy_modular_round4:make_measured_policy` |

Rebuild the contact head and grasp normal from the labeled archives (`runs` is the saved results directory):

```bash
out=$(mktemp -d)
OPENBLAS_NUM_THREADS=1 uv run --locked python -m method.modular.fit_refinement \
  contact "$out/contact_v1.npz" "$runs/astra_r3/training" "$runs/astra_r3/training_obs"
uv run --locked python ops_modular/astra_calibrate_normal.py "$runs/grasp_hand_frame.fit.json" \
  --contact "$out/contact_v1.npz" --output "$out"
```

Retrain the estimator (CUDA, with the original data mounted; the saved checkpoint is from step 8,000 of a 10,000-step run):

```bash
MUJOCO_GL=egl ORT_DISABLE_TELEMETRY=1 uv run --locked python -m method.harness.label_replay \
  /data/dagger3 --out /data/dagger3_priv
python -m method.harness.train_estimator --device cuda --cache-dir /cache \
  --run-dir /runs/est_v1b \
  --data-dirs /data/teacher_v1_dart0 /data/teacher_v1_gen2 /data/dagger3_priv \
  --steps 10000 --batch-size 64 --microbatch-size 32 --val-episodes 6 \
  --val-every 1000 --num-workers 2 --seed 42
```

## Credits

- **SWIRL BiGym2:** task, robot model, simulator and demonstrations. Upstream NVIDIA controller and asset terms still apply ([notices](../THIRD_PARTY_NOTICES.md)).
- **Advaitic:** evaluator, scoring and the observation/action interface.
- **Torchvision:** ImageNet ResNet-18 weights.
- **AI assistance:** Codex workers Astra (kinematics, perception heads, integration, clearance), Sol (phase control, release fallback, paired tests) and Astra-grasp (pickup and grasp calibration). Claude led coordination, candidate selection and evaluation.
