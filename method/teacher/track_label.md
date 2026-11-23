# Full-action tracking labeler (Brief 16)

`track_label.py` is an analysis/data tool. It replays only each source episode's
`action_executed`, and never executes a generated label. Evaluator, physics,
success predicates, source episodes, original demonstrations, and `Teacher`
are unchanged. Both prohibited seed ranges are rejected before simulator
construction. Run no more than two workers, always with `ADV_THROTTLE=1`.

## Design

The labeler uses `Teacher(mode="v1", max_wait=0)`. After resetting from the
source seed, it calls the original `reset` and `_act_replay` to initialize demo
selection and grasp/place shifts. The initialization action is discarded.

At every saved **pre-action** state, the replayed observation is converted to
the source format (`float32`) and compared byte for byte, including signed zero.
A mismatch identifies seed, step, dimensions, and maximum difference and fails
before that episode can be published. Early environment termination also fails.
Output files are compressed to a temporary file, flushed, fsynced, and published
with an exclusive hard link; existing files are never replaced. Each shard has
its own config and report. A failed run may leave already verified episodes and
its config, but never publishes the failing episode.

The phase cost, configured in `track_config.json`, is:

```text
arm_weight * mean(((student_q_left - demo_q_left[t]) / arm_scale_rad)^2)
  + gripper_weight * ((student_gripper - demo_gripper[t]) / gripper_scale)^2
```

Defaults: left arm qpos `state[0:7]`, left gripper `state[44]`, arm weight 1,
gripper weight 0.1, arm scale 0.2 radians, gripper scale 0.6. The left arm is
active in these plate-transfer demos. These are observation indices, not action
indices: `state[7:9]` are adjacent finger joints; pelvis xy/yaw are 18:20/21.
The mean over joints prevents the arm term's magnitude depending on arm DOF.
The weak gripper term helps distinguish open/closed poses without dominating
the arm geometry. No elapsed-time prior forces the student's clock forward.

Choose the earliest argmin in the inclusive window
`[previous_phase, min(previous_phase + W, last_action_index)]`, initially from
phase 0. `W=25` allows up to 0.5 seconds of demo phase advancement per student
step at 50 Hz. Phase may stay still and cannot decrease. It can become stuck
at a local minimum; this is visible in the report and is not automatically
recovered by jumping outside the configured window.

**Plate progress limitation:** the fixed demonstration archive contains only
`a{k}`, `s{k}`, and `seed{k}`. It does not contain aligned held/lifted/site-distance
observations. A camera-free replay probe of demo 0, seed 18, differed from
`s0[0]` already at reset (maximum observed-state difference 0.017852873).
Consequently this implementation does not invent privileged progress references
from that replay. Phase cost uses archived arm/gripper observations. Live
`PrivilegedSensors` supply the strict gripper rule below. Reliable aligned demo
geometry would be needed before adding a plate-progress phase term.

For a matched phase `t*`, call the original `_act_replay` at that phase with
waits disabled. This retains its complete base behavior: desired pelvis xy
`states[t*,18:20] + shift_at(t*)`, actual student's pelvis/yaw, body-frame P
feedback (`kp=2`), demo base feed-forward at `t*`, normalization and clipping.
Dimensions 2:4 also retain the demo's base commands at `t*`. Replace dimensions
4:20 with demo actions at `min(t*+1,last_action_index)`, preserving absolute
targets for both arms and the other gripper. Then override dimension 18:

- Open (-1): target-rack contact **and** site distance <= 0.04 m **and** tilt <= 10 degrees.
- Closed (+1): held **and** height above this episode's reset plate >= 0.03 m **and** no target-rack contact.
- Otherwise: the demo's gripper at `t*+1`.

This is the strict rule from `analysis/relabel_dagger.py` with its closed rule
unchanged. It is a training label, not a modification to task success.

Each output archive preserves every source array except `action` and
`label_mask`, and adds `track_phase` (int64) and `track_cost` (float64). It keeps
the original observations, RGB alignment, and executed actions. With no filter,
`label_mask` is True in all 20 dimensions. `--drop-cost` alone selects threshold
4; `--drop-cost X` selects X. Entire rows with cost > X receive False masks;
rows and their RGB alignment remain present. Threshold 4 is roughly an arm RMS
error of 0.4 rad (23 degrees) when the gripper agrees. This is a practical
heuristic, not a calibrated guarantee that retained labels are reliable.

The JSON `-o` report contains design/provenance, SHA256s, exact CLI arguments,
all phase trajectories, per-dimension mean/max absolute label differences,
worst-step indices, counts differing by more than 1e-6, cost quantiles, strict
gripper counts, and masked fractions. The source success fields are copied
metadata about the original rollouts; they do not evaluate the generated labels.

## Commands

Normal locked-environment invocation (OUT must be new):

```bash
ADV_THROTTLE=1 ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl uv run --frozen python -m method.teacher.track_label /home/keyu/advaitic/data/dagger3 --out /home/keyu/advaitic/data/dagger3_track --drop-cost -o /home/keyu/advaitic/reports/dagger3_track.json
```

This session cannot write `/home/keyu/advaitic/data` or git metadata. Actual
outputs are staged in this worktree under `results/dagger3_track`. The requested
external destination has not been written. Normal `uv run --frozen` setup was
also blocked: the shared uv cache is read-only and an empty writable cache
requires network access, but GitHub DNS is unavailable. Actual runs reused the
installed environment via uv; its NumPy 2.5.4, MuJoCo 3.8.1, ONNX Runtime 1.31.1,
pytest 9.2.0, ruff 0.17.4, and BiGym commit
`d71501359b215b9d78e27dcb70b220412d474dfa` match the committed lock.
All 58 installed applicable lock packages were checked, with no version
mismatches; absent entries were platform/optional packages.

Actual command environment (from worktree root):

```bash
export VIRTUAL_ENV=/home/keyu/advaitic/robot-learning-assignment/.venv
export UV_CACHE_DIR=/tmp/uv-track-cache
export ADV_THROTTLE=1 ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl

uv run --active --no-sync python -m method.teacher.track_label /home/keyu/advaitic/data/teacher_v1_dart0 --out results/track_label/teacher_sanity --limit 3 -o results/track_label/teacher_sanity.json
uv run --active --no-sync python -m method.teacher.track_label /home/keyu/advaitic/data/dagger3 --out results/track_label/student_sanity --limit 3 --drop-cost -o results/track_label/student_sanity.json

# The following two commands run concurrently, with no other heavy workers:
uv run --active --no-sync python -m method.teacher.track_label /home/keyu/advaitic/data/dagger3 --out results/dagger3_track --shard 0/2 --drop-cost
uv run --active --no-sync python -m method.teacher.track_label /home/keyu/advaitic/data/dagger3 --out results/dagger3_track --shard 1/2 --drop-cost

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --active --no-sync python -m pytest -q method/teacher/test_teacher.py method/teacher/test_teacher_data.py method/teacher/test_ablation.py method/teacher/test_track_label.py
uv run --active --no-sync ruff check method/teacher/track_label.py method/teacher/test_track_label.py
uv run --active --no-sync ruff format --check method/teacher/track_label.py method/teacher/test_track_label.py
git diff --check

# After both labeling workers finish: compare preserved .npy members byte for
# byte (including RGB), and independently check phases, masks and arm targets.
PYTHONPATH=. uv run --active --no-sync python results/track_label/verify_outputs.py
```

Disabling pytest plugin autoload avoids the unrelated system ROS plugin, which
otherwise fails before test collection due to missing `lark`.

## Measured sanity results

See `track_label_report.md` for per-dimension tables, phase trajectories, full
dataset totals, and verification results. These are offline replay/labeling
results. No generated-label rollout, student training, or policy success target
is claimed.
