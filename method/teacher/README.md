# Privileged demo-transfer teacher

This is a training-demonstration generator, **not an evaluated student policy**.
It reads the real plate pose and destination rack origin at reset. This privileged
geometry is unavailable to a student during evaluation. It never changes physics,
poses, termination, success checks, or the evaluator. Human demonstrations come
from the supplied BiGym2-derived `method/data/demos_lowdim.npz` and
`method/data/demo_meta.json`; the controller and tooling were implemented with Codex.

## Controller

`Teacher.reset(env)` reads geometry after the environment's ordinary reset and
warmup. The first `act(ts)` uses that reset timestep's pelvis xy to select the demo
with nearest plate-relative-to-starting-pelvis xy. It excludes demo 45, which was
reported to fail even on its original seed. Selection uses neither seed nor source
slot. `nearest_demos(plate_xy, pelvis_xy, top_n=N)` also returns ranked candidates.

For reference demo k, the world-frame translations are:

```text
g = (plate_xy - pelvis0_xy) - (demo_plate_xy - demo_pelvis0_xy)
h = target_rack_offset_xy - demo_target_rack_offset_xy
desired_pelvis_xy(t) = demo_pelvis_xy(t) + shift(t)
```

The destination rack's nominal xy origin is `[0.55, -0.24]`, read from the pinned
`MovePlateG1` implementation. The source plate comes from `get_pose()[:2]` after
warmup. The destination remains the reference demo's placement location translated
with its rack; v1 does not infer a destination slot or replay demos to label one.
The sampled source slot is recorded by the runner for analysis only.

`shift(t)` is g through `grasp_step + 30`, blends linearly to h through
`release_step - 40`, then stays at h. The default proportional gain is 2/s. Rotate
the world xy position error by minus the observed pelvis yaw, add feedback to the
demo's raw body-frame vx/vy, clip to the raw action bounds, and normalize again.
Height, yaw command, both arms and both grippers retain the demo's targets.

The single demo clock stops **before** the closing/opening action when xy error is
at least 1 cm, holding the preceding arm/gripper targets for at most 100 additional
environment steps per event. During a wait, velocity feed-forward is zero and the
pelvis target is fixed. The event proceeds when error drops below 1 cm or the wait
budget expires. These waits count toward the unchanged 1,700-step episode budget.
After the demo ends, hold its final action with both grippers open, without further
base corrections, until the environment terminates. There is no grasp detection,
retry, or candidate switching in v1; measure v1 before adding those features.

API example (inside an operator-run simulator process):

```python
from advaitic_eval.environment import make_environment
from method.teacher.teacher import Teacher

env = make_environment()
teacher = Teacher()
try:
    ts = env.reset(seed=100000)
    teacher.reset(env)
    while not ts.last():
        ts = env.step(teacher.act(ts))
    print(env.episode_succeeded(), env.episode_termination())
finally:
    env.close()
```

## Commands

Run from the worktree root. Use the committed lockfile; no dependency changes are
needed. If this worktree has no environment, the operator can prepare one with:

```bash
cd /home/keyu/advaitic/wt-teacher
uv sync --locked
```

CPU-only implementation checks (fake environments only; no rendering or rollout):

```bash
ORT_DISABLE_TELEMETRY=1 uv run --locked ruff check method
ORT_DISABLE_TELEMETRY=1 uv run --locked ruff format --check method
ORT_DISABLE_TELEMETRY=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  uv run --locked python -m unittest discover -s method/teacher -p 'test_*.py' -v
uv run --locked python -m method.teacher.run_teacher --help
```

Generate and measure on fresh **training** seeds; the upper range endpoint is
exclusive. These example seeds are outside the provided demo seed set. Keep them
separate from any final reviewer evaluation block:

```bash
ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl uv run --locked \
  python -m method.teacher.run_teacher \
  --seeds 100000:100030 \
  --out method/teacher/runs/v1.jsonl \
  --save-demos method/teacher/generated/v1
```

Run the identical command to resume: completed seeds, including failures, are
skipped. Explicit lists, comma-separated lists and stepped ranges also work, e.g.
`--seeds 100000,100003 100008 100010:100020:2`. Logs include per-slot counts; a
consecutive seed block is not necessarily balanced across slots. For a balanced
30-episode diagnostic block, print a seed list using the unchanged protocol and
paste those integers after `--seeds` with a **new** output filename:

```bash
uv run --locked python -c 'from advaitic_eval.protocol import make_plan; print(" ".join(str(ep.seed) for ep in make_plan(30, 100000)))'
```

For an open-loop nearest-demo ablation, run the same seeds with
`--kp 0 --max-wait 0` and a new output filename. Change `--kp`, `--tolerance`
(meters), or `--max-wait` only in separate experiment logs. The log stores these
settings and a hash of the controller, runner, data, metadata and action bounds;
resuming with a changed configuration is rejected.

The runner is always single-process and launches no workers. To use two processes,
the operator can run the following in **two terminals**, at most two at once, with
disjoint seeds and separate logs. No `--workers` argument is needed:

```bash
# Terminal 1
ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl uv run --locked \
  python -m method.teacher.run_teacher --seeds 100000:100060:2 \
  --out method/teacher/runs/v1_even.jsonl --save-demos method/teacher/generated/v1

# Terminal 2
ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl uv run --locked \
  python -m method.teacher.run_teacher --seeds 100001:100060:2 \
  --out method/teacher/runs/v1_odd.jsonl --save-demos method/teacher/generated/v1
```

The runner disables ONNX telemetry and hides CUDA devices before importing the
simulator. EGL graphics may still use the operator's GPU for rendering. On a host
with CPU rendering available, replace `MUJOCO_GL=egl` with `MUJOCO_GL=osmesa`.

## Logs and successful demonstrations

Each completed episode adds one JSONL record: seed, zero-based sampled source slot,
chosen demo, grasp/place xy shifts in meters, success, termination, fall flag,
steps, maximum pre-action pelvis xy tracking error in meters, event wait counts,
demo clock, saved path, configuration hash and wall time in seconds. Wall time
includes reset and saving. Success requires `episode_succeeded()` and no fall;
the runner steps until the real environment terminates, including its held-success
requirement. Ordinary falls/timeouts are logged failures. An exception or Ctrl-C
leaves the unfinished seed pending and stops the run. Completed episodes alone
enter the summary, and failures are never removed when resuming.

The log is append-only and flushed/fsynced after each completed episode. A file
lock rejects simultaneous writers to the same log. A kill during append can leave
a partial line: readers warn and ignore malformed lines; the next append preserves
that fragment and starts its record on a new line. Never share overlapping seeds
between processes writing into the same demonstration directory.

With `--save-demos DIR`, successful episodes produce `DIR/ep_<seed>.npz`:

| Key | Shape / contents |
| --- | --- |
| `state` | `(T, 50)` float32, every **pre-action** observation |
| `action` | `(T, 20)` float32, normalized action paired with each state |
| `rgb` | `(ceil(T/2), 3, 3, 84, 84)` uint8, pre-action RGB at indices 0, 2, 4, ... |
| `rgb_stride` | scalar integer 2; `rgb[j]` pairs with `state[2*j]` |
| `seed`, `slot` | scalar integers; slot is zero-based |
| `success` | scalar boolean true |

T is the number of executed control steps through environment termination. There
is no terminal post-action observation in these arrays. Unsuccessful episodes
produce log records but no NPZ. Saving uses a temporary file in DIR, fsync, then
`os.replace`; the completed record is appended afterward. A kill can leave an
ignored `.ep_*.npz` temporary file or a complete NPZ without a log record. Rerunning
that pending seed atomically replaces its NPZ. Enable `--save-demos` from the start
if data are wanted: resuming a log does not backfill already-completed episodes.

## Verification in the coding sandbox

Five CPU-only checks passed: all 59 usable demo action streams are reproduced when
fed their recorded states with zero residual; selection/shift blending/body-frame
feedback; bounded grasp/release waits; interrupted JSONL recovery; and runner
resume/failure/interruption plus NPZ dtype, stride and temporal alignment. These
use fake environments and are **not task-solving rollouts**.

The worktree lacked `.venv`. Checks used the existing neighboring environment with
the same `pyproject.toml` and `uv.lock` (both compared byte-for-byte):

```bash
export UV_PROJECT_ENVIRONMENT=/home/keyu/advaitic/robot-learning-assignment/.venv
export UV_CACHE_DIR=/tmp/teacher-uv-cache
export RUFF_CACHE_DIR=/tmp/teacher-ruff-cache
ORT_DISABLE_TELEMETRY=1 uv run --locked --no-sync ruff check method
ORT_DISABLE_TELEMETRY=1 uv run --locked --no-sync ruff format --check method
ORT_DISABLE_TELEMETRY=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  uv run --locked --no-sync python -m unittest discover -s method/teacher -p 'test_*.py' -v
```

No GPU was used. A CPU OSMesa probe failed on import with
`AttributeError: 'NoneType' object has no attribute 'glGetError'`. No simulator
episodes were run. **The teacher's real success rate and the ≥90% goal remain
unverified.** Privileged teacher results are training-workflow measurements and
cannot establish the student policy's assignment target.
