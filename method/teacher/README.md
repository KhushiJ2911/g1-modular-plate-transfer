# Privileged demo-transfer teacher: v1 / v2

This generates training demonstrations. It is **not an evaluated student policy**:
the teacher reads live simulator plate poses, target sites, contacts and robot
kinematics that the student cannot access. It sends ordinary normalized actions;
it never writes simulator poses, physics, success rules or evaluator state.
No evaluator, dependency, fixture or integrity-manifest changes are needed.

The operator measured v1 at **27/30** on seeds 2000000–2000029. All three failures
were source slot 0; their grasps succeeded and their placement/release failed.
That is the supplied v1 measurement, not a new result from these implementation
checks, and does not demonstrate the student policy's 90% target.

## Shared replay and v1 ablation

After ordinary reset/warmup, choose the nearest demonstration by plate xy relative
to starting pelvis xy, excluding demo 45. Neither seed nor sampled source slot
selects the demonstration. `nearest_demos(..., top_n=N)` exposes the ranking.
The runner records the sampled slot only for analysis.

For demo k, the initial translation g aligns its grasp with the actual plate;
the placement translation h aligns its nominal rack with the actual target rack:

```text
g = (plate_xy - pelvis0_xy) - (demo_plate_xy - demo_pelvis0_xy)
h = target_rack_offset_xy - demo_target_rack_offset_xy
desired_pelvis_xy(t) = demo_pelvis_xy(t) + shift(t)
```

The nominal destination-rack origin is `[0.55, -0.24]`, from pinned MovePlateG1.
`shift(t)` holds g through `grasp_step + 30`, blends linearly to h through
`release_step - 40`, then holds h. Arms, grippers, height and yaw follow a single
demo clock except during the explicitly described v2 corrections/recovery.

`--mode v1` preserves the original action controller: recorded base velocity plus
2/s pelvis feedback in the observed body frame, a 1 cm pelvis gate before closing
and opening, each with a 100-step wait limit. After replay it holds the last demo
action with open grippers. There is no object feedback, wrist correction or
retreat guard in v1. Diagnostics are now collected in both modes.

## V2 (default)

V2 closes the placement loop on the plate instead of assuming the plate follows
the pelvis. Its phases are:

1. **Approach/grasp/carry:** follow the same shifted pelvis path. V2 base control
   uses position feedback without demo planar velocity feed-forward: stop inside
   the 1.5 cm pelvis tolerance; otherwise command the error direction at
   `max(0.06 m/s, 2/s * distance)`, transformed into the body frame and clipped to
   action bounds. The minimum exceeds the WBC's approximately 0.05 m/s walk-entry
   threshold. The grasp gate retains its 100-step limit.
2. **Placement alignment:** from the end of the shift blend (`release_step - 40`)
   until opening, recompute the nearest of all six destination sites using the
   current plate position. Control **plate xy** toward that site's xy with the
   same deadzone-aware feedback, stopping within **1 cm**. Do not add the recorded
   base velocity while trying to center the plate.
3. **Lowering gates:** pause at `release_step - 40` if plate xy error is at least
   1.5 cm, holding that frame's arm targets; pause again immediately before the
   release action, holding the preceding closed-gripper frame. Each placement
   gate has a 150-step budget. The first gate preserves the final 40 demo frames
   as a distinct lowering/settling segment and aligns before replaying them; the
   second catches movement during that segment. This uses the brief's explicit
   timing fallback, **not a claim that frame -40 is a measured z-velocity boundary
   in every demo**. Arm height/lowering remains demonstration driven. If a wait
   expires, continue and retain its count in the log.
4. **Tilt before release:** use exactly the task convention: rotate local +z by
   the plate's wxyz quaternion and take the angle to world −y. Above **14°**, try
   a bounded correction through **left wrist roll/pitch, action dims 8/9**.
   The G1 XML uses local x/y hinge axes. Read their live world axes and use
   `d(plate_up)/dq = axis × plate_up`; damped least squares rotates the measured
   normal toward −y, assuming a rigid grasp. This determines the sign and relative
   influence at the actual posture, rather than guessing a universal roll sign.
   Distal wrist corrections avoid the larger hand translations of shoulder/elbow
   rotation. The live reset-pose probe found normal sensitivities of approximately
   **1.000/rad for wrist roll and 0.112/rad for wrist pitch** (not a held-release
   probe); the controller recomputes its Jacobian at the actual release posture.
   Resolve joints by name: the WBC's `limb_joints` also contains legs and waist,
   so indexing that list by arm action offset would select the wrong joints.
   Limit the joint correction vector to 0.01 rad per step and each accumulated
   offset to ±0.18 rad. Try at most 60 steps within the release gate's 150-step
   budget; stop early when tilt improves to ≤14°. Keep the offset in subsequent
   targets to avoid a wrist snap. If tilt remains high, open anyway and log the
   unresolved tilt, correction steps, offsets and initial normal sensitivities.
   No finite-difference pose injection or simulation-state write is used.
5. **Opening/settling:** stop planar locomotion, fully open both grippers, then
   hold the release targets for **25 additional steps**. V2 keeps planar base
   commands zero throughout retreat.
6. **Guarded retreat:** replay the demo retreat with fingers open. Compare the
   plate with its command-time release pose every step. A position change over
   **1.5 cm in 3D**, or a tilt increase over **5°**, stops the arm at its last
   commanded targets for **50 steps**, with fingers open. Still touching either
   gripper at 50 steps after opening triggers the same recovery. Then interpolate
   a vertical hand goal upward **6 cm over 50 steps**, changing only **left
   shoulder pitch/elbow, action dims 4/7**. Use their positional hinge Jacobian
   `dp/dq = axis × (hand − anchor)` with damping, a 0.015 rad per-step bound and
   ±0.35 rad total offsets. The hand point is the G1's `left_end_effector` site
   returned by `robot.get_hand_pos`, at the pad center 0.148 m along wrist yaw;
   the generic gripper's default `pinch_site` is at its wrist-body origin.
   Those two joints provide an **approximate** vertical
   path; exact xyz tracking is generally impossible with only two joints. Other
   arm targets, height and yaw remain held during recovery. Rejoin the retreat
   over 50 steps with monitoring still active. After each lift, re-arm the guard
   from the latest plate pose to detect additional disturbance without looping
   on the same old displacement. Every trigger retains its pose and reason.

All pauses use physical steps from the unchanged 1,700-step episode budget. There
is no grasp retry or demo switching. Tilt correction assumes the plate follows
the hand; rack contact, slipping, joint saturation and limited reach can defeat
it. The diagnostics expose those cases rather than declaring them solved.

## Run and resume

Use the committed lockfile, from the repository root (`uv sync --locked` if an
environment needs preparing). The runner is single-process and starts no workers.
The runner disables ORT telemetry and hides CUDA before importing the simulator.
For **CPU rendering** on this Linux host, force Mesa's vendor and software rasterizer:

```bash
export ORT_DISABLE_TELEMETRY=1 CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void
export MUJOCO_GL=egl LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=1 OMP_NUM_THREADS=1
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export EGL_PLATFORM=surfaceless PYTHONDONTWRITEBYTECODE=1

uv run --locked python -m method.teacher.run_teacher --mode v2 \
  --seeds 2000000:2000030 --out method/teacher/runs/v2.jsonl \
  --save-demos method/teacher/generated/v2

# Same seeds, separate ablation log and demonstration directory.
uv run --locked python -m method.teacher.run_teacher --mode v1 \
  --seeds 2000000:2000030 --out method/teacher/runs/v1_ablation.jsonl \
  --save-demos method/teacher/generated/v1_ablation

# Compact per-25-physical-step trace, plus terminal sample and JSON diagnostics.
uv run --locked python -m method.teacher.diag_trace --mode v2 \
  --seeds 2000000 2000007 2000024
```

Mesa EGL was verified to report `llvmpipe (LLVM 15.0.7, 256 bits)` here. Ordinary
EGL can use a GPU; **keep the software/vendor settings for a no-GPU run**. OSMesa
is another CPU option where installed, but its import fails on this host.

Ranges use an exclusive upper endpoint; integers, comma lists and stepped ranges
work: `--seeds 100000,100003 100008 100010:100020:2`. Keep development seeds separate
from the final reviewer evaluation block. Each physical trace row reports step,
next demo clock, plate xyz, nearest site's 3D and xy distances (meters), tilt
(degrees), rack/table/floor contacts, held predicate and recovery phase.
`diag_trace.py` executes real actions without saving demonstrations or resuming;
its final JSON contains the same episode diagnostics as the runner.

Repeat the identical runner command to resume; completed failures are also skipped.
The `run_id` hashes the runner, teacher, diagnostic helper, input demos/metadata,
fixed action layout and all controller settings, **including mode**. Changed code
or settings require a fresh JSONL, including for the newly instrumented v1.
`--kp` sets both pelvis and object feedback gains. `--tolerance` and `--max-wait`
configure pelvis gates; in v2 the object gates remain fixed at 1.5 cm/150 steps. Defaults are 1 cm for v1 pelvis gates and 1.5 cm
for v2. The v1 open-loop ablation is `--mode v1 --kp 0 --max-wait 0`.

## Records and saved demonstrations

Each completed JSONL record retains seed, source slot, chosen demo, shifts,
`success`, termination, robot fall flag, physical steps, maximum pelvis tracking
error, gate waits, demo clock, saved path, config/hash and wall time. It adds:

- `plate_at_grasp_plus_60`, `plate_at_release`, `plate_at_end`: snapshots with
  `pose=[x,y,z,qw,qx,qy,qz]`, physical `step`, nearest-site index and xyz, 3D/xy
  distance, tilt, either-gripper `held`, and rack/table/floor contacts.
- Release/end distance and tilt also appear as explicit top-level fields;
  `held_at_end` and `contacts_at_end` expose the remaining task predicates.
- `plate_50_steps_after_opening` preserves the held-contact deadline even during
  another recovery. Actual first-close/open physical indices, carry-drop flag, tilt-correction
  evidence and recovery triggers. A never-reached milestone is `null`.

Grasp+60 means 60 **executed control steps** from the first closing command,
including any later demo-clock pauses. The height increase is relative to the
post-warmup reset pose. The release snapshot is **immediately before the first
opening action**, so `held` can correctly be true there. Carry contact is sampled
every step from grasp+60 through that pre-opening snapshot; a single false
predicate sets `drop_in_carry` (this is the task's finger-contact predicate, not a
stronger physical grasp estimator).

`failure_category` has this precedence, with raw snapshots retained for ambiguity:
`success`; `fell` (robot fall or plate-floor contact); `grasp_miss` (measured rise
at grasp+60 <3 cm); `drop_in_carry`; `knocked_after_release` (release had ≤5 cm,
≤20°, rack contact and no table/floor contact, but end geometry/contact is bad);
`place_position` (>5 cm at end); `place_tilt` (>20° at end); `not_released` (held at
end); `other`. Release goodness excludes the held predicate because it precedes
opening. Unreached grasp+60 is not invented; other available failure evidence
still applies. A timeout can be `other` if the terminal geometry is good but the
unchanged consecutive-success timer was not met.

The runner steps through real termination and reports success only when
`episode_succeeded()` and no robot fall. Fake-environment checks cannot establish
that success. Exceptions/interruption leave the unfinished seed pending.

Logs remain append-only, locked against simultaneous writers, flushed and fsynced
per completed episode. Partial interrupted lines are preserved and ignored on
resume. With `--save-demos DIR`, only successful episodes atomically publish
`ep_<seed>.npz` via a temporary file, fsync and `os.replace`, before logging:

| Key | Contents |
| --- | --- |
| `state`, `action` | `(T,50)` / `(T,20)` float32, paired **pre-action** observations/actions |
| `rgb` | `(ceil(T/2),3,3,84,84)` uint8, pre-action frames 0,2,4,... |
| `rgb_stride` | scalar 2; `rgb[j]` pairs with `state[2*j]` |
| `seed`, `slot`, `success` | integer seed/slot; boolean true |

No terminal post-action observation is saved in these arrays. Resuming does not
backfill successful demos if saving was originally disabled. Use disjoint seeds,
logs and directories if manually running multiple processes.

## Implementation verification

The worktree has no `.venv`; the existing neighboring environment's `pyproject.toml`
and `uv.lock` were compared byte-for-byte with this worktree. Exact CPU check
commands (no changes to either lockfile or environment):

```bash
export UV_PROJECT_ENVIRONMENT=/home/keyu/advaitic/robot-learning-assignment/.venv
export UV_CACHE_DIR=/tmp/teacher-uv-cache RUFF_CACHE_DIR=/tmp/teacher-ruff-cache
ORT_DISABLE_TELEMETRY=1 uv run --locked --no-sync ruff check method/teacher
ORT_DISABLE_TELEMETRY=1 uv run --locked --no-sync ruff format --check method/teacher
ORT_DISABLE_TELEMETRY=1 CUDA_VISIBLE_DEVICES='' MUJOCO_GL=disable \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src uv run --locked --no-sync \
  python -m unittest discover -s method/teacher -p 'test_*.py' -v
uv run --locked --no-sync python -m method.teacher.run_teacher --help
uv run --locked --no-sync python -m method.teacher.diag_trace --help
```

Twelve CPU contract checks cover all 59 usable demos' exact v1 replay, selection,
blend/body-frame feedback, bounded gates, interrupted logs, atomic demonstration
alignment/resume, cross-mode run-ID rejection, object versus pelvis feedback,
deadzone behavior, wrist sign/bounds/timeout, guarded retreat/lift, held-at-50,
physical diagnostic timing, failure categories and named joint lookup with legs first. They use fake environments,
never injected simulator fixture poses, and are **not task-solving rollouts**.

The corrected adapter was also checked on a real, unmodified seed-2000000 reset:
8/9 resolved to left wrist roll/pitch, 4/7 to left shoulder pitch/elbow, world
axes/anchors matched the model's named joints, and the NumPy plate normal matched
`pyquaternion` to 1e-12. This was read-only geometry inspection, without injected
poses or a placement attempt. The exact probe script for this session is
`/tmp/teacher_live_kinematics_probe.py`; with the CPU-rendering exports above:

```bash
PYTHONPATH=src:. uv run --locked --no-sync python /tmp/teacher_live_kinematics_probe.py
```

A real CPU-rendered trace was started with the command below and interrupted after
its **100-step trace row**, before grasp. It exercised normal reset/actions,
privileged snapshots and trace formatting; it produced no completed episode or
success claim. That early smoke trace preceded the named-joint adapter fix, but
never reached either wrist correction or recovery. The corrected adapter was
verified separately as described above.

```bash
uv run --locked --no-sync python -m method.teacher.diag_trace --mode v2 \
  --seeds 2000000 > /tmp/teacher-v2-trace-2000000.txt 2>&1
```

**Unverified:** no complete v2 task-solving episode or new 30-seed comparison was
completed in this session. Live wrist/contact response, placement convergence,
the recovery lift's actual path and v2 success rate remain to be measured. The
contract checks are not evidence of the student policy's ≥90% target.
