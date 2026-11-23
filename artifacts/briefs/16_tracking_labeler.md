# Brief #16: tracking teacher labeler -> full-action DAgger labels for bce1 rollouts. TODAY (~1-1.5 h).

Worktree `/home/keyu/advaitic/wt-teacher` (branch `teacher`, HEAD 2cc0622). Commit when done (if git metadata
is read-only, leave files and say so). Rules: analysis/data tool only; never modify the evaluator, physics,
success checks or source data; never seeds in [2600000,2700000) or [7310000,7320000); `ADV_THROTTLE=1` and at
most 2 processes for anything heavier than unit tests.

## Context
`method/teacher/teacher.py` `_act_replay`: teacher v1 picks a demo k at reset (nearest_demos), then replays the
demo's absolute arm/gripper actions `actions[k][t]` step by step, with base feedback only (desired pelvis xy =
demo pelvis path + shift_at(t), P-control `kp`, feed-forward from the demo's base command, grasp/release waits).
So it can't label arbitrary states. Our student (bce1) drifts off the teacher manifold; DAgger so far only had
gripper labels (`method/dagger/`, `/home/keyu/advaitic/analysis/relabel_dagger.py`) and never helped.

## Build `method/teacher/track_label.py`
Given a directory of student episodes (`/home/keyu/advaitic/data/dagger3`: 120 bce1 rollouts, each with `seed`,
`action_executed`, states, rgb), replay each episode deterministically from its seed (camera-free, exactly as
relabel_dagger.py does; assert the replayed `state` equals the stored state bit-for-bit, fail loudly otherwise),
and at every step compute a FULL 20-dim teacher label:
1. Demo choice and grasp/place shifts exactly as teacher v1 does at the episode start (reuse Teacher code;
   don't fork the math).
2. Phase matching: at step i, choose demo phase t* = argmin over a forward window [t_prev, t_prev + W] of a
   weighted distance between the student's state and the demo state at t (arm joint positions for the active
   arm + gripper state, plus plate-relative progress if available from PrivilegedSensors: e.g. held / lifted /
   distance to target). Phase must be monotone non-decreasing. W, weights in config; document the choice.
3. Label = demo action at t*+1 for arms (absolute targets) with the same base feedback as `_act_replay`
   (desired pelvis xy from demo path at t* + shift, P-control from the student's actual pelvis), no
   grasp/release waits. Gripper (dim 18): use the strict DAgger rule from
   `/home/keyu/advaitic/analysis/relabel_dagger.py` (open: touching target rack, site dist <= 4 cm, tilt <= 10 deg;
   closed: held and lifted >= CLOSED_LIFT_M and not touching), and where that rule is undefined, the demo's
   gripper at t*+1.
4. Write new episodes to a NEW dir (`/home/keyu/advaitic/data/dagger3_track`): same arrays as the source, with
   `action` replaced by the labels and `label_mask` all True (so the BC trainer uses every dim), plus
   `track_phase` (the t* sequence) and `track_cost`. Never overwrite.
5. Sanity check (must report): run the labeler on 3 TEACHER episodes (e.g. `/home/keyu/advaitic/data/teacher_v1_dart0`,
   which have `action_executed`); the labels should reproduce the teacher's own executed arm actions almost
   exactly (report per-dim max/mean abs diff and where they differ). Also report, for 3 dagger3 episodes, the
   phase trajectories and label-vs-executed differences (where does bce1 deviate most?).
6. A flag to drop steps whose `track_cost` is above a threshold (student too far from any demo state ->
   label unreliable); report the fraction dropped at a sensible default. Dropped steps get label_mask False.
CLI: `python -m method.teacher.track_label SRC --out DST [--shard i/n] [--limit N]`. Unit tests for phase
monotonicity and the window. Report (`-o` file): design, sanity numbers, exact commands. Keep it simple.
