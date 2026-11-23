# Brief #24 (Sol, ~60 min): can we estimate full task state from RGB + proprioception across ALL phases?

Worktree `/home/keyu/advaitic/wt-est` (branch est2; your est_v2 work is committed at 2726967). Rules as before:
obs-only at inference, no evaluator/physics changes, training seeds only (never 1000000-block dev, 2600000+ confirm,
7310000+ test), `ADV_THROTTLE=1` + nice locally, leave changes uncommitted. Modal allowed (`advaitic-est3-*`, smoke
first, ≤ $6 total).

## Why
Imitation is capped by the teacher (~80–90%) and the student loses 15–20 points (60–70%; target 90%). Plan B: a
teacher-like controller on *estimated* state. est_v2 is good near the rack (dist MAE 0.39 cm, tilt 1.24°) but only
outputs plate-to-slot distance/tilt near placement. The teacher needs more (Astra is measuring the exact
inventory + tolerance budget in brief #23; assume for now: plate position + orientation relative to the robot/hand
in all phases, especially pre-grasp/grasp where the gripper occludes, and the target slot pose).

## Do
1. Audit est_v2 (and est_v1b) outside its training regime: per phase (approach, pre-grasp, grasp, lift, carry,
   place), how well does it track what it outputs? Use your audit tooling on teacher/flywheel/dagger frames with
   replayed ground truth.
2. Quick feasibility training (cheap, ≤ ~2k steps on A10): extend the head to regress plate position (3) +
   orientation (6D or quaternion) in the robot/pelvis frame and the target slot position, across all phases,
   from the same inputs. Use the available labelled data (teacher_v1, teacher_v1_dart0, flywheel2, dagger*) —
   training seeds only. Hold out whole episodes.
3. Report per phase: position MAE (cm), orientation MAE (deg), 90th percentile, and the occluded-grasp slice.
   Note what limits it (data, occlusion, camera views, resolution) and what a full version would need.
