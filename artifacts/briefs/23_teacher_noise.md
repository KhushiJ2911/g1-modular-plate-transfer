# Brief #23 (Astra, ~60 min): how accurate must a state estimator be for the teacher controller to still work?

Worktree `/home/keyu/advaitic/wt-teacher` (HEAD 961a6d3). Rules: no evaluator/physics/success changes; training seeds
only (2000000–2999999; never 1000000-block dev, 2600000+ confirm, 7310000+ test); `ADV_THROTTLE=1` + nice locally;
leave changes uncommitted (lead commits). Modal allowed (app names `advaitic-tnoise-*`, smoke first, ≤ $5 total; teacher on
Modal: `ops/modal_teacher.py`). Network: Modal CLI works.

## Why
Our imitation student is capped by the teacher (teacher v1 ≈ 79–90%, student ≈ 60–70%; target 90%). Plan: run a
teacher-like state-machine controller on state *estimated from RGB + proprioception* instead of privileged state.
Before building the estimator we need the tolerance budget.

## Do
1. Inventory exactly which privileged quantities `method/teacher/teacher.py` (and the sensors it calls,
   `method/teacher/diagnostics.py`) reads at each phase (approach, pre-grasp, grasp, lift, carry, place, release):
   plate pose (pos + orientation), rack/slot pose, contacts, anything else. Table: quantity, phase, how it's used.
2. Add a default-off noise-injection hook (e.g. `--state-noise` config): perturb those quantities as an estimator
   would: per-episode bias + per-step noise (position cm, orientation deg), optionally a short lag. Smoke it.
3. Sweep on 30 training seeds (2000000–2000029, paired with the v1 27/30 baseline; same mode/flags as that run —
   find them in `/home/keyu/advaitic/robot-learning-assignment/EXPERIMENTS.md` line ~33): position noise
   {0.25, 0.5, 1.0, 2.0} cm and orientation {1, 2, 4} deg (one-at-a-time from a sensible base, not a full grid),
   plus "bias only" vs "jitter only" at 1 cm. Report success and failure mode per setting.
4. Separately: why does slot 1 fail (3/6 slot-1 seeds in the baseline)? One paragraph with evidence.
5. Report: tolerance table → "an estimator needs position error ≤ X cm and orientation ≤ Y deg in phase Z to keep
   ≥ 85% of teacher success", plus Modal spend.
