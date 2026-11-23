# Codex brief #3: teacher v2, object-centric placement + diagnostics

Context: read `method/teacher/teacher.py`, `method/teacher/run_teacher.py`, `method/teacher/README.md` (your previous work, committed). Same rules: do not modify `src/`, `tests/`, `pyproject.toml`, `uv.lock`. No GPU, no git.

## Measured result of v1 (30 fresh seeds 2000000–2000029)
27/30 success. **All 3 failures are source slot index 0** (the farthest reach; 3 of 6 slot-0 seeds failed, 24/24 on other slots). Traces of the plate pose (privileged) show **every grasp succeeded**; all failures are at placement/release:
- 2000000: plate tilted up to 32° during carry; placed 3.9 cm from the nearest target site, touching the rack, but **23.6° tilt** (limit 20°). After the gripper opened, `is_gripper_holding_object` was still True (fingers in contact).
- 2000007: plate lowered **6.5 cm** from the nearest target site (limit 5 cm), clipped the rack, toppled (80°) onto the table.
- 2000024: placed well (4.6 cm, 4.8°), but during the **demo's retreat after release** the plate was dragged +17 cm in x and fell over.
- The pelvis gate tolerance of 1 cm almost never settles: 18/30 episodes hit the 100-step release wait. The WBC has a walk-entry speed of ≈0.05 m/s (dataset metadata: `planar_walk_entry_speed: 0.055`, `policy_walk_switch_speed: 0.05`, `stick_deadzone: 0.15`), so small P-commands likely produce no motion.

Root cause: v1 tracks the **robot's** pelvis trajectory and assumes the plate follows. It does not close the loop on the **object**.

## Changes (keep v1 selectable via `--mode v1|v2`, default v2, for an ablation)
1. **Object-centric placement.** From the end of the shift blend until release, use the privileged plate position: find the nearest target-rack site to the plate (any of the 6) and command the base so the **plate xy** converges to that site xy (target error ≤ 1.0 cm). Keep replaying the demo's arm trajectory for height/lowering, but **gate the demo clock just before the lowering phase completes** until plate-xy error < 1.5 cm (max wait 150 steps). Think carefully about which demo steps are "lowering into the slot". Use the plate z velocity / demo arm motion, or simply gate at `release_step - pre_release_steps` and again at `release_step`. Explain your choice.
2. **Tilt check before release.** Compute the plate tilt with the same formula as the success check (`move_plates.py::_success`: angle between plate-up and world −y). If tilt > 14° at the release gate, try to reduce it with a small correction to the left **wrist roll/pitch** joint targets (find which of action dims 4–10 changes tilt most with a finite-difference probe in sim, or reason from the kinematics; document it). If it cannot be reduced, release anyway and log it.
3. **Deadzone-aware base control.** If the position error > tolerance, enforce a minimum commanded planar speed of 0.06 m/s in the error direction (like a bang-bang near the target), else command zero. Tolerance 1.5 cm for the pelvis gates.
4. **Safe release and retreat.** After the gripper opens, wait 25 steps. Then, instead of blindly replaying the demo's retreat: replay it, but if the plate moves > 1.5 cm or tilts > 5° more than at release, **stop the arm** (hold current arm targets with the gripper open) for 50 steps, then raise the hand straight up by interpolating only the shoulder pitch/elbow (document which) before continuing. If `is_gripper_holding_object` is still True 50 steps after opening, apply the same "lift straight up" move.
5. **Diagnostics in each JSONL record:** plate pose at grasp+60, at release, at the end; nearest-target-site distance and tilt at release and end; `held_at_end`; contacts (target rack, table) at the end; and an automatic `failure_category` in {`success`, `grasp_miss` (plate z not raised ≥ 3 cm by grasp+60), `drop_in_carry` (not held between grasp+60 and release), `place_position` (> 5 cm at end), `place_tilt` (> 20° at end), `not_released`, `knocked_after_release` (good at release, bad at end), `fell`, `other`}.
6. Keep resumability, atomic saves and the run_id guard (the mode must be part of run_id). Update the README with the v1/v2 difference in plain language.

Also add `method/teacher/diag_trace.py`: run the teacher on given seeds and print a compact per-25-step trace (step, demo clock, plate xyz, nearest-site distance, tilt, contacts, held). Useful for debugging.

Report: what changed and why, anything unverified, exact commands.
