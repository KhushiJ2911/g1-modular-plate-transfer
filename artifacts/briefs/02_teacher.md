# Codex brief #2: privileged demo-transfer teacher (training-data generator)

You are working in a git worktree of a robot-learning assignment repo. Read `README.md`, `METHODS.md`, `src/advaitic_eval/environment.py`, `src/advaitic_eval/runner.py` and `.venv/lib/python3.12/site-packages/bigym/envs/move_plates.py` first. **Do not modify anything under `src/`, `tests/`, `pyproject.toml` or `uv.lock`.** Put all new code under `method/teacher/`. You cannot use the GPU or write git metadata; the human operator runs the sim and commits.

## Goal
A **privileged teacher**: a controller that is allowed to read true simulator state (plate pose, rack positions, slot sites) because it is used ONLY to generate training demonstrations on fresh seeds, never at evaluation. It must reach ≥ 90% success under the real success check.

## Facts already measured (trust these)
- Env: `from advaitic_eval.environment import make_environment, observed_source_slot`; `env = make_environment()`; `ts = env.reset(seed=s)`; `ts = env.step(a)` with `a` a normalized float32 (20,) in [-1,1]; `ts.last()`; `env.episode_succeeded()`, `env.episode_fell()`, `env.episode_termination()`. Episode: 1,700 steps max at 50 Hz. Observations: `ts.rgb_obs` uint8 (3,3,84,84), `ts.low_dim_obs` float32 (50,).
- Privileged access (teacher only): `inner = env.inner_env`; `inner.plates[0].get_pose()`; `inner.rack_start.get_position()`, `inner.rack_target.get_position()`; slot sites `inner.data.bind(site).xpos` for `site in inner.rack_start.sites` / `inner.rack_target.sites` (6 each, spaced 0.042 m in y).
- State layout (`low_dim_obs`): 0–6 left arm q, 7–8 left fingers, 9–15 right arm q, 16–17 right fingers, **18–21 pelvis x, y, z, yaw (world frame)**, 22–43 velocities, 44–45 gripper states, 46–49 base-controller pelvis.
- Action layout: `src/advaitic_eval/action_layout.json`. 0 vx, 1 vy, 2 pelvis height, 3 yaw rate, 4–10 left arm targets, 11–17 right arm targets, 18 left gripper, 19 right gripper. Normalize raw with `2*(raw-low)/(high-low)-1`.
- `method/data/demos_lowdim.npz`: for i in 0..59, `a{i}` (T,20) normalized actions, `s{i}` (T+1,50) states, `seed{i}` reset seed. Replaying `a{i}` open-loop on `seed{i}` succeeds 59/60. On a new seed with the same slot it succeeds ~1/8.
- `method/data/demo_meta.json`: per demo: seed, source slot (0-based), start/target rack offsets from nominal, all 12 slot-site positions, plate pose after warmup, grasp_step / release_step (left gripper action > 0), pelvis at start/grasp/release.
- **Key finding:** humans barely walk differently per episode. Pelvis xy at grasp has std ≈ 2 cm while plate xy has std ≈ 3 cm in x and 7 cm in y; correlation ≈ 0.2–0.3. They compensated with the **arm**, not the base. Only the left arm + left gripper + base vx/vy are used; right gripper never closes; height and yaw are constant.

## Teacher design ("nearest demo + residual base shift")
1. On reset, read the true plate position `p` and target-rack offset `dT` (privileged).
2. **Pick the reference demo** `k` minimizing `|| (p - pelvis0) - (p_k - pelvis0_k) ||` in xy (plate position relative to the starting pelvis), excluding demo 45 (fails even on replay). Make this a function so it can also return the top-N candidates.
3. **Residual shifts:** grasp shift `g = (p - pelvis0) - (p_k - pelvis0_k)` (xy). Place shift `h = dT - dT_k` (target rack offset difference; destination = same slot index as demo k used. Compute which target slot demo k actually placed into from its replayed final plate pose if needed, but the offset difference is enough to start).
4. **Closed-loop base tracking:** the desired pelvis trajectory is `pelvis_k(t) + shift(t)`, where `shift(t) = g` until the demo's grasp_step + ~30 steps (after lift), then linearly blended to `h` over the carry (until ~40 steps before release_step), then `h`. Base command = demo's own vx/vy (feed-forward) + P-control on the pelvis position error, rotated into the body frame using pelvis yaw, clipped to the action range. Keep height and yaw rate from the demo.
5. Arms and grippers: replay demo k's arm/gripper targets **indexed by the demo's time**, but allow **time warping**: do not advance the arm-trajectory clock past the grasp step until the pelvis tracking error is < 1 cm (or a max wait of ~100 steps). Same before release. This avoids grasping while still walking into position.
6. After the demo ends, hold its final action (gripper open) until the episode terminates.
7. Optional v2 (only after v1 is measured): privileged grasp check after lift (is the plate lifted with the gripper?). If not, re-open, back off and retry with the next candidate demo.

## Deliverables
1. `method/teacher/teacher.py`: `class Teacher` with `reset(env)` (reads privileged state) and `act(ts) -> (20,) normalized action`. Pure numpy. Docstring explains the design in plain language.
2. `method/teacher/run_teacher.py`: CLI to evaluate the teacher on a list/range of seeds. Single process; use `--workers` only via separate OS processes launched by the operator (the operator will run ≤ 2 at once for thermal reasons). Writes **append-only JSONL** (one line per episode: seed, slot, chosen demo, shifts, success, termination, fell, steps, max pelvis tracking error, wall time) and **skips seeds already present** in the output (resumable). Must handle Ctrl-C / kill safely: write each line only after the episode completes.
3. `--save-demos DIR` option: for each **successful** episode, save `DIR/ep_<seed>.npz` with `rgb` (T,3,3,84,84) uint8 **subsampled every 2nd step** (`rgb_stride=2`), `state` (T,50) float32 at every step, `action` (T,20) float32 at every step, `seed`, `slot`, `success`, using `np.savez_compressed`. Write to a temp name then `os.replace` (atomic). Episode length is the step count up to success (the env terminates on held success).
4. `method/teacher/README.md`: what it is, why it's privileged, exact commands.
5. A short report at the end of your run: what you implemented, any assumptions, what you could not verify, and the exact commands for me to run.

## Constraints
- No changes to the evaluator, physics, success checks. The teacher may read privileged state; the student policy (not your job) never will.
- Keep it simple and readable; no new dependencies beyond numpy/scipy already in `.venv`.
- Run `uv run ruff check method && uv run ruff format --check method` if available. Test what you can on CPU without rendering: if `MUJOCO_GL=egl` fails in your sandbox, try `MUJOCO_GL=osmesa`; if no rendering works, say so and stop at static checks. The operator will run the real tests.
