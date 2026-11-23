# Brief #12: Modal launcher for more teacher_v1 demos (SFT data)

You work in `/home/keyu/advaitic/ops` (not a git repo; just write files). Read-only references:
- `/home/keyu/advaitic/ops/modal_sim.py`: the existing Modal sim image (wt-vla code, `uv sync --frozen`,
  `bigym-download`, `advaitic-check` at build time) and `collect_shard_gpu` (T4, NVIDIA EGL via `GPU_GL`,
  write to local `/tmp`, then copy to the volume, `commit()`). Copy its patterns.
- `/home/keyu/advaitic/wt-teacher` (branch teacher, HEAD 2cc0622): `method/teacher/run_teacher.py` is the
  generator. Default `--mode v1`, `--dart-noise 0` = exactly the controller that produced
  `data/teacher_v1_dart0` (112 demos, ~84% yield). `--save-demos DIR` writes successful episodes as
  `ep_<seed>.npz`; `--out` is an append-only JSONL of every attempt (success or not).

## Goal
Write `/home/keyu/advaitic/ops/modal_teacher.py`, which generates new teacher_v1 (DART 0) demos on Modal
T4 containers in parallel into the `advaitic-data` volume as a new dataset dir `teacher_v1_gen2`.

## Requirements
1. Image: same recipe as `modal_sim.py`'s `image`, but the code dir is a snapshot of
   `/home/keyu/advaitic/wt-teacher` (same ignores; also write its git SHA into the image as `.git_sha`,
   computed locally at launch with `git -C ... rev-parse --short HEAD`, and fail the launch if
   `git status --porcelain` in wt-teacher is non-empty, apart from ignored files).
   Keep `advaitic-check` at build time. Do not modify anything under wt-teacher or wt-vla.
2. `gen_shard(start: int, stop: int)` on `gpu="T4"`: runs
   `{GPU_GL} uv run python -m method.teacher.run_teacher --seeds {start}:{stop} --out /tmp/gen/attempts_{start}.jsonl --save-demos /tmp/gen/demos`
   (check the module path and the seed-range syntax in run_teacher.py; use the real ones). Rendering MUST be
   NVIDIA EGL (the `GPU_GL` prefix), so the images match the evaluator's rendering. Log stdout/stderr to
   `/tmp/gen/log_{start}.txt`. Afterwards copy the demos to `/data/teacher_v1_gen2/` and the jsonl and log to
   `/data/teacher_v1_gen2/_meta/`; refuse to overwrite an existing `ep_*.npz`; `data.commit()`.
   Return (start, stop, n_attempts, n_success, returncode, seconds).
3. `@app.local_entrypoint() gen(start: int = 2800000, count: int = 240, per_shard: int = 10)`: starmap
   over shards, print each result as it arrives, then a total line (attempts, successes, yield).
   Default block 2800000–2800239. **Never allow seeds in [1000000, 1100000) (dev), [2600000, 2700000)
   (confirm) or [7310000, 7320000) (test)**: assert this before launching anything.
4. Also add `@app.local_entrypoint() check(start: int = 2800000)`: a 1-seed dry run (stop = start+1)
   that writes to `/data/teacher_v1_gen2_check/` instead, so I can verify one episode before the big launch.
5. Timeouts: per shard 2 h. CPU 4, memory 16 GB (copy what collect_shard_gpu uses if different).
6. Keep it short and readable; module docstring with the usage lines.

## Deliverable
`/home/keyu/advaitic/ops/modal_teacher.py` plus a short report (the output file of this run): what you
verified by reading code (exact generator CLI, file naming, the success-only save behaviour), anything
you were unsure of, and the exact commands to run. You cannot run Modal; do run
`python -m py_compile` on the file. Do not touch the evaluator, physics or success checks.
