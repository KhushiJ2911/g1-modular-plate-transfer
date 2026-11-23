Implemented v1 under `method/teacher/`: nearest-demo selection, residual base tracking, bounded grasp/release waits, resumable JSONL logging, and atomic successful-demo saves. Details and assumptions are in [README.md](/home/keyu/advaitic/wt-teacher/method/teacher/README.md).

Ruff checks and all five CPU contract tests passed. OSMesa failed during import, so no simulator rollouts ran and **≥90% success remains unverified**. No GPU was used or evaluator files changed.

Run from the worktree root using the existing environment:

```bash
export UV_PROJECT_ENVIRONMENT=/home/keyu/advaitic/robot-learning-assignment/.venv

ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl uv run --locked --no-sync \
  python -m method.teacher.run_teacher \
  --seeds 100000:100030 \
  --out method/teacher/runs/v1.jsonl \
  --save-demos method/teacher/generated/v1
```

Repeat the identical command to resume. The README includes balanced-seed selection, CPU checks, and commands for two separately launched processes.