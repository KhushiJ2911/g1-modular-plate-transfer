Implemented v2 as the default, with `--mode v1` preserving the original controller.

- Plate-centered placement uses all six sites, deadzone-aware base control, and 150-step gates at release−40 and release.
- Tilt correction uses wrist roll/pitch dimensions 8/9. Guarded retreat adds settling, contact checks, arm holds, and a shoulder/elbow lift using dimensions 4/7.
- Added JSONL diagnostics, failure categories, and [diag_trace.py](/home/keyu/advaitic/wt-teacher/method/teacher/diag_trace.py). Atomic saves, resumability, and the mode-sensitive run-ID guard remain intact.

Validation passed: lint, formatting, 12 method checks, and live named-joint/tilt geometry checks. A CPU-rendered trace reached 100 logged steps before interruption. **Full v2 placement, recovery behavior, and success rate remain unverified.** No GPU or git was used; changes stayed under `method/teacher/`.

Exact verification commands:

```bash
export UV_PROJECT_ENVIRONMENT=/home/keyu/advaitic/robot-learning-assignment/.venv
export UV_CACHE_DIR=/tmp/teacher-uv-cache RUFF_CACHE_DIR=/tmp/teacher-ruff-cache
uv run --locked --no-sync ruff check method/teacher
uv run --locked --no-sync ruff format --check method/teacher
ORT_DISABLE_TELEMETRY=1 CUDA_VISIBLE_DEVICES='' MUJOCO_GL=disable \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src uv run --locked --no-sync \
  python -m unittest discover -s method/teacher -p 'test_*.py' -v
```

The [README](/home/keyu/advaitic/wt-teacher/method/teacher/README.md) documents the gate choice, kinematics, diagnostic definitions, and exact CPU rollout/ablation/trace commands.