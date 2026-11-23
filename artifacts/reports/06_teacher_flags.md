Implemented in [method/teacher](/home/keyu/advaitic/wt-teacher/method/teacher/README.md:41): independent flags defaulting to v1, flag values in JSONL/run IDs, the DART release fix, `ablate.sh`, and paired `compare.py`.

Bare `--mode v2` preserves legacy continuous placement feedback. Enabling flags on v1 uses the new align/freeze sequence.

The CPU demo audit found **all 59 demos already vertically overlap the rack at the old gate**. New carry-height gates are **88–189 frames earlier**, with **8.2–19.0 cm clearance**.

**Verified:** 29 contract tests, Ruff check/format, shell syntax. **Unverified:** v3 rollout success and actual two-worker throughput. No protected files changed; no GPU or git used.

Exact checks and experiment commands:

```bash
export UV_PROJECT_ENVIRONMENT=/home/keyu/advaitic/robot-learning-assignment/.venv
export UV_CACHE_DIR=/tmp/teacher-uv-cache RUFF_CACHE_DIR=/tmp/teacher-ruff-cache

uv run --locked --no-sync ruff check method/teacher
uv run --locked --no-sync ruff format --check method/teacher
ORT_DISABLE_TELEMETRY=1 CUDA_VISIBLE_DEVICES='' MUJOCO_GL=disable \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src uv run --locked --no-sync \
  python -m unittest discover -s method/teacher -p 'test_*.py' -v
bash -n method/teacher/ablate.sh

method/teacher/ablate.sh method/teacher/runs/abl_v1.jsonl 2000000 2000030
method/teacher/ablate.sh method/teacher/runs/abl_align.jsonl 2000000 2000030 --align-then-lower
uv run --locked --no-sync python -m method.teacher.compare \
  method/teacher/runs/abl_v1.jsonl method/teacher/runs/abl_align.jsonl
```