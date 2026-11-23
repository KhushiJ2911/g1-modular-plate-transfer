Implemented entirely in `method/teacher/`:

- Seeded, guarded OU noise; separate clean/executed actions; DART scale included in `run_id`.
- All privileged labels and resumable human sidecars with per-step alignment checks.
- Updated [README](/home/keyu/advaitic/wt-teacher/method/teacher/README.md:111) with exact commands and data conventions.

Verification: Ruff passes and all 19 contract checks pass. Six real CPU-rendered steps passed; human-state maximum deviation was **0.02036**.

Unverified: complete DART rollouts and full human replays. The single-demo attempt reached its 180-second limit without publishing a sidecar; the full collection was not produced. No GPU or git used; protected files were untouched.

Exact launch commands, after the README’s CPU environment exports:

```bash
uv run --locked --no-sync python -m method.teacher.run_teacher \
  --mode v2 --dart-noise 1 --seeds 2000100:2000130 \
  --out method/teacher/runs/v2_dart1.jsonl \
  --save-demos method/teacher/generated/v2_dart1

uv run --locked --no-sync python -m method.teacher.label_human \
  --demos-dir /home/keyu/advaitic/data/human60 --indices 0:60
```

Exact validation commands are recorded in the [verification section](/home/keyu/advaitic/wt-teacher/method/teacher/README.md:289).