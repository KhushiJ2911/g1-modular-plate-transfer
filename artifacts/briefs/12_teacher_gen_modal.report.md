Created [modal_teacher.py](/home/keyu/advaitic/ops/modal_teacher.py).

Verified from source:

- Generator: `uv run python -m method.teacher.run_teacher`, pinned to `--mode v1 --dart-noise 0`.
- `--seeds START:STOP` excludes STOP. Successful episodes alone produce `ep_<seed>.npz`.
- JSONL appends completed attempts, including failures; interrupted episodes aren’t logged.
- Worktree is clean at `2cc0622`. The image retains `advaitic-check` and embeds the launch-time SHA.
- T4 resources match `collect_shard_gpu`: 4 CPU, 8192 MiB RAM, with the requested two-hour timeout.

`py_compile`, offline SDK import, and mocked checks passed for protected seed ranges, sharding, overwrite refusal, counts, failure logs, and check-dataset isolation. No Modal jobs ran; NVIDIA EGL and end-to-end generation remain unverified.

Run:

```bash
cd /home/keyu/advaitic/ops
modal run modal_teacher.py::check --start 2800000
```

After inspecting the check output:

```bash
modal run modal_teacher.py::gen --start 2800000 --count 240 --per-shard 10
```

This launches 24 shards covering seeds **2800000–2800239** into `teacher_v1_gen2`.