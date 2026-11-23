# Codex brief #9: per-episode image cache (stop duplicating frames per data mix)

Context: `method/bc/dataset.py` (committed at 5233dc7). Rules unchanged: do not modify `src/`, `tests/`, `pyproject.toml`, `uv.lock`; no GPU, no git. A training process is currently running with the existing code; do not touch `/home/keyu/advaitic/data/cache`.

## Problem
`build_cache` writes one uncompressed uint8 memmap per **set of files**, so every new data mix (adding one directory) writes another full copy of all frames (≈5 GB each). On a 147 GB disk with ~12 GB free, this filled the disk and crashed a training run.

## Change
- Cache **per episode file**: key = sha256 of (cache format version, episode path, size, mtime_ns, and the sidecar's if present). Each entry: `<key>.u8` (frames) + `<key>.npz` (state, action, label_mask, priv arrays, lengths). Write atomically with a lock per entry; skip if it exists and is complete.
- At dataset construction, build the in-memory index by concatenating the small per-episode arrays and keep **one memmap per episode** (or open lazily in workers); sampling must stay fast (the trainer is GPU-bound at ~5 it/s with batch 64 and 2 workers; the data loader currently does ~350 it/s; keep it above ~50 it/s). Measure and report loader it/s for the old and new formats on the same files.
- Add `method/bc/cache_tool.py` with `du` (total size, entries, how many are referenced by a given config's data dirs) and `prune --keep-config CFG...` (delete entries not referenced by the given configs; dry run by default).
- Keep the old monolithic format readable? Not needed; but do not delete old caches automatically.
- Smoke test: two configs that share directories reuse the same entries (no new bytes written for the shared part).

Report: what changed, measured loader speed, exact commands.
