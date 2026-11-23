Implemented per-episode SHA-256 caches with atomic writes, per-entry locks, completeness checks, and lazy worker memmaps. Added `cache_tool.py` with `du` and dry-run-by-default `prune --keep-config …`; deletion requires `--apply`. Old caches remain untouched.

| Loader | Median it/s |
|---|---:|
| Old | 198.20 |
| New | 192.58 |

Same 16 real files, batch 64, two workers, three trials. Both used NumPy pipe IPC because the sandbox blocks PyTorch’s normal tensor IPC. New throughput exceeds 50 it/s.

Overlap smoke: **3 shared entries, 593,994 bytes reused, zero shared bytes rewritten**. Cache recovery/concurrency checks, full CPU training smoke, and Ruff passed.

Exact benchmark, validation, and maintenance commands are in [README.md](/home/keyu/advaitic/wt-bc/method/bc/README.md:86).

No git or GPU use. Prohibited files and `/home/keyu/advaitic/data/cache` were untouched.