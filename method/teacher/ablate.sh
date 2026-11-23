#!/usr/bin/env bash
# Usage: ablate.sh OUTPUT.jsonl START STOP [run_teacher controller/noise options]
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
export ORT_DISABLE_TELEMETRY=1 CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void
export MUJOCO_GL=egl LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=1 OMP_NUM_THREADS=1
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export EGL_PLATFORM=surfaceless PYTHONDONTWRITEBYTECODE=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/teacher-uv-cache}"
uv run --locked --no-sync python - "$@" <<'PY'
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from method.teacher.compare import load_records

if len(sys.argv) < 4 or sys.argv[1] in ('-h', '--help'):
    print('Usage: ablate.sh OUTPUT.jsonl START STOP [controller/noise options]')
    raise SystemExit(0 if len(sys.argv) == 2 else 2)
output = Path(sys.argv[1]).resolve()
start, stop = map(int, sys.argv[2:4])
options = sys.argv[4:]
if not (0 <= start < stop <= 2**32) or stop - start < 2:
    raise SystemExit('Require at least two seeds in [START, STOP), each uint32')
if any(v.split('=')[0] in ('--out', '--seeds') for v in options):
    raise SystemExit('--out and --seeds are managed by ablate.sh')
output.parent.mkdir(parents=True, exist_ok=True)
parts = Path(str(output) + '.parts')
with Path(str(output) + '.lock').open('a') as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('This output is already running')
    if output.exists() and not parts.exists():
        raise SystemExit('Output already exists without resume parts; choose a new name')
    parts.mkdir(exist_ok=True)
    invocation = {'start': start, 'stop': stop, 'options': options}
    manifest = parts / 'invocation.json'
    if manifest.exists() and json.loads(manifest.read_text()) != invocation:
        raise SystemExit('Seed range/options changed; choose a new output name')
    manifest.write_text(json.dumps(invocation) + '\n')
    workers = []
    streams = []
    try:
        for i in range(2):
            stream = (parts / f'worker{i}.log').open('a')
            streams.append(stream)
            cmd = [sys.executable, '-m', 'method.teacher.run_teacher',
                   '--seeds', f'{start+i}:{stop}:2', '--out', str(parts / f'worker{i}.jsonl'),
                   *options]
            workers.append(subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT))
        print(f'Two CPU workers; progress: {parts}/worker{{0,1}}.log', flush=True)
        statuses = [worker.wait() for worker in workers]
        if any(statuses):
            raise SystemExit(f'Workers exited {statuses}; inspect logs and rerun to resume')
        records = {}
        for i in range(2):
            rows = load_records(parts / f'worker{i}.jsonl')
            if records.keys() & rows.keys():
                raise SystemExit('Overlapping worker seeds')
            records.update(rows)
        if set(records) != set(range(start, stop)):
            raise SystemExit('Worker seed sets do not match the requested range')
        if len({r.get('run_id') for r in records.values()}) != 1:
            raise SystemExit('Worker configurations differ')
        with tempfile.NamedTemporaryFile(mode='w', dir=output.parent, delete=False) as merged:
            temporary = Path(merged.name)
            for seed in sorted(records):
                merged.write(json.dumps(records[seed], allow_nan=False) + '\n')
            merged.flush()
            os.fsync(merged.fileno())
        os.replace(temporary, output)
        print(f"Wrote {output}: {sum(r['success'] for r in records.values())}/{len(records)} successes")
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
        for worker in workers:
            worker.wait()
        for stream in streams:
            stream.close()
PY
