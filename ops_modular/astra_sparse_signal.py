"""Measure public v1b errors on archived frames where exact stateful replay diverged."""

import hashlib
import inspect
import json
import os
import subprocess
import time

import modal

from ops_modular.astra_fault_inputs import diagnose
from ops_modular.astra_gate_trace import options
from ops_modular.astra_signal_audit import HEAD, original_failure
from ops_modular.lead_shard import CHECKPOINT, CODE, WEIGHTS_SHA

app = modal.App("modular-astra-archived-inputs-bd951bc")


@app.function(**{**options, "cpu": 1}, timeout=1000, max_containers=1, single_use_containers=True)
def audit(folder, event, step):
    from pathlib import Path

    archive = original_failure(folder)
    if not 0 <= step <= 1700 or event["seed"] != int(folder.split("-")[1]):
        raise ValueError("Require one action divergence on the same original failed episode")
    if (
        hashlib.sha256(Path(HEAD).read_bytes()).hexdigest() != Path(HEAD).stem
        or hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != WEIGHTS_SHA
    ):
        raise ValueError("Unexpected frozen checkpoint")
    event = {**event, "audit_divergence": step}
    src = "import json\nfrom pathlib import Path\n" + inspect.getsource(diagnose)
    src += f"\nprint(json.dumps(diagnose({str(archive)!r},{event!r},{HEAD!r})))"
    start = time.monotonic()
    cmd = subprocess.run(
        ["/root/policy/bin/python", "-c", src],
        cwd=CODE,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "MUJOCO_GL": "osmesa",
            "PYOPENGL_PLATFORM": "osmesa",
            "OMP_NUM_THREADS": "1",
            "LP_NUM_THREADS": "2",
        },
    )
    if cmd.returncode:
        raise RuntimeError(cmd.stderr[-2000:])
    return {
        **json.loads(cmd.stdout.splitlines()[-1]),
        "seconds": time.monotonic() - start,
        "exact_until": step,
    }
