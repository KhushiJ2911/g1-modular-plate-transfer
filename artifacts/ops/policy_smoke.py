"""Fast pre-flight check for a policy, using the evaluator's own PolicyProcess and environment.

python policy_smoke.py SPEC POLICY_PYTHON [CALLS] [TIMEOUT]
One reset at dev seed 1000000; times the first exchange (= policy construction, which the evaluator
bounds by --policy-timeout, default 120 s); then CALLS act() calls, each validated and stepped.
Analysis only: no score, nothing written except /tmp/smoke_stderr.log.
"""
import json
import sys
import time
from pathlib import Path

from advaitic_eval.environment import make_environment, observation, validate_actions
from advaitic_eval.policy_process import PolicyProcess

spec, py = sys.argv[1], sys.argv[2]
calls = int(sys.argv[3]) if len(sys.argv) > 3 else 20
timeout = float(sys.argv[4]) if len(sys.argv) > 4 else 120.0
out = {"spec": spec, "ok": False}
env = make_environment()
proc = None
try:
    ts = env.reset(seed=1000000)
    proc = PolicyProcess(spec, Path("."), Path("/tmp/smoke_stderr.log"), timeout=timeout, python=py)
    t0 = time.monotonic()
    proc.reset()
    out["load_seconds"] = round(time.monotonic() - t0, 1)
    times, steps = [], 0
    for _ in range(calls):
        t1 = time.monotonic()
        acts = validate_actions(proc.act(observation(ts)))
        times.append(time.monotonic() - t1)
        for a in acts:
            ts = env.step(a)
            steps += 1
            if ts.last():
                break
        if ts.last():
            break
    out.update(ok=True, calls=len(times), steps=steps, chunk=len(acts),
               call_seconds_max=round(max(times), 3), call_seconds_mean=round(sum(times) / len(times), 3))
except Exception as exc:  # report, don't hide
    out["error"] = f"{type(exc).__name__}: {exc}"
    out["stderr_tail"] = Path("/tmp/smoke_stderr.log").read_text()[-1500:] if Path("/tmp/smoke_stderr.log").exists() else ""
finally:
    if proc is not None:
        proc.close()
    env.close()
print("SMOKE " + json.dumps(out), flush=True)
