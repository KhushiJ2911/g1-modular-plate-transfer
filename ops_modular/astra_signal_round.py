"""Detach post-hoc diagnostics and the paired public generation run after their individual smokes."""

import json
import time
from pathlib import Path

import modal

app = modal.App("modular-astra-signal-round")
records = modal.Volume.from_name("modular-runs")
root = Path("/records/astra-sig1")
audit_app = "modular-astra-signal-audit-bd951bc"
gen_app = "modular-astra-clearance-gen-bd951bc-528c854"
head = "/runs/heads/64dfe105e7f716c0ab5b7a9c487b7a8a1cac06565070b390e75992dfbee4deff.npz"
policy = "method.modular.base_clearance:make_policy"


@app.function(cpu=0.125, memory=256, timeout=43200, max_containers=3, volumes={"/records": records})
def supervise(kind, items, smoke_id):
    if kind not in ("physical", "decisions", "generation") or len(set(items)) != len(items):
        raise ValueError("Unexpected audit or comparison group")
    records.reload()
    dest = root / kind
    dest.mkdir(parents=True, exist_ok=True)

    def store(key, value):
        (dest / key).write_text(json.dumps(value))
        records.commit()

    def load(key):
        p = dest / key
        return json.loads(p.read_text()) if p.is_file() else {}

    if kind == "generation":
        if len(items) != 30 or not all(3002200 <= s < 3002350 for s in items):
            raise ValueError("Expected all 30 predeclared generation seeds")
        fn = modal.Function.from_name(gen_app, "episode_cpu")
    else:
        if not all(isinstance(x, str) and x.startswith("sig1_bd951bc-") for x in items):
            raise ValueError("Only archived failures may be diagnosed")
        fn = modal.Function.from_name(audit_app, kind)
    smoke = modal.FunctionCall.from_id(smoke_id).get()
    if kind == "generation" and (
        smoke["seed"] != 3002199
        or not smoke["completed"]
        or smoke["policy_error"]
        or "error" in smoke
    ):
        raise ValueError("Generation smoke did not return a complete physical outcome")
    if kind == "physical" and (smoke["replay_success"] or not smoke["bucket"]):
        raise ValueError("Incorrect camera-free smoke")
    if kind == "decisions" and not smoke["path"].endswith(items[0] + ".json"):
        raise ValueError("Incorrect action comparison smoke")
    store("smoke.json", smoke)
    completed = load("completed.json")
    errors = load("errors.json")
    calls = load("calls.json")
    if kind != "generation":
        calls[items[0]] = smoke_id
        completed[items[0]] = smoke
        store("completed.json", completed)
    for item in items:
        k = str(item)
        if k in calls:
            continue
        args = (
            ("astra_clearance_fresh_528c854", item, policy, head)
            if kind == "generation"
            else (item,)
        )
        calls[k] = fn.spawn(*args).object_id
        store("calls.json", calls)
    while len(completed) + len(errors) < len(items):
        for k, cid in calls.items():
            if k in completed or k in errors:
                continue
            try:
                result = modal.FunctionCall.from_id(cid).get(timeout=0)
            except modal.exception.FunctionTimeoutError as exc:
                errors[k] = str(exc)
                store("errors.json", errors)
            except TimeoutError:
                continue
            except Exception as exc:
                errors[k] = f"{type(exc).__name__}: {str(exc)[-1800:]}"
                store("errors.json", errors)
            else:
                completed[k] = result
                store("completed.json", completed)
        if len(completed) + len(errors) < len(items):
            time.sleep(20)
    return {"finished": len(completed), "errors": errors}
