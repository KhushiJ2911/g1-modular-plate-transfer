"""Detached sharded dev supervision, including recovery, counting and failure replay."""

import concurrent.futures
import json
import re
import time
from collections import Counter
from pathlib import Path

import modal

app = modal.App("modular-astra-integration-round")
records = modal.Volume.from_name("modular-runs")
official = modal.Volume.from_name("advaitic-runs")


def graph_failures(cid):
    """Statuses belong to individual calls, including detached children of a scheduler."""
    graph = modal.FunctionCall.from_id(cid).get_call_graph()
    statuses = {}
    while graph:
        node = graph.pop()
        statuses.setdefault(node.function_call_id, set()).add(node.status.name)
        graph.extend(node.children)
    failed = {"FAILURE", "INIT_FAILURE", "TIMEOUT", "TERMINATED"}
    return {key for key, values in statuses.items() if values <= failed}


def attempt_status(cid):
    c = modal.FunctionCall.from_id(cid)
    try:
        c.get(timeout=0)
    except modal.exception.FunctionTimeoutError:
        return "ended"
    except TimeoutError:
        return "running"
    except modal.exception.ConnectionError:
        return "running"
    except Exception:
        return "ended"
    return "success"


def count(rows, plan):
    def pairs(key):
        tot = Counter(r[key] for r in rows)
        ok = Counter(r[key] for r in rows if r["success"])
        return {k: [ok[k], tot[k]] for k in sorted({r[key] for r in plan})}

    b = pairs("block")
    failed = any(total - passed > 3 for passed, total in b.values())
    passed = len(rows) == 90 and all(ok >= 27 for ok, _ in b.values())
    return dict(
        blocks=b,
        slots=pairs("slot"),
        done=len(rows),
        successes=sum(r["success"] for r in rows),
        decision="no-go" if failed else ("pass" if passed else "pending"),
    )


@app.function(
    cpu=0.125, memory=512, timeout=86400, volumes={"/records": records, "/runs": official}
)
def supervise(
    name, app_name, policy, head, plan, existing, keep, prior="", smoke=False, recover_empty=""
):
    if not re.fullmatch("astra_[a-z0-9_]+", name) or not app_name.startswith(
        "modular-shard-astra-"
    ):
        raise ValueError("Expected private integrator run")
    if (
        len(plan) != 90
        or len({r["seed"] for r in plan}) != 90
        or set(Counter((r["block"], r["slot"]) for r in plan).values()) != {5}
    ):
        raise ValueError("Expected three complete balanced dev plans")
    fold = Path("/records/astra-integration") / name
    fold.mkdir(parents=True, exist_ok=True)
    if prior:
        p = fold.parent / prior / "summary.json"
        while True:
            records.reload()
            if p.exists() and json.loads(p.read_text())["decision"] in {"no-go", "pass"}:
                break
            time.sleep(50)
    score = modal.Function.from_name("modular-astra-integration-scores", "scores")
    fn = modal.Function.from_name(app_name, "episode_cpu")
    # Shared slots for the supervisor, score reader, DAgger and diagnostics are
    # reserved separately. At most one official development round is active.
    replay = modal.Function.from_name("modular-astra-integration-replay", "replay")
    replay.autoscaler.update(max_containers=2)
    by_seed = {r["seed"]: r for r in plan}
    ids = {int(seed): cid for seed, cid in existing.items()}
    keep = set(keep)
    journal = fold / "calls.json"
    retries = Counter()
    replay_ids = {}
    replays = {}

    def persist():
        journal.write_text(json.dumps(ids))
        (fold / "replay_calls.json").write_text(json.dumps(replay_ids))
        (fold / "replays.json").write_text(json.dumps(replays))
        records.commit()

    identity = fold / "candidate.json"
    candidate = dict(app=app_name, policy=policy, checkpoint=head, plan=plan)
    if journal.exists():
        if not identity.is_file() or json.loads(identity.read_text()) != candidate:
            raise ValueError("Another candidate already owns this journal")
        if not recover_empty:
            return {"status": "already supervised", "run": name}
        if json.loads(journal.read_text()) or attempt_status(recover_empty) != "ended":
            raise ValueError("Only a failed supervisor with an empty call journal can restart")
    identity.write_text(json.dumps(candidate))
    hybrid_pending = (fold / "hybrid_calls.json").is_file() and not (fold / "hybrid.json").is_file()
    fn.autoscaler.update(max_containers=15 if hybrid_pending else 17)
    persist()
    if smoke and not existing:
        s = plan[0]["seed"]
        reserved = fold / "official_smoke.call"
        if reserved.exists():
            ids[s] = reserved.read_text().strip()
            c = modal.FunctionCall.from_id(ids[s])
        else:
            c = fn.spawn(name, s, policy, head)
            ids[s] = c.object_id
        persist()
        r = c.get()
        if r["seed"] != s or not r["completed"] or r.get("policy_error"):
            raise ValueError(f"Unusable policy smoke for {s}: {r.get('policy_error')}")
    submitted = False
    last_graph_check = 0.0
    with concurrent.futures.ThreadPoolExecutor(10) as pool:
        while True:
            if hybrid_pending:
                records.reload()
                if (fold / "hybrid.json").is_file():
                    hybrid_pending = False
                    fn.autoscaler.update(max_containers=17)
            rows = score.remote(name, [(r["block"], r["seed"], r["slot"]) for r in plan])
            summary = count(rows, plan)
            seen = {r["seed"] for r in rows}
            for r in rows:
                if not r["success"] and r["seed"] not in replay_ids:
                    if r.get("policy_error"):
                        replays[r["seed"]] = dict(bucket="policy_error", detail=r["policy_error"])
                    else:
                        replay_ids[r["seed"]] = replay.spawn(r["folder"]).object_id
            for seed, cid in replay_ids.items():
                if seed in replays:
                    continue
                try:
                    replays[seed] = modal.FunctionCall.from_id(cid).get(timeout=0)
                except (TimeoutError, modal.exception.ConnectionError):
                    pass
                except Exception as exc:
                    replays[seed] = {"error": str(exc)[-400:]}
            (fold / "summary.json").write_text(json.dumps(dict(**summary, rows=rows), indent=2))
            persist()
            if summary["decision"] != "pending":

                def cancel(cid):
                    c = modal.FunctionCall.from_id(cid)
                    try:
                        c.get(timeout=0)
                    except TimeoutError:
                        c.cancel(terminate_containers=True)
                    except Exception:
                        pass

                list(pool.map(cancel, [cid for s, cid in ids.items() if s not in seen]))
                # Failure replays finish independently; wait for all diagnostic records.
                for s, cid in replay_ids.items():
                    if s not in replays or "error" in replays[s]:
                        try:
                            replays[s] = modal.FunctionCall.from_id(cid).get()
                        except Exception as exc:
                            replays[s] = {"error": str(exc)[-400:]}
                persist()
                return summary
            retained = [ids[s] for s in keep if s not in seen and s in ids]
            if not submitted and not any(
                status == "running" for status in pool.map(attempt_status, retained)
            ):
                # All former calls have either finished or were canceled after
                # the account cap changed; successful report files take precedence.
                pending = [s for s in by_seed if s not in seen]

                def submit(s):
                    if s in ids and attempt_status(ids[s]) == "running":
                        return ids[s]
                    return fn.spawn(name, s, policy, head).object_id

                for s, cid in zip(pending, pool.map(submit, pending), strict=True):
                    ids[s] = cid
                submitted = True
                persist()
            elif submitted:
                pending = [s for s in ids if s not in seen]
                failures = set()
                if pending and time.monotonic() - last_graph_check >= 300:
                    # All new shards belong to this supervisor's graph. Query
                    # it once per window instead of once for every queued shard.
                    last_graph_check = time.monotonic()
                    try:
                        failures = graph_failures(ids[pending[-1]])
                    except (OSError, modal.exception.Error):
                        pass
                failed = [
                    s
                    for s, status in zip(
                        pending, pool.map(attempt_status, [ids[s] for s in pending]), strict=True
                    )
                    if status == "ended" or ids[s] in failures
                ]
                if failed:
                    # A worker may have committed a report just before timing
                    # out while publishing its return value.
                    fresh = score.remote(name, [(r["block"], r["seed"], r["slot"]) for r in plan])
                    completed = {r["seed"] for r in fresh}
                    for s in failed:
                        if s in completed:
                            continue
                        if retries[s] >= 2:
                            raise RuntimeError(f"Repeated infrastructure failure on {s}")
                        ids[s] = fn.spawn(name, s, policy, head).object_id
                        retries[s] += 1
                persist()
            time.sleep(50)
