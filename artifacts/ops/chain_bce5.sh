#!/bin/bash
# bce5 chain: wait for the tracking labeler output (data/dagger3_track), sanity-check it, upload it to
# advaitic-data, train bce5a/bce5b on Modal (exact bce1 code export bc_5233dc7), promote last.pt to
# advaitic-ckpts, smoke each, then spawn 3 dev evals each on the deployed advaitic-sim app.
#
#   bce5a = bce1 recipe + dagger3_track @0.67, init student_v1b.pt,      6000 steps
#   bce5b = same data,                       init student_v1b_bce1.pt, 3000 steps
#
# Run:  nohup bash ops/chain_bce5.sh > ops/chain_bce5.out 2>&1 &
# Env overrides: EXPECT (episode count, default 120), STABLE_MIN (fallback minutes, default 15),
#                SKIP_EVALS=1 (stop after smoke).
set -u
cd /home/keyu/advaitic || exit 1

SRC=/home/keyu/advaitic/data/dagger3_track
VOLDIR=dagger3_track
EXPECT=${EXPECT:-120}
STABLE_MIN=${STABLE_MIN:-15}
CODE=/home/keyu/advaitic/export/bc_5233dc7
MODAL=/home/keyu/.local/bin/modal
MPY=/home/keyu/.local/share/uv/tools/modal/bin/python
MLPY=/home/keyu/advaitic/mlenv/bin/python
CK=/home/keyu/advaitic/ckpts
DATA_NAMES="human60 teacher_v1 teacher_v1_dart0 dagger $VOLDIR"
COMMON="--config method/bc/student_v1b_bce.yaml --data-weights 1 0.4 0.6 0.67 0.67 --val-episodes 6 --microbatch-size 64 --num-workers 2 --warmup-steps 200 --gripper-pos-weight 2"
declare -A ARGS=(
  [bce5a]="$COMMON --steps 6000 --init-from /ckpts/student_v1b.pt"
  [bce5b]="$COMMON --steps 3000 --init-from /ckpts/student_v1b_bce1.pt"
)
declare -A STEPS=([bce5a]=6000 [bce5b]=3000)
RUNS="bce5a bce5b"
TMP=$(mktemp -d /tmp/chain_bce5.XXXXXX)

log(){ echo "[$(date '+%F %T %Z')] $*"; }
die(){ log "FATAL: $*"; exit 1; }

# retry N TIMEOUT_S cmd... : run cmd under timeout, retrying with backoff; stdout kept in $TMP/last.out
retry(){
  local n=$1 to=$2; shift 2; local i
  for ((i = 1; i <= n; i++)); do
    local rc=0
    timeout -k 30 "$to" "$@" > "$TMP/last.out" 2> "$TMP/last.err" || rc=$?
    [ $rc = 0 ] && return 0
    log "  attempt $i/$n failed (rc=$rc): ${*:0:160} :: $(tail -c 400 "$TMP/last.err" | tr '\n' ' ')"
    sleep $((20 * i))
  done
  return 1
}

# vol_json VOLUME PATH -> JSON listing in $TMP/last.out ("[]" if path absent)
vol_json(){
  local i
  for ((i = 1; i <= 6; i++)); do
    if timeout -k 30 120 "$MODAL" volume ls --json "$1" "$2" > "$TMP/last.out" 2> "$TMP/last.err"; then return 0; fi
    if cat "$TMP/last.out" "$TMP/last.err" | grep -qiE "no such file|not found|does not exist"; then echo "[]" > "$TMP/last.out"; return 0; fi
    log "  volume ls $1 $2 attempt $i failed: $(tail -c 300 "$TMP/last.err" | tr '\n' ' ')"; sleep $((20 * i))
  done
  return 1
}

# vol_size VOLUME PATH FILENAME -> bytes (or -1)
vol_size(){
  vol_json "$1" "$2" || { echo -1; return; }
  "$MLPY" -I -c '
import json, sys
name = sys.argv[1]
for e in json.load(open(sys.argv[2])):
    if e["filename"].rstrip("/").split("/")[-1] == name:
        print(e["size"]); break
else:
    print(-1)' "$3" "$TMP/last.out"
}

log "chain_bce5 start; code=$CODE ($(cat $CODE/.git_sha 2>/dev/null)); tmp=$TMP"
[ -f "$CODE/method/bc/student_v1b_bce.yaml" ] || die "code export missing"
for f in student_v1b.pt student_v1b_bce1.pt; do [ -f "$CK/$f" ] || log "WARN: local $CK/$f missing (only /ckpts copy is used)"; done

# ---------------------------------------------------------------- 0. wait for the labeler
# track_label.py writes ep_*.npz atomically (tmp + os.link), track_config[.shardI].json at start and
# track_report[.shardI].json at the very end (default location: inside OUT). Done when EXPECT episodes
# exist AND every started shard has its report; fallback (report written elsewhere via -o): EXPECT
# episodes, no track_label process alive, and no new episode for STABLE_MIN minutes.
log "waiting for $EXPECT ep_*.npz + final track_report in $SRC"
last_count=-1; stable_since=$(date +%s); last_note=0
while true; do
  count=$(ls "$SRC"/ep_*.npz 2>/dev/null | grep -cE '/ep_[0-9]+\.npz$')
  now=$(date +%s)
  [ "$count" != "$last_count" ] && { stable_since=$now; last_count=$count; log "  $count episodes"; }
  if [ "$count" -ge "$EXPECT" ]; then
    reports_ok=0
    if [ -f "$SRC/track_report.json" ]; then reports_ok=1
    else
      shopt -s nullglob; cfgs=("$SRC"/track_config.shard*.json); shopt -u nullglob
      if [ ${#cfgs[@]} -gt 0 ]; then
        reports_ok=1
        for c in "${cfgs[@]}"; do r=${c/track_config/track_report}; [ -f "$r" ] || reports_ok=0; done
      fi
    fi
    alive=$(pgrep -f 'method[.]teacher[.]track_label' | wc -l)
    if [ $reports_ok = 1 ] && [ "$alive" = 0 ]; then log "labeler finished (reports present, no labeler process)"; break; fi
    if [ $reports_ok = 1 ] && [ $((now - stable_since)) -ge 300 ]; then log "reports present, count stable 5 min (labeler pid still alive? $alive)"; break; fi
    if [ "$alive" = 0 ] && [ $((now - stable_since)) -ge $((STABLE_MIN * 60)) ]; then
      log "WARN: no track_report in $SRC but $count episodes, no labeler process, stable ${STABLE_MIN} min -> proceeding"; break; fi
  fi
  [ $((now - last_note)) -ge 1800 ] && { log "  still waiting ($count/$EXPECT)"; last_note=$now; }
  sleep 60
done
[ "$count" -eq "$EXPECT" ] || log "WARN: $count episodes (expected $EXPECT)"
ls "$SRC" | grep -vE '^ep_[0-9]+\.npz$' | sed 's/^/  extra file: /'
for r in "$SRC"/track_report*.json; do [ -f "$r" ] && log "report $r summary: $("$MLPY" -I -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1])).get("summary")))' "$r")"; done

# ---------------------------------------------------------------- 1. local sanity check
log "sanity check (every file loads; same validations as method/bc/dataset.py build_cache)"
ADV_THROTTLE=1 "$MLPY" -I - "$SRC" "$EXPECT" <<'PY' || die "sanity check failed"
import glob, re, sys
import numpy as np
src, expect = sys.argv[1], int(sys.argv[2])
files = sorted(p for p in glob.glob(f"{src}/ep_*.npz") if re.fullmatch(r".*/ep_\d+\.npz", p))
assert len(files) >= expect, f"{len(files)} < {expect} episodes"
ACTIVE = [0, 1, 4, 5, 6, 7, 8, 9, 10, 18]
INACTIVE = [d for d in range(20) if d not in ACTIVE]
steps = kept = 0; kept_dims = np.zeros(20); act_sum = np.zeros(20); bad = []; low = []
for f in files:
    try:
        with np.load(f, allow_pickle=False) as d:
            for k in ("state", "action", "rgb", "rgb_stride", "label_mask", "track_phase", "track_cost", "seed"):
                assert k in d.files, f"missing {k}"
            arrays = {k: d[k] for k in d.files}  # force-decompress every array
        a, s, m, rgb = arrays["action"], arrays["state"], arrays["label_mask"], arrays["rgb"]
        T = len(a)
        assert T >= 1 and a.shape == (T, 20), f"action {a.shape}"
        assert s.shape == (T, 50), f"state {s.shape}"
        assert m.shape == (T, 20) and m.dtype == np.bool_, f"label_mask {m.shape} {m.dtype}"
        assert int(arrays["rgb_stride"]) == 2, "rgb_stride"
        assert rgb.dtype == np.uint8 and rgb.shape == ((T + 1) // 2, 3, 3, 84, 84), f"rgb {rgb.shape}"
        assert np.isfinite(a).all() and np.isfinite(s).all(), "nonfinite"
        assert np.abs(a).max() <= 1.00001, f"|action| max {np.abs(a).max():.4f} > 1 (trainer would reject)"
        assert len(arrays["track_phase"]) == T and len(arrays["track_cost"]) == T, "track_* length"
        assert (np.diff(arrays["track_phase"]) >= 0).all(), "track_phase not monotone"
        rows_all = m.all(1); rows_none = ~m.any(1)
        if not (rows_all | rows_none).all():
            print(f"  note {f}: {int((~(rows_all | rows_none)).sum())} rows partially masked")
        if m[:, ACTIVE].sum() == 0:
            low.append(f)
        steps += T; kept += int(rows_all.sum()); kept_dims += m.sum(0)
        act_sum += (a * m).sum(0)
    except Exception as e:
        bad.append((f, repr(e)))
for f, e in bad:
    print("BAD", f, e)
print(f"{len(files)} files, {steps} steps, fully-labeled rows {kept / max(steps, 1):.3f}, "
      f"episodes with no active labels: {len(low)}")
print("per-dim label fraction:", np.round(kept_dims / max(steps, 1), 3).tolist())
mean = act_sum / np.maximum(kept_dims, 1)
print("labeled action mean, inactive dims", {d: round(float(mean[d]), 4) for d in INACTIVE},
      "(enters stats.action_mean = inference defaults for inactive dims; dataset.py:226-230, policy.py:86)")
REF = {2: 0.1333, 3: 0.0, 11: 0.093, 12: -0.2972, 13: 0.0759, 14: -0.6196, 15: 0.1133, 16: 0.1236, 17: -0.1202, 19: -1.0}
diff = {d: round(float(mean[d]) - v, 4) for d, v in REF.items() if abs(float(mean[d]) - v) > 0.02}
print("inactive-dim mean vs student_v1b_bce1 action_mean, |diff|>0.02:", diff or "none")
if kept / max(steps, 1) < 0.5:
    print("WARN: under half of the rows are labeled")
sys.exit(1 if bad else 0)
PY
log "sanity OK"

# ---------------------------------------------------------------- 2. upload to advaitic-data
LOCAL_MAN=$TMP/local_sizes.txt
(cd "$SRC" && find . -maxdepth 1 -type f -printf '%f %s\n' | sort) > "$LOCAL_MAN"
verify_data(){
  vol_json advaitic-data "/$VOLDIR" || return 1
  "$MLPY" -I -c '
import json, sys
remote = {e["filename"].split("/")[-1]: int(e["size"]) for e in json.load(open(sys.argv[1])) if e["type"] == "file"}
local = dict((n, int(s)) for n, s in (l.split() for l in open(sys.argv[2]) if l.strip()))
miss = [n for n in local if remote.get(n) != local[n]]
eps = sum(1 for n in remote if n.startswith("ep_") and n.endswith(".npz"))
print(f"remote eps={eps} files={len(remote)} mismatched/missing={len(miss)} {miss[:5]}")
sys.exit(0 if not miss else 1)' "$TMP/last.out" "$LOCAL_MAN"
}
vol_json advaitic-data "/$VOLDIR" || die "cannot list advaitic-data"
if [ "$(cat "$TMP/last.out" | tr -d ' \n')" != "[]" ]; then
  log "WARN: /$VOLDIR already exists on advaitic-data"
  if verify_data; then log "existing volume copy matches local; skipping upload"; else die "volume /$VOLDIR exists and differs from local; refusing to overwrite (inspect manually)"; fi
else
  ok=0
  for t in 1 2 3 4; do
    F=""; [ $t -gt 1 ] && F="--force"
    log "volume put attempt $t"
    if timeout -k 60 3600 "$MODAL" volume put $F advaitic-data "$SRC" "/$VOLDIR" > "$TMP/put.out" 2>&1; then :; else log "  put rc=$? $(tail -c 400 "$TMP/put.out" | tr '\n' ' ')"; fi
    if verify_data; then ok=1; break; fi
    sleep 60
  done
  [ $ok = 1 ] || die "upload of $VOLDIR failed verification"
fi
log "data on volume verified"

# ---------------------------------------------------------------- 3. launch trainings (detached)
vol_json advaitic-runs / || die "cannot list advaitic-runs"
for N in $RUNS; do
  grep -qE "\"filename\": \"/?$N(\.log)?\"" "$TMP/last.out" && die "/runs/$N or /runs/$N.log already exists; pick new names"
  [ -e "$CK/student_$N.pt" ] && die "$CK/student_$N.pt already exists"
done

launch(){  # launch N: start detached modal run, retrying client-side launch failures
  local N=$1 t L
  for t in 1 2 3; do
    L=ops/modal_$N.log; [ $t -gt 1 ] && { mv "$L" "ops/modal_$N.failed$((t - 1)).log" 2>/dev/null; }
    log "launch $N attempt $t: ${ARGS[$N]}"
    ADV_CODE=$CODE nohup "$MODAL" run --detach ops/modal_train.py::run --gpu A100-40GB --module method.bc.train \
      --run-name "$N" --data "$DATA_NAMES" --args "${ARGS[$N]}" > "$L" 2>&1 &
    local pid=$! w
    for ((w = 0; w < 30; w++)); do  # up to 300 s for the remote command to start
      sleep 10
      grep -aq "python -m method.bc.train" "$L" && { log "$N started (client pid $pid, $(grep -aoE 'ap-[A-Za-z0-9]+' "$L" | head -1))"; return 0; }
      kill -0 $pid 2>/dev/null || break
    done
    log "  $N launch not confirmed: $(tail -c 400 "$L" | tr '\n' ' ')"
    kill $pid 2>/dev/null; sleep 5
    # never double-launch: if the remote job wrote its header, it is running.
    if [ "$(vol_size advaitic-runs / "$N.log")" != "-1" ]; then log "  but /runs/$N.log exists -> treating as launched"; return 0; fi
    sleep 30
  done
  return 1
}

# Stagger: bce5a builds the new image cache first (both runs share one digest); bce5b starts after
# bce5a prints its dataset line so it can (usually) reuse the committed cache instead of racing it.
launch bce5a || die "bce5a failed to launch"
for ((w = 0; w < 180; w++)); do grep -aq "train frames; cache=" ops/modal_bce5a.log && break; grep -aq "Traceback" ops/modal_bce5a.log && break; sleep 10; done
grep -aq "Traceback" ops/modal_bce5a.log && log "WARN: bce5a traceback: $(grep -a -A3 Traceback ops/modal_bce5a.log | tail -c 600 | tr '\n' ' ')"
log "bce5a dataset line: $(grep -a 'train frames' ops/modal_bce5a.log | head -1)"
sleep 90
launch bce5b || die "bce5b failed to launch"

# ---------------------------------------------------------------- 4. wait, download, verify, promote
promote(){
  local N=$1 deadline=$(( $(date +%s) + 6 * 3600 )) R=$TMP/$N.runlog
  log "waiting for exit=0 in /runs/$N.log"
  while true; do
    if timeout -k 30 120 "$MODAL" volume get advaitic-runs "/$N.log" - > "$R" 2>/dev/null; then
      if grep -aq "exit=0" "$R"; then log "$N finished: $(grep -a 'exit=0' "$R" | tail -1)"; break; fi
      starts=$(grep -ac "^# gpu=" "$R"); ends=$(grep -acE "exit=-?[0-9]+" "$R")
      if [ "$ends" -ge 1 ] && [ "$ends" -ge "$starts" ]; then
        log "$N FAILED: $(grep -aE 'exit=|Error|Traceback' "$R" | tail -5 | tr '\n' ' ')"; return 1; fi
    fi
    grep -aqE "App stopped|Traceback" "ops/modal_$N.log" && ! grep -aq "exit=0" "$R" 2>/dev/null && log "  WARN: client log for $N shows stop/traceback; still polling volume"
    [ "$(date +%s)" -gt "$deadline" ] && { log "$N timed out after 6 h"; return 1; }
    sleep 60
  done
  log "$N train tail: $(grep -a '^step=' "$R" | tail -1)"
  retry 5 900 "$MODAL" volume get --force advaitic-runs "/$N/last.pt" "$TMP/$N.last.pt" || { log "download of $N/last.pt failed"; return 1; }
  "$MLPY" -I -c '
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert c["step"] == int(sys.argv[2]), f"step {c['step']} != {sys.argv[2]}"
assert "ema" in c and "stats" in c and c["stats"]["action_mean"].shape == (20,)
dirs = c["config"]["data_dirs"]; assert any("dagger3_track" in d for d in dirs), dirs
print("ckpt ok step", c["step"], "init_from", c.get("init_from"), "val_loss", c.get("val_loss"), "dirs", dirs)
' "$TMP/$N.last.pt" "${STEPS[$N]}" || { log "$N last.pt failed torch.load verification"; return 1; }
  mv -n "$TMP/$N.last.pt" "$CK/student_$N.pt"; local size; size=$(stat -c %s "$CK/student_$N.pt")
  log "$N local ckpt $CK/student_$N.pt size=$size sha256=$(sha256sum "$CK/student_$N.pt" | cut -c1-16)"
  local t
  for t in 1 2 3 4; do
    F=""; [ $t -gt 1 ] && F="--force"
    timeout -k 30 900 "$MODAL" volume put $F advaitic-ckpts "$CK/student_$N.pt" "/student_$N.pt" > "$TMP/put_$N.out" 2>&1 || log "  ckpt put rc=$? $(tail -c 300 "$TMP/put_$N.out" | tr '\n' ' ')"
    [ "$(vol_size advaitic-ckpts / "student_$N.pt")" = "$size" ] && { log "$N uploaded to advaitic-ckpts as student_$N.pt ($size bytes verified)"; return 0; }
    sleep $((30 * t))
  done
  log "$N ckpt upload not verified"; return 1
}

smoke(){
  local N=$1
  log "smoke student_$N.pt"
  if retry 3 1500 "$MPY" -I -c '
import json, sys, modal
f = modal.Function.from_name("advaitic-sim", "smoke_run")
ckpt, res = f.remote(sys.argv[1], "submission.policy:make_policy", "", 20)
print(res)
ok = res.startswith("{") and json.loads(res).get("ok") is True
sys.exit(0 if ok else 3)' "student_$N.pt"; then
    log "$N smoke OK: $(cat "$TMP/last.out")"; return 0
  fi
  log "$N smoke FAILED: $(cat "$TMP/last.out") $(tail -c 300 "$TMP/last.err")"; return 1
}

spawn_evals(){
  local N=$1 r name
  vol_json advaitic-runs /evals || true; cp "$TMP/last.out" "$TMP/evals_ls.json"
  for r in 1 2 3; do
    name=dev_${N}_r$r
    if grep -q "\"evals/$name\"" "$TMP/evals_ls.json"; then log "WARN: /runs/evals/$name exists; skipping (no duplicate spawn)"; continue; fi
    if retry 3 180 "$MPY" -I /home/keyu/advaitic/ops/spawn.py advaitic-sim eval_run \
         "[[\"student_$N.pt\", \"$name\", 1000000, 30, \"submission.policy:make_policy\", \"\", \"\"]]"; then
      log "spawned $(cat "$TMP/last.out")"
    else
      log "FAILED to spawn $name"
    fi
  done
}

# Both runs are handled in parallel (whichever finishes first is promoted/evaluated first);
# each pipeline gets its own TMP so the shared last.out/last.err scratch files never collide.
pipeline(){
  local N=$1; TMP=$TMP/$N; mkdir -p "$TMP"
  if promote "$N" && smoke "$N"; then
    if [ "${SKIP_EVALS:-0}" = 1 ]; then log "SKIP_EVALS=1: not spawning evals for $N"; else spawn_evals "$N"; fi
    return 0
  fi
  log "$N: no evals (promotion or smoke failed)"; return 1
}
declare -A PIDS
for N in $RUNS; do pipeline "$N" & PIDS[$N]=$!; done
rc=0
for N in $RUNS; do wait "${PIDS[$N]}" || rc=1; done
log "chain_bce5 done rc=$rc. Eval results land in advaitic-runs:/evals/dev_<name>_r{1,2,3} (+ .log)."
rm -rf "$TMP"
exit $rc
