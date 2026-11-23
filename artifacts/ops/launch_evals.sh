#!/bin/bash
# Launch each eval as its own detached Modal app (survives client/network loss), retrying launch timeouts.
# usage: launch_evals.sh ENTRY "spec1;spec2;..."   ENTRY = evals | evals_lr ; each spec as in modal_sim.py
cd /home/keyu/advaitic
ENTRY=$1; IFS=';' read -ra SPECS <<< "$2"
one(){ s="$1"; name=$(echo "$s" | cut -d, -f2); log=ops/evallogs/$name.log; mkdir -p ops/evallogs
  for t in 1 2 3; do modal run --detach ops/modal_sim.py::$ENTRY --spec "$s" > $log 2>&1 & pid=$!; sleep 75
    if /usr/bin/grep -aq 'Created Function eval_run' $log && ! /usr/bin/grep -aqE 'RetryTimeoutError|ConnectionError|Deadline' $log; then
      echo "$name launched $(/usr/bin/grep -aoE 'ap-[A-Za-z0-9]+' $log | head -1)"; return; fi
    kill $pid 2>/dev/null; echo "$name retry $t"; done; echo "$name FAILED TO LAUNCH"; }
for s in "${SPECS[@]}"; do [ -n "$s" ] && { one "$s" & sleep 3; }; done; wait
