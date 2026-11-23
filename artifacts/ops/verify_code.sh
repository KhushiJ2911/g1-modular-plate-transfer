#!/bin/bash
# verify_code.sh KEY NAME... : print whether evals/dev_NAME/policy.stderr.log shows harness config containing KEY
# (stale warm containers from an older deployment silently ignore new ADV_H_* flags).
S=/tmp/claude-1000/-home-keyu/3b9c1008-0c21-4094-b791-662588c20839/scratchpad/verify; mkdir -p $S; K=$1; shift
for N in "$@"; do timeout 120 modal volume get --force advaitic-runs evals/dev_$N/policy.stderr.log $S/$N.err >/dev/null 2>&1
  if [ ! -s $S/$N.err ]; then echo "$N: no log"; elif /usr/bin/grep -aq "$K" $S/$N.err; then echo "$N: NEW code ($K present)";
  elif /usr/bin/grep -aq 'ModuleNotFoundError' $S/$N.err; then echo "$N: STALE (module missing)"; else echo "$N: STALE? ($K absent)"; fi; done
