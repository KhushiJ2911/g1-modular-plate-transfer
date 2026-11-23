#!/bin/bash
# collect.sh NAME... : fetch evals/dev_<NAME>/report.json, 8 in parallel, with one retry; print results.
S=/tmp/claude-1000/-home-keyu/3b9c1008-0c21-4094-b791-662588c20839/scratchpad/collect; mkdir -p $S
get(){ N=$1; [ -s $S/$N.json ] && python3 -c "import json;json.load(open('$S/$N.json'))" 2>/dev/null && return
  for t in 1 2; do timeout 90 modal volume get --force advaitic-runs evals/dev_$N/report.json $S/$N.json >/dev/null 2>&1 && return; done; }
export -f get; export S
printf '%s\n' "$@" | xargs -P 8 -I{} bash -c 'get {}'
for N in "$@"; do python3 -c "
import json
try: d=json.load(open('$S/$N.json')); print('$N', d['successes'], d['completed_episodes'], d.get('checkpoint'))
except Exception: print('$N', 'pending')"; done
