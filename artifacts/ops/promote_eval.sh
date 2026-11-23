#!/bin/bash
# promote_eval.sh NAME ENVSTRING : wait for advaitic-runs NAME.log exit=0 -> download+verify last.pt -> upload to ckpts
#   -> smoke (harness app) -> spawn 3 dev evals dev_NAME_ens_r{1,2,3} with ENVSTRING.
cd /home/keyu/advaitic; N=$1; E=$2; MP=~/.local/share/uv/tools/modal/bin/python
S=/tmp/claude-1000/-home-keyu/3b9c1008-0c21-4094-b791-662588c20839/scratchpad; L(){ echo "[$(date '+%T')] $N $*"; }
until timeout 120 modal volume get --force advaitic-runs $N.log $S/$N.vol.log >/dev/null 2>&1 && /usr/bin/grep -aq 'exit=' $S/$N.vol.log; do sleep 60; done
/usr/bin/grep -aq 'exit=0' $S/$N.vol.log || { L "train failed: $(tail -n 3 $S/$N.vol.log)"; exit 1; }
L "trained: $(/usr/bin/grep -a '^step=' $S/$N.vol.log | tail -n 1 | cut -c1-120)"
for t in 1 2 3; do timeout 900 modal volume get --force advaitic-runs $N/last.pt ckpts/student_$N.pt >/dev/null 2>&1 && mlenv/bin/python -I -c "import torch; torch.load('/home/keyu/advaitic/ckpts/student_$N.pt', map_location='cpu', weights_only=False)" 2>/dev/null && break; L "download retry $t"; done
timeout 900 modal volume put --force advaitic-ckpts ckpts/student_$N.pt student_$N.pt >/dev/null && L "promoted $(sha256sum ckpts/student_$N.pt | cut -c1-12)"
SM=$(timeout 900 $MP -c "import modal; print(modal.Function.from_name('advaitic-sim-harness2','smoke_run').remote('student_$N.pt','submission.policy_harness:make_policy','$E',20))" 2>&1 | tail -n 1)
L "smoke $SM"; echo "$SM" | /usr/bin/grep -q '"ok": true' || exit 1
$MP ops/spawn.py advaitic-sim-harness2 eval_run "$(python3 -c "import json; print(json.dumps([['student_$N.pt','dev_${N}_ens_r%d'%r,1000000,30,'submission.policy_harness:make_policy','$E',''] for r in (1,2,3)]))")"
L "evals spawned"
