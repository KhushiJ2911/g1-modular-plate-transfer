#!/bin/bash
# flywheel1 collection done -> filter successes -> upload -> train bce6c/b -> verify -> smoke -> 3 ensembled evals each.
cd /home/keyu/advaitic
MP=~/.local/share/uv/tools/modal/bin/python
L(){ echo "[$(date '+%F %T')] $*"; }
until [ "$(timeout 120 modal volume ls advaitic-runs flywheel2 2>/dev/null | /usr/bin/grep -c 'shard_.*\.log')" -ge 24 ]; do sleep 120; done
L "24 shards done"; rm -rf data/flywheel2_raw; mkdir -p data/flywheel2_raw
for t in 1 2 3; do timeout 1800 modal volume get --force advaitic-runs flywheel2 data/flywheel2_raw/ >/dev/null 2>&1 && break; L "get retry $t"; done
SRC=$(dirname $(find data/flywheel2_raw -name 'shard_*' -type d | head -n 1)); L "src $SRC"
ADV_THROTTLE=1 mlenv/bin/python -I ops/flywheel_filter.py $SRC data/flywheel2_succ || { L "filter failed"; exit 1; }
for t in 1 2 3 4; do timeout 3000 modal volume put advaitic-data data/flywheel2_succ /flywheel2_succ && break; L "put retry $t"; done
D="human60 teacher_v1 teacher_v1_dart0 dagger flywheel2_succ"
A="--config method/bc/student_v1b_bce.yaml --data-weights 1 0.4 0.6 0.67 0.67 --val-episodes 6 --microbatch-size 64 --num-workers 2 --warmup-steps 200 --gripper-pos-weight 2"
ADV_CODE=/home/keyu/advaitic/export/bc_5233dc7 setsid nohup modal run --detach ops/modal_train.py::run --gpu A100-40GB --module method.bc.train --run-name bce6c --data "$D" --args "$A --steps 6000 --init-from /ckpts/student_v1b.pt" > ops/modal_bce6c.log 2>&1 &
sleep 120
ADV_CODE=/home/keyu/advaitic/export/bc_5233dc7 setsid nohup modal run --detach ops/modal_train.py::run --gpu A100-40GB --module method.bc.train --run-name bce6d --data "$D" --args "$A --steps 3000 --init-from /ckpts/student_v1b_bce1.pt" > ops/modal_bce6d.log 2>&1 &
S=/tmp/claude-1000/-home-keyu/3b9c1008-0c21-4094-b791-662588c20839/scratchpad
for N in bce6c bce6d; do
  until timeout 120 modal volume get --force advaitic-runs $N.log $S/$N.vol.log >/dev/null 2>&1 && /usr/bin/grep -aq 'exit=' $S/$N.vol.log; do sleep 60; done
  /usr/bin/grep -aq 'exit=0' $S/$N.vol.log || { L "$N failed"; continue; }
  for t in 1 2 3; do timeout 900 modal volume get --force advaitic-runs $N/last.pt ckpts/student_$N.pt >/dev/null 2>&1 && mlenv/bin/python -I -c "import torch; torch.load('/home/keyu/advaitic/ckpts/student_$N.pt', map_location='cpu', weights_only=False)" && break; L "$N download retry $t"; done
  timeout 900 modal volume put --force advaitic-ckpts ckpts/student_$N.pt student_$N.pt >/dev/null && L "$N promoted $(sha256sum ckpts/student_$N.pt | cut -c1-12)"
  E="ADV_ESTIMATOR=/runs/est_v1b/last.pt ADV_H_TEMPORAL_ENSEMBLE=1 ADV_H_SUPERVISE=0"
  SM=$(timeout 900 $MP -c "import modal; print(modal.Function.from_name('advaitic-sim-harness','smoke_run').remote('student_$N.pt','submission.policy_harness:make_policy','$E',20))" 2>&1 | tail -n 1)
  L "$N smoke $SM"; echo "$SM" | /usr/bin/grep -q '"ok": true' || continue
  $MP ops/spawn.py advaitic-sim-harness eval_run "[[\"student_$N.pt\",\"dev_${N}_ens_r1\",1000000,30,\"submission.policy_harness:make_policy\",\"$E\",\"\"],[\"student_$N.pt\",\"dev_${N}_ens_r2\",1000000,30,\"submission.policy_harness:make_policy\",\"$E\",\"\"],[\"student_$N.pt\",\"dev_${N}_ens_r3\",1000000,30,\"submission.policy_harness:make_policy\",\"$E\",\"\"]]"
done
L "done"
