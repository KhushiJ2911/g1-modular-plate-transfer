#!/bin/bash
# snapshot last.pt of a run at given steps: snap.sh RUN_DIR NAME STEP...
run=$1; name=$2; shift 2
for s in "$@"; do
  until grep -q "^$s," $run/log.csv 2>/dev/null; do sleep 30; done
  sleep 20; cp $run/last.pt /home/keyu/advaitic/ckpts/${name}_s$s.pt; echo "snap $s $(date +%H:%M)"
done
