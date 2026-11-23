#!/bin/bash
# Wait for bce4a/bce4b on Modal, promote last.pt, run 3 dev evals each (one parallel map).
set -u
cd /home/keyu/advaitic
for N in bce4a bce4b; do
  until /usr/bin/grep -aqE "App completed|App stopped|Traceback" ops/modal_$N.log; do sleep 30; done
  echo "$(date) $N ended"
  modal volume get --force advaitic-runs $N/last.pt ckpts/student_$N.pt >/dev/null || { echo "no last.pt for $N"; exit 1; }
  sha256sum ckpts/student_$N.pt
  modal volume put --force advaitic-ckpts ckpts/student_$N.pt student_$N.pt >/dev/null
done
SPEC=""; for N in bce4a bce4b; do for r in 1 2 3; do SPEC="${SPEC}student_$N.pt,dev_${N}_r$r,1000000;"; done; done
echo "$(date) evals: $SPEC"
modal run ops/modal_sim.py::evals --spec "$SPEC" > ops/modal_evals_bce4.log 2>&1
echo "$(date) done"; /usr/bin/grep -aE 'exit=|all evals' ops/modal_evals_bce4.log
