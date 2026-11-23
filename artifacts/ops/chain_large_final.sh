#!/bin/bash
# Waits for mvla_large_final to finish on Modal, promotes last.pt to the ckpts volume, launches 3 dev evals.
set -u
cd /home/keyu/advaitic
N=mvla_large_final
until /usr/bin/grep -qE "App completed|App stopped|Traceback|Error" ops/modal_$N.log; do sleep 30; done
echo "$(date) train ended: $(tail -1 ops/modal_$N.log)"
modal volume get --force advaitic-runs $N/log.jsonl results/$N.log.jsonl
modal volume get --force advaitic-runs $N/last.pt /home/keyu/advaitic/ckpts/$N.pt || { echo "no last.pt"; exit 1; }
sha256sum ckpts/$N.pt
modal volume put --force advaitic-ckpts ckpts/$N.pt $N.pt
SPEC=""; for r in 1 2 3; do SPEC="$SPEC$N.pt,dev_${N}_r$r,1000000,submission.policy_vla:make_policy;"; done
echo "$(date) launching evals: $SPEC"
modal run ops/modal_sim.py::evals --spec "$SPEC" > ops/modal_evals_$N.log 2>&1
echo "$(date) evals done"; tail -5 ops/modal_evals_$N.log
