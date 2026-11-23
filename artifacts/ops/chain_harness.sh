#!/bin/bash
# est_v1 done -> smoke harness with the real estimator -> 5 ablation configs x 3 dev runs (harness sim image).
set -u
cd /home/keyu/advaitic
S=/tmp/claude-1000/-home-keyu/3b9c1008-0c21-4094-b791-662588c20839/scratchpad
until timeout 120 modal volume get --force advaitic-runs est_v1b.log $S/est_v1b.vol.log >/dev/null 2>&1 && /usr/bin/grep -aq 'exit=' $S/est_v1b.vol.log; do sleep 120; done
echo "$(date) est_v1b: $(/usr/bin/grep -a 'exit=' $S/est_v1b.vol.log)"
/usr/bin/grep -a '"step"' $S/est_v1b.vol.log | tail -n 1 | cut -c1-900
/usr/bin/grep -aq 'exit=0' $S/est_v1b.vol.log || { echo "estimator failed"; exit 1; }
E="ADV_ESTIMATOR=/runs/est_v1b/last.pt"
P=submission.policy_harness:make_policy
export ADV_SIM_CODE=/home/keyu/advaitic/wt-harness
SM=$(timeout 1800 modal run ops/modal_sim.py::smoke --spec "student_v1b_bce1.pt,$P,$E" 2>&1 | /usr/bin/grep -a '\.pt ')
echo "smoke: $SM"
echo "$SM" | /usr/bin/grep -q '"ok": true' || { echo "smoke failed"; exit 1; }
declare -A CFG=( [h_pass]="student_v1b_bce1.pt|$E ADV_H_SUPERVISE=0" [h_full]="student_v1b_bce1.pt|$E" \
  [h_exec10]="student_v1b_bce1.pt|$E ADV_H_EXEC_STEPS=10" [h_bce4b]="student_bce4b.pt|$E" \
  [h_ens]="student_v1b_bce1.pt|$E ADV_H_TEMPORAL_ENSEMBLE=1" )
SP=""; for k in h_pass h_full h_exec10 h_bce4b h_ens; do ck=${CFG[$k]%%|*}; ev=${CFG[$k]#*|}; for r in 1 2 3; do SP="${SP}$ck,dev_${k}_r$r,1000000,$P,$ev;"; done; done
echo "$(date) launching harness ablations"
ops/launch_evals.sh evals "$SP"
