#!/bin/bash
# Sequential laptop evals (free; NVIDIA EGL = Modal rendering). One at a time, thermal-throttled.
cd /home/keyu/advaitic/wt-harness
EV=/home/keyu/advaitic/robot-learning-assignment/.venv/bin/advaitic-eval
B="ADV_THROTTLE=1 MUJOCO_GL=egl ORT_DISABLE_TELEMETRY=1 OMP_NUM_THREADS=2 ADV_ESTIMATOR=/home/keyu/advaitic/ckpts/est_v1b.pt ADV_H_SUPERVISE=0 ADV_H_TEMPORAL_ENSEMBLE=1"
run(){ name=$1; ck=$2; env=$3; out=/home/keyu/advaitic/results/laptop/$name
  [ -e $out/report.json ] && { echo "$name exists"; return; }
  echo "$(date +%T) start $name"
  env $B $env ADV_CKPT=/home/keyu/advaitic/ckpts/$ck $EV --policy submission.policy_harness:make_policy \
    --policy-python /home/keyu/advaitic/mlenv/bin/python --episodes 30 --seed-start 1000000 \
    --method $name --checkpoint $ck --out $out > /home/keyu/advaitic/results/laptop/$name.log 2>&1
  echo "$(date +%T) done $name: $(/usr/bin/grep -a '"successes"' $out/report.json 2>/dev/null)"; }
mkdir -p /home/keyu/advaitic/results/laptop
run lap_ens_d005_r1 student_v1b_bce1.pt "ADV_H_ENSEMBLE_DECAY=0.05"
run lap_ens_d010_r1 student_v1b_bce1.pt "ADV_H_ENSEMBLE_DECAY=0.1"
run lap_bce4b_ens_r1 student_bce4b.pt ""
