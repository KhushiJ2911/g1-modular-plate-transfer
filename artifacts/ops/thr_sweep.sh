#!/bin/bash
# Gripper threshold sweep on dev for student_v1b_bce1 (one eval at a time)
cd /home/keyu/advaitic/wt-bc
for thr in 0.4 0.3; do
  tag=dev_student_v1b_bce1_thr${thr/./}
  PYTHONPATH= MUJOCO_GL=egl ORT_DISABLE_TELEMETRY=1 ADV_CKPT=/home/keyu/advaitic/ckpts/student_v1b_bce1.pt ADV_GRIPPER_THRESHOLD=$thr \
    nice -n 5 /home/keyu/advaitic/robot-learning-assignment/.venv/bin/advaitic-eval --policy submission.policy:make_policy \
    --policy-python /home/keyu/advaitic/mlenv/bin/python --episodes 30 --seed-start 1000000 \
    --method student-v1b-bce1-thr$thr --checkpoint student_v1b_bce1 --out /home/keyu/advaitic/results/$tag > /home/keyu/advaitic/results/$tag.log 2>&1
  echo "$thr done $(date +%H:%M) $(grep -m1 '"successes"' /home/keyu/advaitic/results/$tag/report.json)"
done
