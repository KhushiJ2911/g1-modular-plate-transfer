#!/bin/bash
# Wait for BCE fine-tune to finish, save ckpt, then stop the DAgger collector (resumable).
until grep -qE "Peak CUDA memory|Traceback" /home/keyu/advaitic/runs/student_v1b_bce1.stdout; do sleep 30; done
cp /home/keyu/advaitic/runs/student_v1b_bce1/last.pt /home/keyu/advaitic/ckpts/student_v1b_bce1.pt
for p in $(pgrep -f "^/home/keyu/advaitic/robot-learning-assignment/.venv/bin/python -m method.dagger.collect"); do kill -CONT $p; kill $p; done
echo "stopped $(date +%H:%M); dagger2 episodes: $(ls /home/keyu/advaitic/data/dagger2/*.npz | wc -l)"
