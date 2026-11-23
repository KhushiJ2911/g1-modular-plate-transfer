#!/bin/bash
# Wait for ft_smolvla_v2 step 15000 -> adapter smoke (lr_check) -> spawn 3 dev evals on the deployed sim app.
cd /home/keyu/advaitic
MP=~/.local/share/uv/tools/modal/bin/python
C=ft_smolvla_v2/checkpoints/015000/pretrained_model
until timeout 120 modal volume ls advaitic-lerobot runs/$C 2>/dev/null | /usr/bin/grep -q model.safetensors; do sleep 180; done
sleep 90
echo "$(date) checkpoint ready"
timeout 1500 $MP -c "
import modal; print(modal.Function.from_name('advaitic-sim','lr_check_job').remote('/lr/runs/$C'))" > ops/lr_check_smolvla_v2.log 2>&1
tail -n 12 ops/lr_check_smolvla_v2.log
/usr/bin/grep -qiE 'Traceback|Error' ops/lr_check_smolvla_v2.log && { echo "SMOKE FAILED"; exit 1; }
$MP ops/spawn.py advaitic-sim eval_run_lr "[[\"$C\",\"dev_smolvla_v2_r1\",1000000],[\"$C\",\"dev_smolvla_v2_r2\",1000000],[\"$C\",\"dev_smolvla_v2_r3\",1000000]]"
echo "$(date) evals spawned"
