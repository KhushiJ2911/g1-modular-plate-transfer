#!/bin/bash
# Zero-shot benchmark of the released pi0.5 and MolmoAct2 base checkpoints (no training on our data).
set -u
cd /home/keyu/advaitic
NORM="'--policy.normalization_mapping={\"ACTION\":\"MEAN_STD\",\"STATE\":\"MEAN_STD\",\"VISUAL\":\"IDENTITY\"}'"
COMMON="--batch_size=2 --steps=1 --save_freq=1 --log_freq=1 --num_workers=2 --policy.chunk_size=20 --policy.n_action_steps=10"
PI="$COMMON --policy.type=pi05 --policy.pretrained_path=/lr/models/pi05_base --policy.train_expert_only=true --policy.freeze_vision_encoder=true --policy.dtype=bfloat16 $NORM"
MO="$COMMON --policy.type=molmoact2 --policy.checkpoint_path=allenai/MolmoAct2 --policy.train_mode_vlm=freeze --policy.action_mode=continuous --policy.image_keys='[\"observation.images.cam0\",\"observation.images.cam1\",\"observation.images.cam2\"]' --policy.dtype=bfloat16 $NORM"
echo "$(date) building step-0 checkpoints"
modal run ops/modal_lerobot.py::train --gpu L4 --name zs_pi05 --timeout 3600 --exe "python /root/lerobot_zeroshot.py" --args "$PI" > ops/modal_lr_zs_pi05.log 2>&1 &
modal run ops/modal_lerobot.py::train --gpu L4 --name zs_molmoact2 --timeout 3600 --exe "python /root/lerobot_zeroshot.py" --args "$MO" > ops/modal_lr_zs_molmoact2.log 2>&1 &
wait
for n in zs_pi05 zs_molmoact2; do
  modal volume ls advaitic-lerobot runs/$n/checkpoints/000001/pretrained_model >/dev/null 2>&1 || { echo "no checkpoint for $n"; tail -20 ops/modal_lr_$n.log; exit 1; }
done
echo "$(date) verify pi0.5 weights vs base"
modal run ops/modal_lerobot.py::verify --saved /lr/runs/zs_pi05/checkpoints/000001/pretrained_model/model.safetensors --ref /lr/models/pi05_base/model.safetensors 2>&1 | /usr/bin/grep -aE 'saved=|only_in|Error'
P=zs_pi05/checkpoints/000001/pretrained_model; M=zs_molmoact2/checkpoints/000001/pretrained_model; X="--policy-timeout 600"
echo "$(date) launching 6 evals"
modal run ops/modal_sim.py::evals_lr --spec "$P,dev_zs_pi05_r1,1000000,$X;$P,dev_zs_pi05_r2,1000000,$X;$P,dev_zs_pi05_r3,1000000,$X;$M,dev_zs_molmoact2_r1,1000000,$X;$M,dev_zs_molmoact2_r2,1000000,$X;$M,dev_zs_molmoact2_r3,1000000,$X" > ops/modal_evals_lr_zs.log 2>&1
echo "$(date) evals done"; /usr/bin/grep -aE 'exit=|all evals' ops/modal_evals_lr_zs.log
