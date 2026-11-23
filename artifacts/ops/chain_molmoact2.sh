#!/bin/bash
# Fine-tune MolmoAct2 (LoRA VLM + action expert) on A100, then immediately run 3 dev evals.
set -u
cd /home/keyu/advaitic
N=ft_molmoact2
ARGS="--batch_size=8 --steps=1500 --save_freq=1500 --log_freq=50 --num_workers=8 --policy.chunk_size=20 --policy.n_action_steps=10 --policy.type=molmoact2 --policy.checkpoint_path=allenai/MolmoAct2 --policy.train_mode_vlm=lora --policy.action_mode=continuous --policy.image_keys='[\"observation.images.cam0\",\"observation.images.cam1\",\"observation.images.cam2\"]' --policy.gradient_checkpointing=true --policy.dtype=bfloat16 --dataset.image_transforms.enable=true '--policy.normalization_mapping={\"ACTION\":\"MEAN_STD\",\"STATE\":\"MEAN_STD\",\"VISUAL\":\"IDENTITY\"}'"
echo "$(date) train start"
modal run ops/modal_lerobot.py::train --gpu A100-40GB --name $N --timeout 10800 --args "$ARGS" > ops/modal_lr_$N.log 2>&1
echo "$(date) train end: $(tail -3 ops/modal_lr_$N.log | tr '\n' ' ')"
C="$N/checkpoints/001500/pretrained_model"
modal volume ls advaitic-lerobot runs/$C >/dev/null 2>&1 || { echo "no checkpoint $C"; exit 1; }
X="--policy-timeout 600"
modal run ops/modal_sim.py::evals_lr --spec "$C,dev_molmoact2_r1,1000000,$X;$C,dev_molmoact2_r2,1000000,$X;$C,dev_molmoact2_r3,1000000,$X" > ops/modal_evals_lr_molmoact2.log 2>&1
echo "$(date) evals done"; /usr/bin/grep -aE 'exit=|all evals' ops/modal_evals_lr_molmoact2.log
