#!/bin/bash
# Relaunch (detached) SmolVLA continued SFT on plate_transfer_v2, then evals when the 15k checkpoint exists.
set -u
cd /home/keyu/advaitic
ARGS="--num_workers=8 --policy.chunk_size=20 --policy.n_action_steps=10 --log_freq=100 --batch_size=32 --steps=15000 --save_freq=5000 --policy.path=/lr/runs/ft_smolvla/checkpoints/005000/pretrained_model --rename_map='{\"observation.images.cam0\":\"observation.images.camera1\",\"observation.images.cam1\":\"observation.images.camera2\",\"observation.images.cam2\":\"observation.images.camera3\"}'"
setsid nohup modal run --detach ops/modal_lerobot.py::train --gpu A100-40GB --name ft_smolvla_v2 --timeout 14400 --ds /lr/datasets/advaitic/plate_transfer_v2 --args "$ARGS" > ops/modal_lr_ft_smolvla_v2.log 2>&1 &
echo "$(date) launched detached"
C=ft_smolvla_v2/checkpoints/015000/pretrained_model
sleep 600
until modal volume ls advaitic-lerobot runs/$C 2>/dev/null | /usr/bin/grep -q model.safetensors; do sleep 120; done
sleep 60  # let the checkpoint write/commit finish
echo "$(date) checkpoint ready; evals"
modal run ops/modal_sim.py::evals_lr --spec "$C,dev_smolvla_v2_r1,1000000;$C,dev_smolvla_v2_r2,1000000;$C,dev_smolvla_v2_r3,1000000" > ops/modal_evals_lr_smolvla_v2.log 2>&1
echo "$(date) done"; /usr/bin/grep -aE 'exit=|all evals' ops/modal_evals_lr_smolvla_v2.log
