#!/bin/bash
# SmolVLA continued SFT: convert teacher_v1_gen2 -> merge with plate_transfer -> continue ft_smolvla@5000 for 15k steps -> 3 dev evals.
set -eu
cd /home/keyu/advaitic
echo "$(date) convert gen2"
modal run ops/modal_lerobot.py::convert --shards 6 --src /data/teacher_v1_gen2 --prefix gen2_s > ops/modal_lr_convert_gen2.log 2>&1
/usr/bin/grep -a '^shard' ops/modal_lr_convert_gen2.log
echo "$(date) merge"
modal run ops/modal_lerobot.py::merge --ids "plate_transfer gen2_s0 gen2_s1 gen2_s2 gen2_s3 gen2_s4 gen2_s5" --new plate_transfer_v2 > ops/modal_lr_merge_v2.log 2>&1
S=/tmp/claude-1000/-home-keyu/3b9c1008-0c21-4094-b791-662588c20839/scratchpad/v2info.json
modal volume get --force advaitic-lerobot datasets/advaitic/plate_transfer_v2/meta/info.json $S >/dev/null
python3 -c "import json;d=json.load(open('$S'));print('episodes',d['total_episodes'],'frames',d['total_frames']);assert d['total_episodes']==262+191, 'episode count mismatch'"
echo "$(date) train"
ARGS="--num_workers=8 --policy.chunk_size=20 --policy.n_action_steps=10 --log_freq=100 --batch_size=32 --steps=15000 --save_freq=5000 --policy.path=/lr/runs/ft_smolvla/checkpoints/005000/pretrained_model --rename_map='{\"observation.images.cam0\":\"observation.images.camera1\",\"observation.images.cam1\":\"observation.images.camera2\",\"observation.images.cam2\":\"observation.images.camera3\"}'"
modal run ops/modal_lerobot.py::train --gpu A100-40GB --name ft_smolvla_v2 --timeout 14400 --ds /lr/datasets/advaitic/plate_transfer_v2 --args "$ARGS" > ops/modal_lr_ft_smolvla_v2.log 2>&1
C=ft_smolvla_v2/checkpoints/015000/pretrained_model
modal volume ls advaitic-lerobot runs/$C >/dev/null
echo "$(date) evals"
modal run ops/modal_sim.py::evals_lr --spec "$C,dev_smolvla_v2_r1,1000000;$C,dev_smolvla_v2_r2,1000000;$C,dev_smolvla_v2_r3,1000000" > ops/modal_evals_lr_smolvla_v2.log 2>&1
echo "$(date) done"; /usr/bin/grep -aE 'exit=|all evals' ops/modal_evals_lr_smolvla_v2.log
