#!/bin/bash
cd /eos/user/d/dhnaik/treebeard

nohup python -m treebeard.train_sdt \
    --dataset JET_CV2 \
    --data_dir /eos/user/d/dhnaik/C2V_jet_training_data \
    --epochs 120 \
    --depth 6 \
    --lr 8e-4 \
    --weight_decay 1e-4 \
    --lamda 8e-3 \
    --lr_step_size 40 \
    --lr_gamma 0.7 \
    --batch_size 1024 \
    --use_cuda \
    --save_model_path distilled_deepset.pth \
    --save_probs_path dist_deepset_probs_nocw \
    --save_preds_path dist_deepset_preds_nocw \
    --save_targets_path dist_deepset_targets_nocw \
    --save_num_test_parts dist_deepset_n_test_nocw \
    --distill \
    --teacher_epochs 120 \
    --teacher_lr 7.5e-4 \
    --teacher_lr_step_size 40 \
    --teacher_lr_gamma 0.8 \
    --temperature 3 > /eos/user/d/dhnaik/treebeard/logs/train_log_distilled_deepset_nocw.txt 2>&1 &