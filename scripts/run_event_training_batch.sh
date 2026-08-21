#!/bin/bash
cd /eos/user/d/dhnaik/treebeard

nohup python -m treebeard.train_sdt \
    --dataset EVENT_C2V \
    --data_dir /eos/user/d/dhnaik/C2V_event_training_data \
    --epochs 10 \
    --depth 7 \
    --lr 1e-3 \
    --weight_decay 5e-5 \
    --lamda 1e-5 \
    --lr_step_size 8 \
    --lr_gamma 0.5 \
    --batch_size 1024 \
    --use_cuda \
    --X_train_override /eos/user/d/dhnaik/C2V_event_training_data/event_cache/X_train_6.npy \
    --y_train_override /eos/user/d/dhnaik/C2V_event_training_data/event_cache/y_train_6.npy \
    --save_model_path sdt_depth7_p6.pth \
    --save_probs_path sdt_probs_depth7_p6 \
    --save_preds_path sdt_preds_depth7_p6 \
    --save_targets_path sdt_targets_depth7_p6 \
    --save_num_test_parts sdt_n_test_depth7_p6 > /eos/user/d/dhnaik/treebeard/logs/final/train_log_depth7_p6.txt 2>&1 &