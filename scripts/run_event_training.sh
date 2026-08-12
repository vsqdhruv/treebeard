#!/bin/bash
cd /eos/user/d/dhnaik/treebeard

nohup python -m treebeard.train_sdt \
    --dataset EVENT_C2V \
    --data_dir /eos/user/d/dhnaik/C2V_event_training_data \
    --epochs 30 \
    --depth 2 \
    --lr 1e-3 \
    --weight_decay 5e-5 \
    --lamda 1e-5 \
    --lr_step_size 8 \
    --lr_gamma 0.5 \
    --batch_size 1024 \
    --use_cuda \
    --save_model_path sdt_depth2_v2.pth \
    --save_probs_path sdt_probs_depth2_v2 \
    --save_preds_path sdt_preds_depth2_v2 \
    --save_targets_path sdt_targets_depth2_v2 \
    --save_num_test_parts sdt_n_test_depth2_v2 > /eos/user/d/dhnaik/treebeard/logs/train_log_event_classifier_depth2_v2.txt 2>&1 &