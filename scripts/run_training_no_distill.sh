#!/bin/bash

nohup python train_sdt.py \
    --dataset JET_CV2 \
    --data_dir /eos/user/d/dhnaik/C2V_jet_training_data \
    --epochs 100 \
    --depth 6 \
    --lr 5e-4 \
    --weight_decay 1e-4 \
    --lamda 1e-5 \
    --lr_step_size 30 \
    --lr_gamma 0.6 \
    --batch_size 1024 \
    --use_cuda \
    --save_model_path sdt_flat.pth \
    --save_probs_path sdt_probs_flat \
    --save_preds_path sdt_preds_flat \
    --save_targets_path sdt_targets_flat \
    --save_num_test_parts sdt_n_test_flat > logs/train_log_flat_plainSDT.txt 2>&1 &