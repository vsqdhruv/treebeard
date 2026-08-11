"""
Quantised numpy forward pass for Frosst-Hinton style SDT (SDT.py)

Loads a pre-trained .pth checkpoint, folds beta parameter into weights and biases, quantises with Fxp.fxpmath.
Goal is to match Vitis HLS ap_fixed/ap_ufixed semantics.   
Runs inference on quantised test data (dataset.quantise_dataset), compares against float model performance.
"""

## train_sdt.py

import argparse
from dataset import get_jets, get_events
import numpy as np
import os
from SDT import SDT, DeepSetsTeacher
from quantSDT import QuantisedSDT
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from quantSDT import QuantisedSDT
from dataset import quantise_dataset, FIXED_POINT_SPECS

def unique_path(path):
    """If path exists, append _1, _2, ... before the extension until it doesn't."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 1
    while os.path.exists(f"{base}_{i}{ext}"):
        i += 1
    return f"{base}_{i}{ext}"

def train_and_evaluate(args):
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    print('Using CUDA' if device.type == 'cuda' else 'Using CPU')

    if args.dataset == 'JET_C2V':
        args.output_dim = 4

        loaders = get_jets(data_dir=args.data_dir, batch_size=args.batch_size, distill=args.distill)

        n_train, n_val, n_test = loaders['n']

        train_loader, val_loader, test_loader = loaders['agg']
        #train_loader, val_loader, test_loader = loaders['flat']
        #train_loader, val_loader, test_loader = loaders['raw']

        args.input_dim = train_loader.dataset.X.shape[1]

    elif args.dataset == 'EVENT_C2V':
        args.output_dim = 2

        loaders = get_events(data_dir=args.data_dir, batch_size=args.batch_size, distill=args.distill)

        n_train, n_val, n_test = None, None, None

        train_loader, val_loader, test_loader = loaders['flat']

        args.input_dim = train_loader.dataset.X.shape[1]

    if args.distill:
        # temp
        if args.dataset != 'JET_C2V':
            raise ValueError("--distill is only supported for JET_C2V (requires 'raw' set-level data)")
        
        train_loader, val_loader, test_loader = loaders['raw']
        args.input_dim = loaders['flat'][0].dataset.X.shape[1]
        n_features = train_loader.dataset.X.shape[-1]

        # initialising teacher model
        teacher = DeepSetsTeacher(n_features=n_features, output_dim=args.output_dim).to(device)
        teacher_optimizer = torch.optim.Adam(teacher.parameters(), lr=args.teacher_lr, weight_decay=1e-4)
        teacher_scheduler = torch.optim.lr_scheduler.StepLR(teacher_optimizer, step_size=args.teacher_lr_step_size, gamma=args.teacher_lr_gamma)
        teacher_criterion = nn.CrossEntropyLoss()
        best_teacher_acc = 0.0

        # training teacher model
        for epoch in range(args.teacher_epochs):
            teacher.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                output = teacher(data)
                loss = teacher_criterion(output, target)
                teacher_optimizer.zero_grad()
                loss.backward()
                teacher_optimizer.step()
                if batch_idx % args.log_interval == 0:
                    print(f"Teacher Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                          f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
                    
            # validation loop
            teacher.eval()
            correct = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                    output = teacher(data)
                    pred = output.argmax(dim=1)
                    target_idx = target.argmax(dim=1)
                    correct += pred.eq(target_idx).sum().item()
            teacher_acc = 100. * correct / len(val_loader.dataset)
            if teacher_acc > best_teacher_acc:
                best_teacher_acc = teacher_acc
            print(f'Teacher Epoch: {epoch} Val Acc: {teacher_acc:.2f}% Best: {best_teacher_acc:.2f}%\n')
            teacher_scheduler.step()
        
        # test evaluation
        teacher.eval()
        all_preds_t, all_targets_t = [], []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                output = teacher(data)
                pred = output.argmax(dim=1)
                target_idx = target.argmax(dim=1)
                all_preds_t.append(pred.cpu().numpy())
                all_targets_t.append(target_idx.cpu().numpy())

        all_preds_t = np.concatenate(all_preds_t)
        all_targets_t = np.concatenate(all_targets_t)
            
        print("\n----- Teacher (DeepSets) confusion matrix on test set -----")
        cm = confusion_matrix(all_targets_t, all_preds_t)
        print(cm)
        print(classification_report(all_targets_t, all_preds_t, zero_division=0))

        # GENERATING SOFT LABELS FROM RAW TRAINING DATA 
        unshuffled_raw_dataloader = DataLoader(loaders['raw'][0].dataset, batch_size=args.batch_size, shuffle=False)

        teacher.eval()
        soft_labels = []
        with torch.no_grad():
            for data, _ in unshuffled_raw_dataloader:
                logits = teacher(data.to(device, non_blocking=True))
                soft_labels.append(torch.softmax(logits, dim=1).cpu())
        soft_labels = torch.cat(soft_labels)

        if args.dataset == 'JET_C2V':
            print('\n----- Soft Label Stats ----')
            print("Soft label argmax distribution:", soft_labels.argmax(dim=1).bincount())
            print("Mean soft prob for class 1 (charm):", soft_labels[:, 1].mean().item())

        # hard label mixing - deviation from frosst and hinton
        y_tr_int = loaders['flat'][0].dataset.y.argmax(dim=1)
        assert y_tr_int.shape[0] == soft_labels.shape[0], \
            "y_tr / soft_labels length mismatch — alignment assumption broken"
        
        y_tr_hard = torch.nn.functional.one_hot(y_tr_int, num_classes=args.output_dim).float()
        mixed_target = args.alpha * y_tr_hard + (1 - args.alpha) * soft_labels

        print('\n---- Hard and Mixed Label Stats ----')
        print(f"Hard label bincount:        {y_tr_int.bincount(minlength=args.output_dim).tolist()}")
        print(f"Mixed target argmax bincount (alpha={args.alpha}): "
              f"{mixed_target.argmax(dim=1).bincount(minlength=args.output_dim).tolist()}")
        
        if args.dataset == 'JET_C2V':
            print(f"Mean mixed prob for class 1 (charm): {mixed_target[:, 1].mean().item():.4f}")

        # replace train_loader targets with mixed labels
        X_tr_flat_tensor = loaders['flat'][0].dataset.X  # (batch_size, num_candidates*num_features)
        assert X_tr_flat_tensor.shape[0] == mixed_target.shape[0], \
            "raw/flat train set length mismatch — cannot align soft labels"
        
        # calculate class weighting
        class_counts = y_tr_int.bincount(minlength=args.output_dim).float()
        class_weights = class_counts.sum() / (args.output_dim * class_counts)  # inverse-freq, mean weight ~1
        class_weights = class_weights.to(device, non_blocking=True)
        print(f"\n Class weights (inverse freq): {class_weights.tolist()}")

        # student trains on flattened features + mixed labels = can shuffle here
        train_loader = DataLoader(
            TensorDataset(X_tr_flat_tensor, mixed_target),
            batch_size=args.batch_size, shuffle=True
        )

        _, val_loader, test_loader = loaders['flat']

    # SDT student training - initialising model 
    tree = SDT(args.input_dim, args.output_dim, args.depth, args.lamda, args.use_cuda).to(device)
    optimizer = torch.optim.Adam(tree.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    best_testing_acc = 0.0

    # training loop
    for epoch in range(args.epochs):
        tree.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.view(-1, args.input_dim).to(device, non_blocking=True), target.to(device, non_blocking=True)
            output, penalty = tree(data, is_training_data=True)

            if args.use_class_weights:
                loss = -(class_weights * target * torch.log(output + 1e-8)).sum(dim=1).mean() + penalty
            else:
                loss = -(target * torch.log(output + 1e-8)).sum(dim=1).mean() + penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # diagnostics for class 1 leaf collapse
            if args.leaf_collapse_diagnostics:
                if epoch == 0 and batch_idx < 50:
                    with torch.no_grad():
                        pred_classes = output.argmax(dim=1)
                        argmax_counts = pred_classes.bincount(minlength=args.output_dim)
                        mean_probs = output.mean(dim=0)
                    print(f"[epoch0 batch {batch_idx:02d}] argmax_bincount={argmax_counts.tolist()} "
                        f"mean_prob_per_class={[f'{p:.4f}' for p in mean_probs.tolist()]} "
                        f"loss={loss.item():.4f}")

            if batch_idx % args.log_interval == 0:
                print(f"Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                    f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
                print("train argmax bincount:", output.argmax(dim=1).detach().cpu().bincount(minlength=args.output_dim))

        # validation loop
        tree.eval()
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.view(-1, args.input_dim).to(device, non_blocking=True), target.to(device, non_blocking=True)
                output = tree(data, is_training_data=False)
                pred = output.argmax(dim=1, keepdim=True)
                target_indices = target.argmax(dim=1, keepdim=True)
                correct += pred.eq(target_indices).sum().item()
        accuracy = 100. * correct / len(val_loader.dataset)
        if accuracy > best_testing_acc:
            best_testing_acc = accuracy
        print(f'\nVal set: Epoch: {epoch} Accuracy: {correct}/{len(val_loader.dataset)} '
              f'({accuracy:.0f}%) Best: {best_testing_acc:.0f}%\n')
        scheduler.step()

    # test evaluation
    tree.eval()
    correct = 0
    all_probs_sdt, all_preds, all_targets = [], [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.view(-1, args.input_dim).to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = tree(data, is_training_data=False)
            #probs = torch.softmax(output, dim=1)
            probs = output
            preds = output.argmax(dim=1, keepdim=True)
            target_indices = target.argmax(dim=1, keepdim=True)
            correct += preds.eq(target_indices).sum().item()
            all_probs_sdt.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(target_indices.cpu().numpy())

    all_probs_sdt = np.concatenate(all_probs_sdt)
    all_preds     = np.concatenate(all_preds)
    all_targets   = np.concatenate(all_targets)
    accuracy = 100. * correct / len(test_loader.dataset)
            
    if accuracy > best_testing_acc:
        best_testing_acc = accuracy

    print(f'\nTest set: Epoch: {epoch} Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.0f}%) Best: {best_testing_acc:.0f}%\n')
    if args.output_dim == 2:
        auc = roc_auc_score(all_targets, all_probs_sdt[:, 1])
    else:
        auc = roc_auc_score(all_targets, all_probs_sdt, multi_class='ovr')

    print(f'ROC-AUC (OvR): {auc:.4f}\n')

    print("\n----- Student SDT confusion matrix on test set -----")
    cm_stu = confusion_matrix(all_targets, all_preds)
    print(cm_stu)
    print(classification_report(all_targets, all_preds, zero_division=0))

    path_output_dir = os.path.join('/eos/user/d/dhnaik/SDT/path_output', args.dataset)
    os.makedirs(path_output_dir, exist_ok=True)
 
    model_save_path = unique_path(os.path.join(path_output_dir, args.save_model_path))
    torch.save(tree.state_dict(), model_save_path)
 
    test_outputs_dir = os.path.join('/eos/user/d/dhnaik/SDT/test_outputs', args.dataset)
    os.makedirs(test_outputs_dir, exist_ok=True)

    if args.quantise:
            all_columns = loaders['all_columns']
            
            X_test_raw = test_loader.dataset.X.numpy()
            X_test_q = quantise_dataset(X_test_raw, all_columns, FIXED_POINT_SPECS)
    
            weight_spec = dict(n_word=args.w_n_word, int_bits=args.w_int_bits,
                            signed=True, overflow="wrap")
            mu_spec = dict(n_word=args.mu_n_word, int_bits=args.mu_int_bits,
                            signed=False, overflow="saturate")
            acc_spec = (dict(n_word=args.acc_n_word, int_bits=args.acc_int_bits,
                            signed=True, overflow="wrap")
                        if args.acc_n_word is not None else None)
            leaf_spec = (dict(n_word=args.leaf_n_word, int_bits=args.leaf_int_bits,
                            signed=True, overflow="wrap")
                        if args.leaf_n_word is not None else None)
    
            qsdt = QuantisedSDT(tree, weight_spec, mu_spec, 
                                acc_spec=acc_spec, leaf_spec=leaf_spec, 
                                requant_mu_every_layer=True)
    
            all_probs_q = qsdt.forward(X_test_q)
            all_preds_q = all_probs_q.argmax(axis=1)
    
            acc_q = accuracy_score(all_targets, all_preds_q)
            auc_q = roc_auc_score(all_targets, all_probs_q[:, 1]) if args.output_dim == 2 \
                else roc_auc_score(all_targets, all_probs_q, multi_class='ovr')
            agreement = (all_preds.flatten() == all_preds_q).mean()
    
            print(f"\n----- Quantised SDT test results -----")
            print(f"Quantised accuracy: {acc_q*100:.2f}%  (float: {accuracy:.2f}%)")
            print(f"Quantised ROC-AUC: {auc_q:.4f}  (float: {auc:.4f})")
            print(f"Float vs quantised prediction agreement: {agreement*100:.2f}%")
            print(confusion_matrix(all_targets, all_preds_q))
    
            probs_q_path = unique_path(os.path.join(test_outputs_dir, 'sdt_probs_quantised.npy'))
            preds_q_path = unique_path(os.path.join(test_outputs_dir, 'sdt_preds_quantised.npy'))
            np.save(probs_q_path, all_probs_q)
            np.save(preds_q_path, all_preds_q)
 
    probs_path = unique_path(os.path.join(test_outputs_dir, args.save_probs_path))
    preds_path = unique_path(os.path.join(test_outputs_dir, args.save_preds_path))
    targets_path = unique_path(os.path.join(test_outputs_dir, args.save_targets_path))
    num_test_parts_path = unique_path(os.path.join(test_outputs_dir, args.save_num_test_parts))
 
    np.save(probs_path, all_probs_sdt)
    np.save(preds_path, all_preds)
    np.save(targets_path, all_targets)
    if n_test is not None:
        np.save(num_test_parts_path, n_test)

    print(f'Saved model to: {path_output_dir}')
    print(f'Saved test outputs to: {test_outputs_dir}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training a Soft Decision Tree on MNIST or CELEBA")
    parser.add_argument('--data_dir', type=str, default=os.path.join(os.getcwd(), 'datasets'),
                        help='Directory for storing input data')
    parser.add_argument('--dataset', type=str, choices=[
                        'JET_C2V', 'EVENT_C2V'], default='JET_C2V', help='Dataset to use.')
    parser.add_argument('--feature_idx', type=int, default=0,
                        help='Feature index for CelebA dataset')
    
    parser.add_argument('--input_dim', type=int, default=28*28,
                        help='Input dimension size. Will be overridden based on dataset.')
    parser.add_argument('--output_dim', type=int, default=10,
                        help='Output dimension size (number of classes). This will be overriden based on dataset')
    parser.add_argument('--depth', type=int, default=5,
                        help='Depth of the tree.')
    
    parser.add_argument('--lamda', type=float, default=1e-3,
                        help='Regularization coefficient.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate.')
    parser.add_argument('--lr_step_size', type=int, default=10,
                        help='Epochs between learning rate decay steps.')
    parser.add_argument('--lr_gamma', type=float, default=0.5,
                        help='Learning rate decay factor')
    
    parser.add_argument('--weight_decay', type=float,
                        default=5e-4, help='Weight decay.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of epochs to train.')
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                        help='Number of times to wait before early stopping')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='How many batches to wait before logging training status.')
    parser.add_argument('--use_cuda', action='store_true',
                        default=False, help='Enable CUDA if available.')
    
    parser.add_argument('--save_model_path', type=str,
                        default='stl_star_model.pth', help='Path to save the trained model.')
    parser.add_argument('--save_probs_path', type=str,
                        default='sdt_probs.npy', help='Path to save predicted probabilities.')
    parser.add_argument('--save_preds_path', type=str,
                        default='sdt_preds.npy', help='Path to save predicted predictions.')
    parser.add_argument('--save_targets_path', type=str,
                        default='sdt_targets.npy', help='Path to save predicted targets.')
    parser.add_argument('--save_num_test_parts', type=str,
                        default='sdt_n_test.npy', help='Path to save number of particles per jet in test dataset.')

    parser.add_argument('--distill', action='store_true', default=False,
                        help='Use teacher MLP to generate soft labels for SDT training.')
    parser.add_argument('--quantise', action='store_true', default=False,
                        help='Quantise test dataset and SDT, run quantised model inference.')
    parser.add_argument('--teacher_epochs', type=int, default=50,
                        help='Number of epochs to train the teacher MLP.')
    parser.add_argument('--teacher_lr', type=float, default=7.5e-4,
                        help='Learning rate for teacher MLP.')
    parser.add_argument('--teacher_lr_step_size', type=int, default=8,
                        help='Step size for teacher LR scheduler.')
    parser.add_argument('--teacher_lr_gamma', type=float, default=0.85,
                        help='Gamma for teacher LR scheduler.')
    parser.add_argument('--temperature', type=float, default=4.0,
                        help='Temperature for soft label generation.')
    parser.add_argument('--alpha', type=float, default=0.3,
                    help='Weight on hard one-hot label in mixed_target = alpha*hard + (1-alpha)*soft. '
                         'alpha=0 recovers pure Frosst-Hinton distillation. Explicit deviation from the paper, '
                         'added to counteract charm-class argmax collapse under soft-CE-only training.')
    parser.add_argument('--use_class_weights', action='store_true', default=False,
                        help='Apply inverse-frequency class weighting to the CE loss to counter '
                             'population imbalance (charm ~15%% of data). Explicit deviation from '
                             'Frosst-Hinton, additive to --alpha hard-label mixing.')
    parser.add_argument('--leaf_collapse_diagnostics', action='store_true', default=False,
                        help='Print per-batch outputs for first 50 batch updates of 0th epoch of student training.')

    parser.add_argument('--w_n_word', type=int, default=12)
    parser.add_argument('--w_int_bits', type=int, default=4)
    parser.add_argument('--mu_n_word', type=int, default=8)
    parser.add_argument('--mu_int_bits', type=int, default=1)
    parser.add_argument('--acc_n_word', type=int, default=None)   # None -> no accumulator quantization
    parser.add_argument('--acc_int_bits', type=int, default=None)
    parser.add_argument('--leaf_n_word', type=int, default=None)  # None -> no leaf_logits quantization
    parser.add_argument('--leaf_int_bits', type=int, default=None)
    
    args = parser.parse_args()

    train_and_evaluate(args)
