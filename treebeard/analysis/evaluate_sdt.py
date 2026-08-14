import torch
import numpy as np
from treebeard.models.sdt import SDT
from treebeard.data.dataset import get_jets, get_events
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

def evaluate(input_dim, 
             output_dim, 
             depth, 
             lamda, 
             model=SDT, 
             model_path='sdt_old.pth', 
             device=torch.device('cpu'),
             dataloader = get_events,
             data_path='./C2V_jet_training_dataset',
             batch_size=128):
    
    # load model
    tree = model(input_dim, output_dim, depth, lamda, use_cude=False).to(device)
    tree.load_state_dict(torch.load(model_path, map_location=device))
    tree.eval()

    # generate test data loader
    _,_,test_loader = dataloader(data_dir=data_path, batch_size=batch_size)

    # eval loop
    correct = 0 
    all_probs_sdt = []
    all_preds     = []
    all_targets   = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.view(-1, input_dim).to(device), target.to(device)
            output = tree(data, is_training_data=False)
            probs = torch.softmax(output, dim=1)
            preds = output.argmax(dim=1, keepdim=True)
            target_indices = target.argmax(dim=1, keepdim=True)
            correct += preds.eq(target_indices).sum().item()

            all_probs_sdt.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(target_indices.cpu().numpy())

    all_probs_sdt  = np.concatenate(all_probs_sdt)
    all_preds      = np.concatenate(all_preds)
    all_targets    = np.concatenate(all_targets)

    print(f'Accuracy: {100.*correct/len(test_loader.dataset):.1f}%')
    print(f'ROC-AUC: {roc_auc_score(all_targets, all_probs_sdt, multi_class="ovr"):.4f}')
    print(classification_report(all_targets, all_preds))
    print(confusion_matrix(all_targets, all_preds))

    return all_probs_sdt, all_preds, all_targets

def hls_csim_evaluate(csim_results_path='tb_data/csim_results.log', 
                      truth_path='/eos/user/d/dhnaik/C2V_event_training_data/event_cache/y_test.npy'):
    # load HLS output (2 columns: score_0, score_1 per row)
    hls_scores = np.loadtxt(csim_results_path)
    hls_preds = hls_scores.argmax(axis=1)

    # load true labels — must be same order/rows as tb_input_features.dat
    y_true = np.load(truth_path).flatten()

    accuracy = (hls_preds == y_true).mean()
    print(f"HLS CSIM accuracy: {accuracy:.4f}")

    return hls_scores, hls_preds, y_true

