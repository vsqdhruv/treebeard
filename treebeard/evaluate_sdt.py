import torch
import numpy as np
from SDT import SDT
from dataset import get_jets
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

input_dim  = 49
output_dim = 4
depth      = 6
lamda      = 1e-4
device     = torch.device('cpu')

# load model
tree = SDT(input_dim, output_dim, depth, lamda, use_cude=False).to(device)
tree.load_state_dict(torch.load('sdt_old.pth', map_location=device))
tree.eval()

# generate test data loader
_,_,test_loader = get_jets(data_dir='./C2V_jet_training_dataset', batch_size=128)

# eval loop
correct = 0 
all_probs_sdt = []
all_preds     = []
all_targets   = []

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.view(-1, args.input_dim).to(device), target.to(device)
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