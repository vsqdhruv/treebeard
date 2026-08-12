"""
Single quantized-run sanity check. Not a precision sweep -- just confirms
QuantisedSDT runs end-to-end against a real checkpoint and real data, and
that its accuracy is in the right ballpark vs the float model.

Fill in the paths below and run directly:
    python single_quant_test.py
"""

import numpy as np

from treebeard.quant.quant_sdt import load_sdt_checkpoint, QuantisedSDT, evaluate_against_float
from treebeard.data.dataset import quantise_dataset, FIXED_POINT_SPECS

PTH_PATH = "/eos/user/d/dhnaik/SDT/path_output/EVENT_C2V/sdt_flat.pth"
INPUT_DIM = 57      # must match what the checkpoint was trained with
OUTPUT_DIM = 2
DEPTH = 6           # must match the checkpoint's depth

X_TEST_FLOAT_PATH = "/eos/user/d/dhnaik/C2V_event_training_data/event_cache/X_test_j10_m4_e4.npy" 
Y_TEST_PATH = "/eos/user/d/dhnaik/C2V_event_training_data/event_cache/y_test.npy"

jet_feature_list = ['L1T_JetPuppiAK4_PT','L1T_JetPuppiAK4_Eta','L1T_JetPuppiAK4_Phi']
muon_feature_list = ['L1T_MuonTight_PT','L1T_MuonTight_Eta','L1T_MuonTight_Phi']
electron_feature_list = ['L1T_Electron_PT','L1T_Electron_Eta','L1T_Electron_Phi']
met_feature_list = ['L1T_PUPPIMET_MET','L1T_PUPPIMET_Eta','L1T_PUPPIMET_Phi']

max_number_of_jets = 10
max_number_of_muons = 4
max_number_of_electrons = 4

top_x_jets = [feature + str(i) for i in range(max_number_of_jets) for feature in jet_feature_list ]
top_x_muons = [feature + str(i) for i in range(max_number_of_muons) for feature in muon_feature_list]
top_x_electrons = [feature + str(i) for i in range(max_number_of_electrons) for feature in electron_feature_list]
ALL_COLUMNS = top_x_jets + top_x_muons + top_x_electrons + met_feature_list

# Generous bit-widths -- expect acc_quant to be close to acc_float here.
# If it isn't, the bug is in QuantisedSDT/plumbing, not in "quantization is hard".
WEIGHT_SPEC = dict(n_word=16, n_frac=11, signed=True, overflow="wrap")
MU_SPEC = dict(n_word=14, n_frac=13, signed=False, overflow="saturate")  # n_frac = n_word-1
LEAF_SPEC = dict(n_word=8, n_frac=5, signed=True, overflow="wrap")


def main():
    if ALL_COLUMNS is None:
        raise ValueError("Set ALL_COLUMNS to the feature-name list from get_events(...)['all_columns'].")

    print("Loading checkpoint...")
    tree = load_sdt_checkpoint(PTH_PATH, INPUT_DIM, OUTPUT_DIM, DEPTH)

    print("Loading test data...")
    X_test_float = np.load(X_TEST_FLOAT_PATH)
    y_test = np.load(Y_TEST_PATH).flatten()
    print(f"  X_test_float shape: {X_test_float.shape}")
    print(f"  y_test shape: {y_test.shape}")

    assert X_test_float.shape[1] == INPUT_DIM, \
        f"X_test has {X_test_float.shape[1]} columns, expected INPUT_DIM={INPUT_DIM}"
    assert X_test_float.shape[1] == len(ALL_COLUMNS), \
        f"X_test has {X_test_float.shape[1]} columns but ALL_COLUMNS has {len(ALL_COLUMNS)}"

    print("Quantizing test dataset...")
    X_test_q = quantise_dataset(X_test_float, ALL_COLUMNS, FIXED_POINT_SPECS)
    print(f"  X_test_q shape: {X_test_q.shape}, dtype: {X_test_q.dtype}")

    print("Building QuantisedSDT (generous bit-widths)...")
    qsdt = QuantisedSDT(
        tree, WEIGHT_SPEC, MU_SPEC,
        acc_spec=None, leaf_spec=LEAF_SPEC,
        requant_mu_every_layer=True,
    )

    print("Running forward pass on a small slice first (sanity check shapes)...")
    y_pred_small = qsdt.forward(X_test_q[:8])
    print(f"  y_pred_small shape: {y_pred_small.shape}")
    print(f"  y_pred_small (first 3 rows):\n{y_pred_small[:3]}")
    assert y_pred_small.shape == (8, OUTPUT_DIM), "Output shape wrong -- check data augmentation / weight shapes"
    row_sums = y_pred_small.sum(axis=1)
    print(f"  row sums (should be ~1.0): {row_sums}")

    print("\nRunning full evaluation against float model...")
    metrics = evaluate_against_float(tree, qsdt, X_test_float, X_test_q, y_test)

    print(f"\n  acc_float = {metrics['acc_float']:.4f}")
    print(f"  acc_quant = {metrics['acc_quant']:.4f}")
    print(f"  auc_float = {metrics['auc_float']:.4f}")
    print(f"  auc_quant = {metrics['auc_quant']:.4f}")
    print(f"  agreement = {metrics['agreement_float_vs_quant']:.4f}")

    gap = abs(metrics['acc_float'] - metrics['acc_quant'])
    if gap > 0.02:
        print(f"\n  WARNING: accuracy gap ({gap:.4f}) is large for generous bit-widths. "
              f"Suspect a bug (shape mismatch, wrong axis, missing augmentation) "
              f"rather than genuine quantization error at this precision.")
    else:
        print(f"\n  Gap ({gap:.4f}) looks reasonable for a first pass -- plumbing seems correct.")


if __name__ == "__main__":
    main()