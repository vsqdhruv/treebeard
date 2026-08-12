"""
quantisation sensitivity sweep for SDT, using quantSDT.py
sweeps weight and mu bit-widths (optionally accumulator/leaf) across tree depths
"""

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
from treebeard.quant.quant_sdt import load_sdt_checkpoint, QuantisedSDT, evaluate_against_float
from treebeard.data.dataset import quantise_dataset, FIXED_POINT_SPECS

### config ###

INPUT_DIM  = 57
OUTPUT_DIM = 2

CHECKPOINTS = {
    "depth=4": "/eos/user/d/dhnaik/SDT/path_output/EVENT_C2V/sdt_depth4.pth",
    "depth=3": "/eos/user/d/dhnaik/SDT/path_output/EVENT_C2V/sdt_depth3.pth",
    "depth=2": "/eos/user/d/dhnaik/SDT/path_output/EVENT_C2V/sdt_depth2.pth",
}

DEPTHS = {"depth=4": 4, "depth=3": 3, "depth=2": 2}

X_TEST_FLOAT_PATH = "/eos/user/d/dhnaik/C2V_event_training_data/event_cache/X_test_j10_m4_e4.npy" 
Y_TEST_PATH       = "/eos/user/d/dhnaik/C2V_event_training_data/event_cache/y_test.npy"

jet_feature_list      = ['L1T_JetPuppiAK4_PT','L1T_JetPuppiAK4_Eta','L1T_JetPuppiAK4_Phi']
muon_feature_list     = ['L1T_MuonTight_PT','L1T_MuonTight_Eta','L1T_MuonTight_Phi']
electron_feature_list = ['L1T_Electron_PT','L1T_Electron_Eta','L1T_Electron_Phi']
met_feature_list      = ['L1T_PUPPIMET_MET','L1T_PUPPIMET_Eta','L1T_PUPPIMET_Phi']

max_number_of_jets      = 10
max_number_of_muons     = 4
max_number_of_electrons = 4

top_x_jets      = [feature + str(i) for i in range(max_number_of_jets) for feature in jet_feature_list ]
top_x_muons     = [feature + str(i) for i in range(max_number_of_muons) for feature in muon_feature_list]
top_x_electrons = [feature + str(i) for i in range(max_number_of_electrons) for feature in electron_feature_list]
ALL_COLUMNS     = top_x_jets + top_x_muons + top_x_electrons + met_feature_list

WEIGHT_GRID = [(13,9)] # ap_fixed<13,4>
MU_GRID     = [(6, 5)] # ap_ufixed<6,1>
REQUANT_EVERY_LAYER_OPTIONS = [True]

ACC_GRID    = [(9,4)] # ap_fixed<9,5>
LEAF_GRID = [(8,5)] # ap_fixed<8,3>
 
def make_spec(n_word, n_frac, signed, overflow):
    return dict(n_word=n_word, n_frac=n_frac, signed=signed, overflow=overflow)

def run_sweep():
    X_test_float = np.load(X_TEST_FLOAT_PATH)
    y_test = np.load(Y_TEST_PATH).flatten()
 
    if ALL_COLUMNS is None:
        raise ValueError("Set ALL_COLUMNS to the feature-name list from get_events(...)['all_columns'].")
 
    X_test_q = quantise_dataset(X_test_float, ALL_COLUMNS, FIXED_POINT_SPECS)
 
    results = []
    trees = {name: load_sdt_checkpoint(path, INPUT_DIM, OUTPUT_DIM, DEPTHS[name])
             for name, path in CHECKPOINTS.items()}
 
    for depth_name, tree in trees.items():
        print(f"\n=== {depth_name} ===")
 
        for (w_word, w_frac), (mu_word, mu_frac),(acc_word,acc_frac),(leaf_word,leaf_frac),requant_every_layer in itertools.product(
            WEIGHT_GRID, MU_GRID, ACC_GRID, LEAF_GRID, REQUANT_EVERY_LAYER_OPTIONS
        ):
            weight_spec = make_spec(w_word, w_frac, signed=True, overflow="wrap")
            mu_spec = make_spec(mu_word, mu_frac, signed=False, overflow="saturate")
            acc_spec = make_spec(acc_word, acc_frac, signed=True, overflow='saturate')
            leaf_spec = make_spec(leaf_word, leaf_frac, signed=True, overflow='wrap')
 
            qsdt = QuantisedSDT(
                tree, weight_spec, mu_spec,
                acc_spec=acc_spec, leaf_spec=leaf_spec,
                requant_mu_every_layer=requant_every_layer,
            )
 
            metrics = evaluate_against_float(tree, qsdt, X_test_float, X_test_q, y_test)
 
            row = {
                "depth": depth_name,
                "w_n_word": w_word, "w_n_frac": w_frac,
                "mu_n_word": mu_word, "mu_n_frac": mu_frac,
                "acc_n_word": acc_word, "acc_n_frac": acc_frac,
                "leaf_n_word": leaf_word, "leaf_n_frac": leaf_frac,
                "requant_every_layer": requant_every_layer,
                "acc_float": metrics["acc_float"],
                "acc_quant": metrics["acc_quant"],
                "auc_float": metrics["auc_float"],
                "auc_quant": metrics["auc_quant"],
                "agreement": metrics["agreement_float_vs_quant"],
            }
            results.append(row)
            #print(f"  w=<{w_word},{w_word-w_frac}> mu=<{mu_word},{mu_word-mu_frac}> accu=<{acc_word},{acc_word-acc_frac}>  "
            print(f"   leaf=<{leaf_word},{leaf_word-leaf_frac}>  "
                  #f"requant_every_layer={requant_every_layer} "
                  f"-> acc={metrics['acc_quant']:.4f} (float={metrics['acc_float']:.4f}) "
                  f"agree={metrics['agreement_float_vs_quant']:.4f}")
 
    df = pd.DataFrame(results)
    df.to_csv("quant_sweep_results_leaf_int_sweep_frac5.csv", index=False)
    return df
 
 
def plot_sweep(df: pd.DataFrame, requant_every_layer: bool = True):
    """Accuracy vs total weight bit-width, one line per depth, best mu-config
    at each weight bit-width, filtered to one requant_every_layer setting."""
    sub = df[df["requant_every_layer"] == requant_every_layer].copy()
    sub["w_total_bits"] = sub["w_n_word"]
 
    plt.figure(figsize=(10, 6))
    for depth_name, group in sub.groupby("depth"):
        best = group.groupby("w_total_bits")["acc_quant"].max().reset_index()
        plt.plot(best["w_total_bits"], best["acc_quant"], marker="o", label=depth_name)
 
    baselines = sub.groupby("depth")["acc_float"].first()
    for depth_name, acc in baselines.items():
        plt.axhline(acc, linestyle="--", alpha=0.3)
 
    plt.xlabel("Weight bit-width (n_word)")
    plt.ylabel("Test accuracy")
    plt.title(f"Fractional Bit Weight Quantization Sweep (n_int=4)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("quant_sweep_results_leaf_int_sweep_frac5.png", dpi=150)
    plt.show()
 
 
if __name__ == "__main__":
    df = run_sweep()
    print(df.sort_values("acc_quant", ascending=False).head(10))
    #plot_sweep(df, requant_every_layer=True)
 