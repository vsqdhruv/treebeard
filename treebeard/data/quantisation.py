## quantisation.py

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import json
from fxpmath import Fxp

FIXED_POINT_SPECS = {
    'pt':  {'n_word': 14, 'n_frac': 2,  'signed': False},  # ap_ufixed<14,12>
    'eta': {'n_word': 12, 'n_frac': 7,  'signed': True},   # ap_fixed<12,4>
    'phi': {'n_word': 12, 'n_frac': 9,  'signed': True},
    'met' : {'n_word': 14, 'n_frac': 2,  'signed': False},
}

def get_feature_type(colname: str) -> str:
    """
    Map a column name from `all_columns` to its physical feature type.
    """
    if colname.startswith('L1T_PUPPIMET_'):
        suffix = colname.split('_')[-1]
        if suffix == 'MET':
            return 'met'
        elif suffix == 'Eta':
            return 'eta'
        elif suffix == 'Phi':
            return 'phi'
        else:
            raise ValueError(f"Unrecognized PUPPIMET suffix in column: {colname}")
 
    if 'PT' in colname:
        return 'pt'
    elif 'Eta' in colname:
        return 'eta'
    elif 'Phi' in colname:
        return 'phi'
    else:
        raise ValueError(f"Could not classify column: {colname}")

def quantise_fxp(x: np.ndarray, n_word: int, n_frac: int, signed: bool = True,
                        overflow: str = "wrap") -> np.ndarray:
    """
    quantise a numpy array to fixed point representation w/ fxpmath
    returns float64 which equal what the fixed-point representation stores 
    (i.e. simulates assignment to an ap_fixed/ap_ufixed
    variable: round-to-nearest + saturate).
    """
    assert overflow in ("wrap", "saturate")

    fxp = Fxp(x, signed, n_word=n_word, n_frac=n_frac,
              rounding='floor', overflow=overflow) # 'floor' -> AP_TRN, 'wrap' -> AP_WRAP (default)

    return np.array(fxp, dtype=np.float64)

def quantise_dataset(X: np.ndarray, feature_names: list, specs: dict = FIXED_POINT_SPECS) -> np.ndarray:
    """
    apply fixed point quantisation to (N, n_features) array
    """
    assert X.shape[1] == len(feature_names), \
        f'X has {X.shape[1]} columns but {len(feature_names)} have been given'

    X_fxp = np.empty_like(X, dtype=np.float64)
    for col_idx, feature in enumerate(feature_names):
        ftype = get_feature_type(feature)
        spec = specs[ftype]
        X_fxp[:, col_idx] = quantise_fxp(X[:, col_idx], n_word=spec['n_word'],
                                                 n_frac=spec['n_frac'], signed=spec['signed'])

    return X_fxp

def quantisation_error_report(X: np.ndarray, X_fxp: np.ndarray, feature_names: list):
    """
    per-feature-type mean absolute quantisation error
    """

    types = {}
    for i, name in enumerate(feature_names):
        ftype = get_feature_type(name)
        types.setdefault(ftype, []).append(i)
 
    for ftype, idxs in types.items():
        mae = np.mean(np.abs(X[:, idxs] - X_fxp[:, idxs]))
        max_err = np.max(np.abs(X[:, idxs] - X_fxp[:, idxs]))
        print(f"{ftype:>4s}: mean_abs_err={mae:.6f}  max_abs_err={max_err:.6f}")
