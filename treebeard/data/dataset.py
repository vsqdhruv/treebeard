## dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import json
from fxpmath import Fxp
from treebeard.data.quantisation import quantise_dataset, quantisation_error_report, FIXED_POINT_SPECS

def onehot_coding(target, output_dim):
    """Convert class labels into one-hot encoded vectors."""
    target_onehot = torch.zeros(output_dim)
    target_onehot[int(target)] = 1
    return target_onehot

def just_candidate_vector(X_cand, X_jet, puppi_cand_features, cand_features=[]):
    selected_cand_features = [puppi_cand_features[cand_feature] for cand_feature in cand_features]
    n_particles = (X_cand[:, :, puppi_cand_features['pT']] != 0).sum(axis=1)
    X = X_cand[:, :, selected_cand_features]
    
    return X, n_particles

def preprocess_jets(X_cand,X_jet,y,create_and_slice_features, num_candidates= 128, test_size=0.3):
    X, n_particles = create_and_slice_features(X_cand,X_jet)
    
    X = X[:,:num_candidates,:]
    n_particles = np.minimum(n_particles, num_candidates)

    X_train, X_test, y_train, y_test, n_train, n_test = train_test_split(X, y, n_particles, 
                                                                         test_size=test_size, random_state=42, 
                                                                         stratify=y, shuffle = True)

    return X_train, X_test, y_train, y_test, n_train, n_test

def aggregate_advanced(X):
    mask = (X[:, :, 0] != 0)[:, :, np.newaxis]
    X_masked = np.where(mask, X, np.nan)
    
    n_nonzero = mask.sum(axis=1).astype(float)  # (N, 1) broadcast across features

    mean  = np.nanmean(X_masked, axis=1)
    std   = np.nanstd(X_masked, axis=1)
    percs = np.nanpercentile(X_masked, [25, 50, 75], axis=1)
    skew_approx = (mean - percs[0]) / (std + 1e-8)
    
    agg = np.concatenate([
        mean, std,
        np.nanmin(X_masked,  axis=1),
        np.nanmax(X_masked,  axis=1),
        percs[0], percs[1], percs[2],
        skew_approx, n_nonzero,
    ], axis=1)
    
    return np.nan_to_num(agg, nan=0.0)
    
def get_jets(
    data_dir, batch_size, distill=False, output_dims=4, num_candidates=16, cand_features=['pT','eta','phi','z0','dxy','puppiweight']): #cand_features=['pT','dxy']

    print("loading jet dataset")
    cache_dir = os.path.join(data_dir, 'jet_cache/')
    os.makedirs(cache_dir, exist_ok=True)

    raw_cache = os.path.join(cache_dir, 'X_train_raw.npy')

    if os.path.exists(raw_cache):
        print("Loading cached features")
        X_train_agg   = np.load(os.path.join(cache_dir, 'X_train_agg.npy'))
        X_val_agg     = np.load(os.path.join(cache_dir, 'X_val_agg.npy'))
        X_test_agg    = np.load(os.path.join(cache_dir, 'X_test_agg.npy'))
        
        X_tr_raw      = np.load(os.path.join(cache_dir, 'X_train_raw.npy'))
        X_val_raw     = np.load(os.path.join(cache_dir, 'X_val_raw.npy'))
        X_test_raw    = np.load(os.path.join(cache_dir, 'X_test_raw.npy'))

        X_tr_flat     = np.load(os.path.join(cache_dir, 'X_train_flat.npy'))
        X_val_flat    = np.load(os.path.join(cache_dir, 'X_val_flat.npy'))
        X_test_flat   = np.load(os.path.join(cache_dir, 'X_test_flat.npy'))

        y_tr   = np.load(os.path.join(cache_dir, 'y_train.npy'))
        y_val  = np.load(os.path.join(cache_dir, 'y_val.npy'))
        y_test = np.load(os.path.join(cache_dir, 'y_test.npy'))

        n_tr   = np.load(os.path.join(cache_dir, 'n_train.npy'))
        n_val  = np.load(os.path.join(cache_dir, 'n_val.npy'))
        n_test = np.load(os.path.join(cache_dir, 'n_test.npy'))
        
    else:
        X_puppi_cands = np.load(os.path.join(data_dir, 'train/X_part.npy'))
        X_jets        = np.load(os.path.join(data_dir, 'train/X_jet.npy'))
        y_labels      = np.load(os.path.join(data_dir, 'train/y_label.npy'))
        
        with open(os.path.join(data_dir, 'meta_dict.json')) as f:
            meta_dict = json.load(f)
        puppi_cand_features = meta_dict['part_input_vars']

        # Raw (N, 16, 6)
        X_train_raw, X_test_raw, y_train, y_test, n_train, n_test = preprocess_jets(
            X_puppi_cands, X_jets, y_labels,
            lambda X_cand, X_jet: just_candidate_vector(X_cand, X_jet, puppi_cand_features, cand_features=cand_features),
            num_candidates=num_candidates)

        X_tr_raw, X_val_raw, y_tr, y_val, n_tr, n_val = train_test_split(X_train_raw, y_train, n_train, test_size=0.1, random_state=42) 

        # Aggregated (N, 49)
        X_train_agg = aggregate_advanced(X_tr_raw)
        X_val_agg   = aggregate_advanced(X_val_raw)
        X_test_agg  = aggregate_advanced(X_test_raw)

        scaler = StandardScaler()

        X_train_agg = scaler.fit_transform(X_train_agg)
        X_test_agg  = scaler.transform(X_test_agg)
        X_val_agg   = scaler.transform(X_val_agg)

        X_tr_flat   = X_tr_raw.reshape(X_tr_raw.shape[0], -1)
        X_val_flat  = X_val_raw.reshape(X_val_raw.shape[0], -1)
        X_test_flat = X_test_raw.reshape(X_test_raw.shape[0], -1)

        np.save(os.path.join(cache_dir, 'X_train_agg.npy'), X_train_agg)
        np.save(os.path.join(cache_dir, 'X_val_agg.npy'),   X_val_agg)
        np.save(os.path.join(cache_dir, 'X_test_agg.npy'),  X_test_agg)
        
        np.save(os.path.join(cache_dir, 'X_train_raw.npy'), X_tr_raw)
        np.save(os.path.join(cache_dir, 'X_val_raw.npy'),   X_val_raw)
        np.save(os.path.join(cache_dir, 'X_test_raw.npy'),  X_test_raw)

        np.save(os.path.join(cache_dir, 'X_train_flat.npy'), X_tr_flat)
        np.save(os.path.join(cache_dir, 'X_val_flat.npy'),   X_val_flat)
        np.save(os.path.join(cache_dir, 'X_test_flat.npy'),  X_test_flat)
        
        np.save(os.path.join(cache_dir, 'y_train.npy'),     y_tr)
        np.save(os.path.join(cache_dir, 'y_val.npy'),       y_val)
        np.save(os.path.join(cache_dir, 'y_test.npy'),      y_test)

        np.save(os.path.join(cache_dir, 'n_train.npy'),     n_tr)
        np.save(os.path.join(cache_dir, 'n_val.npy'),       n_val)
        np.save(os.path.join(cache_dir, 'n_test.npy'),      n_test)

    aggregated_loaders = make_dataloader(X_train_agg, X_val_agg, X_test_agg, 
                                         y_tr, y_val, y_test, 
                                         output_dims=output_dims, batch_size=batch_size, distill=distill)

    raw_loaders = make_dataloader(X_tr_raw, X_val_raw, X_test_raw,
                                    y_tr, y_val, y_test,
                                    output_dims=output_dims, batch_size=batch_size, distill=distill)
    
    flat_loaders = make_dataloader(X_tr_flat, X_val_flat, X_test_flat, 
                                    y_tr, y_val, y_test,
                                    output_dims=output_dims, batch_size=batch_size, distill=distill)
    
    print(f"Loaded jets — Train: {len(X_train_agg)}, Val: {len(X_val_agg)}, Test: {len(X_test_agg)}")    

    return {
        'raw':  raw_loaders,
        'flat': flat_loaders,
        'agg':  aggregated_loaders,
        'n':    (n_tr, n_val, n_test),
    }

### EVENT CLASSIFIER ###
class JetSet(Dataset):
    def __init__(self, X, y, output_dims):
        self.X = torch.FloatTensor(X)
        y_int = torch.as_tensor(y, dtype=torch.long)
        self.y = torch.nn.functional.one_hot(y_int, num_classes=output_dims).float()
        self.output_dims = output_dims

    def __len__(self):
        return len(self.X)

    def __getitem__(self,idx):
        return self.X[idx], self.y[idx]

def make_dataloader(X_train, X_val, X_test, y_train, y_val, y_test, output_dims, batch_size, distill):
    train_loader = DataLoader(JetSet(X_train, y_train, output_dims), batch_size=batch_size, shuffle=not distill, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(JetSet(X_test, y_test, output_dims), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(JetSet(X_val, y_val, output_dims), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader

def preprocess_events(X,y,create_and_slice_features, test_size=0.3):
    X = create_and_slice_features(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y, shuffle=True)

    return X_train, X_test, y_train, y_test

def just_feature_vector(X_cand,features=[],features_list=[]):
    selected_features = [features_list[feature] for feature in features ]
    X = X_cand[:,selected_features]
    
    return X

def get_events(
    data_dir, batch_size, distill=False, output_dims=2,
    max_num_jets=10, max_num_muons=4, max_num_electrons=4, quantise: bool = False, quant_specs: dict = None
):

    with open(os.path.join(data_dir, 'meta_dict_corrected.json')) as f:
        meta_dict = json.load(f)

    feature_dict = meta_dict['input_vars']
    class_labels = meta_dict['class_labels']

    jet_feature_list = ['L1T_JetPuppiAK4_PT','L1T_JetPuppiAK4_Eta','L1T_JetPuppiAK4_Phi']
    muon_feature_list = ['L1T_MuonTight_PT','L1T_MuonTight_Eta','L1T_MuonTight_Phi']
    electron_feature_list = ['L1T_Electron_PT','L1T_Electron_Eta','L1T_Electron_Phi']
    met_feature_list = ['L1T_PUPPIMET_MET','L1T_PUPPIMET_Eta','L1T_PUPPIMET_Phi']

    MAX_AVAILABLE = {'jets': 10, 'muons': 4, 'electrons': 4}
    assert max_num_jets <= MAX_AVAILABLE['jets']
    assert max_num_muons <= MAX_AVAILABLE['muons']
    assert max_num_electrons <= MAX_AVAILABLE['electrons']

    top_x_jets = [feature + str(i) for i in range(max_num_jets) for feature in jet_feature_list ]
    top_x_muons = [feature + str(i) for i in range(max_num_muons) for feature in muon_feature_list]
    top_x_electrons = [feature + str(i) for i in range(max_num_electrons) for feature in electron_feature_list]
    all_columns = top_x_jets + top_x_muons + top_x_electrons + met_feature_list
        
    print("loading event dataset")
    cache_dir = os.path.join(data_dir, 'event_cache/')
    os.makedirs(cache_dir, exist_ok=True)

    tag = f"j{max_num_jets}_m{max_num_muons}_e{max_num_electrons}"
    train_cache = os.path.join(cache_dir, f'X_train_{tag}.npy')

    if os.path.exists(train_cache):
        print("Loading cached features")
        X_tr   = np.load(train_cache)
        X_val  = np.load(os.path.join(cache_dir, f'X_val_{tag}.npy'))
        X_test = np.load(os.path.join(cache_dir, f'X_test_{tag}.npy'))

        y_tr   = np.load(os.path.join(cache_dir, 'y_train.npy'))
        y_val  = np.load(os.path.join(cache_dir, 'y_val.npy'))
        y_test = np.load(os.path.join(cache_dir, 'y_test.npy'))

    else:
        X_features = np.load(os.path.join(data_dir, 'train/X_features.npy'))
        y_labels   = np.load(os.path.join(data_dir, 'train/y_label.npy'))

        X_train, X_test, y_train, y_test = preprocess_events(
            X_features,y_labels, 
            lambda X_features: just_feature_vector(X_features,features=all_columns,features_list=feature_dict))
        
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42) 

        np.save(train_cache, X_tr)
        np.save(os.path.join(cache_dir, f'X_val_{tag}.npy'), X_val)
        np.save(os.path.join(cache_dir, f'X_test_{tag}.npy'), X_test)

        np.save(os.path.join(cache_dir, 'y_train.npy'), y_tr)
        np.save(os.path.join(cache_dir, 'y_val.npy'),   y_val)
        np.save(os.path.join(cache_dir, 'y_test.npy'),  y_test)

    if quantise:
        specs = quant_specs or FIXED_POINT_SPECS
        tag_q = tag + '_fxp'
        fxp_cache = os.path.join(cache_dir, f'X_test_{tag_q}.npy')

        if os.path.exists(fxp_cache):
            X_test_fxp = np.load(fxp_cache)
        else:
            X_test_fxp = quantise_dataset(X_test, all_columns, specs)
            quantisation_error_report(X_test, X_test_fxp, all_columns)
            np.save(fxp_cache, X_test_fxp)
        X_test = X_test_fxp

    loaders = make_dataloader(X_tr, X_val, X_test, 
                                y_tr, y_val, y_test,
                                output_dims=output_dims, batch_size=batch_size, distill=distill)

    print(f"Loaded events — Train: {len(X_tr)}, Val: {len(X_val)}, Test: {len(X_test)}")

    return {'flat': loaders,
            'raw': loaders,
            'all_columns': all_columns}

def get_events_from_arrays(X_train, y_train, X_val, y_val, X_test, y_test,
                            output_dims, batch_size, distill=False, all_columns=None):
    """
    Build loaders directly from arrays you supply. Use this for bootstrap/subsample 
    variance reps, or any time you want to hand the model a specific dataset rather 
    than have get_events generate/cache one itself.
    """
    loaders = make_dataloader(X_train, X_val, X_test,
                               y_train, y_val, y_test,
                               output_dims=output_dims, batch_size=batch_size, distill=distill)
    print(f"Loaded events (from arrays) — Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return {'flat': loaders, 'raw': loaders, 'all_columns': all_columns}

