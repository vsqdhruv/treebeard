## quantSDT.py

import numpy as np
import torch
from fxpmath import Fxp
from treebeard.models.sdt import SDT
from treebeard.data.quantisation import quantise_fxp
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

# quantSDT.py

class SigmoidLUT: # direct index addressing 
    def __init__(self, low: float, high: float, table_bits: int):
        self.low = low
        self.high = high
        self.n_entries = 2 ** table_bits
        self.lut_x = np.linspace(low, high, self.n_entries)

        self.lut_y = 1.0 / (1.0 + np.exp(-self.lut_x))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x_clipped = np.clip(x, self.low, self.high)
        idx = np.round(
            (x_clipped - self.low) / (self.high - self.low) * (self.n_entries - 1)
        ).astype(np.int64)

        return self.lut_y[idx]

class QuantisedSDT:
    """
    weight_spec : dict(n_word, n_frac, signed, overflow) for W_eff/bias (Fxp backend)
    mu_spec     : dict(n_word, n_frac, signed, overflow) for path_prob/mu (Fxp backend
    acc_spec    : optional dict, same fields, for quantizing the dense-layer
                  accumulator (pre-sigmoid logits). None = float accumulator.
    leaf_spec   : optional dict, same fields, for quantizing leaf_logits (Fxp backend).
                  Softmax itself is computed in float64 -- see note below.)
    requant_mu_every_layer : bool, quantize mu after every routing layer (True,
        hardware-realistic) vs only once at the end (False).
    """

    def __init__(
        self,
        tree : SDT,
        weight_spec : dict,
        mu_spec : dict,
        acc_spec:  dict | None = None,
        leaf_spec: dict | None = None,
        path_prob_spec: dict | None = None,
        requant_mu_every_layer: bool = True,
        sigmoid_low: float = -8.0,
        sigmoid_high: float = 8.0,
        sigmoid_table_bits: int = 10
    ):

        self.depth = tree.depth
        self.internal_node_num = tree.internal_node_num_
        self.leaf_node_num_   = tree.leaf_node_num_

        beta = torch.clamp(tree.beta.detach(), max=5.0).numpy()
        W_raw = tree.inner_nodes[0].weight.detach().numpy()
        W_full_eff = beta[:, None] * W_raw

        #bias_eff = W_full_eff[:,0]
        #W_eff = W_full_eff[:,1:]

        self.W_full_fxp = quantise_fxp(W_full_eff, **weight_spec)
        #self.bias_fxp = quantise_fxp(bias_eff, **weight_spec)
        #self.W_fxp = quantise_fxp(W_eff, **weight_spec)

        self.mu_spec = mu_spec
        self.acc_spec = acc_spec
        self.path_prob_spec = path_prob_spec if path_prob_spec is not None else mu_spec
        self.requant_mu_every_layer = requant_mu_every_layer
        self.sigmoid = SigmoidLUT(sigmoid_low, sigmoid_high, sigmoid_table_bits)

        self.leaf_logits = tree.leaf_logits.detach().numpy()
        if leaf_spec is not None:
            self.leaf_logits = quantise_fxp(self.leaf_logits, **leaf_spec)

        ## need to implement softmax quantisation
        #exp = np.exp(self.leaf_logits - self.leaf_logits.max(axis=1, keepdims=True))
        #self.leaf_probs = exp / exp.sum(axis=1, keepdims=True)

        logit_diff = self.leaf_logits[:,0] - self.leaf_logits[:,1]
        p0 = self.sigmoid(logit_diff)
        p1 = 1 - p0
        self.leaf_probs = np.stack([p0, p1], axis=1)
        
    def _quantise_mu(self, mu: np.ndarray) -> np.ndarray:
        return quantise_fxp(mu, **self.mu_spec)
 
    def _quantise_path_prob(self, p: np.ndarray) -> np.ndarray:
        return quantise_fxp(p, **self.path_prob_spec)

    def forward(self, X_fxp: np.ndarray) -> np.ndarray:
        """
        X_fxp   : already quantised dataset
        Returns : predictions (N, output_dim) 
        """
        batch_size = X_fxp.shape[0]
        X_aug = np.concatenate([np.ones((batch_size, 1), dtype=np.float64), X_fxp], axis=1)
    
        logits = X_aug @ self.W_full_fxp.T
        if self.acc_spec is not None:
            logits=quantise_fxp(logits, **self.acc_spec)

        p_right = self.sigmoid(logits)
        p_left = 1 - p_right

        p_right_fxp = quantise_fxp(p_right, **self.path_prob_spec)
        p_left_fxp = quantise_fxp(p_left, **self.path_prob_spec)

        path_prob = np.stack([p_right_fxp, p_left_fxp], axis=2)

        _mu = np.ones((batch_size,1), dtype=np.float64)
        begin_idx, end_idx = 0, 1
        for layer_idx in range(self.depth):
            _path_prob = path_prob[:, begin_idx:end_idx, :]
            _mu = _mu[:, :, None] * _path_prob
            _mu = _mu.reshape(batch_size, -1)

            if self.requant_mu_every_layer:
                _mu = quantise_fxp(_mu, **self.mu_spec) 

            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (layer_idx + 1)

        if not self.requant_mu_every_layer:
            _mu = quantise_fxp(_mu, **self.mu_spec) 

        _mu = _mu.reshape(batch_size, self.leaf_node_num_)
        print('leaf logit shape : ', self.leaf_logits.shape)
        return _mu @ self.leaf_probs

def load_sdt_checkpoint(pth_path: str, input_dim: int, output_dim: int, 
                        depth: int, lamda: float = 1e-3) -> SDT:
    tree = SDT(input_dim, output_dim, depth, lamda, use_cuda=False)
    state_dict = torch.load(pth_path, map_location="cpu")
    tree.load_state_dict(state_dict)
    tree.eval()
    return tree

def evaluate_against_float(tree: SDT, qsdt: QuantisedSDT, 
                        X_test_float: np.ndarray, X_test_fxp: np.ndarray, y_true: np.ndarray):

    with torch.no_grad():
        y_pred_float = tree(torch.FloatTensor(X_test_float), is_training_data=False).numpy()
    y_pred_fxp = qsdt.forward(X_test_fxp)

    preds_float = y_pred_float.argmax(axis=1)
    preds_fxp = y_pred_fxp.argmax(axis=1)

    return {
        "acc_float": accuracy_score(y_true, preds_float),
        "acc_quant": accuracy_score(y_true, preds_fxp),
        "auc_float": roc_auc_score(y_true, y_pred_float[:, 1]),
        "auc_quant": roc_auc_score(y_true, y_pred_fxp[:, 1]),
        "confusion_quant": confusion_matrix(y_true, preds_fxp),
        "agreement_float_vs_quant": (preds_float == preds_fxp).mean(),
        "y_pred_float": y_pred_float,
        "y_pred_quant": y_pred_fxp,
    }