"""
Soft Decision Tree implementation, adapted from Ethan Shapiro's
Soft-Decision-Tree-Feature-Learning repository:
https://github.com/Ethan-Shapiro/Soft-Decision-Tree-Feature-Learning

Original architecture: Frosst & Hinton, "Distilling a Neural Network Into a
Soft Decision Tree" (2017), https://arxiv.org/abs/1711.09784

Modifications for this project: beta clamping, 
                                running_alpha buffer for penalty calculation, 
                                adaptation to jet and event classifcation from CMS L1 Trigger Primitives
                                DeepSet Teacher Distillation functionionality
"""

##SDT.py

import torch
import torch.nn as nn


class SDT(nn.Module):
    """
    Fast implementation of a soft decision tree in PyTorch.

    Attributes:
        input_dim (int): Number of input dimensions.
        output_dim (int): Number of output dimensions, e.g., number of classes in classification.
        depth (int): Depth of the tree, affecting its complexity.
        lamda (float): Regularization coefficient for the loss function.
        device (torch.device): Computation device (CPU or GPU).
        internal_node_num_ (int): Number of internal nodes in the tree.
        leaf_node_num_ (int): Number of leaf nodes in the tree.
        penalty_list (List[float]): Coefficients for regularization penalty of nodes at different depths.
        inner_nodes (nn.Sequential): Sequential model for internal nodes.
        leaf_nodes (nn.Linear): Linear layer representing leaf nodes.
    """

    def __init__(self, input_dim: int, output_dim: int, depth: int = 5, lamda: float = 1e-3, use_cuda: bool = False):
        """
        Initializes the Soft Decision Tree model.

        Parameters:
            input_dim (int): The number of features in the input data.
            output_dim (int): The number of target outputs or classes.
            depth (int): The depth of the tree, affecting the number of nodes.
            lamda (float): Regularization coefficient to control model complexity.
            use_cuda (bool): Flag to enable CUDA (GPU) computation.
        """
        super(SDT, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.lamda = lamda
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self._validate_parameters()

        self.internal_node_num_ = 2 ** self.depth - 1
        self.leaf_node_num_ = 2 ** self.depth

        self.penalty_list = [self.lamda * (2 ** (-depth)) for depth in range(self.depth)]
        self.beta = nn.Parameter(torch.ones(self.internal_node_num_))

        self.inner_nodes = nn.Sequential(nn.Linear(self.input_dim + 1, self.internal_node_num_, bias=False))
        self.leaf_logits = nn.Parameter(torch.randn(self.leaf_node_num_, self.output_dim) * 0.1)

        self.register_buffer('running_alpha', torch.full((self.internal_node_num_,), 0.5))
        self.alpha_update_rate = [2 ** (-layer_idx) for layer_idx in range(self.depth)]

    def forward(self, X: torch.Tensor, is_training_data: bool = False) -> torch.Tensor:
        """
        Performs a forward pass of the model.

        Parameters:
            X (torch.Tensor): Input data tensor.
            is_training_data (bool): Indicates if the pass is for training.

        Returns:
            torch.Tensor: The model's predictions. Includes penalty if is_training_data is True.
        """
        _mu, _penalty = self._forward(X)                     # (batch, leaf_node_num)
        leaf_probs = torch.softmax(self.leaf_logits, dim=1)  # per-leaf Q^l, own softmax
        y_pred = _mu @ leaf_probs

        if is_training_data:
            return y_pred, _penalty
        else:
            return y_pred
        
    def _forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Core implementation of the model's forward pass.

        Parameters:
            X (torch.Tensor): Input data tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of path probabilities and total penalty.
        """
        batch_size = X.size(0)
        X = self._data_augment(X)

        path_prob = torch.sigmoid(torch.clamp(self.beta, max=5.0) * self.inner_nodes(X))
        path_prob = torch.unsqueeze(path_prob, dim=2)
        path_prob = torch.cat((path_prob, 1 - path_prob), dim=2)

        _mu = X.data.new(batch_size, 1, 1).fill_(1.0)
        _penalty = torch.tensor(0.0).to(self.device)

        begin_idx = 0
        end_idx = 1
        for layer_idx in range(self.depth):
            _path_prob = path_prob[:, begin_idx:end_idx, :]
            _penalty += self._cal_penalty(layer_idx, _mu, _path_prob)
            _mu = _mu.view(batch_size, -1, 1).repeat(1, 1, 2) * _path_prob

            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (layer_idx + 1)

        mu = _mu.view(batch_size, self.leaf_node_num_)
        return mu, _penalty 
    
    def _cal_penalty(self, layer_idx: int, _mu: torch.Tensor, _path_prob: torch.Tensor) -> torch.Tensor:
        """
        Computes regularization penalty for a given layer.

        Parameters:
            layer_idx (int): Index of the current tree layer.
            _mu (torch.Tensor): Path probabilities up to the current layer.
            _path_prob (torch.Tensor): Probabilities for routing at the current layer.

        Returns:
            torch.Tensor: Computed regularization penalty for the layer.
        """
        penalty = torch.tensor(0.0).to(self.device)
        batch_size = _mu.size(0)
        _mu = _mu.view(batch_size, 2 ** layer_idx)
        _path_prob = _path_prob.view(batch_size, 2 ** (layer_idx + 1))

        for node in range(2 ** (layer_idx + 1)):
            alpha_batch = torch.sum(_path_prob[:, node] * _mu[:, node // 2], dim=0) / (torch.sum(_mu[:, node // 2], dim=0) + 1e-8)
            momentum = self.alpha_update_rate[layer_idx]
            global_idx = 2 ** layer_idx - 1 + node // 2

            if self.training:
                self.running_alpha[global_idx] = (1 - momentum) * self.running_alpha[global_idx] + momentum * alpha_batch.detach()
            alpha = self.running_alpha[global_idx]

            coeff = self.penalty_list[layer_idx]
            penalty -= 0.5 * coeff * (torch.log(alpha + 1e-8) + torch.log(1 - alpha + 1e-8))
            
        return penalty
    
    def _data_augment(self, X: torch.Tensor) -> torch.Tensor:
        """
        Adds a constant bias term to the input data.

        Parameters:
            X (torch.Tensor): Original input data.

        Returns:
            torch.Tensor: Augmented input data.
        """
        batch_size = X.size(0)
        X = X.view(batch_size, -1)
        bias = torch.ones(batch_size, 1).to(self.device)
        X = torch.cat((bias, X), 1)
        return X    
    
    def _validate_parameters(self):
        """
        Validates model parameters.
        """
        if not self.depth > 0:
            raise ValueError(
                f"The tree depth should be strictly positive, got {self.depth} instead.")
        if not self.lamda >= 0:
            raise ValueError(
                f"The coefficient of the regularization term should not be negative, got {self.lamda} instead.")

class DeepSetsTeacherOld(nn.Module):
    def __init__(self, n_features, output_dim):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(n_features, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32), nn.LayerNorm(32), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
        )
        self.rho = nn.Sequential(
            nn.Linear(16, 64), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 16), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, output_dim),
        )

    def forward(self, x):
        # x: (batch, n_candidates, n_features)
        mask = (x[:, :, 0] != 0).float().unsqueeze(-1)     # (batch, n_candidates, 1), assumes feature 0 is pT
        phi_out = self.phi(x)                              # (batch, n_candidates, 16)
        phi_out = phi_out * mask                           # zero out padded slots' contribution
        summed = phi_out.sum(dim=1)                        # (batch, 16)
        counts = mask.sum(dim=1).clamp(min=1)              # (batch, 1), avoid div-by-zero
        pooled = summed / counts                           # true masked mean

        return self.rho(pooled)
    

class DeepSetsTeacher(nn.Module):
    def __init__(self, n_features, output_dim):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(n_features, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32), nn.LayerNorm(32), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
        )
        # input is 32-dim: masked-mean (16) concat masked-max (16)
        self.rho = nn.Sequential(
            nn.Linear(32, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32), nn.LayerNorm(32), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        # x: (batch, n_candidates, n_features)
        mask = (x[:, :, 0] != 0).float().unsqueeze(-1)   # (batch, n_candidates, 1) — assumes feature 0 is pT

        phi_out = self.phi(x)                              # (batch, n_candidates, 16)
        phi_out_masked = phi_out * mask

        counts = mask.sum(dim=1).clamp(min=1)               # (batch, 1)
        masked_mean = phi_out_masked.sum(dim=1) / counts    # (batch, 16)

        # masked max: set padded slots to -inf so they never win the max, then guard
        # against jets with zero real candidates (counts clamped to min=1 above, but the
        # max itself still needs an explicit fallback since -inf would otherwise propagate)
        phi_out_for_max = phi_out.masked_fill(mask == 0, float('-inf'))
        masked_max = phi_out_for_max.max(dim=1).values
        masked_max = torch.where(torch.isinf(masked_max), torch.zeros_like(masked_max), masked_max)

        pooled = torch.cat([masked_mean, masked_max], dim=1)  # (batch, 32)
        return self.rho(pooled)