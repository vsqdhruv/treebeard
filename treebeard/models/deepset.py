import torch
import torch.nn as nn

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