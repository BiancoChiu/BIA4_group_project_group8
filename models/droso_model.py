
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureEncoder(nn.Module):
    def __init__(self, in_dim=74, hidden_dim=128, out_dim=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True),
        )
        self.emb_dim = out_dim

    def forward(self, x):
        # x: [B, in_dim]
        return self.net(x)

