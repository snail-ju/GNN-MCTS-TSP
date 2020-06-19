# -*- coding: UTF-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear_1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, 1)
        self.bn = nn.BatchNorm1d(hidden_dim)

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, encoder_outputs):
        # (B, L, H) -> (B , L, 1)
        x = self.linear_1(encoder_outputs)
        x = F.tanh(torch.einsum('bhl->blh', self.bn(torch.einsum('blh->bhl', x))))
        x = self.linear_2(x)
        weights = F.softmax(x.squeeze(-1), dim=1)
        # (B, L, H) * (B, L, 1) -> (B, L, H)
        outputs = encoder_outputs * weights.unsqueeze(-1)
        return outputs
