import torch.nn as nn
import torch
from transformers import RobertaModel


class RegressModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=101, out_features=25),
            # nn.ReLU(), # For experiment 2
            nn.Linear(in_features=25, out_features=10),
            # nn.ReLU(), # For experiment 2
            nn.Linear(in_features=10, out_features=1)
        )

    def forward(self, x):
        return self.linear_layer_stack(x)


class ClassifyModel(nn.Module):
    def __init__(self, in_feat, out_feat, hidden_units=200):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=in_feat, out_features=hidden_units),
            # nn.ReLU(), # For experiment 2
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            # nn.ReLU(), # For experiment 2
            nn.Linear(in_features=hidden_units, out_features=out_feat)
        )

    def forward(self, x):
        return self.linear_layer_stack(x)


