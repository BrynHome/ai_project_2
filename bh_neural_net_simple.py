import torch.nn as nn
import torch
from transformers import RobertaModel


class RegressModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=101, out_features=25),
            nn.ReLU(),
            nn.Linear(in_features=25, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=4)
        )

    def forward(self, x):
        return self.linear_layer_stack(x)


class ClassifyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=101, out_features=25),
            nn.Linear(in_features=25, out_features=5)
        )

    def forward(self, x):
        return self.linear_layer_stack(x)


