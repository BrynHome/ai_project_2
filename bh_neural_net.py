import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch import nn


class PositionalEncoding(nn.Module) :
    """
    Positional Encoding based on Attention-is-all-you-need positional encoding.
    From torch

    Injection of data about the relative or absolute position of
    tokens in the sequence.

    d_model = embed dim
    dropout = dropout value
    max_len = max length of incoming seq
    """

    def __init__(self, d_model, drop_out=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=drop_out)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """

        :param x: sequence to fed to the position encoder
        :return: sequence
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformModel(nn.Module):
    """
    Transformer container model. Contains a encoder and decoder

    """

    def __init__(self, num_classes: int, max_out_len: int, dim: int = 128):
        super().__init__()