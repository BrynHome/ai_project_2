import math

import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch import nn


class embed(nn.Module):
    """
    Simple word embedding.
    Using passed network, a lookup for its corresponding embedding
    vector is received. The vectors will be learnt by the model.
    """
    def __init__(self, vocab_s, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_s, d_model)

    def forward(self, x):
        return self.embed(x)


class PositionalEncoding(nn.Module):
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
        """

        :param d_model: dimension of embedding
        :param drop_out:
        :param max_len: length of input sequence
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=drop_out)
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / self.d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / self.d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """

        :param x: sequence to fed to the position encoder
        :return: sequence
        """
        # make embeddings relatively larger. This makes the positional
        # encoding relatively smaller, so the original meaning in the embedding
        # vector is not lost when added together.
        x = x * math.sqrt(self.d_model)
        s_l = x.size(1)
        x = x + x + torch.autograd.Variable(self.pe[:, :s_l], requires_grad=False)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout = 0.1):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.s_h_m = int(self.d_model / self.n_heads)
        self.dropout = nn.Dropout(dropout)

        # key,query and value matrixes    #64 x 64
        self.query_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        # single key matrix for all 8 keys #512x512
        self.key_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.value_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.out = nn.Linear(self.n_heads * self.single_head_dim, self.embed_dim)

    def forward(self, key, query, value, mask=None):
        batch_s = key.size(0)
        seq_len = key.size(1)

        seq_len_q = query.size(1)
        # perform linear operation and split into h heads

        key = self.k_linear(key).view(batch_s, seq_len, self.n_heads, self.s_h_m)
        query = self.q_linear(query).view(batch_s, seq_len_q, self.n_heads, self.s_h_m)
        value = self.v_linear(value).view(batch_s, seq_len, self.n_heads, self.s_h_m)

        k = self.key_matrix(key)
        q = self.query_matrix(query)
        v = self.value_matrix(value)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(query, k.transpose(-1, -2))

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float("-1e20"))
            scores = scores / math.sqrt(self.s_h_m)
            scores = nn.functional.softmax(scores, dim=-1)

        if self.dropout is not None:
            scores = self.dropout(scores)

        output = torch.matmul(scores, value)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(batch_s, seq_len_q, self.d_model*self.n_heads)

        output = self.out(concat)

        return output

    
class TransformModel(nn.Module):
    """
    Transformer container model. Contains a encoder and decoder

    """

    def __init__(self, num_classes: int, max_out_len: int, dim: int = 128):
        super().__init__()
