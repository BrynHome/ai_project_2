import copy
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
    def __init__(self, d_model, n_heads, dropout=0.1):
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
        concat = scores.transpose(1, 2).contiguous().view(batch_s, seq_len_q, self.d_model * self.n_heads)

        output = self.out(concat)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, dff=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dff, d_model)

    def forward(self, x):
        x = self.dropout(nn.functional.relu(self.linear1(x)))
        x = self.linear2(x)
        return x


class Normalize(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.size = d_model
        self.a = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        normalized = self.a * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return normalized


class EncodeLayer(nn.Module):
    """
    Encoder layer.
    One multi-head layer
    One feed-forward layer
    """

    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm1 = Normalize(d_model)
        self.norm2 = Normalize(d_model)
        self.attention = MultiHeadAttention(heads, d_model)
        self.feed = FeedForward(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm1(x)
        x = x + self.drop1(self.attention(x2, x2, x2, mask))
        x2 = self.norm2(x)
        x = x + self.drop2(self.feed(x2))
        return x


class DecoderLayer(nn.Module):
    """
        Decoder layer.
        Two multi-head layers
        One feed-forward layer
        """

    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm1 = Normalize(d_model)
        self.norm2 = Normalize(d_model)
        self.norm3 = Normalize(d_model)

        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

        self.attention1 = MultiHeadAttention(heads, d_model)
        self.attention2 = MultiHeadAttention(heads, d_model)
        self.feed = FeedForward(d_model).cuda()

    def forward(self, x, outputs, src_mask, target_mask):
        x2 = self.norm1(x)
        x = x + self.drop1(self.attention1(x2, x2, x2, target_mask))
        x2 = self.norm2(x)
        x = x + self.drop2(self.attention2(x2, outputs, outputs, src_mask))
        x2 = self.norm3(x)
        x = x + self.drop3(self.feed(x2))
        return x


def clone(module, n):
    return nn.ModuleList([copy.deepcopy(module) for i in range(n)])


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n, heads):
        super().__init__()
        self.n = n
        self.embed = embed(vocab_size, d_model)
        self.positional = PositionalEncoding(d_model)
        self.layers = clone(EncodeLayer(d_model, heads), n)
        self.norm = Normalize(d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.n):
            x = self.layers[i](x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n, heads):
        super().__init__()
        self.n = n
        self.embed = embed(vocab_size, d_model)
        self.positional = PositionalEncoding(d_model)
        self.layers = clone(EncodeLayer(d_model, heads), n)
        self.norm = Normalize(d_model)

    def forward(self, target, outputs, src_mask, target_mask):
        x = self.embed(target)
        x = self.positional(x)
        for i in range(self.n):
            x = self.layers[i](x, outputs, src_mask, target_mask)
        return self.norm(x)


class TransformModel(nn.Module):
    """
    Transformer container model. Contains a encoder and decoder

    """

    def __init__(self, src, target, d_model, n, heads):
        super().__init__()
        self.encoder = Encoder(src, d_model, n, heads)
        self.decoder = Decoder(target, d_model, n, heads)
        self.out = nn.Linear(d_model, target)

    def forward(self, src, target, src_mask, target_mask):
        encoder_out = self.encoder(src, src_mask)
        decoder_out = self.decoder(target, encoder_out, src_mask, target_mask)
        out = self.out(decoder_out)
        return out

d_model = 512
heads = 8
n = 6
vocab = len(EN_TEXT)