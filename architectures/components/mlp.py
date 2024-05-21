import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn_utils import weights_init_normal

class DenseBlock(nn.Module):
    def __init__(self, channels_in, channels_out, dropout, activation='relu'):
        super(DenseBlock, self).__init__()        
        self.dense = nn.Linear(channels_in, channels_out)
        
        if activation=='relu': self.act = nn.ReLU()
        elif activation=='gelu': self.act = nn.GELU()
        elif activation=='lrelu': self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.block = nn.Sequential(self.dense, self.act, self.dropout)
        self.apply(weights_init_normal)

    def forward(self, x):        
        return self.block(x)


class MLP(nn.Module):
    def __init__(self, chan_in, chan_out, dropout=0.1, activation='relu'):
        super(MLP, self).__init__()
        layers = []
        num_levels = len(chan_out)
        for i in range(num_levels):
            in_channels = chan_in if i == 0 else chan_out[i-1]
            out_channels = chan_out[i]
            layers += [DenseBlock(in_channels, out_channels, dropout, activation)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
        

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1, activation='relu'):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        if activation=='relu': self.a = F.relu
        elif activation=='gelu': self.a = F.gelu
        elif activation=='lrelu': self.a = F.leaky_relu()
        self.apply(weights_init_normal)

    def forward(self, x):
        return self.w_2(self.dropout(self.a(self.w_1(x))))
