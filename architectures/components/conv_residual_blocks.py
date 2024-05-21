import torch
import torch.nn as nn

from components.causal_conv1d import CausalConv1d

class ConvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1, groups=1, bias=True, dropout=0.1):
        super(ConvResBlock, self).__init__()
        layers = []
        numlevels = len(out_channels)        
        for i in range(numlevels):                
            in_chans = in_channels if i == 0 else out_channels[i-1]
            out_chans = out_channels[i]
            layers += [CausalConv1d(in_chans,
                                    out_chans,
                                    kernel_size=kernel_size,
                                    stride=stride,            
                                    dilation=dilation,
                                    groups=groups,
                                    bias=bias)]
            if i<(numlevels-1):
                layers += [#nn.BatchNorm1d(out_chans),
                           nn.GELU(),
                           nn.Dropout(dropout)]

        self.block = nn.Sequential(*layers)        

    def forward(self, x):
        return self.block(x)
        

class IdentityBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(IdentityBlock, self).__init__()                
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            #nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        return self.block(x)
