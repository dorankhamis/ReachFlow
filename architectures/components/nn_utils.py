import numpy as np
import torch
import torch.nn as nn
import copy

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

@torch.no_grad()
def weights_init_normal(m):
    if isinstance(m, nn.Linear):
        y = m.in_features        
        m.weight.data.normal_(0.0, 1/np.sqrt(y))        
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Conv1d):
        y = m.in_features        
        m.weight.data.normal_(0.0, 1/np.sqrt(y))        
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Conv2d):
        y = m.in_features        
        m.weight.data.normal_(0.0, 1/np.sqrt(y))        
        m.bias.data.fill_(0)        
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0 # lower triangular is True
    
def alltrue_mask(size):
    "Mask template"
    attn_shape = (1, size, size)
    no_mask = np.zeros(attn_shape, dtype='uint8')
    return torch.from_numpy(no_mask) == 0
    
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])  
