import torch
import numpy as np
import torch.nn as nn

class StandardMLP(nn.Module):
    '''Creates a simple MLP module using a single linear layer (in_dim-->h_dim-->ReLU-->out_dim)'''
    def __init__(self, in_dim, h_dim, out_dim):
        super(StandardMLP, self).__init__()
        self.linear = nn.Linear(in_dim, h_dim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(h_dim, out_dim)

    def forward(self, X):
        '''Assumes X is in (batch, in_vector) format [e.g. input is pre-flattened if necessary]'''
        out = self.linear(X) # dim = (batch, 16)
        out = self.activation(out) # dim = (batch, 16)
        return self.linear2(out) # dim = (batch, 2)

