import torch
import numpy as np
import einops
import torch.nn as nn

class SimpleMLP(nn.Module):
    '''Creates a simple MLP module using a single linear layer (out_dim-->in_dim-->ReLU)'''
    def __init__(self, in_dim, out_dim):
        super(SimpleMLP, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.ReLU()

    def forward(self, X):
        out = self.linear(X)
        return self.activation(out)

