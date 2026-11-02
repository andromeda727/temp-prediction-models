import torch
import numpy as np
import einops
import torch.nn as nn

class SimpleMLP(nn.Module):
    '''Creates a simple MLP module using a single linear layer (out_dim-->in_dim-->ReLU)'''
    def __init__(self, in_dim, out_dim):
        super(SimpleMLP, self).__init__()
        self.linear = nn.Linear(in_dim, 16)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(16, 2)

    def forward(self, X):
        # out = self.linear(X.flatten(start_dim=1))
        # return out
        out = self.linear(X)
        out = self.activation(out)
        return self.linear2(out)

