import torch
import numpy as np
import einops
import torch.nn as nn
import math

class CustomCNN(nn.Module):
    '''Custom CNN Model utilizing 3 convolution layers, 2 Conv2d and 1 Conv1d, as well as an MLP projection "head" following convolution. 
    
    Architecture/Psuedocode: 
    
    Take input of batch_size x sequence_length x num_features
	Apply padding of 1 in last two dimensions → dims = b, s+2, n+2
	Pass 3 3x3 kernels over data to create 3 (s, n) feature maps --> dims = b, 3, s, n
	Apply non-standard activation (tanh)
	Apply padding of 1 to each feature map (no padding on batch or channel dimensions)
	Pass 1 more 3x3 kernel over (3, s, n) stacked feature map to create one (s, n) feature map --> dims = b, s, n
	Apply non-standard activation (tanh)
	Apply padding of 5 in dimension s (add 5 padding rows to top and bottom of matrix)
	Do 1d convolution of length/size = 11 over (s+10, n) feature map to create len = s feature map (need to transpose into n x s for conv1d to treat n as channel dimension)
	Pass length s vector into simple two layer MLP:
		Sequential(Linear(s, h_dim), ReLU(),Linear(h_dim, out_dim))
    Final outputted vector of length out_dim represents predicted values.
    '''
    def __init__(self, seq_length, f_dim, h_dim, out_dim):
        super(CustomCNN, self).__init__()
        self.pad2d = nn.ZeroPad2d(1)

        self.conv2d_1 = nn.Conv2d(1, 3, 3) # TODO
        self.tanh = nn.Tanh()
        self.conv2d_2 = nn.Conv2d(3, 1, 3) # TODO

        self.pad1d = nn.ZeroPad2d((0, 0, 5, 5))
        self.conv1d = nn.Conv1d(f_dim, 1, 11) # TODO

        self.MLP_in = nn.Linear(seq_length, h_dim)
        self.ReLU = nn.ReLU()
        self.MLP_out = nn.Linear(h_dim, out_dim)

    def forward(self, x):
        '''Assumes x is in batch-first format'''
        # do first convolution (2d)
        in_convL1 = self.pad2d(x) # dims = (batch, seq_len + 2, feature_dim + 2)
        in_convL1 = einops.rearrange(in_convL1, 'b s f -> b 1 s f') # add empty channel dimension (1 channel)
        out_convL1 = self.conv2d_1(in_convL1) # dims = (batch, 3, seq_len, feature_dim)

        # apply activation and do second convolution (2d)
        in_convL2 = self.tanh(out_convL1) # dims = (batch, 3, seq_len, feature_dim)
        in_convL2 = self.pad2d(in_convL2) # dims = (batch, 3, seq_len + 2, feature_dim + 2)
        out_convL2 = self.conv2d_2(in_convL2) # dims = (batch, 1, seq_len, feature_dim)

        # apply activation and do third convolution (1d)
        in_convL3 = self.tanh(out_convL2) # dims = (batch, 1, seq_len, feature_dim)
        in_convL3 = einops.rearrange(in_convL3, 'b c s f -> b s f', c=1) # dims = (batch, seq_len, feature_dim)
        in_convL3 = self.pad1d(in_convL3) # dims = (batch, seq_len + 10, feature_dim)
        in_convL3 = einops.rearrange(in_convL3, 'b s f -> b f s') # dims = (batch, feature_dim, seq_len + 10)
        out_convL3 = self.conv1d(in_convL3) # dims = (batch, 1, seq_len)

        # pass into final MLP prediction layer
        out_MLP = self.MLP_in(out_convL3) # dims = (batch, 1, h_dim)
        out_MLP = self.ReLU(out_MLP) # dims = (batch, 1, h_dim)
        out_MLP = self.MLP_out(out_MLP) # dims = (batch, 1, out_dim)

        # return output of MLP layer
        return out_MLP
        