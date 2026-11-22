import torch
import numpy as np
import einops
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    '''Classic Sinusoidal Positional Encoding Implementation, all credit to GeeksForGeeks for this particular implementation: https://www.geeksforgeeks.org/deep-learning/transformer-using-pytorch/'''
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class STF(nn.Module):
    '''Standard encoder transformer model with mlp prediction head. Three encoder layers. 
    
     Projects input_dim to given d_model, and adds sinusoidal encoding ala "All You Need is Attention". Inputs into encoder layers with given d_model,
     dim_feedforward, and nheads. Simple ReLU activation for FFN. MLP prediction head takes d_model input, h_dim hidden dimension/width, and out_dim output.
     Architecture/Layers: input --> projection --> add positional encoding --> encoder layer 1 --> encoder layer 2 --> encoder layer 3 --> MLP prediction head'''

    def __init__(self, input_dim, d_model, dim_feedforward, nheads, h_dim, out_dim):
        super(STF, self).__init__()
        self.projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, 200) # max of 200 just in case, should only ever have seq_len = 144 however.
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nheads, dim_feedforward, batch_first=True) # initialize encoder layer with given parameters
        self.full_encoder = nn.TransformerEncoder(self.encoder_layer, 3) # create transformer stack from defined encoder layer
        self.prediction_in = nn.Linear(d_model, h_dim)
        self.ReLU = nn.ReLU()
        self.prediction_out = nn.Linear(h_dim, out_dim)

    def forward(self, x):
        enc_in = self.projection(x) # shape = batch, seq_len, d_model
        enc_in = self.pos_encoding(enc_in) # shape = batch, seq_len, d_model
        enc_out: torch.Tensor = self.full_encoder(enc_in) # shape = batch, seq_len, d_model -- define as torch.Tensor explicitly for pylance to recognize when calling mean() below
        pred_in = enc_out.mean(dim=1, keepdim=True) # mean of each feature  # shape = batch, 1, d_model
        pred_out = self.prediction_in(pred_in) # shape = batch, 1, h_dim
        pred_out = self.prediction_out(pred_out) # shape = batch, 1, out_dim
        return pred_out