import torch
import torch.nn as nn


class CETN(nn.Module):
    '''Custom encoder transformer with learned positional encodings and a CNN/MLP prediction head.'''
    def __init__(self, input_dim, d_model, dim_feedforward, nheads, h_dim, out_dim, seq_length=144):
        super(CETN, self).__init__()
        self.projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_length, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nheads, dim_feedforward, batch_first=True)
        self.full_encoder = nn.TransformerEncoder(encoder_layer, 6)

        self.conv1d_1 = nn.Conv1d(d_model, 64, 5, padding=2)
        self.conv1d_2 = nn.Conv1d(64, 16, 3, padding=1)
        self.ReLU = nn.ReLU()
        self.prediction_in = nn.Linear(seq_length * 16, h_dim)
        self.dropout = nn.Dropout(0.1)
        self.prediction_out = nn.Linear(h_dim, out_dim)

    def forward(self, x):
        # Project features, add learned positions, and run the transformer.
        enc_in = self.projection(x) # shape = batch, seq_length, d_model
        enc_in = enc_in + self.pos_encoding
        enc_out = self.full_encoder(enc_in)

        # Treat embedding features as channels and convolve along time.
        conv_in = enc_out.transpose(1, 2) # shape = batch, d_model, seq_length
        conv_out = self.conv1d_1(conv_in)
        conv_out = self.ReLU(conv_out) # shape = batch, 64, seq_length
        conv_out = self.conv1d_2(conv_out)
        conv_out = self.ReLU(conv_out) # shape = batch, 16, seq_length

        # Flatten the CNN output and predict tomorrow's minimum and maximum.
        pred_in = conv_out.flatten(start_dim=1) # shape = batch, 16 * seq_length
        pred_out = self.prediction_in(pred_in)
        pred_out = self.ReLU(pred_out)
        pred_out = self.dropout(pred_out)
        return self.prediction_out(pred_out) # shape = batch, out_dim
