import torch
import torch.nn as nn
from dataset import N_MELS  # Import N_MELS constant

class AudioEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=2):
        super().__init__()
        # Change projection to handle mel features correctly
        self.input_projection = nn.Linear(N_MELS, d_model)  # Project from n_mels to d_model
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, inputs):
        # inputs: (batch_size, time, n_mels)
        x = inputs.transpose(1, 2)  # (batch_size, n_mels, time)
        x = x.transpose(1, 2)  # (batch_size, time, n_mels) 
        x = self.input_projection(x)  # (batch_size, time, d_model)
        x = self.transformer_encoder(x)  # (batch_size, time, d_model)
        return x
