import torch
import torch.nn as nn

class AudioEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=2):
        super(AudioEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.input_projection = nn.Linear(40, d_model)  # e.g., projecting 40 mel features to d_model

    def forward(self, inputs):
        # inputs: (batch_size, seq_len, feature_dim)
        x = self.input_projection(inputs)
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, d_model)
        out = self.transformer_encoder(x)
        out = out.permute(1, 0, 2)
        return out
