import torch
import torch.nn as nn

class AudioDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=2):
        super(AudioDecoder, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, tgt_tokens, memory):
        # tgt_tokens: (batch_size, tgt_seq_len)
        # memory: (batch_size, src_seq_len, d_model)
        embedded = self.embedding(tgt_tokens)              # (batch_size, tgt_seq_len, d_model)
        out = self.transformer_decoder(embedded, memory)   # (batch_size, tgt_seq_len, d_model)
        logits = self.output_projection(out)               # (batch_size, tgt_seq_len, vocab_size)
        return logits