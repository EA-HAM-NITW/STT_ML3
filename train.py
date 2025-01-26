import torch
import torch.nn as nn
import torch.optim as optim
from encoder import AudioEncoder
from decoder import AudioDecoder

class SpeechToTextModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=2):
        super(SpeechToTextModel, self).__init__()
        self.encoder = AudioEncoder(d_model=d_model, nhead=nhead, num_layers=num_layers)
        self.decoder = AudioDecoder(vocab_size, d_model=d_model, nhead=nhead, num_layers=num_layers)
        # CTC expects a linear projection, but let's do a placeholder layer
        self.ctc_projection = nn.Linear(d_model, vocab_size)

    def forward(self, spectrogram, tgt_tokens):
        # Encode audio
        encoder_out = self.encoder(spectrogram)
        # Decode to text logits
        decoder_out = self.decoder(tgt_tokens, encoder_out)
        # Optionally for CTC-based approach
        ctc_out = self.ctc_projection(encoder_out)
        return decoder_out, ctc_out

def train_speech_to_text(model, dataloader, epochs=5):
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            # Example batch: {'log_mel_spec': spec, 'tokens': tokens}
            spectrogram = batch['log_mel_spec']        # (batch_size, seq_len, feature_dim)
            tgt_tokens = batch['tokens']               # (batch_size, tgt_seq_len)
            input_lengths = torch.full((spectrogram.size(0),), spectrogram.size(1), dtype=torch.long)
            target_lengths = torch.full((tgt_tokens.size(0),), tgt_tokens.size(1), dtype=torch.long)

            decoder_out, ctc_out = model(spectrogram, tgt_tokens)

            # Typical usage: cross-entropy if we want teacher-forcing decoding,
            # but here we'll show CTC:
            ctc_out_perm = ctc_out.permute(1, 0, 2) # (seq_len, batch_size, vocab_size)
            loss = ctc_loss_fn(ctc_out_perm, tgt_tokens, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")

# Example usage:
# (Create a DataLoader from dataset.py outputs, then run train_speech_to_text(model, dataloader))