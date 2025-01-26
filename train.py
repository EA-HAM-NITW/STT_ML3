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
        self.ctc_projection = nn.Linear(d_model, vocab_size)

    def forward(self, spectrogram, tgt_tokens):
        encoder_out = self.encoder(spectrogram)
        decoder_out = self.decoder(tgt_tokens, encoder_out)
        ctc_out = self.ctc_projection(encoder_out)
        return decoder_out, ctc_out

def train_speech_to_text(model, dataloader, epochs=5, device='cpu'):
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            spectrogram = batch['log_mel_spec'].to(device)  # Move to device
            tgt_tokens = batch['tokens'].to(device)         # Move to device
            
            decoder_out, ctc_out = model(spectrogram, tgt_tokens)
            
            input_lengths = torch.full((spectrogram.size(0),), spectrogram.size(1), dtype=torch.long).to(device)
            target_lengths = torch.full((tgt_tokens.size(0),), tgt_tokens.size(1), dtype=torch.long).to(device)

            ctc_out_perm = ctc_out.permute(1, 0, 2)
            loss = ctc_loss_fn(ctc_out_perm, tgt_tokens, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")

# Example usage:
# (Create a DataLoader from dataset.py outputs, then run train_speech_to_text(model, dataloader))