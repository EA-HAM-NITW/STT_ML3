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

def train_speech_to_text(model, dataloader, epochs=10, device='cuda'):
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for i, batch in enumerate(dataloader):
            spectrogram = batch['log_mel_spec'].to(device)
            tgt_tokens = batch['tokens'].to(device)
            
            optimizer.zero_grad()
            decoder_out, ctc_out = model(spectrogram, tgt_tokens)
            
            input_lengths = torch.full((spectrogram.size(0),), spectrogram.size(1), dtype=torch.long).to(device)
            target_lengths = torch.full((tgt_tokens.size(0),), tgt_tokens.size(1), dtype=torch.long).to(device)

            ctc_out_perm = ctc_out.permute(1, 0, 2)
            loss = ctc_loss_fn(ctc_out_perm, tgt_tokens, input_lengths, target_lengths)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {i}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

def create_dataloader(dataset_dir, url="train-clean-100", batch_size=32):
    dataset = MyLibriSpeechDataset(dataset_dir, url=url)
    return DataLoader(dataset, 
                     batch_size=batch_size, 
                     shuffle=True, 
                     collate_fn=collate_fn,
                     num_workers=2,  # Parallel data loading
                     pin_memory=True)  # Faster data transfer to GPU