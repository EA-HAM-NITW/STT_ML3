import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
# ...existing code...
from dataset import preprocess_audio, augment_audio, extract_mfcc, extract_log_mel_spectrogram, clean_text, tokenize_text

class MyLibriSpeechDataset(Dataset):
    def __init__(self, dataset_dir, url="train-clean-100"):
        super().__init__()
        self.data = torchaudio.datasets.LIBRISPEECH(dataset_dir, url=url, download=False)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        waveform, sample_rate, transcript, _, _, _ = self.data[idx]
        processed_audio = preprocess_audio(waveform, sample_rate)
        augmented_audio = augment_audio(processed_audio)
        log_mel_spec = extract_log_mel_spectrogram(augmented_audio)
        cleaned_text = clean_text(transcript)
        tokens = tokenize_text(cleaned_text)
        return {
            'log_mel_spec': log_mel_spec,   # (seq_len, feature_dim)
            'tokens': torch.tensor([ord(c) for c in cleaned_text], dtype=torch.long)
        }

def collate_fn(batch):
    # Simple collate that pads sequences if needed
    specs = [item['log_mel_spec'] for item in batch]
    tokens = [item['tokens'] for item in batch]
    specs_padded = torch.nn.utils.rnn.pad_sequence(specs, batch_first=True)
    tokens_padded = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
    return {'log_mel_spec': specs_padded, 'tokens': tokens_padded}

def create_dataloader(dataset_dir, url="train-clean-100", batch_size=4):
    dataset = MyLibriSpeechDataset(dataset_dir, url=url)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

if __name__ == "__main__":
    # Example usage (assumes setup.sh has run to prepare data)
    loader = create_dataloader("/content/drive/MyDrive/LibriSpeech")
    sample_batch = next(iter(loader))
    print("Batch shapes:", sample_batch['log_mel_spec'].shape, sample_batch['tokens'].shape)