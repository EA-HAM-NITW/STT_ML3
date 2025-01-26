import torch
from train import SpeechToTextModel, train_speech_to_text
from dataloader import create_dataloader

# Create DataLoader
dataloader = create_dataloader("ds", url="train-clean-100", batch_size=4)

# Create model
vocab_size = 128  # adjust as needed
model = SpeechToTextModel(vocab_size=vocab_size)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Train
train_speech_to_text(model, dataloader, epochs=5, device=device)