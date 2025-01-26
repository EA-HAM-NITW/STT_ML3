import torch
from train import SpeechToTextModel, train_speech_to_text
from dataloader import create_dataloader

# Create DataLoader
dataloader = create_dataloader("/workspace/STT_ML3/ds", url="train-clean-100", batch_size=4)

# Create model
vocab_size = 128  # adjust as needed
model = SpeechToTextModel(vocab_size=vocab_size)

# Train
train_speech_to_text(model, dataloader, epochs=5)