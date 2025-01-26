import torch
import torchaudio
from torchaudio.datasets import LIBRISPEECH
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio
import random
import string
import os

from google.colab import drive
drive.mount('/content/drive')

dataset_dir = '/content/drive/MyDrive/LibriSpeech'
os.makedirs(dataset_dir, exist_ok=True)

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH = SAMPLE_RATE * 10  # 10 seconds max
VOCAB = "abcdefghijklmnopqrstuvwxyz' "
NUM_MFCC = 13
N_FFT = 512
HOP_LENGTH = 128
N_MELS = 40

def preprocess_audio(waveform, sample_rate):
    # Ensure correct sample rate
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Normalize audio
    waveform = waveform / torch.max(torch.abs(waveform))
    
    # Pad or truncate to MAX_AUDIO_LENGTH
    if waveform.shape[1] > MAX_AUDIO_LENGTH:
        waveform = waveform[:, :MAX_AUDIO_LENGTH]
    elif waveform.shape[1] < MAX_AUDIO_LENGTH:
        padding = torch.zeros(1, MAX_AUDIO_LENGTH - waveform.shape[1])
        waveform = torch.cat([waveform, padding], dim=1)
    
    return waveform

def add_noise(waveform, noise_factor=0.005):
    noise = torch.randn_like(waveform) * noise_factor
    return waveform + noise

def change_speed(waveform, speed_factor):
    return torchaudio.functional.speed(waveform, speed_factor)

def shift_pitch(waveform, n_steps):
    return torchaudio.functional.pitch_shift(waveform, SAMPLE_RATE, n_steps)

def augment_audio(waveform):
    augmentations = [add_noise, change_speed, shift_pitch]
    num_augmentations = random.randint(1, len(augmentations))
    
    for _ in range(num_augmentations):
        aug_func = random.choice(augmentations)
        if aug_func == add_noise:
            waveform = aug_func(waveform, noise_factor=random.uniform(0.001, 0.01))
        elif aug_func == change_speed:
            waveform = aug_func(waveform, speed_factor=random.uniform(0.9, 1.1))
        elif aug_func == shift_pitch:
            waveform = aug_func(waveform, n_steps=random.randint(-2, 2))
    
    return waveform

def extract_mfcc(waveform):
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=NUM_MFCC,
        melkwargs={
            'n_fft': N_FFT,
            'n_mels': N_MELS,
            'hop_length': HOP_LENGTH
        }
    )
    return mfcc_transform(waveform)

def extract_log_mel_spectrogram(waveform):
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH
    )
    log_mel_spectrogram = torch.log(mel_spectrogram(waveform) + 1e-9)
    return log_mel_spectrogram

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation.replace("'", "")))
    return text

def tokenize_text(text):
    return text.split()

def create_vocabulary(tokenized_texts):
    vocab = set()
    for tokens in tokenized_texts:
        vocab.update(tokens)
    vocab = sorted(list(vocab))
    word_to_id = {word: i for i, word in enumerate(vocab)}
    id_to_word = {i: word for word, i in word_to_id.items()}
    return word_to_id, id_to_word

def preprocess_audio_text_pair(waveform, sample_rate, text):
    preprocessed_audio = preprocess_audio(waveform, sample_rate)
    augmented_audio = augment_audio(preprocessed_audio)
    
    mfccs = extract_mfcc(augmented_audio)
    log_mel_spec = extract_log_mel_spectrogram(augmented_audio)
    
    cleaned_text = clean_text(text)
    tokenized_text = tokenize_text(cleaned_text)
    
    return {
        'mfccs': mfccs,
        'log_mel_spec': log_mel_spec,
        'tokens': tokenized_text
    }

if __name__ == "__main__":
    librispeech_train = torchaudio.datasets.LIBRISPEECH(dataset_dir, url="train-clean-100", download=True)
    waveform, sample_rate, utterance, _, _, _ = librispeech_train[0]

    preprocessed_audio = preprocess_audio(waveform, sample_rate)
    augmented_audio = augment_audio(preprocessed_audio)

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.title('Original Audio')
    plt.plot(waveform.numpy()[0])
    plt.subplot(3, 1, 2)
    plt.title('Preprocessed Audio')
    plt.plot(preprocessed_audio.numpy()[0])
    plt.subplot(3, 1, 3)
    plt.title('Augmented Audio')
    plt.plot(augmented_audio.numpy()[0])
    plt.tight_layout()
    plt.show()

    mfccs = extract_mfcc(augmented_audio)
    log_mel_spec = extract_log_mel_spectrogram(augmented_audio)

    
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.title('MFCC')
    plt.imshow(mfccs.numpy().squeeze(), aspect='auto', origin='lower')
    plt.colorbar()
    plt.subplot(2, 1, 2)
    plt.title('Log-Mel Spectrogram')
    plt.imshow(log_mel_spec.numpy().squeeze(), aspect='auto', origin='lower')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    sample_size = 1000
    texts = [librispeech_train[i][2] for i in range(sample_size)]

    cleaned_texts = [clean_text(text) for text in texts]
    tokenized_texts = [tokenize_text(text) for text in cleaned_texts]

    word_to_id, id_to_word = create_vocabulary(tokenized_texts)

    print("Sample cleaned and tokenized texts:")
    for tokens in tokenized_texts[:5]:
        print(tokens)

    print("\nVocabulary size:", len(word_to_id))
    print("Sample vocabulary items:")
    print(list(word_to_id.items())[:10])

    encoded_texts = [[word_to_id[word] for word in tokens] for tokens in tokenized_texts]

    print("\nSample encoded texts:")
    for encoded_text in encoded_texts[:5]:
        print(encoded_text)

    preprocessed_data = preprocess_audio_text_pair(waveform, sample_rate, utterance)

    print("\nMFCC shape:", preprocessed_data['mfccs'].shape)
    print("Log-Mel spectrogram shape:", preprocessed_data['log_mel_spec'].shape)
    print("Tokenized text:", preprocessed_data['tokens'])

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.title('MFCC')
    plt.imshow(preprocessed_data['mfccs'].numpy().squeeze(), aspect='auto', origin='lower')
    plt.colorbar()
    plt.subplot(2, 1, 2)
    plt.title('Log-Mel Spectrogram')
    plt.imshow(preprocessed_data['log_mel_spec'].numpy().squeeze(), aspect='auto', origin='lower')
    plt.colorbar()
    plt.tight_layout()
    plt.show()