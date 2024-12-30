# Speech To Text

## Overview

This Speech-To-Text Model uses deep learning technique to convert spoken language into textual form. We will use a sequence-to-sequence architecture, an **encoder-decoder** model to map the audio input to textual output.

## Dataset 
- **Source:** Publicly available repository [LibriSpeech](https://www.tensorflow.org/datasets/catalog/librispeech)

- **Size:** Large-scale (1000 hours) corpus of read English speech

- **Sampling Rate:** 16kHz

- **Format:** Audio files given in Free Lossless Audio Codec (.flac) format, corresponding text data is given as plain text (.txt) files

## Data Pre-Processing

### Audio Processing:

- **Augmentation:**
Apply noise, speed variations, and pitch shifts to increase data diversity.

 - Simulate **real-world conditions** like background noise.

- **Feature Extraction:** Extract acoustic features from the audio, such as Mel-Frequency Cepstral Coefficients (MFCCs) or log-Mel spectrograms.

### Text Processing:

- **Cleaning:** Remove punctuation, special characters, and convert to lowercase.

- **Tokenization:** Split text into individual words or sub-word units

- **Labeling:** Create a vocabulary of all unique words and assign an integer ID to each word.


## Model Architecture 

### Encoder

- **Pre-trained Transformer Encoder:** Use a pre-trained transformer model Wav2Vec 2.0 specifically designed for audio tasks. 
- This encoder takes the extracted audio features as input and learns contextual representations of the spoken language.
- Outputs a **context vector**

### Decoder 

- **Transformer Decoder**
- This decoder takes the encoded representation (context vector) from the pre-trained encoder as input. 
- Generates the output text sequence one token at a time, conditioned on the previous generated tokens
- Use **attention mechanism** to focus on relevant part of the input

## Model Training

- **Loss function:** Connectionist  Temporal Classification (CTC)
- **Optimization:** AdamW with appropriate learning rate and weight decay
- **Training Procedure:**

    - Train the model on the training data for a specified number of epochs.
    - Monitor performance on a validation set.
    - Adjust hyperparameters (learning rate, batch size, etc.) to optimize performance.

## Evaluation:

### Metrics:
- **Word Error Rate (WER):** Measures the percentage of words that are incorrectly transcribed.
- **Character Error Rate (CER):** Measures the percentage of characters that are incorrectly transcribed.
- **Accuracy:** Measures the overall accuracy of the model.
- **Evaluation Data:** Use a separate test set to evaluate the model's performance on unseen data.