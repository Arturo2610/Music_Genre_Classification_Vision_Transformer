# Music Genre Classification with Vision Transformer

This repository contains the full implementation, training workflow, and evaluation utilities for a Vision Transformer–based model designed to classify audio into 10 music genres using Mel spectrograms. Spectrograms are treated as images, enabling the model to leverage Transformer-based attention mechanisms for high-quality predictions.

---

## Project Overview

The system processes audio tracks by converting them into Mel spectrograms and feeding them into a Vision Transformer architecture. The model outputs a probability distribution across ten genres:

**blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock**

The pipeline includes full-song temporal analysis, visualizations, and a refined preprocessing setup targeted at music classification tasks.

---

## Features

- **10-Class Genre Recognition**  
  Classifies audio into one of ten genres based on spectral features.

- **Vision Transformer Backbone**  
  A ViT-like architecture with spectrogram-adapted preprocessing and positional encoding.

- **Full Track Temporal Analysis**  
  Songs are split into **4-second segments** with a **2-second hop**, allowing time-resolved probability tracking.

- **Genre Probability Visualization**  
  Generates probability curves showing how the predicted genre changes throughout the track.

- **High Model Accuracy**  
  Achieves **92.03%** accuracy on the test set derived from GTZAN spectrograms.

---

## Installation

```bash
git clone https://github.com/your-repo/VisionTransformer_MusicGenre.git
cd VisionTransformer_MusicGenre
```

Ensure you have a CUDA-enabled PyTorch build installed if you plan to run computations on the GPU.

---

## Dataset

This project is based on Mel spectrograms computed from the **GTZAN** dataset, a long-established benchmark for music genre classification.

- Contains **1000** audio files, each **30 seconds** long.  
- Ten genres with balanced class distribution.  
- Used for research and educational purposes only.  
- Users must comply with the dataset’s licensing terms.

---

## Model Architecture

A ViT-inspired architecture adapted for Mel spectrograms is used for classification.  
For the complete architecture specification, refer to `MODEL.md`.

### Key components

#### SpecAugment
Applies time and frequency masking to improve generalization during training.

#### Convolutional projection
A `Conv2d` layer transforms the spectrogram: (1, 216, 144) → (512, 13, 9)


This serves as a patch-embedding stage.

#### 2D positional encoding
Learnable encodings capture time–frequency spatial structure.

#### Transformer encoder (12 blocks)
Each block consists of:

- `LayerNorm`  
- Multi-Head Self-Attention (8 heads)  
- Residual / DropPath connection  
- `LayerNorm`  
- Feed-Forward Network: `Linear → GELU → Linear`  
- `Dropout`

Token sequence length: **118 tokens**, each **512-dimensional**.

#### Classifier head
A pooled embedding is passed through a linear layer to produce **10 class logits**.

### Parameter summary

- **Total parameters:** 25,356,810  
- **Trainable parameters:** 100%  
- **Approx. model size:** ~197 MB

---

## Workflow

The processing pipeline converts raw audio into model-ready spectrograms and performs classification.

1. **Load audio**  
   Decode audio files using `librosa`.

2. **Segmentation**  
   Split audio into **4-second windows** with a **2-second stride**.  
   - Short windows → padded  
   - Long windows → trimmed

3. **Mel spectrogram extraction**  
   Convert each segment into a 2D Mel spectrogram.

4. **Normalization**  
   Convert to decibel scale and normalize to the **[0, 1]** range.

5. **Classification**  
   Resize spectrograms to **(216, 144)** and feed them to the Vision Transformer.  
   Obtain probability scores for each of the 10 genres.

---

## Performance

Evaluation metrics for each genre:

| Genre     | Precision | Recall | F1-Score | AUC    |
|-----------|-----------:|-------:|---------:|-------:|
| blues     | 0.9412     | 0.9302 | 0.9357   | 0.9972 |
| classical | 0.9855     | 0.9906 | 0.9881   | 1.0000 |
| country   | 0.9167     | 0.9252 | 0.9209   | 0.9961 |
| disco     | 0.8879     | 0.8962 | 0.8921   | 0.9941 |
| hiphop    | 0.9286     | 0.8506 | 0.8879   | 0.9926 |
| jazz      | 0.9553     | 0.9061 | 0.9300   | 0.9959 |
| metal     | 0.9077     | 0.9776 | 0.9414   | 0.9983 |
| pop       | 0.9178     | 0.9331 | 0.9254   | 0.9972 |
| reggae    | 0.9567     | 0.9298 | 0.9431   | 0.9977 |
| rock      | 0.8316     | 0.8759 | 0.8532   | 0.9918 |

**Overall accuracy:** **0.9203**

## How to Use

A helper function `analyze_full_song` provides long-form analysis of entire audio tracks:

```python
audio_path = "path/to/song.mp3"
analyze_full_song(audio_path)
```

This function will:

- Load the pretrained model

- Split the track into 4-second windows with 2-second steps

- Generate average class probabilities

- Produce a probability-over-time plot

## License

This repository is released under the MIT License.

Audio datasets (e.g., GTZAN) retain their original licenses and are not included in this repository.