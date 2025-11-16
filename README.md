# Music Genre Classification with Vision Transformer

A Vision Transformer-based deep learning model for classifying audio into 10 music genres using Mel spectrograms as visual representations. Achieves **92.03% accuracy** on the GTZAN benchmark dataset.

---

## Project Overview

This project treats music classification as a computer vision problem by converting audio signals into Mel spectrograms and processing them with a Vision Transformer architecture. The model analyzes spectral-temporal patterns to classify songs into one of ten genres:

**blues • classical • country • disco • hiphop • jazz • metal • pop • reggae • rock**

### Key Features

- **Vision Transformer Architecture** adapted for audio spectrograms with 8 transformer blocks
- **Hybrid Design** combining convolutional projection with self-attention mechanisms
- **Advanced Regularization** using SpecAugment, Dropout (0.3), and Stochastic Depth (0.2)
- **Temporal Analysis** of full songs via 4-second overlapping segments
- **High Performance** with 92.03% test accuracy and AUC scores >0.99 across all classes

---

## Performance

### Overall Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | 92.03% |
| **Precision (Macro)** | 92.13% |
| **Recall (Macro)** | 92.07% |
| **F1-Score (Macro)** | 91.99% |
| **AUC (Macro)** | 99.65% |

### Per-Class Results

| Genre | Precision | Recall | F1-Score | AUC | Support |
|-------|-----------|--------|----------|-----|---------|
| blues | 95.71% | 96.35% | 96.03% | 0.9994 | 301 |
| classical | 92.38% | 97.98% | 95.10% | 0.9986 | 297 |
| country | 95.62% | 81.37% | 87.92% | 0.9950 | 322 |
| disco | 85.16% | 91.67% | 88.29% | 0.9937 | 288 |
| hiphop | 95.54% | 91.13% | 93.28% | 0.9971 | 282 |
| jazz | 95.53% | 90.61% | 93.00% | 0.9959 | 330 |
| metal | 90.77% | 97.76% | 94.14% | 0.9983 | 312 |
| pop | 91.78% | 93.31% | 92.54% | 0.9972 | 299 |
| reggae | 95.67% | 92.98% | 94.31% | 0.9977 | 285 |
| rock | 83.16% | 87.59% | 85.32% | 0.9918 | 282 |

**Best performing:** classical (F1: 95.10%), metal (94.14%), reggae (94.31%)  
**Most challenging:** rock (85.32%) due to stylistic overlap with country and pop

---

## Model Architecture

### Specifications

Total Parameters: 25,356,810 (~197 MB)
Input Shape: (1, 216, 144) # Mel spectrogram
Output Classes: 10
Sequence Length: 118 tokens (after patch embedding)


### Architecture Components

1. **SpecAugment** - Time/frequency masking for data augmentation
2. **Convolutional Projection** - Conv2d(1→512) reduces spatial dimensions and extracts local features
3. **2D Positional Encoding** - Learnable embeddings capture temporal-spectral structure
4. **8 Transformer Encoder Blocks** - Each containing:
   - Layer Normalization (Ba et al., 2016)
   - Multi-Head Self-Attention with 8 heads (Vaswani et al., 2017)
   - Residual connections with DropPath (stochastic depth)
   - Feed-Forward Network: Linear(512→2048) + GELU + Linear(2048→512)
   - Dropout (0.3) for regularization
5. **Classification Head** - Global pooling + Linear(512→10) + Softmax

### Key Design Decisions

**Hybrid CNN-Transformer Architecture:** Following the approach of Wu et al. (2021), the model combines convolutional layers for local feature extraction with transformer blocks for global context modeling. This hybrid design is particularly effective for limited training data scenarios like GTZAN.

**Why Vision Transformer for Audio?** As demonstrated by Dosovitskiy et al. (2021), Vision Transformers excel at capturing long-range dependencies through self-attention mechanisms. When audio is represented as 2D Mel spectrograms, similar spatial-temporal patterns emerge that transformers can effectively model.

For detailed architecture description, see [`MODEL.md`](MODEL.md).

---

## Quick Start

### Prerequisites

- Python ≥3.12
- CUDA-capable GPU (12GB+ VRAM recommended)
- Poetry package manager

### Installation

```bash
# Clone repository
git clone https://github.com/Arturo2610/Music_Genre_Classification_Vision_Transformer.git
cd Music_Genre_Classification_Vision_Transformer

# Install dependencies
poetry install

# Activate environment
poetry shell
```

### Dependencies
Core libraries (managed via Poetry):

python = "^3.12"
torch = { version = "^2.0.1", extras = ["cuda126"] }
torchvision = "^0.20.1"
torchaudio = "^2.5.1"
librosa = "^0.10.2.post1"
numpy = "^1.24.4"
scipy = "^1.14.1"
matplotlib = "^3.9.3"
seaborn = "^0.13.2"
soundfile = "^0.12.1"
tqdm = "^4.67.1"

## Dataset

This project uses the GTZAN Genre Collection (Tzanetakis & Cook, 2002), a widely-used benchmark dataset for music genre classification research.

### Dataset Structure

GTZAN/\
├── blues/     (100 files × 30s each)\
├── classical/ (100 files)\
├── country/   (100 files)\
├── disco/     (100 files)\
├── hiphop/    (100 files)\
├── jazz/      (100 files)\
├── metal/     (100 files)\
├── pop/       (100 files)\
├── reggae/    (100 files)\
└── rock/      (100 files)

- Total: 1000 WAV files (22050 Hz, mono, 16-bit)
- Duration: 30 seconds per file
- Split: 80% train (800 files) / 20% test (200 files)
- Augmentation: 4-second segments with 2-second overlap → ~15,000 training samples

### Obtaining the Dataset

1. Download from Kaggle GTZAN Dataset
2. Extract to project root:

Music_Genre_Classification_Vision_Transformer/
├── Music_Genre_Classification_Vision_Transformer_PL.ipynb
├── GTZAN/
│   ├── blues/
│   ├── classical/
│   └── ...

## Training

### Running the Notebook

Open and execute the Jupyter notebook:

```bash
jupyter notebook Music_Genre_Classification_Vision_Transformer_PL.ipynb
```

Or in VS Code: Click "Run All" cells.

### Training Configuration

# Hyperparameters
batch_size = 64
num_epochs = 150
learning_rate = 1e-4

# Model architecture
model = AudioTransformer(
    patch_size=16,
    dim=512,
    depth=8,              # 8 transformer blocks
    heads=8,              # multi-head attention
    hidden_dim=2048,      # FFN inner dimension (4×dim)
    num_classes=10,
    dropout=0.3,
    drop_path_rate=0.2    # stochastic depth
)

# Loss function - Focal Loss (Lin et al., 2017)
criterion = FocalLoss(gamma=2.5)

# Optimizer - AdamW with weight decay
optimizer = AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=0.1
)

# Learning rate scheduler - Cosine Annealing
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,
    eta_min=1e-7
)

# Early stopping
patience = 10  # Stop if no val improvement for 10 epochs


### Preprocessing Pipeline

Audio is converted to Mel spectrograms with the following parameters:

# Segmentation
chunk_duration = 4       # seconds
overlap_duration = 2     # seconds (50% overlap)

# Mel spectrogram computation
n_mels = 128            # frequency bins
n_fft = 2048            # FFT window size
hop_length = 64         # samples between frames
sample_rate = 22050     # Hz
window = 'hann'         # Hann window (Testa et al., 2004)

# Normalization
1. Convert to dB scale: librosa.amplitude_to_db()
2. Min-max normalize to [0, 1]
3. Resize to (216, 144) pixels using bilinear interpolation


Rationale for parameter choices:

  - chunk_duration=4s: Captures complete melodic phrases while maintaining computational efficiency
  - overlap=2s: Ensures temporal continuity and increases training sample diversity
  - n_mels=128: Standard for music information retrieval (Bahuleyan, 2018)
  - hop_length=64: Provides fine temporal resolution (2.9 ms per frame @ 22050 Hz)
  - Output size (216×144): Chosen to maintain aspect ratio while fitting GPU memory constraints

### Training Details

**Hardware & Duration:**
- **GPU:** NVIDIA GeForce RTX 4070 SUPER (12GB VRAM)
- **CPU:** AMD Ryzen 5 7600
- **RAM:** 32GB DDR5
- **Total time:** ~35 minutes for 124 epochs
- **Early stopping:** Triggered at epoch 124 (patience=10)
- **Best checkpoint:** Epoch 114 (val_acc: 93.92%, val_loss: 0.0718)

**Training Progression:**

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1 | 1.70 | 14.78% | - | - |
| 10 | 1.19 | 35.52% | - | 37.92% |
| 50 | ~0.50 | ~75% | - | ~78% |
| 100 | ~0.12 | ~89% | - | ~91% |
| **114** | **0.0857** | **91.69%** | **0.0718** | **93.92%** ✓ |
| 124 | 0.0863 | 91.71% | 0.0721 | 93.90% |

**Note:** Validation accuracy exceeding training accuracy is expected due to dropout/DropPath being active only during training (regularization effect).

### Saved Model

Training automatically saves the best-performing model:

```python
torch.save(model.state_dict(), "audio_transformer_early_stop.pth")
```
File size: ~200 MB (25.4M parameters × 4 bytes/param + metadata)

## Inference - Analyzing Full Songs

Use the analyze_full_song() function to classify entire audio files:

# Example usage
audio_path = "path/to/your/song.mp3"  # or .wav, .flac, etc.
model_path = "audio_transformer_early_stop.pth"

# Analyze song with temporal predictions
predictions, segment_predictions = analyze_full_song(
    audio_path=audio_path,
    model_path=model_path,
    segment_duration=4,    # seconds
    overlap=2              # seconds (50% overlap)
)

# Returns:
# - predictions: Dict of average probabilities per genre
# - segment_predictions: List of per-segment predictions
# - Auto-generates visualizations: bar chart + probability evolution


How It Works

- Load audio: Resample to 22050 Hz using librosa
- Segment: Split into 4-second windows with 2-second stride
- Extract features: Compute Mel spectrogram for each segment
- Classify: Run inference on each segment independently
- Aggregate: Average probabilities across all segments
- Visualize: Display results with temporal evolution plot

### Example Output

Analyzing: song.mp3 (Duration: 3:24)
Number of segments: 51

Processing segments... [████████████████████] 100%

Genre Probabilities (averaged):
  metal:     78.3%
  rock:      12.1%
  blues:      4.2%
  classical:  2.1%
  hiphop:     1.8%
  ...

Predicted Genre: metal (confidence: 78.3%)

Plus interactive matplotlib visualizations showing:

- Bar chart of final probabilities
- Line plot of genre probabilities evolving through the song


## Comparison with State-of-the-Art

Performance comparison on GTZAN test set:

| Model | Accuracy | Architecture | Reference |
|-------|----------|--------------|-----------|
| EAViT | 93.99% | External Attention ViT | Iqbal et al., 2024 |
| **This Work** | **92.03%** | **ViT-8 + CNN Hybrid** | **-** |
| CNN-TE | 87.41% | CNN + Transformer Encoder | Chen et al., 2024 |
| Improved ViT | 86.8% | Pure ViT Variant | - |

### Detailed Per-Genre Comparison

Comparison with EAViT (best-performing model) and CNN-TE:

| Genre | This Work | EAViT | CNN-TE | Improvement vs CNN-TE |
|-------|-----------|-------|--------|----------------------|
| blues | 96.03% | 94% | 90% | +6.03% |
| classical | 95.10% | 97% | **100%** | -4.90% |
| country | 87.92% | 91% | 90% | -2.08% |
| disco | 88.29% | 97% | 95% | -6.71% |
| hiphop | 93.28% | 90% | 86% | +7.28% |
| jazz | 93.00% | 93% | 92% | +1.00% |
| metal | **94.14%** | 93% | 94% | +0.14% |
| pop | **92.54%** | 92% | 88% | +4.54% |
| reggae | **94.31%** | 92% | 63% | **+31.31%** |
| rock | 85.32% | 94% | 69% | +16.32% |

**Key Observations:**

1. **More stable than CNN-TE:** No extreme per-class variance (CNN-TE: 63%-100% range vs. This work: 85%-96%)
2. **Competitive with EAViT:** Within 2% on most genres despite simpler architecture
3. **Significant reggae improvement:** +31% over CNN-TE, demonstrating better generalization
4. **Outperforms EAViT:** On metal, pop, reggae classification

---

## Project Structure

Music_Genre_Classification_Vision_Transformer/\
├── Music_Genre_Classification_Vision_Transformer_PL.ipynb # Main implementation\
├── pyproject.toml # Poetry dependencies\
├── poetry.lock # Locked versions\
├── README.md # This file\
├── MODEL.md # Architecture details\
└── GTZAN/ # Dataset (download separately)\
├── blues/\
├── classical/\
├── country/\
├── disco/\
├── hiphop/\
├── jazz/\
├── metal/\
├── pop/\
├── reggae/\
└── rock/



**Design Philosophy:** Entire implementation in single Jupyter notebook for:
- Educational transparency
- Easy experimentation
- Reproducibility
- Self-contained analysis

---

## ⚠️ Known Limitations

### 1. Rock Genre Confusion (F1: 85.32%)

**Problem:** Lowest per-class performance, frequently confused with:
- Country (similar guitar-based instrumentation)
- Pop (modern rock overlaps with pop-rock)

**Root cause:** Genre boundary ambiguity in modern music (Pachet et al., 2001)

### 2. GTZAN Dataset Issues

As documented by Sturm (2013):
- **Replications:** Some songs appear across multiple genres
- **Mislabeling:** Subjective genre assignments
- **Distortions:** Audio artifacts in some files
- **Limited diversity:** Only 100 songs per genre

**Impact:** Despite these issues, GTZAN remains the standard benchmark for comparison.

### 3. Small Training Data

- Only 1000 original songs (800 for training)
- Segment augmentation partially mitigates but doesn't replace true diversity
- Modern deep learning models typically require 10,000+ samples for optimal performance

**Mitigation strategies employed:**
- Aggressive regularization (Dropout 0.3, DropPath 0.2)
- SpecAugment data augmentation
- Hybrid architecture (reduces parameter count vs. pure ViT)

### 4. Hardware Requirements

**Training:**
- Requires CUDA GPU (12GB+ VRAM for batch_size=64)
- CPU training possible but ~20× slower

**Inference:**
- GPU: ~0.5-1 second per segment
- CPU: ~2-5 seconds per segment
- Full song (3 min): ~10-30 seconds on GPU, ~2-5 minutes on CPU

### 5. Generalization Beyond GTZAN

Model trained specifically on GTZAN characteristics:
- 22050 Hz sample rate
- 30-second excerpts
- Specific recording conditions
- Western popular music focus

**Caution:** Performance may degrade on:
- Live recordings
- Non-Western music
- Electronic/experimental genres not in GTZAN
- Low-quality/compressed audio

---

## Technical Deep Dive

### Why Vision Transformer for Audio?

**1. Global Context Modeling**

Traditional CNNs have limited receptive fields, requiring deep stacking to capture long-range patterns. As shown by Vaswani et al. (2017), self-attention mechanisms allow each position to directly attend to all other positions, enabling:
- Song structure recognition (verse-chorus patterns)
- Long-term harmonic progressions
- Rhythmic patterns spanning multiple measures

**2. Spectral-Temporal Pattern Recognition**

Mel spectrograms exhibit 2D structures analogous to natural images:
- **Horizontal axis:** Temporal evolution (rhythm, note sequences)
- **Vertical axis:** Spectral content (pitch, timbre)
- **Local patterns:** Note onsets, harmonic overtones
- **Global patterns:** Genre-specific production styles

**3. Hybrid Architecture Advantages**

Following Wu et al. (2021)'s Convolutional Vision Transformer (CvT):
- **CNN stage:** Extracts local features (reduces dimensionality from 216×144 to 13×9)
- **Transformer stage:** Models global dependencies on reduced feature map
- **Benefit:** Combines CNN's inductive bias with Transformer's flexibility

**Result:** 60% fewer parameters than pure ViT while maintaining comparable performance.

### Key Training Techniques

#### Focal Loss (Lin et al., 2017)

Standard cross-entropy treats all examples equally. Focal Loss adds modulating factor:

`FL(p_t) = -α_t (1 - p_t)^γ log(p_t)`


Where:
- `p_t`: predicted probability for true class
- `γ = 2.5`: focusing parameter (higher = more focus on hard examples)
- `α_t`: class balancing weight

**Impact on this work:**
- Reduces weight of easy examples (e.g., obvious classical music)
- Increases focus on hard cases (rock/country confusion)
- Improved rock F1-score by ~3-5% over standard cross-entropy

#### SpecAugment

Random masking during training:
- **Time masking:** Zero out random time segments (simulates missing audio)
- **Frequency masking:** Zero out random frequency bands (simulates EQ changes)

**Effect:** Forces model to learn robust features not dependent on specific time-frequency regions.

#### Stochastic Depth (drop_path_rate=0.2)

Randomly drops entire transformer blocks during training:
- Reduces overfitting in deep models
- Enables training deeper networks than otherwise possible
- Acts as ensemble of models with different depths

**Implementation:** 20% probability of skipping any transformer block during forward pass.

#### Cosine Annealing LR Schedule

Learning rate decreases following cosine curve:
`η_t = η_min + (η_max - η_min) × (1 + cos(πt/T)) / 2`


**Benefits:**
- Smooth convergence (no sharp drops like step decay)
- Final epochs use very small LR (1e-7) for fine-tuning
- Reduces oscillations near convergence

### Reproducibility Notes

For reproducible results, set random seeds:

```python
import random
import numpy as np
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# For full reproducibility (may reduce performance):
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Note: Exact reproduction may still vary due to:

- CUDA non-determinism in certain operations
- Hardware-specific floating-point precision
- Operating system thread scheduling

Expect ±0.5% accuracy variance across runs.

### License

This project is licensed under the MIT License.