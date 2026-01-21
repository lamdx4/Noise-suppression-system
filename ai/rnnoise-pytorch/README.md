# RNNoise PyTorch - Custom Training Project

Clean, customizable PyTorch implementation of RNNoise for speech enhancement.

## Features

- ✅ PyTorch-only (no TensorFlow/Keras)
- ✅ Modular architecture (easy to customize)
- ✅ Sparsification support (50% model size reduction)
- ✅ JSON logging (for documentation/reports)
- ✅ Vietnamese-friendly comments

## Project Structure

```
ai/rnnoise-pytorch/
├── rnnoise/           # Core model package
│   ├── model.py      # RNNoise architecture
│   └── __init__.py
│
├── sparsification/    # Sparse training
│   ├── gru_sparsifier.py
│   ├── common.py
│   └── __init__.py
│
├── scripts/          # Training scripts
│   ├── train.py     # Training script (to be added)
│   └── training_logger.py
│
├── configs/          # Configuration files
│   └── default.yaml # Default config (to be added)
│
├── examples/         # Usage examples
│   └── basic_training.py (to be added)
│
├── README.md        # This file
└── requirements.txt # Dependencies
```

## Quick Start

### Installation

```bash
cd ai/rnnoise-pytorch
pip install -r requirements.txt
```

### Usage

```python
from rnnoise.model import RNNoise
import torch

# Create model
model = RNNoise(
    input_dim=42,      # Number of features
    output_dim=22,     # Number of gains
    cond_size=128,     # Conv layer size
    gru_size=384       # GRU hidden size
)

# Forward pass
features = torch.randn(1, 100, 42)  # [batch, sequence, features]
gains, vad, states = model(features)

print(f"Gains shape: {gains.shape}")  # [1, 100, 22]
print(f"VAD shape: {vad.shape}")      # [1, 100, 1]
```

### Training

See `scripts/train.py` (to be added)

## Customization Points

### 1. Model Size

```python
# Smaller model (faster, less quality)
model = RNNoise(gru_size=256)

# Larger model (slower, better quality)
model = RNNoise(gru_size=512)
```

### 2. Sparsification

```python
# Enable sparse training (reduces model size 50%)
model = RNNoise(gru_size=384)
model.sparsify()  # Call after optimizer.step()
```

### 3. Custom Loss

See `rnnoise/loss.py` (to be added) for perceptual loss functions.

## Documentation

See `../docs/` for detailed guides:

- Architecture explanation
- Training workflow
- Inference pipeline
- Preprocessing details

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- tqdm

Full list in `requirements.txt`

## Notes

This is a clean extraction from Mozilla's RNNoise reference implementation,
organized for easy customization and Vietnamese development.

## License

Based on RNNoise by Mozilla/Xiph (BSD-3-Clause)
