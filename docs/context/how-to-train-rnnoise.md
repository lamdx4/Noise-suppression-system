# RNNoise Training Guide

Complete workflow for training RNNoise models using `references/rnnoise` repository.

---

## RNNoise Repository Structure

```
references/rnnoise/
├── src/                    # C source code
│   ├── denoise.c          # Core denoising algorithm
│   ├── rnn.c              # GRU implementation
│   └── dump_features.c    # Feature extraction tool
├── torch/                  # PyTorch training (NEW)
│   ├── rnnoise/
│   │   ├── rnnoise.py           # Model architecture
│   │   ├── train_rnnoise.py     # Training script
│   │   └── dump_rnnoise_weights.py  # Export to C
│   ├── sparsification/    # Sparse training tools
│   └── weight-exchange/   # Cross-platform export
├── training/              # Keras training (DEPRECATED)
├── examples/              # Demo applications
└── README                 # Official documentation
```

---

## Prerequisites

### System Requirements

**Minimum:**

- Python 3.8-3.11
- 16GB RAM
- 50GB free storage
- Build tools (gcc, make, autotools)

**Recommended:**

- NVIDIA GPU (GTX 1060+ with 6GB VRAM)
- 32GB RAM
- 100GB free storage
- CUDA 11.0+

### Install Dependencies

```bash
# Navigate to rnnoise repo
cd references/rnnoise

# Install Python packages
pip install torch>=2.0.0 numpy>=1.20.0 tqdm

# Build tools (Linux/WSL)
sudo apt-get install build-essential autoconf automake libtool

# Build tools (Windows)
# Use WSL2 or MinGW
```

---

## Dataset Format Requirements

### Strict Format Specifications

All audio files MUST be:

- **Sample rate**: 48,000 Hz (RNNoise requirement)
- **Bit depth**: 16-bit signed integer
- **Channels**: Mono (1 channel)
- **Format**: Raw PCM (no WAV header)
- **Endianness**: Little-endian

### Three Input Files Needed

**File 1: Clean Speech**

```
Content: Pure speech without any noise
Size: Minimum 1-2 hours (600-1200 MB PCM)
Important: NO noise, NO music, NO echo
```

**File 2: Background Noise**

```
Content: Continuous stationary noise
Examples: AC hum, rain, wind, traffic
Size: Minimum 30 minutes (~200 MB PCM)
Important: NO speech in this file!
```

**File 3: Foreground Noise**

```
Content: Transient burst noise
Examples: Clapping, keyboard, door slam
Size: Minimum 10 minutes (~60 MB PCM)
Can reuse: Same file as background noise works
```

---

## Step-by-Step Training Workflow

### Step 1: Build Feature Extraction Tool

```bash
cd references/rnnoise

# Generate build scripts
./autogen.sh

# Configure build
./configure

# Compile
make

# Verify installation
./dump_features --help
```

**Expected output:**

```
Usage: dump_features <speech> <background_noise> <foreground_noise> <output.f32> <count>
```

### Step 2: Prepare Audio Data

**Convert WAV to PCM format:**

```bash
# Install SoX if not available
sudo apt-get install sox

# Convert single file
sox input.wav -r 48000 -c 1 -b 16 -e signed-integer output.pcm

# Batch convert directory
for file in *.wav; do
    sox "$file" -r 48000 -c 1 -b 16 -e signed-integer "${file%.wav}.pcm"
done

# Concatenate multiple PCM files
cat file1.pcm file2.pcm file3.pcm > combined.pcm
```

**Verify PCM format:**

```bash
# File size must be divisible by 2 (16-bit samples)
ls -lh audio.pcm

# Play back to verify (convert to WAV first)
sox -r 48000 -c 1 -b 16 -e signed-integer audio.pcm test.wav
```

### Step 3: Generate Training Features

```bash
# Syntax
./dump_features <clean_speech.pcm> <bg_noise.pcm> <fg_noise.pcm> <output.f32> <count>

# Start with small test (5000 sequences)
./dump_features \
    clean_speech.pcm \
    background_noise.pcm \
    foreground_noise.pcm \
    test_features.f32 \
    5000

# Full training set (30000 sequences recommended)
./dump_features \
    clean_speech.pcm \
    background_noise.pcm \
    foreground_noise.pcm \
    training_features.f32 \
    30000
```

**What happens during feature generation:**

1. Randomly picks 1-second chunks from each input file
2. Applies random gains (SNR: -30 to +10 dB)
3. Mixes: `noisy = clean + background + foreground (87.5% chance)`
4. Applies random filtering, clipping, quantization
5. Extracts 42 hand-crafted features from noisy audio
6. Computes ideal Wiener gains from clean reference
7. Saves training pairs: `[42 features, 22 gains, 1 VAD]`

**Timeline:**

- 5K sequences: 15-30 minutes
- 30K sequences: 1-3 hours

**Output file size:**

- 5K: ~1-2 GB
- 30K: ~8-12 GB

### Step 4: Train PyTorch Model

```bash
cd torch

# Basic training
python3 train_rnnoise.py \
    ../training_features.f32 \
    ./output \
    --epochs 100 \
    --batch-size 128

# Production training with sparsification (RECOMMENDED)
python3 train_rnnoise.py \
    ../training_features.f32 \
    ./output_sparse \
    --sparse \
    --epochs 150 \
    --batch-size 128 \
    --lr 1e-3 \
    --gru-size 384 \
    --gamma 0.25
```

**Key Training Arguments:**

| Argument            | Default | Description                    |
| ------------------- | ------- | ------------------------------ |
| `--sparse`          | False   | Enable 50% sparsification      |
| `--epochs`          | 200     | Training epochs                |
| `--batch-size`      | 128     | Batch size (reduce if OOM)     |
| `--lr`              | 1e-3    | Learning rate                  |
| `--gru-size`        | 384     | GRU hidden units (128/256/384) |
| `--gamma`           | 0.25    | Perceptual loss exponent       |
| `--sequence-length` | 2000    | Frames per sequence            |

**Expected training output:**

```
model: 1534976 weights
training epoch 1...
100%|████████| 234/234 [01:23<00:00]
loss: 0.12345  gain_loss: 0.11234  vad_loss: 0.01234

training epoch 50...
loss: 0.04567  gain_loss: 0.04123  vad_loss: 0.00543

training epoch 150...
loss: 0.01234  gain_loss: 0.01098  vad_loss: 0.00321
```

**Training timeline:**

- GPU (GTX 1060+): 4-8 hours for 150 epochs
- CPU: 24-48 hours (not recommended)

**Checkpoints saved at:**

```
output_sparse/checkpoints/
├── rnnoise_1.pth
├── rnnoise_50.pth
├── rnnoise_100.pth
└── rnnoise_150.pth
```

### Step 5: Export Model to C

```bash
# Convert PyTorch checkpoint to C code
python3 dump_rnnoise_weights.py \
    --quantize \
    ./output_sparse/checkpoints/rnnoise_150.pth \
    ../src/

# Output files created:
# - ../src/rnnoise_data.c  (model weights in C array)
# - ../src/rnnoise_data.h  (header file)
```

**Model quantization:**

- Input: Float32 PyTorch model (6 MB)
- Output: Int8 C arrays (1.5 MB dense, 850 KB sparse)

### Step 6: Build and Test

```bash
cd ..

# Rebuild RNNoise with new weights
make clean
make

# Test denoise functionality
./examples/rnnoise_demo \
    noisy_input.pcm \
    clean_output.pcm

# Convert output to WAV for listening
sox -r 48000 -c 1 -b 16 -e signed-integer \
    clean_output.pcm \
    clean_output.wav
```

---

## Advanced Features

### Sparsification Training

**Enable with `--sparse` flag:**

```python
# From rnnoise.py
sparse_params = {
    'W_hr': (0.3, [8, 4], True),  # 30% sparsity, 8×4 blocks
    'W_hz': (0.2, [8, 4], True),  # 20% sparsity
    'W_hn': (0.5, [8, 4], True),  # 50% sparsity
    # ...
}
```

**Progressive sparsification:**

- Steps 0-6000: Dense training
- Steps 6000-20000: Gradual pruning (0% → 50%)
- Steps 20000+: Fine-tuning with fixed sparsity

**Benefits:**

- Model size: 1.5 MB → 850 KB (43% reduction)
- Inference speed: +30-50% faster
- Quality loss: <3% (PESQ 2.45 → 2.42)

### Multi-Scale Processing

**Architecture feature:**

```python
# From rnnoise.py line 106
out_cat = torch.cat([tmp, gru1_out, gru2_out, gru3_out], dim=-1)
```

Concatenates outputs from ALL GRU layers:

- Conv output (96 dims)
- GRU1 output (384 dims)
- GRU2 output (384 dims)
- GRU3 output (384 dims)
- **Total**: 1248 dims → Dense layer

Captures multi-scale temporal patterns.

### Resume Training

```bash
# Resume from checkpoint
python3 train_rnnoise.py \
    ../features.f32 \
    ./output \
    --initial-checkpoint ./output/checkpoints/rnnoise_50.pth \
    --epochs 200
```

---

## Deployment

### Model Sizes

| Configuration     | Float32 | Int8   | Int8 Sparse |
| ----------------- | ------- | ------ | ----------- |
| PyTorch (dense)   | 6 MB    | 1.5 MB | -           |
| PyTorch (sparse)  | 6 MB    | 1.5 MB | 850 KB      |
| C export (dense)  | -       | 1.5 MB | -           |
| C export (sparse) | -       | -      | 850 KB      |

### ESP32 Integration

**Hardware requirements:**

| Device           | Flash | RAM    | Verdict       |
| ---------------- | ----- | ------ | ------------- |
| ESP32            | 4 MB  | 520 KB | ❌ Need PSRAM |
| ESP32 + PSRAM    | 4 MB  | 4 MB   | ✅ Works      |
| ESP32-S3 + PSRAM | 4 MB  | 8 MB   | ✅ Optimal    |

**Integration steps:**

```bash
# 1. Copy C files to your project
cp src/rnnoise_data.c /path/to/arduino/project/
cp src/rnnoise_data.h /path/to/arduino/project/

# 2. Include headers in your code
#include "rnnoise.h"
#include "rnnoise_data.h"

# 3. See examples/rnnoise_demo.c for usage
```

---

## Troubleshooting

### Build Issues

**Error: `./autogen.sh: command not found`**

```bash
# Install autotools
sudo apt-get install autoconf automake libtool
```

**Error: `make: No rule to make target`**

```bash
# Clean and rebuild
make distclean
./autogen.sh
./configure
make
```

### Training Issues

**CUDA out of memory:**

```bash
--batch-size 64  # Reduce from 128
--batch-size 32  # If still failing
```

**Loss not decreasing:**

```bash
# Verify data format
sox -r 48000 -c 1 -b 16 test.pcm test.wav
# Listen to verify quality

# Reduce learning rate
--lr 1e-4

# Check feature file integrity
ls -lh features.f32
# Should be ~400MB per 5K sequences
```

**Training too slow:**

```bash
# Check GPU usage
nvidia-smi

# If GPU not used, install CUDA:
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Data Format Issues

**File size not divisible by 2:**

```bash
# Indicates wrong format - reconvert
sox input.wav -r 48000 -c 1 -b 16 -e signed-integer output.pcm
```

**Clippy/distorted output:**

```bash
# Lower SNR in mixing
# Edit dump_features.c lines 386-388 to reduce gain ranges
```

---

## Performance Benchmarks

**Model metrics:**

- Parameters: 1.5M (dense), 750K active (sparse)
- PESQ: 2.3-2.5
- Latency: <10ms per 10ms frame
- Real-time factor: <1.0 (real-time capable)

**ESP32-S3 inference:**

- Latency: 10-15ms per frame
- RAM usage: ~800 KB
- Flash usage: 850 KB (sparse model)

---

## Quick Reference

**Complete workflow:**

```bash
# 1. Build tools
cd references/rnnoise && ./autogen.sh && ./configure && make

# 2. Prepare data (convert to 48kHz PCM)
sox speech.wav -r 48000 -c 1 -b 16 speech.pcm
sox noise.wav -r 48000 -c 1 -b 16 noise.pcm

# 3. Generate features
./dump_features speech.pcm noise.pcm noise.pcm features.f32 30000

# 4. Train
cd torch
python3 train_rnnoise.py ../features.f32 ./output --sparse --epochs 150

# 5. Export
python3 dump_rnnoise_weights.py --quantize ./output/checkpoints/rnnoise_150.pth ../src/

# 6. Test
cd .. && make clean && make
./examples/rnnoise_demo test_noisy.pcm test_clean.pcm
```

**Total time**: 2-3 days (mostly waiting for downloads/training)

---

## Resources

- Official repo: https://github.com/xiph/rnnoise
- Paper: [A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement](https://jmvalin.ca/papers/rnnoise_mmsp2018.pdf)
- RNNoise README: `references/rnnoise/README`
