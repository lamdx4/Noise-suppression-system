# AI Workspace

This folder contains all AI/training related components for the Speech Enhancement project.

## Structure

```
ai/
├── docs/              # Training documentation and guides
├── references/        # Reference implementations
│   ├── rnnoise/      # RNNoise training code
│   ├── DNS-Challenge/ # Dataset preparation tools
│   └── nam/          # U-Net model code
├── notebooks/         # Jupyter notebooks for experiments
├── scripts/          # Training scripts
├── models/           # Saved model checkpoints
└── src/              # Custom training code
```

## Documentation

See `docs/` for comprehensive guides:

- **how-to-train-rnnoise.md** - RNNoise training workflow
- **rnnoise-inference-flow.md** - How RNNoise processes audio
- **rnnoise-preprocessing-and-model.md** - Feature extraction & architecture
- **rnnoise-pytorch-architecture.md** - Deep dive into PyTorch implementation

## Getting Started

### Prerequisites

```bash
# Python 3.8-3.11
python --version

# Install dependencies
pip install torch>=2.0.0 numpy>=1.20.0 tqdm
```

### Training RNNoise

1. **Prepare dataset** (48kHz PCM format)
2. **Build dump_features:**
   ```bash
   cd references/rnnoise
   ./autogen.sh && ./configure && make
   ```
3. **Generate features:**
   ```bash
   ./dump_features clean.pcm noise_bg.pcm noise_fg.pcm features.f32 30000
   ```
4. **Train model:**
   ```bash
   cd torch
   python3 train_rnnoise.py ../features.f32 ./output --sparse --epochs 150
   ```
5. **Export to C:**
   ```bash
   python3 dump_rnnoise_weights.py --quantize ./output/checkpoints/rnnoise_150.pth ../src/
   ```

## Notes

- This workspace is separate from embedded/firmware code
- Trained models go in `models/` folder
- Large datasets should stay in project root `data/` folder
- Use `.gitignore` to exclude large files (models, checkpoints)
