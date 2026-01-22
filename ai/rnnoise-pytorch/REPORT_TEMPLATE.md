# RNNoise Training Report Template

**Experiment:** [Experiment Name]  
**Date:** [YYYY-MM-DD]  
**Author:** [Your Name]

---

## 1. Dataset Specification

### Audio Sources

**Clean Speech:**

- Language: Vietnamese [X%], English [Y%]
- Duration: X hours
- Sample rate: 48kHz
- Format: Clean studio recordings
- Sources:
  - [Dataset 1]: X hours
  - [Dataset 2]: Y hours

**Background Noise:**

- Duration: X minutes
- Types:
  - Street noise: X min
  - Office noise: Y min
  - Café noise: Z min
- Sources: [List sources]

**Foreground Noise:**

- Duration: X minutes
- Types:
  - Keyboard: X min
  - Phone rings: Y min
  - Other: Z min

### Generated Training Data

**dump_features parameters:**

```bash
./dump_features speech.pcm bg_noise.pcm fg_noise.pcm features.f32 30000
```

**Feature file:**

- Filename: `features_30k.f32`
- Number of sequences: 30,000
- Sequence length: 2,000 frames (20 seconds)
- Total duration: ~166 hours mixed data
- File size: 11.5 GB
- SNR range: -40 to +45 dB (randomized)

**Augmentation:**

- Random filtering: ✅ Enabled
- Clipping (25%): ✅ Enabled
- Quantization (50%): ✅ Enabled
- Reverberation: ❌ Disabled

---

## 2. Model Configuration

### Architecture

**Model:** RNNoise (GRU-based)

**Parameters:**

```yaml
model:
  gru_size: 384
  cond_size: 128
  input_dim: 42
  output_dim: 22
```

**Layer breakdown:**

- Conv1D #1: 42 → 128
- Conv1D #2: 128 → 96
- GRU #1: 96 → 384
- GRU #2: 384 → 384
- GRU #3: 384 → 384
- Dense (gains): 1248 → 22
- Dense (VAD): 1248 → 1

**Total parameters:**

- Dense: 1,500,000 params
- Sparse (50%): 750,000 active params

---

## 3. Training Configuration

### Hyperparameters

```yaml
training:
  epochs: 150
  batch_size: 128
  learning_rate: 1e-3
  lr_decay: 5e-5
  sequence_length: 2000
  gamma: 0.25 # Perceptual loss exponent

optimizer:
  type: AdamW
  betas: [0.8, 0.98]
  eps: 1e-8

scheduler:
  type: LambdaLR
  formula: 1 / (1 + 5e-5 * step)

sparsification:
  enabled: true
  start_step: 6000
  stop_step: 20000
  interval: 100
  exponent: 3
  targets:
    W_hn: 0.5
    W_hz: 0.2
    W_hr: 0.3
```

### Hardware

- **GPU:** NVIDIA RTX 3060 (12GB)
- **CUDA:** 11.8
- **PyTorch:** 2.0.1
- **System:** Ubuntu 22.04 / Windows 11 WSL2

---

## 4. Training Process

### Execution Command

```bash
python scripts/train.py \
    ../references/rnnoise/features_30k.f32 \
    ./output_sparse_384 \
    --sparse \
    --epochs 150 \
    --batch-size 128 \
    --gru-size 384 \
    --lr 1e-3 \
    --gamma 0.25 \
    --log-dir ../logs \
    --experiment-name rnnoise_vn_384_sparse
```

### Training Timeline

| Phase       | Epochs  | Time        | Loss Range        |
| ----------- | ------- | ----------- | ----------------- |
| Initial     | 1-20    | 45 min      | 0.15 → 0.05       |
| Convergence | 21-100  | 3.5 hrs     | 0.05 → 0.015      |
| Fine-tuning | 101-150 | 2 hrs       | 0.015 → 0.0098    |
| **Total**   | **150** | **6.5 hrs** | **0.15 → 0.0098** |

### Loss Curves

![Training Loss](path/to/loss_curve.png)

**Key observations:**

- Rapid decrease in first 20 epochs
- Sparsification from epoch 25-80 (steps 6K-20K)
- Stable convergence after epoch 100
- Best checkpoint: Epoch 145 (loss: 0.0097)

### Resource Usage

- **GPU Memory:** 8.2 GB peak
- **GPU Utilization:** 85-95%
- **Training speed:** ~2.5 seconds/epoch
- **Checkpoints saved:** 150 files (~1.5MB each)

---

## 5. Results

### Model Performance

**Best checkpoint:** `rnnoise_sparse_384_145.pth`

| Metric | Value   | Baseline | Improvement |
| ------ | ------- | -------- | ----------- |
| PESQ   | 2.43    | 1.95     | +24.6%      |
| STOI   | 0.89    | 0.72     | +23.6%      |
| SI-SDR | 12.3 dB | 5.1 dB   | +7.2 dB     |

### Model Size

| Configuration     | Parameters | Size       | Sparsity |
| ----------------- | ---------- | ---------- | -------- |
| Dense (float32)   | 1.5M       | 6.0 MB     | 0%       |
| Dense (int8)      | 1.5M       | 1.5 MB     | 0%       |
| **Sparse (int8)** | **750K**   | **850 KB** | **50%**  |

### Inference Performance

**Hardware:** ESP32-S3 @ 240MHz

| Configuration | Latency    | RAM        | Flash      |
| ------------- | ---------- | ---------- | ---------- |
| Dense         | 8.5 ms     | 1.2 MB     | 1.5 MB     |
| **Sparse**    | **5.2 ms** | **800 KB** | **850 KB** |

**Real-time capable:** ✅ Yes (10ms frame, <6ms processing)

---

## 6. Quality Analysis

### Subjective Evaluation

**Test setup:**

- 20 Vietnamese speech samples
- SNR: 0, 5, 10 dB
- Listeners: 5 native Vietnamese speakers

**MOS (Mean Opinion Score):**

| SNR   | Noisy | Denoised | Improvement |
| ----- | ----- | -------- | ----------- |
| 0 dB  | 2.1   | 3.8      | +1.7        |
| 5 dB  | 2.8   | 4.2      | +1.4        |
| 10 dB | 3.5   | 4.5      | +1.0        |

### Audio Samples

**Sample 1: Street noise (SNR = 0 dB)**

- Noisy: [link to audio]
- Denoised: [link to audio]
- Clean: [link to audio]

**Observations:**

- Effective noise reduction
- Minimal speech distortion
- Natural sound quality

---

## 7. Comparison

### vs. Baseline (Dense Model)

| Metric | Dense  | Sparse | Change      |
| ------ | ------ | ------ | ----------- |
| PESQ   | 2.45   | 2.43   | -0.8%       |
| Size   | 1.5 MB | 850 KB | -43%        |
| Speed  | 8.5 ms | 5.2 ms | +38% faster |

**Trade-off:** +38% speed for -0.8% quality → Excellent!

### vs. U-Net (Previous Model)

| Metric     | U-Net   | RNNoise | Change      |
| ---------- | ------- | ------- | ----------- |
| PESQ       | 2.38    | 2.43    | +2.1%       |
| Latency    | 2500 ms | 5.2 ms  | 99% faster  |
| Model size | 45 MB   | 850 KB  | 98% smaller |

**Verdict:** RNNoise far superior for real-time

---

## 8. Deployment

### Export Process

```bash
python scripts/export_to_c.py \
    --quantize \
    ./output_sparse_384/checkpoints/rnnoise_145.pth \
    ./exported
```

**Output:**

- `rnnoise_data.c` (842 KB)
- `rnnoise_data.h` (2 KB)

### Integration

**Platform:** ESP32-S3 (8MB PSRAM)

**Test results:**

- Compilation: ✅ Success
- Runtime: ✅ Stable (tested 24 hours)
- Real-time: ✅ Yes (5.2ms < 10ms frame)

---

## 9. Conclusion

### Achievements

✅ Successfully trained Vietnamese-focused RNNoise model  
✅ 50% sparsity with <1% quality loss  
✅ Real-time capable on ESP32-S3  
✅ 850KB model size (fits embedded constraints)  
✅ PESQ 2.43 (high quality)

### Limitations

- Requires PSRAM (800KB RAM)
- Vietnamese-specific (may not generalize)
- Stateful processing (not parallelizable)

### Future Work

- [ ] Multi-language support
- [ ] Larger model for cloud deployment
- [ ] Online learning/adaptation
- [ ] More aggressive sparsity (70%)

---

## 10. Appendix

### Training Logs

**Location:** `ai/logs/rnnoise_vn_384_sparse_*`

**Files:**

- `*_config.json` - Training configuration
- `*_metrics.json` - Per-epoch metrics
- `*_summary.json` - Final statistics

### Checkpoint Files

**Location:** `ai/rnn oise-pytorch/output_sparse_384/checkpoints/`

**Important checkpoints:**

- `rnnoise_145.pth` - Best (loss: 0.0097)
- `rnnoise_150.pth` - Final
- `rnnoise_100.pth` - Fallback

### References

1. Valin, J. M. (2018). A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement
2. RNNoise repository: https://github.com/xiph/rnnoise
3. DNS Challenge: https://github.com/microsoft/DNS-Challenge

---

**Report generated:** [YYYY-MM-DD HH:MM]  
**Training duration:** 6.5 hours  
**Final verdict:** ✅ Production-ready for ESP32 deployment
