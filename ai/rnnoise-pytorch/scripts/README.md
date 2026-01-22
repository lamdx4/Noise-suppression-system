# Evaluation & Reporting Scripts

Scripts để đánh giá quality và tạo báo cáo từ training results.

---

## 📋 Scripts Overview

| Script               | Purpose                  | Input                   | Output                        |
| -------------------- | ------------------------ | ----------------------- | ----------------------------- |
| **evaluate.py**      | Compute PESQ/STOI/SI-SDR | Checkpoint + test audio | Metrics JSON + denoised audio |
| **plot_training.py** | Generate training charts | Metrics JSON            | PNG charts                    |
| **dataset_info.py**  | Document dataset         | Audio directories       | Dataset JSON                  |

---

## 1. evaluate.py - Model Quality Metrics

### Purpose

Đánh giá chất lượng model trained với objective metrics.

### Usage

```bash
python scripts/evaluate.py \
    output/checkpoints/rnnoise_150.pth \
    ../test_data \
    ../evaluation_results
```

### Arguments

- `checkpoint`: Path to .pth model file
- `test_dir`: Directory with test files (must have `noisy/` and `clean/` subdirs)
- `output_dir`: Where to save results
- `--num-samples`: Number of test files (default: 20)
- `--sample-rate`: Sample rate in Hz (default: 48000)

### Test Directory Structure

```
test_data/
├── noisy/
│   ├── test001.wav
│   ├── test002.wav
│   └── ...
└── clean/
    ├── test001.wav  # Same filename as noisy
    ├── test002.wav
    └── ...
```

### Output

```
evaluation_results/
├── evaluation_results.json  # Metrics
└── denoised/
    ├── test001.wav         # Processed audio
    ├── test002.wav
    └── ...
```

### Output JSON Format

```json
{
  "checkpoint": "rnnoise_150.pth",
  "timestamp": "2026-01-21T18:00:00",
  "num_samples": 20,
  "average_metrics": {
    "pesq": {
      "mean": 2.43,
      "std": 0.15,
      "min": 2.15,
      "max": 2.68
    },
    "stoi": {
      "mean": 0.89,
      "std": 0.04
    },
    "si_sdr": {
      "mean": 12.3,
      "std": 2.1
    }
  },
  "per_sample": [...]
}
```

### Metrics Explained

- **PESQ** (1.0-4.5): Perceptual quality
  - \> 4.0: Excellent
  - 3.0-4.0: Good
  - 2.0-3.0: Fair
  - < 2.0: Poor

- **STOI** (0.0-1.0): Intelligibility
  - \> 0.90: Excellent
  - 0.80-0.90: Good
  - 0.70-0.80: Fair
  - < 0.70: Poor

- **SI-SDR** (dB): Signal quality
  - \> 15 dB: Excellent
  - 10-15 dB: Good
  - 5-10 dB: Fair
  - < 5 dB: Poor

---

## 2. plot_training.py - Training Visualization

### Purpose

Tạo charts từ JSON logs để phân tích training và báo cáo.

### Usage

```bash
python scripts/plot_training.py \
    ../logs/rnnoise_20260121_143000_metrics.json \
    ../plots
```

### With Summary File

```bash
python scripts/plot_training.py \
    ../logs/rnnoise_20260121_143000_metrics.json \
    ../plots \
    --summary-file ../logs/rnnoise_20260121_143000_summary.json
```

### Arguments

- `metrics_file`: Path to \*\_metrics.json (from training_logger)
- `output_dir`: Where to save charts
- `--summary-file`: Optional \*\_summary.json for additional info

### Output Charts

```
plots/
├── loss_curves.png         # 4 subplots: total, components, log, smoothed
├── learning_rate.png       # LR schedule
├── convergence.png         # Convergence analysis
└── training_summary.txt    # Text summary
```

### Charts Explained

**1. loss_curves.png:**

- Top-left: Total loss (linear scale)
- Top-right: Gain vs VAD loss (log scale)
- Bottom-left: Total loss (log scale)
- Bottom-right: Smoothed loss (moving average)

**2. learning_rate.png:**

- LR decay over epochs
- Log scale for visibility

**3. convergence.png:**

- Left: Loss with best epoch marker
- Right: Improvement rate per epoch

**4. training_summary.txt:**

```
==================================================
TRAINING SUMMARY
==================================================

Total Epochs        : 150
Best Epoch          : 145
Best Loss           : 0.009700
Final Loss          : 0.009800
Average Loss        : 0.023400
Loss Reduction      : 93.47%
Training Time       : 6.50 hours

==================================================
```

---

## 3. dataset_info.py - Dataset Documentation

### Purpose

Document dataset used cho reproducibility và báo cáo.

### Usage

```bash
python scripts/dataset_info.py \
    --clean-speech ../data/clean_vietnamese/ \
    --background-noise ../data/background_noise/ \
    --foreground-noise ../data/foreground_noise/ \
    --output ../dataset_info.json \
    --description "Vietnamese speech enhancement dataset" \
    --language "Vietnamese"
```

### Arguments

- `--clean-speech`: Directory with clean .wav files (required)
- `--background-noise`: Directory with background noise .wav (required)
- `--foreground-noise`: Directory with foreground noise .wav (optional)
- `--output`: Output JSON file (required)
- `--description`: Dataset description
- `--language`: Primary language

### Output JSON

```json
{
  "metadata": {
    "created": "2026-01-21T14:00:00",
    "description": "Vietnamese speech enhancement dataset",
    "language": "Vietnamese"
  },
  "sources": {
    "clean_speech": {
      "directory": "../data/clean_vietnamese/",
      "summary": {
        "num_files": 523,
        "total_duration_hours": 2.54,
        "avg_duration_sec": 17.5,
        "avg_rms_db": -23.4,
        "sample_rates": [48000]
      },
      "files": [...]
    },
    "background_noise": {...},
    "foreground_noise": {...}
  },
  "overall": {
    "total_clean_hours": 2.54,
    "total_noise_hours": 0.87,
    "clean_to_noise_ratio": 2.92
  }
}
```

### Console Output

```
=== Clean Speech ===
Analyzing 523 files...
Total: 2.54 hours

=== Background Noise ===
Analyzing 87 files...
Total: 0.87 hours

============================================================
DATASET SUMMARY
============================================================
Clean Speech:      2.54 hours (523 files)
Background Noise:  0.87 hours (87 files)

Language: Vietnamese
Description: Vietnamese speech enhancement dataset
============================================================

✅ Dataset info saved to: ../dataset_info.json
```

---

## 🔄 Complete Workflow

### Before Training

```bash
# 1. Document dataset
python scripts/dataset_info.py \
    --clean-speech ../data/clean/ \
    --background-noise ../data/noise/ \
    --output ../dataset_info.json \
    --language "Vietnamese"
```

### During Training

Training script automatically logs to JSON if `--log-dir` specified.

### After Training

```bash
# 2. Generate training charts
python scripts/plot_training.py \
    ../logs/*_metrics.json \
    ../plots

# 3. Evaluate model quality
python scripts/evaluate.py \
    output/checkpoints/rnnoise_145.pth \
    ../test_data \
    ../evaluation_results

# 4. Copy results to report
# - plots/*.png → Report figures
# - evaluation_results.json → Report metrics table
# - dataset_info.json → Report dataset section
```

---

## 📊 For Report Generation

### Required Files

1. ✅ `dataset_info.json` - Dataset section
2. ✅ `*_config.json` - Training config
3. ✅ `*_metrics.json` - Loss data
4. ✅ `*_summary.json` - Training summary
5. ✅ `evaluation_results.json` - Quality metrics
6. ✅ `plots/*.png` - Charts

### Fill Report Template

```markdown
## Dataset

[Copy from dataset_info.json]

## Training

[Copy from *_config.json + *_summary.json]
[Insert plots/loss_curves.png]

## Results

[Copy from evaluation_results.json]
PESQ: 2.43 ± 0.15
STOI: 0.89 ± 0.04
SI-SDR: 12.3 ± 2.1 dB

## Audio Samples

[Link files from evaluation_results/denoised/]
```

---

## Dependencies

Install all evaluation dependencies:

```bash
cd ai/rnnoise-pytorch
pip install -r requirements.txt
```

Includes:

- `pesq` - PESQ metric
- `pystoi` - STOI metric
- `scipy` - Signal processing
- `soundfile` - Audio I/O
- `librosa` - Audio analysis
- `matplotlib` - Plotting

---

## Troubleshooting

### evaluate.py fails

**Issue:** PESQ computation error

**Solution:**

```bash
pip install --upgrade pesq
```

**Issue:** "Expected noisy/ and clean/ subdirs"

**Solution:** Organize test files:

```bash
mkdir -p test_data/noisy test_data/clean
mv noisy*.wav test_data/noisy/
mv clean*.wav test_data/clean/
```

### plot_training.py fails

**Issue:** "cannot open display"

**Already handled** - Uses `Agg` backend (non-interactive)

### dataset_info.py slow

**Normal** - Analyzes every audio file  
**Speedup:** Reduce number of files or use sampling

---

## Example Complete Report Workflow

```bash
# Setup
cd ai/rnnoise-pytorch
pip install -r requirements.txt

# Before training
python scripts/dataset_info.py \
    --clean-speech ../../data/clean_vn/ \
    --background-noise ../../data/noise/ \
    --output ../../dataset_info.json \
    --language "Vietnamese" \
    --description "Vietnamese TTS + noise dataset for RNNoise"

# Train (with logging)
python scripts/train.py \
    ../references/rnnoise/features.f32 \
    ./output \
    --sparse \
    --epochs 150 \
    --log-dir ../logs \
    --experiment-name rnnoise_vn

# After training - Generate plots
python scripts/plot_training.py \
    ../logs/rnnoise_vn_*_metrics.json \
    ../plots \
    --summary-file ../logs/rnnoise_vn_*_summary.json

# Evaluate quality
python scripts/evaluate.py \
    output/checkpoints/rnnoise_145.pth \
    ../../test_data \
    ../evaluation

# Now fill REPORT_TEMPLATE.md with all the JSON data!
```

---

✅ **HOÀN CHỈNH CHO BÁO CÁO!** 📊
