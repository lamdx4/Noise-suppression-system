# RNNoise Project Documentation Index

**Context files for AI assistants and developers**

---

## 📚 Documentation Files

### **1. rnnoise-pytorch-complete.md** ⭐ MAIN GUIDE

**Complete end-to-end workflow**

- Project setup and structure
- C tools (dump_features) usage
- Training process
- Export to C for ESP32
- Troubleshooting

**When to use:** First-time setup, complete reference

---

### **2. dataset-guide.md** 📊 DATASET REFERENCE

**Everything about data preparation**

- Dataset sources (VIVOS, DNS-Challenge)
- Audio format requirements
- Conversion workflow (WAV → PCM)
- Why to mix all noise types
- Common mistakes

**When to use:** Preparing training data

---

### **3. PROJECT_AUDIT.md** ✅ STATUS CHECK

**Project readiness and completeness**

- Code integrity verification
- Missing components checklist
- Evaluation scripts overview
- Files for reporting

**When to use:** Verify project is ready to train

---

## 🎯 Quick Navigation

**Want to:**

- **Start from scratch?** → Read `rnnoise-pytorch-complete.md`
- **Prepare dataset?** → Read `dataset-guide.md`
- **Check if ready?** → Read `PROJECT_AUDIT.md`
- **Generate reports?** → See `rnnoise-pytorch/scripts/README.md`

---

## 📁 Project Structure

```
ai/
├── docs/context/              ← You are here
│   ├── README.md             ← This file
│   ├── rnnoise-pytorch-complete.md
│   ├── dataset-guide.md
│   └── PROJECT_AUDIT.md
│
├── rnnoise-pytorch/           ← Custom PyTorch project
│   ├── scripts/
│   │   ├── train.py          ← Training
│   │   ├── export_to_c.py    ← PyTorch → C
│   │   ├── evaluate.py       ← Quality metrics
│   │   └── plot_training.py  ← Visualization
│   ├── rnnoise/              ← Model package
│   └── REPORT_TEMPLATE.md    ← For documentation
│
└── references/rnnoise/        ← Original implementation
    ├── dump_features         ← Build from src/
    └── torch/                ← Reference code
```

---

## ⚡ Quick Commands

### Build C Tools

```bash
cd ai/references/rnnoise
./autogen.sh && ./configure && make
```

### Generate Features

```bash
./dump_features speech.pcm noise.pcm noise.pcm features.f32 30000
```

### Train Model

```bash
cd ../../rnnoise-pytorch
python scripts/train.py ../references/rnnoise/features.f32 ./output \
    --sparse --epochs 150 --gru-size 384
```

### Evaluate

```bash
python scripts/evaluate.py output/checkpoints/rnnoise_150.pth \
    ../test_data ../evaluation
```

### Export to C

```bash
python scripts/export_to_c.py --quantize \
    output/checkpoints/rnnoise_150.pth ./exported
```

---

## 🔄 Typical Workflow

1. **Setup** → Read `rnnoise-pytorch-complete.md`
2. **Prepare Data** → Follow `dataset-guide.md`
3. **Verify Ready** → Check `PROJECT_AUDIT.md`
4. **Train** → `python scripts/train.py`
5. **Evaluate** → `python scripts/evaluate.py`
6. **Export** → `python scripts/export_to_c.py`
7. **Deploy** → Integrate C files to ESP32

---

## 📝 For AI Context

**Key Information:**

- **Goal:** Vietnamese speech enhancement on ESP32
- **Approach:** RNNoise (GRU-based denoising)
- **Input:** 48kHz mono audio (10ms frames)
- **Output:** Denoised audio + VAD
- **Model:** 384-unit GRU with 50% sparsity
- **Dataset:** VIVOS (Vietnamese) + DNS-Challenge (noise)

**Critical Files:**

- Training: `scripts/train.py` (1:1 match with reference)
- Export: `scripts/export_to_c.py` (uses weight-exchange)
- Model: `rnnoise/model.py` (65→32 bands)

**All documentation consolidated here for easy AI consumption** ✅

---

Last Updated: 2026-01-22
