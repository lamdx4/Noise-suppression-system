# RNNoise Dataset Guide - Complete Reference

**Last Updated:** 2026-01-22

---

## 🎯 Quick Start

**Goal:** Prepare audio data for RNNoise training

**What You Need:**

1. Clean speech (Vietnamese) - 2+ hours
2. Background noise (DNS-Challenge) - 30GB+
3. Foreground noise (optional - can reuse background)

**End Result:** `features.f32` file for training

---

## 📋 Dataset Strategy

### ✅ **CORRECT: Mix All Noise Types**

```bash
# Combine ALL noise sources into one file
cat dns_noise/*.pcm rnnoise_contrib/*.pcm > background_noise.pcm

# Combine ALL clean speech
cat vivos/*.pcm fpt/*.pcm > clean_speech.pcm

# Foreground can reuse background
cp background_noise.pcm foreground_noise.pcm
```

**Why Mix Everything?**

- ✅ dump_features uses random sampling → automatic diversity
- ✅ Model learns general "noise" pattern → robust
- ✅ Better generalization to unseen conditions
- ✅ Proven by DNS Challenge winners

**Don't:**

- ❌ Train separate models per noise type → overfitting
- ❌ Organize by categories → unnecessary complexity

---

## 🌐 Dataset Sources

### Clean Speech (Vietnamese)

**VIVOS** ⭐ Recommended

- Link: https://ailab.hcmus.edu.vn/vivos
- Size: 15 hours, 65 speakers
- Quality: Studio recordings
- Format: WAV → convert to 48kHz PCM

**CommonVoice Vietnamese**

- Link: https://commonvoice.mozilla.org/vi
- Size: Variable
- Quality: Crowdsourced (mixed)

### Background Noise

**DNS-Challenge** ⭐ BEST FOR RNNOISE

- Link: https://github.com/microsoft/DNS-Challenge
- Size: 58GB (noise_fullband only)
- Sample Rate: **48kHz native!** (no resampling!)
- Types: AudioSet, FreeSound, DEMAND, emotional speech
- Download: See below

**MS-SNSD** (Not Recommended)

- 16kHz only → quality loss when upsampling
- Use only if storage limited

---

## 📥 DNS-Challenge Noise Download

### Quick Commands

```bash
# 1. Clone repo
cd ai/data
git clone https://github.com/microsoft/DNS-Challenge.git
cd DNS-Challenge

# 2. Download noise only (edit script to skip clean speech)
# Edit download-dns-challenge-5-headset-training.sh
# Uncomment noise_fullband download lines
bash download-dns-challenge-5-headset-training.sh

# 3. Extract
cd datasets_fullband
tar -xzf noise_fullband.tar.gz

# Expected: ~58GB noise files
```

### Size Options

| Option      | Size | Files           | Best For    |
| ----------- | ---- | --------------- | ----------- |
| DEMAND only | 5GB  | 16 environments | Quick test  |
| Recommended | 30GB | Mixed sources   | Production  |
| Full        | 58GB | All noise       | Max quality |

---

## 🔧 Audio Format Requirements

### CRITICAL - Must Be Exact

```
Sample Rate: 48,000 Hz (48kHz)
Bit Depth:   16-bit signed integer
Channels:    Mono (1 channel)
Format:      Raw PCM (NO WAV headers!)
Endianness:  Little-endian
```

**dump_features will FAIL if format is wrong!**

---

## ⚙️ Conversion Workflow

### Step 1: Convert to 48kHz WAV

```bash
# Single file
sox input.wav -r 48000 -c 1 output_48k.wav

# Batch convert directory
for f in *.wav; do
    sox "$f" -r 48000 -c 1 "48k_${f}"
done

# Verify
soxi output_48k.wav | grep "Sample Rate"
# Must show: Sample Rate : 48000
```

### Step 2: Convert WAV to PCM

```bash
# Single file
sox input_48k.wav -r 48000 -c 1 -b 16 -e signed-integer output.pcm

# Batch convert
for f in 48k_*.wav; do
    sox "$f" -r 48000 -c 1 -b 16 -e signed-integer "${f%.wav}.pcm"
done
```

### Step 3: Concatenate PCM Files

```bash
# Clean speech
cat clean_speech_pcm/*.pcm > clean_speech.pcm

# Background noise
cat noise_pcm/**/*.pcm > background_noise.pcm

# Foreground (or reuse)
head -c $((100 * 1024 * 1024)) background_noise.pcm > foreground_noise.pcm
```

### Step 4: Verify

```bash
# Test playback (requires SoX)
play -r 48000 -c 1 -b 16 -e signed-integer clean_speech.pcm

# Check file sizes
ls -lh *.pcm
# Expected: 2 hours = ~691 MB
```

---

## ✅ Final Checklist

**Before running dump_features:**

- [ ] All files are 48kHz (verify with soxi)
- [ ] All files are mono (1 channel)
- [ ] PCM format (no WAV headers)
- [ ] clean_speech.pcm exists (2+ GB)
- [ ] background_noise.pcm exists (20+ GB)
- [ ] foreground_noise.pcm exists (100+ MB)
- [ ] Playback test passes (sounds correct speed)

---

## 🚀 Generate Training Features

```bash
cd ai/references/rnnoise

# Build dump_features (if not done)
./autogen.sh && ./configure && make

# Generate features
./dump_features \
    ../../data/clean_speech.pcm \
    ../../data/background_noise.pcm \
    ../../data/foreground_noise.pcm \
    features_vn_30k.f32 \
    30000

# Expected output: ~12GB (30K sequences)
```

**What dump_features does:**

- Random sample from PCM files
- Random SNR mixing (-40 to +45 dB)
- Random filtering
- Extract 65 features per frame
- Output: 30,000 training sequences

---

## 📊 Recommended Sizes

| Component          | Minimum | Recommended | Ideal    |
| ------------------ | ------- | ----------- | -------- |
| Clean Speech       | 1 hour  | 2-3 hours   | 5+ hours |
| Background Noise   | 15 min  | 30-60 min   | Full DNS |
| Training Sequences | 10K     | 30K ⭐      | 100K     |
| features.f32 Size  | 4GB     | 12GB        | 40GB     |

---

## ⚠️ Common Mistakes

### ❌ Wrong Sample Rate

```bash
# WRONG - not forcing 48kHz
sox input.wav output.pcm

# CORRECT
sox input.wav -r 48000 -c 1 -b 16 -e signed-integer output.pcm
```

### ❌ WAV Headers in PCM

```bash
# WRONG - still WAV format!
cp input.wav output.pcm

# CORRECT - use sox to convert
sox input.wav -r 48000 -c 1 -b 16 -e signed-integer output.pcm
```

### ❌ Stereo Instead of Mono

```bash
# WRONG - missing -c 1
sox stereo.wav -r 48000 output.pcm

# CORRECT
sox stereo.wav -r 48000 -c 1 -b 16 -e signed-integer output.pcm
```

---

## 🎓 References

- RNNoise Paper: https://arxiv.org/abs/1709.08243
- DNS-Challenge: https://github.com/microsoft/DNS-Challenge
- VIVOS Dataset: https://ailab.hcmus.edu.vn/vivos
- Bark Scale: https://en.wikipedia.org/wiki/Bark_scale

---

**Last Step:** Run `python scripts/train.py features_vn_30k.f32 output/ --sparse --epochs 150`
