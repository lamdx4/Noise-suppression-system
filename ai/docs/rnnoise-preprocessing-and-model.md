# RNNoise Preprocessing & Model Architecture

Hiểu cách RNNoise biến audio thành features và architecture của neural network.

---

## Big Picture

**Training flow:**

```
Raw Audio (noisy + clean)
    ↓ Preprocessing (dump_features)
42 Features + Ground Truth Gains
    ↓ Training (PyTorch)
GRU Model Weights
    ↓ Export
C Code cho Inference
```

Phần này giải thích 2 bước đầu: **Preprocessing** và **Model Architecture**.

---

# PHẦN 1: PREPROCESSING (dump_features)

## Nhiệm Vụ Chính

Tạo training pairs từ audio thô:

```
Input: 3 files (clean speech, bg noise, fg noise)
Output: File .f32 chứa [42 features, 22 gains, 1 VAD] × 30,000
```

---

## Bước 1: Synthetic Mixing

### **Random Picking**

Mỗi sequence (1 giây):

- Random pick 1s từ clean speech
- Random pick 1s từ background noise
- Random pick 1s từ foreground noise

**Tại sao random:** Tạo diversity, tránh model học "vị trí" thay vì "pattern"

### **Random Gains (SNR Levels)**

```
Speech gain: 10^(random(-45 to 0 dB) / 20)
BG noise gain: 10^(random(-30 to +10 dB) / 20)
FG noise gain: 10^(random(-30 to +10 dB) / 20)
```

**Ý nghĩa:**

- Speech: -45 to 0 dB = từ thì thầm đến nói to
- Noise: -30 to +10 dB = từ cực nhỏ đến át cả speech
- **SNR range:** -40 to +45 dB (cực kỳ diverse!)

**Tại sao random SNR:**

- Real-world có mọi điều kiện
- Model phải robust với mọi noise levels

### **Mixing Formula**

```
Noisy = Speech × speech_gain
      + BG_noise × bg_gain
      + FG_noise × fg_gain (87.5% chance)
```

**87.5% foreground:** Không phải lúc nào cũng có transient noise

---

## Bước 2: Data Augmentation

Không chỉ mixing đơn giản! Có thêm **5 augmentations**:

### **2a. Random Filtering (Spectral Coloring)**

```
Biquad IIR filters với random coefficients:
- Cả speech, bg noise, fg noise đều qua filter khác nhau
- Mô phỏng: phone quality, room acoustics, device characteristics
```

**Tại sao:** Microphone/speaker khác nhau = EQ khác nhau

### **2b. Random Start Position**

```
75% samples: Start từ đầu file
25% samples: Start từ random position (exponential distribution)
```

**Tại sao:** Không có "beginning bias"

### **2c. Reverberation (Optional)**

Nếu có RIR dataset:

```
50% samples: Apply room impulse response
- Speech: Early reflections only (first 10ms)
- Noisy: Full reverb
```

**Tại sao:**

- Clean speech không có late reflections (không echo)
- Noisy có full reverb (realistic)
- Model học suppress reverb tail

### **2d. Input Clipping**

```
25% samples: Clip to ±32767 (0 dBFS)
```

**Tại sao:** Real-world có clipping (ADC saturation), model cần handle

### **2e. Quantization**

```
50% samples: Round to 16-bit integers
```

**Tại sao:** Real audio = 16-bit, không phải float32

---

## Bước 3: Feature Extraction (Same as Inference)

Với **noisy audio**, extract 42 features:

- 22 Bark band energies
- 6 Spectral correlations
- 7 Delta features
- 7 Pitch features

(Chi tiết xem `rnnoise-inference-flow.md`)

**Key:** Features từ **noisy audio**, không phải clean!

---

## Bước 4: Ground Truth Computation

### **Ideal Wiener Gain**

Từ clean và noisy, compute ideal gains:

```
For each Bark band i:
    clean_energy = compute_energy(clean_spectrum, band[i])
    noisy_energy = compute_energy(noisy_spectrum, band[i])

    ideal_gain[i] = sqrt(clean_energy / noisy_energy)

    # Cap at 1.0 (never amplify)
    if ideal_gain[i] > 1.0:
        ideal_gain[i] = 1.0

    # Mark invalid if too quiet
    if clean_energy < threshold or noisy_energy < threshold:
        ideal_gain[i] = -1  # Mask out
```

**Công thức Wiener filter:** Optimal gain trong MSE sense

**Cap at 1.0:** Không amplify (chỉ suppress)

**Masking:** -1 = "don't learn from this band" (too quiet/unreliable)

### **VAD Ground Truth**

```
Viterbi algorithm trên energy:
- Segment thành speech/silence
- Smooth với HMM (tránh flickering)
- Output: 0 (silence) or 1 (speech)
```

**Tại sao Viterbi:** Tránh VAD nhảy liên tục (temporal consistency)

---

## Bước 5: Save Training Pair

```
每 sequence lưu:
[42 features] [22 ideal_gains] [1 VAD] = 65 float32 values
       ↑ Input         ↑ Target      ↑ Auxiliary
```

Repeat 30,000 lần → File ~8-12 GB

---

## Những Điểm Hay Ho (Preprocessing)

### 1. **Synthetic Data > Real Data**

**Tại sao không dùng real noisy recordings?**

❌ **Real recordings:**

- Không có ground truth (không biết clean nào)
- Ít diversity (chỉ vài scenarios)
- Expensive (cần người thu âm)

✅ **Synthetic mixing:**

- Perfect ground truth (có clean + noise riêng)
- Infinite diversity (random combinations)
- Free (code tự generate)

**Trade-off:** Phải careful với augmentation để realistic

---

### 2. **Wiener Filter = Optimal Baseline**

```
Wiener gain = Best possible gain (statistically)

Nếu model predict gần Wiener → Excellent!
```

Model không cần "invent" better method, chỉ cần **learn to approximate Wiener filter**.

**Brilliant:** Target không phải "perfect clean" mà là "statistically optimal gains"

---

### 3. **Masking Invalid Bands**

```
gain = -1 → Loss = 0 (không học từ band này)

Tại sao:
- Band quá yên tĩnh → noise floor, unreliable
- Silence frames → không có speech để học
- Target speaker not active → avoid learning noise
```

Smart masking = better training signal

---

### 4. **Multi-Scale SNR**

```
SNR range: -40 to +45 dB = 85 dB dynamic range!

Covers:
- Whisper in quiet room (+40 dB)
- Normal speech with AC (-5 dB)
- Shouting in construction site (-30 dB)
```

Model phải universal → train với extreme conditions

---

### 5. **Foreground Noise Strategy**

```
12.5% samples: No foreground
87.5% samples: With foreground (random bursts)

Tại sao không 100%?
- Real-world: Transients không liên tục
- Model learn: "Sometimes có, sometimes không"
- Tránh bias: "Luôn expect transient"
```

**Subtle detail = big impact** on generalization

---

# PHẦN 2: MODEL ARCHITECTURE

## GRU Network Design

**Philosophy:** Small model, big context

```
Input: 42 features (1 frame = 10ms)
Output: 22 gains + 1 VAD
Hidden: Persistent state (temporal memory)
```

---

## Layer-by-Layer Breakdown

### **Input Processing**

```
Input: [Batch, Sequence, 42]
       ↓
Dense(42 → 128, tanh) - "Feature compression"
       ↓
Conv1D(128 → 96, kernel=3, tanh) - "Temporal smoothing"
```

**Dense layer:** Combine raw features  
**Conv1D:** Look at 3-frame window (30ms context)

**Output:** [Batch, Sequence, 96]

---

### **GRU Stack (Core)**

```
GRU1: 96 → 384 units
       ↓ (carry hidden state)
GRU2: 384 → 384 units
       ↓ (carry hidden state)
GRU3: 384 → 384 units
       ↓ (carry hidden state)
```

**Tại sao 3 layers?**

- Layer 1: Low-level patterns (phonemes)
- Layer 2: Mid-level patterns (words)
- Layer 3: High-level patterns (sentences)

**Tại sao 384 units?**

- Power of 2 friendly (SIMD optimization)
- Sweet spot (256 = underfitting, 512 = overkill)
- Tested empirically

---

### **Multi-Scale Fusion**

```
conv2_out: [Batch, Seq, 96]
gru1_out:  [Batch, Seq, 384]
gru2_out:  [Batch, Seq, 384]
gru3_out:  [Batch, Seq, 384]

Concatenate all:
fused = [Batch, Seq, 1248]  (96+384+384+384)
```

**Tại sao concatenate tất cả?**

- GRU1: Short-term context
- GRU2: Medium-term context
- GRU3: Long-term context
- Conv: Local features

**All scales matter!** → Combine cả 4

---

### **Output Layers**

```
fused [1248]
    ↓
Dense(1248 → 22, sigmoid) → gains [0-1]
    ↓
Dense(1248 → 1, sigmoid) → VAD [0-1]
```

**Sigmoid activation:** Bound output to [0,1]

**Post-processing (during inference):**

```
# Clip gains to [0.6, 1.0]
gains = 0.6 + 0.4 * sigmoid_output
```

Never suppress below 60%!

---

## Training Strategy

### **Loss Function (PyTorch version)**

**Gain Loss (Perceptual):**

```python
# Gamma = 0.25 (perceptual exponent)
error = predicted_gain^0.25 - target_gain^0.25

# Weight by VAD (speech present = more important)
weighted_error = (1 + 5*VAD) × mask × error²

gain_loss = mean(weighted_error)
```

**Tại sao gamma=0.25?**

- Linear MSE: treat all errors equally
- Power 0.25: penalize errors ở low gains nhiều hơn
- **Perceptual:** Tai người nhạy cảm với small gains hơn large gains

**Tại sao weight by VAD?**

- Speech frames: 6× more important
- Silence frames: Still learn (don't ignore noise-only)

**VAD Loss (Binary Cross-Entropy):**

```python
# Weight by confidence
weight = |2*VAD - 1|  # 1 at extremes, 0 at 0.5

vad_loss = mean(weight × BCE(predicted_VAD, target_VAD))
```

**Total Loss:**

```python
loss = gain_loss + 0.001 × vad_loss
       ↑ Main      ↑ Auxiliary (1000× smaller)
```

VAD = helper task, không phải primary objective

---

### **Optimizer & Schedule**

```
Optimizer: AdamW
- Beta: [0.8, 0.98] (faster momentum decay)
- Epsilon: 1e-8
- LR: 1e-3 (initial)

LR Schedule: Lambda decay
LR(step) = 1 / (1 + 5e-5 × step)

Batch size: 128
Sequence length: 2000 frames (20 seconds)
```

**Tại sao sequence 2000?**

- GRU cần long context để học temporal patterns
- 20s = enough để cover multi-syllable words/phrases

---

### **Regularization**

**Weight Constraints:**

```python
# Clip all weights to [-0.499, 0.499]
W = clip(W, -0.499, 0.499)
```

**Tại sao 0.499?**

- Prevents exploding gradients
- Keep weights quantization-friendly (int8 conversion)
- Empirically stable

**L2 Regularization:**

```python
L2 penalty: 1e-6
```

Very small (just prevents extreme outliers)

---

## Sparsification (Advanced)

### **Progressive Pruning**

```
Step 0-6000: Dense training
    ↓ Model learns with full capacity

Step 6000-20000: Gradual pruning
    progress = (step - 6000) / 14000
    sparsity = target_sparsity × progress³

    Every 100 steps:
        - Find smallest |weights|
        - Zero out bottom X%
        - Continue training

Step 20000+: Fixed sparsity fine-tuning
    ↓ Model adapts to sparse structure
```

**Tại sao cubic progress (³)?**

- Slow start (careful pruning)
- Fast end (aggressive cleanup)
- Smooth transition

### **Block Sparsity**

```
Instead of: Random individual zeros
Use: 8×4 block zeros

Example:
[0 0 0 0  x x x x  0 0 0 0 ...]
 └ block 1┘└block 2┘└block 3┘

Hardware can skip entire blocks → Faster!
```

**Magic numbers:**

- W_hn: 50% sparsity (most aggressive)
- W_hz: 20% sparsity (least aggressive)
- W_hr: 30% sparsity (middle ground)

Different layers = different importance!

---

## Những Điểm Hay Ho (Model)

### 1. **Multi-Scale Architecture**

```
Conv: Local (3 frames = 30ms)
GRU1: Short (phonemes ~50ms)
GRU2: Medium (words ~200ms)
GRU3: Long (phrases ~500ms)

All fused → Comprehensive understanding!
```

Không có layer nào "useless", each captures different timescale.

---

### 2. **Tiny Model, Big Memory**

```
Parameters: 1.5M (PyTorch) or 85K (Keras)
Memory footprint: 30KB + model

But: Infinite temporal context via GRU hidden state!
```

**Recurrence = memory efficiency**

Contrast with U-Net:

- U-Net: 450K params, finite window (1s)
- RNNoise: 85K params, infinite memory

**GRU magic!**

---

### 3. **Auxiliary VAD Task**

```
Main: Predict gains (denoising)
Aux: Predict VAD (speech detection)

Why both?
- VAD helps denoise (know when to suppress)
- Denoise helps VAD (clean features easier to classify)
- Multi-task learning = better representations
```

**Synergy:** Two tasks improve each other

---

### 4. **Perceptual Loss (Gamma=0.25)**

```
Linear loss:    Equal penalty cho mọi errors
Perceptual:     Penalize errors theo tai người

Example:
Gain = 0.2:  delta ±0.1 → VERY noticeable
Gain = 0.9:  delta ±0.1 → Barely noticeable

Gamma = 0.25 captures this!
```

**Loss design = critical** cho quality

---

### 5. **Block Sparsity = Hardware-Aware**

```
Random sparsity: Theoretical speedup
[x 0 x 0 x 0 ...] → Hard to optimize

Block sparsity: Practical speedup
[x x x x 0 0 0 0 ...] → Skip whole blocks!

ESP32 SIMD: Process 4-8 values at once
→ Block size 8×4 perfect!
```

**ML for embedded = must think hardware!**

---

## Training Timeline

```
Feature generation: 1-3 giờ (30K sequences)
    ↓
PyTorch training: 4-8 giờ (150 epochs, GPU)
    ↓ Checkpoints every epoch
Export to C: <5 phút
    ↓
Ready for deployment!

Total: ~6-11 giờ end-to-end
```

---

## Takeaways

### **Preprocessing Insights:**

1. **Synthetic > Real** (with careful augmentation)
2. **Wiener filter = optimal target** (statistically)
3. **Extreme diversity** (SNR -40 to +45 dB)
4. **Smart masking** (invalid bands = don't learn)

### **Model Insights:**

1. **Multi-scale fusion** (local + short + medium + long)
2. **GRU = infinite memory** (stateful processing)
3. **Perceptual loss** (gamma=0.25 matches human hearing)
4. **Block sparsity** (hardware-aware pruning)
5. **Auxiliary VAD** (multi-task synergy)

**Engineering excellence** trong mỗi detail! 🎯
