# RNNoise Inference Flow - Giải Thích Từng Bước

Hiểu cách RNNoise xử lý âm thanh real-time, tập trung vào CONCEPT thay vì code.

---

## Big Picture

RNNoise không "vẽ lại" audio sạch từ đầu như U-Net. Thay vào đó:

**Chiến lược:** Điều chỉnh âm lượng từng dải tần số

- Dải có nhiều speech → giữ nguyên (gain ~1.0)
- Dải có nhiều noise → giảm xuống (gain ~0.6-0.8)

**Processing:** 10ms một frame (480 samples @ 48kHz)

---

## 10 Bước Xử Lý

### **Bước 1: Lọc Cao Tần (High-Pass Filter)**

**Làm gì:** Bỏ các tần số cực thấp (<100-200 Hz)

**Tại sao:**

- Loại DC offset (tín hiệu không dao động)
- Bỏ rumble (tiếng rung cực thấp từ thiết bị)
- Giống như bass cut trên mixer

**Kết quả:** Audio "sạch" hơn ở dải thấp, chuẩn bị cho FFT

---

### **Bước 2: Chia Cửa Sổ (Windowing)**

**Làm gì:** Nhân audio với "cửa sổ" Hamming

**Tại sao:**

- FFT ghét "cạnh sắc" (đầu/cuối frame)
- Cửa sổ làm mượt 2 đầu → giảm artifacts
- Overlap 50% (frame này chồng lên frame trước)

**Hình dung:**

```
Hamming window: ╱‾‾‾‾╲
Audio gốc:     ████████
Sau nhân:      ▁███████▁
               ↑ Mượt hơn
```

**Kết quả:** Audio có biên mượt, FFT chính xác hơn

---

### **Bước 3: FFT - Chuyển Sang Miền Tần Số**

**Làm gì:** Biến 480 samples time-domain → 241 frequency bins

**Tại sao:**

- Noise và speech có "dấu vân tay" khác nhau ở frequency domain
- Dễ thao tác từng tần số riêng biệt
- Giống như "xem màu sắc" thay vì "nghe âm thanh"

**Output:** 241 bins phủ 0-24kHz (mỗi bin ~100 Hz)

---

### **Bước 4: Tìm Pitch (Cao Độ Giọng Nói)**

**Làm gì:** Phân tích autocorrelation để tìm chu kỳ lặp

**Tại sao:**

- Giọng nói = có harmonic (sóng tuần hoàn)
- Noise = không có pattern lặp
- Biết pitch → phân biệt speech vs noise tốt hơn

**Kỹ thuật:**

- Tìm delay làm signal tự tương quan cao nhất
- Delay đó = pitch period (chu kỳ)
- Tạo "bản sao pitch-shifted" để dùng sau

**Kết quả:** Biết giọng nói đang ở tần số nào (94-3000 Hz)

---

### **Bước 5: Trích Xuất 42 Features**

**Làm gì:** Nén 241 bins → 42 số thông minh

Đây là bước **SIÊU QUAN TRỌNG** - sự khác biệt lớn nhất với U-Net!

#### **5a. Band Energy (22 features)**

**Concept:** Nhóm tần số theo "Bark scale" (theo tai người nghe)

```
Tai người KHÔNG nghe đều:
- Dải thấp (0-500 Hz): Nhạy cảm, chia nhỏ
- Dải cao (8-24 kHz): Ít nhạy, gộp lại

Bark bands: [0-100Hz], [100-200Hz], [200-300Hz]...
            (nhỏ ở thấp)
            [8K-10K], [10K-16K], [16K-24K]
            (to ở cao)
```

**Kết quả:** 241 bins → 22 energy values (theo perception)

#### **5b. Spectral Correlation (6 features)**

**Concept:** So sánh frame hiện tại với 6 frames trước

**Tại sao:**

- Speech: Pattern lặp đều (phonemes ~100ms)
- Noise: Random, correlation thấp

**Tính toán:** `correlation(frame_now, frame_6ms_ago)`, `correlation(frame_now, frame_12ms_ago)`...

**Kết quả:** 6 số đo "temporal consistency"

#### **5c. Delta Features (7 features)**

**Concept:** Tốc độ thay đổi energy

**Tại sao:**

- Speech: Năng lượng thay đổi smooth (âm tiết)
- Noise: Thay đổi đột ngột hoặc không đổi

**Tính toán:** `delta = energy_now - energy_previous`

**Kết quả:** 7 số đo dynamics

#### **5d. Pitch Features (7 features)**

**Concept:** Thông tin về harmonic structure

Bao gồm:

- Pitch period (chu kỳ)
- Pitch gain (độ mạnh harmonic)
- Pitch correlation values

**Kết quả:** 7 số mô tả cấu trúc harmonic

**TỔNG: 22 + 6 + 7 + 7 = 42 features**

**Magic:** 42 features này là **expert knowledge** được mã hóa thành số!

---

### **Bước 6: GRU Inference - Trí Tuệ Nhân Tạo**

**Làm gì:** Đưa 42 features vào neural network → ra 22 gains

**Architecture hiểu đơn giản:**

```
42 features → Conv layers (filter patterns)
           → GRU layer 1 (nhớ quá khứ 1)
           → GRU layer 2 (nhớ quá khứ 2)
           → GRU layer 3 (nhớ quá khứ 3)
           → Dense layer → 22 gains [0.6-1.0]
                        → 1 VAD [0-1]
```

**GRU làm gì:** Nhớ context từ frames trước

- "Frame trước có tiếng người → frame này cũng likely có"
- "3 frames trước đang tăng dần → đây là start của phoneme"

**Output:**

- 22 gains: Mỗi band nên giảm bao nhiêu
- 1 VAD: Xác suất có giọng nói (0=silence, 1=speech)

**Tại sao GRU mạnh:** Hiểu temporal context, không chỉ nhìn 1 frame riêng lẻ

---

### **Bước 7: Pitch Filtering - Enhancement**

**Làm gì:** Thêm lại harmonic structure

**Concept:**

- Từ bước 4 có "pitch-shifted spectrum"
- Cộng cái này vào spectrum hiện tại với trọng số nhỏ
- Làm **nổi bật** các tần số harmonic của speech

**Tại sao:**

- Speech quality không chỉ là "bỏ noise"
- Cần "enhance speech" (làm rõ hơn)
- Harmonic = hallmark của giọng nói tốt

**Kết quả:** Speech có "chất" tự nhiên hơn, không khô khan

---

### **Bước 8: Gain Smoothing - Làm Mịn**

**Làm gì:** Không cho gains thay đổi quá nhanh

**Tại sao:**

- Gain nhảy đột ngột = "musical noise" (artifacts)
- Cần smooth temporal transition

**Chiến lược:**

```
Gain hiện tại: 0.7
Gain frame trước: 0.9
→ Không cho drop xuống 0.7 ngay
→ Cho phép tối đa: 0.9 × 0.6 = 0.54
→ Actual gain: max(0.7, 0.54) = 0.7 ✅

(Minimum decay rate = 0.6/frame = RT60 of 135ms)
```

**Kỹ thuật đặc biệt:** Energy compensation

- Nếu signal tăng đột ngột (transient) → điều chỉnh threshold
- Tránh leak noise khi có transient

**Kết quả:** Gains thay đổi tự nhiên, không có artifacts

---

### **Bước 9: Apply Gains - Áp Dụng Lên Spectrum**

**Làm gì:** Nhân spectrum với gains

**Chi tiết:**

1. **Interpolate:** 22 gains → 241 gains (cho từng bin)
2. **Multiply:** `spectrum[i] = spectrum[i] × gain[i]`

**QUAN TRỌNG - Phase Preservation:**

```
Complex spectrum = Magnitude × e^(i×Phase)

Chỉ nhân magnitude:
new_spectrum = (Magnitude × gain) × e^(i×Phase)
                ↑ Modified      ↑ GIỮ NGUYÊN!
```

**Tại sao giữ phase:**

- Phase rất khó predict chính xác
- Magnitude-only modification = ít artifacts
- Phase chứa thông tin speech quan trọng

**Kết quả:** Spectrum với noise suppressed, phase preserved

---

### **Bước 10: IFFT + Overlap-Add - Tổng Hợp**

**Làm gì:** Chuyển spectrum về time-domain

**IFFT:** 241 frequency bins → 480 time samples

**Overlap-Add:**

```
Frame hiện tại: [====240====][====240====]
Frame trước:              [====240====][====240====]
                          ↑ Overlap 50%

Cộng 2 overlap regions lại
→ Smooth transition không có "seam"
```

**Window again:** Nhân Hamming window lần nữa

**Output:** 480 samples denoised audio!

---

## Visual Flow - Toàn Cảnh

```
🎤 Raw Audio (10ms, noisy)
    ↓
🔧 High-Pass Filter → Bỏ DC + rumble
    ↓
🪟 Windowing → Làm mượt biên
    ↓
📊 FFT → 241 frequency bins
    ↓         ↓
🎵 Pitch    📈 Band Energy
   (7 feat)    (22 feat)
    ↓         ↓
🔗 Combine với Correlation (6) + Delta (7)
    ↓
✨ 42 Features (expert-designed)
    ↓
🧠 GRU Neural Network
   (Context-aware prediction)
    ↓
🎚️ 22 Gains + 1 VAD
    ↓
🎼 Pitch Enhancement
    ↓
⏱️ Temporal Smoothing
    ↓
✖️ Apply Gains (magnitude only)
    ↓
🔄 IFFT + Overlap-Add
    ↓
🎧 Clean Audio (10ms)
```

---

## Những Điểm "Siêu Hay Ho"

### 1. **Hybrid Approach (DSP + AI)**

Không phải "pure deep learning":

- **DSP part:** Feature extraction (42 features)
  - Expert knowledge 30+ năm
  - Bark scale, pitch detection, correlation...
- **AI part:** GRU prediction
  - Học từ data
  - Context awareness

**Kết hợp tốt nhất 2 thế giới!**

---

### 2. **Perceptual Compression**

```
241 bins → 42 features = 83% compression!

Nhưng không mất thông tin quan trọng vì:
- Bark scale = theo tai người
- Pitch = cốt lõi của speech
- Correlation = temporal pattern
- Delta = dynamics
```

**Brilliant:** Compress theo cách "có ý nghĩa"

---

### 3. **Minimum Gain = 0.6**

Never suppress below 60%!

**Tại sao:**

- Tránh "over-suppression" (giết cả speech)
- 40% noise reduction đã đủ tốt
- Preserves speech naturalness

**Trade-off thông minh:** Chấp nhận 1 chút noise còn sót, đổi lại speech tự nhiên

---

### 4. **Lookahead = 1 Frame**

Process "delayed" spectrum (từ frame trước):

```
Input frame N → Extract features
             → Predict gains for frame N-1
             → Output frame N-1

Delay: 10ms
```

**Tại sao:** Có context từ future → better decisions

**Acceptable:** 10ms latency không perceptible trong voice calls

---

### 5. **Stateful Processing**

GRU hidden state persists:

```
Frame 1 → GRU state 1
       ↓
Frame 2 → GRU state 2 (affected by state 1)
       ↓
Frame 3 → GRU state 3 (affected by state 2)
```

**Power:** Hiểu "câu chuyện" của audio, không chỉ snapshot

---

### 6. **Energy Compensation**

Khi energy tăng đột ngột:

```
Energy[t-1] = 10
Energy[t] = 100  (transient!)

Naïve threshold: Fixed
Smart threshold: Adjust based on energy change

→ Prevents noise leakage during transients
```

**Subtle nhưng crucial** cho quality!

---

## So Sánh Triết Lý

### **RNNoise Philosophy:**

```
"Đừng cố hoàn hảo. Cố đủ tốt + đủ nhanh."

- Magnitude-only (bỏ phase)
- 22 bands (không phải 241 bins)
- Minimum gain 0.6 (không zero)
- 10ms latency (real-time)

→ Engineering pragmatism ⭐
```

### **U-Net Philosophy:**

```
"Học tất cả từ data. Tái tạo hoàn hảo."

- Full spectrogram reconstruction
- 26,624 values processed
- No domain knowledge constraints
- Quality > speed

→ Academic perfectionism 🎓
```

**Không có đúng sai, chỉ có phù hợp hay không!**

---

## Metrics Thực Tế

**Per-frame processing:**

- Feature extraction: ~0.5ms
- GRU inference: ~2-5ms (SIMD optimized)
- Gain + IFFT: ~1ms
- **Total**: ~4-7ms

**Với input 10ms → processing 7ms → còn 3ms buffer → Real-time ✅**

**Memory:**

- Model: 85KB-1.5MB (tùy version)
- State: ~20KB
- Buffers: ~10KB
- **Total**: ~30KB + model

**ESP32-S3:** ~10-15ms/frame (vẫn real-time!)

---

## Takeaways

1. **Expert features > Raw data** (trong constrained environments)
2. **Magnitude-only = pragmatic** (phase too hard)
3. **Temporal smoothing = essential** (avoid artifacts)
4. **Hybrid DSP+AI = powerful** (best of both)
5. **Perceptual design = efficient** (Bark scale, pitch...)

RNNoise = **Engineering masterpiece** trong real-time audio processing! 🎯
