# Hướng Dẫn Chi Tiết: Luồng Xử Lý Âm Thanh Từ Microphone

> **Mục tiêu:** Giải thích toàn bộ pipeline RNNoise bằng ví dụ cụ thể, có số liệu thực tế.
> Đọc xong bài này, bạn sẽ hiểu chính xác 480 samples đầu vào trở thành 480 samples đầu ra như thế nào.

---

## 📊 TỔNG QUAN: Microphone → RNNoise → Clean Audio

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           LUỒNG XỬ LÝ TỔNG QUÁT                              │
└──────────────────────────────────────────────────────────────────────────────┘

  🎤 MICROPHONE
  │
  │  Thu âm liên tục ở 48,000 samples/giây
  │  (48kHz sample rate)
  │
  ▼
  ┌─────────────────────────────────────────┐
  │  FRAME BUFFER (10ms = 480 samples)     │
  │                                         │
  │  Ví dụ: [0.023, -0.015, 0.041, ..., 0.008]  │
  │          ↑                                ↑
  │       sample 0                        sample 479  │
  └─────────────────────────────────────────┘
  │
  ▼
  ╔════════════════════════════════════════════╗
  ║  GIAI ĐOẠN 1: PHÂN TÍCH (Analysis)        ║
  ║  ├── High-pass filter (loại DC offset)    ║
  ║  ├── FFT (time → frequency)               ║
  ║  ├── Pitch detection (tìm pitch period)    ║
  ║  └── Tính band energies (22 features)     ║
  ╚════════════════════════════════════════════╝
  │
  ▼
  ╔════════════════════════════════════════════╗
  ║  GIAI ĐOẠN 2: NEURAL NETWORK             ║
  ║  └── RNN: 22 features → 32 gain values    ║
  ╚════════════════════════════════════════════╝
  │
  ▼
  ╔════════════════════════════════════════════╗
  ║  GIAI ĐOẠN 3: DSP SYNTHESIS ★            ║
  ║  ├── Lọc răng lược Pitch                 ║
  ║  ├── Nội suy 32 → 481 bins               ║
  ║  ├── Nhân phổ (X * G)                    ║
  ║  └── IFFT + Overlap-Add                   ║
  ╚════════════════════════════════════════════╝
  │
  ▼
  ┌─────────────────────────────────────────┐
  │  OUTPUT FRAME (10ms = 480 samples)      │
  │                                         │
  │  Ví dụ: [0.018, -0.009, 0.035, ..., 0.005]  │
  │          ↑                                ↑
  │       sample 0                        sample 479  │
  │                                         │
  │  ★ Tiếng nói rõ ràng, ít nhiễu hơn ★   │
  └─────────────────────────────────────────┘
  │
  ▼
  🔊 LOA → Tai người nghe thấy "Tiếng nói sạch"
```

---

## 🔢 THÔNG SỐ CƠ BẢN (Cần nhớ)

Trước khi đi vào chi tiết, hãy ghi nhớ các thông số:

| Thông số | Giá trị | Ý nghĩa |
|----------|---------|----------|
| **Sample Rate** | 48,000 Hz | Số mẫu/giây |
| **Frame Size** | 480 samples | 10ms audio = 480 samples |
| **Window Size** | 960 samples | FFT window (2 × frame) |
| **FFT Size** | 960 | Kích thước FFT |
| **Frequency Bins** | 481 | Số bins sau FFT (960/2 + 1) |
| **ERB Bands** | 32 | Số bands theo thang ERB |
| **Features** | 65 | Input cho RNN (2×32 + 1) |
| **RNN Gain** | 32 values | Output của RNN |

---

## 🎤 VÍ DỤ CỤ THỂ: Giọng nói "Hello" + Tiếng ồn quán cafe

### Đầu vào: Frame âm thanh từ microphone

Giả sử microphone thu được 1 frame (10ms) chứa:
- **Tiếng nói:** Tần số cơ bản f₀ ≈ 180 Hz (giọng nam trung bình)
- **Tiếng ồn:** Tiếng ồn nền quán cafe (random noise)

```
Frame đầu vào (480 samples, 10ms):

Index:    0     1     2     3     ...    239   ...    479
Value:  0.023 -0.015  0.041  0.002 ...  0.156  ...  -0.031

Min: -0.847    Max: 0.923    RMS: 0.234
```

---

## ═══════════════════════════════════════════════════════════════
## GIAI ĐOẠN 1: PHÂN TÍCH (Analysis)
## ═══════════════════════════════════════════════════════════════

### Bước 1.1: High-pass Filter (Loại bỏ DC offset)

**Vấn đề:** Microphone có thể thu được DC offset (tần số 0 Hz) do:
- Tín hiệu analog bị lệch
- Nhiệt độ thay đổi
→ DC offset gây ra "tiếng ù" không mong muốn

**Giải pháp:** Bộ lọc IIR biquad high-pass với cutoff ~80 Hz

```
Trước high-pass:  mean = 0.0032  (có DC offset nhỏ)
Sau high-pass:    mean = 0.0001  (DC offset gần như bằng 0)

Ví dụ:
  Input:  [0.023, -0.015, 0.041, 0.002, ...]
  Output: [0.020, -0.018, 0.038, 0.000, ...]  (đã loại bỏ DC)
```

---

### Bước 1.2: FFT (Chuyển sang miền tần số)

**Windowing trước FFT:**
- Nhân 960 samples với Hann window (smooth ở 2 đầu)
- Frame mới nhất + 480 samples từ frame trước (overlap 50%)

```
Window Size = 960 samples (20ms)

Buffer cấu trúc:
[  480 samples cũ (từ frame trước)  |  480 samples mới (frame hiện tại)  ]
         overlap 50%                              new data

Sau khi window:
[   ~~smooth fade in~~    |    ~~~~main signal~~~~    |   ~~smooth fade out~~   ]
     0 ──────────────→ 1.0                   1.0 ──────────────→ 0
```

**FFT 960 điểm:**

```
FFT Input (960 time-domain samples):
  Sample 0:    0.000
  Sample 1:    0.001
  Sample 2:   -0.002
  ...
  Sample 479:  0.023    ← frame trước (fade out)
  Sample 480:  0.041    ← frame hiện tại (fade in)
  ...
  Sample 959: -0.031

FFT Output (481 frequency bins):
┌──────────────────────────────────────────────────────────────────────┐
│ BIN   FREQUENCY    MAGNITUDE    PHASE      NHẬN XÉT                  │
├──────────────────────────────────────────────────────────────────────┤
│  0       0 Hz        0.002        0.0°      DC (đã lọc)             │
│  1      50 Hz        0.023       45.2°      Low freq noise          │
│  2     100 Hz        0.089       12.8°      ← Tiếng ồn nền          │
│  3     150 Hz        0.156       78.3°      ← Tiếng ồn nền          │
│  4     200 Hz        0.834      156.4°      ★ Tiếng nói (f₀ = 180 Hz)│
│  5     250 Hz        0.756       89.2°      ★ Harmonic 2f₀           │
│  6     300 Hz        0.623       34.7°      ★ Harmonic 3f₀           │
│  7     350 Hz        0.512       67.1°      ★ Harmonic 4f₀           │
│  8     400 Hz        0.423      123.5°      ★ Harmonic 5f₀           │
│ ...      ...           ...          ...                                │
│ 50   2500 Hz        0.298       89.4°      ★ Tiếng nói formants     │
│ ...      ...           ...          ...                                │
│ 200 10000 Hz        0.045      234.1°      Tiếng ồn cao tần         │
│ ...      ...           ...          ...                                │
│ 480 24000 Hz        0.001       15.0°      Siêu âm (sẽ bị lọc)     │
└──────────────────────────────────────────────────────────────────────┘

Frequency resolution: 48,000 / 960 = 50 Hz mỗi bin
```

**Biểu diễn phổ (ASCII art):**

```
Magnitude
  1.0 |            ★
      |           ★★★
  0.8 |         ★★★★★
      |        ★★★★★
  0.6 |       ★★★★★★★
      |      ★★★★★★★★★
  0.4 |     ★★★★★★★★★★★  ← Tiếng nói (f₀ + harmonics)
      |    ★★★★★★★★★★★★★
  0.2 |  ★★★★★★★★★★★★★★★  ← Tiếng ồn nền
      | ★★★★★★★★★★★★★★★★★
  0.0 +------------------------------
      0   200  400  600  800  1000  ...  Hz
                  ↑
             Tiếng nói tập trung ở đây
```

---

### Bước 1.3:  

**Mục đích:** Tìm "độ cao" của giọng nói (fundamental frequency f₀)

**Phương pháp:** Correlation-based pitch detection

```
Frame 10ms = 480 samples @ 48kHz

Pitch period (T) = số samples giữa 2 đỉnh liên tiếp
  - Nếu f₀ = 180 Hz → T = 48,000 / 180 = 267 samples
  - Range: PITCH_MIN = 60 samples (Tmax = 800 Hz)
            PITCH_MAX = 768 samples (Tmin = 62.5 Hz)

Correlation analysis:
┌─────────────────────────────────────────────────────────────────────┐
│  pitch_buf (1728 samples buffer)                                    │
│                                                                     │
│  [ 288 samples  |  480 samples  |  960 samples ]                    │
│     (old)        (this frame)   (search window)                     │
│                                                                     │
│  Tìm vị trí trong search window sao cho tương quan max              │
│  với "this frame"                                                   │
└─────────────────────────────────────────────────────────────────────┘

Kết quả:
  Pitch period = 267 samples  (tương ứng f₀ = 180 Hz)
  Pitch gain   = 0.756        (0 = pure noise, 1 = pure tone)
```

**Tại sao cần pitch detection?**
- Giọng nói có cấu trúc periodic (lặp lại theo chu kỳ)
- Tiếng ồn thì random (không periodic)
- Pitch detection giúp phân biệt tiếng nói và tiếng ồn

---

### Bước 1.4: Tính Band Energies (22 Features cho RNN)

**Chia phổ thành 32 bands theo thang ERB:**

```
Bảng band boundaries (eband20ms):
Band  0: bins   0-1     (  0-100 Hz)   → Low bass
Band  1: bins   2-3     (100-200 Hz)
Band  2: bins   4-5     (200-300 Hz)   ★ Tiếng nói f₀
Band  3: bins   6-7     (300-400 Hz)   ★ Harmonics
...
Band  7: bins  12-14    (600-750 Hz)   ★ Formant region
...
Band 15: bins  41-46    (2050-2350 Hz) ★ Tiếng nói rõ ràng nhất
...
Band 31: bins 317-355   (15850-17750 Hz) → Cao tần

Mỗi band có năng lượng = tổng (real² + imag²) của các bins trong band
```

**Tính 22 năng lượng band (Ex[i]):**

```
Band  0:   0.004
Band  1:   0.023
Band  2:   0.089    ← f₀ region (có tiếng nói)
Band  3:   0.156
Band  4:   0.612    ← Harmonics mạnh
Band  5:   0.534
Band  6:   0.423
Band  7:   0.512    ← Formant region
...
Band 15:   0.298    ← Tiếng nói rõ
...
Band 31:   0.012    ← Cao tần (yếu)

Cộng dồn Ex_total = 4.892
```

**Tính 22 correlations (Exp[i]):**

```
Exp[i] = correlation giữa signal spectrum và pitch spectrum
  = Σ (X_bin * P_bin) / sqrt(Ex[i] * Ep[i])
  = Normalized dot product

Band  2: Exp = 0.756    ← ★ Cao! Có pitch (tiếng nói)
Band  4: Exp = 0.712
Band  7: Exp = 0.523
Band 15: Exp = 0.234    ← Thấp! Ít pitch (unvoiced)
Band 31: Exp = 0.023    ← ★ Thấp! Không có pitch (noise)
```

**Tính Pitch Period Feature:**
```
features[2*NB_BANDS] = 0.01 * (pitch_index - 300)
                      = 0.01 * (267 - 300)
                      = -0.33
```

**22 features cuối cùng cho RNN:**
```
features[0..31]     = log10(Ex[i]) sau khi compress
features[32..63]    = DCT của Exp[i]
features[64]        = normalized pitch period
```

---

## ═══════════════════════════════════════════════════════════════
## GIAI ĐOẠN 2: NEURAL NETWORK (RNN)
## ═══════════════════════════════════════════════════════════════

### RNN Inference: 22 Features → 32 Gain Values

```
Input Layer     Hidden Layer(s)      Output Layer
┌──────────┐                       ┌──────────┐
│ Feature 0│                       │  Gain 0  │
│ Feature 1│ ──┐              ┌──→ │  Gain 1  │
│    ...   │   │   ┌─────┐   │    │   ...   │
│ Feature  │   ├──→│ GRU │───┼──→ │  Gain 31 │
│   63     │   │   └─────┘   │    └──────────┘
│ Feature 64│ ──┘              │
└──────────┘                   │
                                │  VAD probability

Kích thước:
  - Input:  22 neurons (22 features)
  - Hidden: 2 × 384 neurons (GRU layers)
  - Output: 32 neurons (32 gain values)
  - VAD:    1 neuron (Voice Activity Detection)
```

**Output của RNN: 32 Gain Values**

```
Mỗi gain = mức độ "giữ" tín hiệu ở band đó
  - g[i] = 1.0 → Giữ nguyên (100% tín hiệu)
  - g[i] = 0.5 → Giảm 50%
  - g[i] = 0.0 → Triệt tiêu hoàn toàn (100% noise)

Ví dụ output:
┌────────────────────────────────────────────────────────────────────┐
│ Band   0:  g = 0.856   │  Band  16:  g = 0.923   │  Band  31: g = 0.123 │
│ Band   1:  g = 0.812   │  Band  17:  g = 0.887   │                  │
│ Band   2:  g = 0.945   │  Band  18:  g = 0.856   │  ★ Cao tần = noise │
│ Band   3:  g = 0.923   │  Band  19:  g = 0.812   │  → Gain thấp      │
│ Band   4:  g = 0.956   │  Band  20:  g = 0.756   │                  │
│ Band   5:  g = 0.934   │  Band  21:  g = 0.623   │                  │
│ Band   6:  g = 0.912   │  Band  22:  g = 0.534   │                  │
│ Band   7:  g = 0.967   │  Band  23:  g = 0.423   │                  │
│ Band   8:  g = 0.923   │  Band  24:  g = 0.298   │                  │
│ Band   9:  g = 0.856   │  Band  25:  g = 0.234   │                  │
│ Band  10:  g = 0.823   │  Band  26:  g = 0.189   │                  │
│ Band  11:  g = 0.789   │  Band  27:  g = 0.145   │                  │
│ Band  12:  g = 0.745   │  Band  28:  g = 0.112   │                  │
│ Band  13:  g = 0.701   │  Band  29:  g = 0.089   │                  │
│ Band  14:  g = 0.756   │  Band  30:  g = 0.056   │                  │
│ Band  15:  g = 0.912   │  Band  31:  g = 0.023   │                  │
└────────────────────────────────────────────────────────────────────┘

VAD probability = 0.923  (→ 92.3% có tiếng nói)
```

**Ý nghĩa:**
- Bands 0-15 (dải thấp-trung): Gain cao (0.7-0.97) → **Có tiếng nói**
- Bands 16-31 (dải cao): Gain thấp dần → **Tiếng nói ít, nhiều noise**

---

## ═══════════════════════════════════════════════════════════════
## GIAI ĐOẠN 3: DSP SYNTHESIS ★ (TRỌNG TÂM)
## ═══════════════════════════════════════════════════════════════

**Sơ đồ luồng dữ liệu:**

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │                     ĐỆM TRỄ 1 FRAME                                   │  │
│   │                                                                      │  │
│   │   delayed_X[0..480] = 0.002, 0.023, ..., 0.234                       │  │
│   │   delayed_P[0..480] = 0.001, 0.012, ..., 0.089                       │  │
│   │                                                                      │  │
│   │   delayed_Ex[0..31] = 0.004, 0.023, ..., 0.012                       │  │
│   │   delayed_Ep[0..31] = 0.002, 0.015, ..., 0.008                       │  │
│   │   delayed_Exp[0..31] = 0.001, 0.456, ..., 0.012                      │  │
│   │                                                                      │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                                     │                                       │
│                                     │ X_d, P_d                              │
│                                     ▼                                       │
│   ┌───────────────────────────────────────────────────────────────────┐   │
│   │              BƯỚC 1: LỌC RĂNG LƯỢC PITCH                          │   │
│   │              (rnn_pitch_filter)                                     │   │
│   │                                                                   │   │
│   │   Công thức: X_new = X + rf * P                                   │   │
│   │                                                                   │   │
│   │   Trộn phổ tín hiệu với phổ pitch theo hệ số rf                  │   │
│   │   → Tăng cường thành phần harmonic                                │   │
│   │                                                                   │   │
│   └───────────────────────────────────────────────────────────────────┘   │
│                                     │                                       │
│                                     │ X_p (đã pitch-enhanced)              │
│                                     ▼                                       │
│   ┌───────────────────────────────────────────────────────────────────┐   │
│   │              BƯỚC 2: NỘI SUY 32 → 481 BINS                       │   │
│   │              (interp_band_gain)                                   │   │
│   │                                                                   │   │
│   │   Input:  g[0..31]    (32 gain values từ RNN)                    │   │
│   │   Output: gf[0..480]  (481 gain values cho 481 FFT bins)          │   │
│   │                                                                   │   │
│   │   Ví dụ:                                                          │   │
│   │     g[2] = 0.945   →  bins 4-5  có gf = 0.945                    │   │
│   │     g[3] = 0.923   →  bins 6-7  có gf = 0.923                    │   │
│   │     g[4] = 0.956   →  bins 8-11 có gf = 0.956                    │   │
│   │                                                                   │   │
│   └───────────────────────────────────────────────────────────────────┘   │
│                                     │                                       │
│                                     │ G_f (Gain mask)                      │
│                                     ▼                                       │
│   ┌───────────────────────────────────────────────────────────────────┐   │
│   │              BƯỚC 3: NHÂN PHỔ                                    │   │
│   │              X_p * G_f                                              │   │
│   │                                                                   │   │
│   │   X_p[bin] *= G_f[bin]                                            │   │
│   │                                                                   │   │
│   │   Ví dụ:                                                          │   │
│   │     Bin  4: X_p = 0.834 * 0.945 = 0.788  (giảm 5.5%)             │   │
│   │     Bin  5: X_p = 0.756 * 0.934 = 0.706  (giảm 6.6%)             │   │
│   │     Bin 30: X_p = 0.089 * 0.056 = 0.005   (giảm 94%!)            │   │
│   │     Bin 31: X_p = 0.045 * 0.023 = 0.001   (giảm 98%!)            │   │
│   │                                                                   │   │
│   │   → Noise ở dải cao bị triệt tiêu mạnh                           │   │
│   │   → Tiếng nói ở dải trung-tần thấp được giữ                      │   │
│   │                                                                   │   │
│   └───────────────────────────────────────────────────────────────────┘   │
│                                     │                                       │
│                                     │ Phổ đã khử nhiễu                      │
│                                     ▼                                       │
│   ┌───────────────────────────────────────────────────────────────────┐   │
│   │              BƯỚC 4: IFFT + OVERLAP-ADD                           │   │
│   │              (frame_synthesis)                                    │   │
│   │                                                                   │   │
│   │   1. IFFT: 481 bins → 960 samples                                 │   │
│   │   2. Window: Áp Hann window                                       │   │
│   │   3. Overlap-Add:                                                  │   │
│   │      output = frame_mới + overlap_từ_frame_trước                 │   │
│   │                                                                   │   │
│   └───────────────────────────────────────────────────────────────────┘   │
│                                     │                                       │
│                                     ▼                                       │
│   output[0..479]: Tiếng nói đã khử nhiễu (clean audio)                   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## ═══════════════════════════════════════════════════════════════
## CHI TIẾT TỪNG BƯỚC GIAI ĐOẠN 3
## ═══════════════════════════════════════════════════════════════

---

### BƯỚC 1: LỌC RĂNG LƯỢC PITCH (Comb Filtering)

**Input cho bước này:**
- `delayed_X` = Phổ tín hiệu đã trễ (từ frame trước)
- `delayed_P` = Phổ pitch đã trễ
- `delayed_Ex`, `delayed_Ep`, `delayed_Exp` = Band energies/correlations
- `g` = Gain mask từ RNN (22 values)

**Thuật toán:**

```
┌─────────────────────────────────────────────────────────────────────┐
│  VỚI MỖI BAND i (0..31):                                            │
│                                                                     │
│  Bước 1.1: Tính hệ số lọc r[i]                                      │
│  ─────────────────────────────────────────                           │
│  if Exp[i] > g[i]       // Correlation cao hơn gain                │
│      r[i] = 1            // → Có pitch rõ, giữ nguyên             │
│  else                                                                │
│      r[i] = (Exp² * (1 - g²)) / (g² * (1 - Exp²) + 0.001)          │
│      r[i] = sqrt(r[i])   // → Không có pitch, suppress              │
│                                                                     │
│  Ví dụ với Band 4 (f₀ region):                                      │
│    Exp[4] = 0.756   (correlation cao → có pitch)                    │
│    g[4]   = 0.956   (RNN muốn giữ)                                  │
│    → Exp > g → r[4] = 1  (giữ nguyên tín hiệu)                     │
│                                                                     │
│  Ví dụ với Band 31 (cao tần):                                       │
│    Exp[31] = 0.023   (correlation thấp → noise)                    │
│    g[31]   = 0.023   (RNN muốn suppress)                           │
│    → Exp ≈ g → r[31] ≈ 0  (triệt tiêu)                             │
│                                                                     │
│  Bước 1.2: Normalize theo energy ratio                              │
│  ─────────────────────────────────────                              │
│  r[i] *= sqrt(Ex[i] / (Ep[i] + ε))                                  │
│                                                                     │
│  Bước 1.3: Nội suy r[i] → rf[0..480]                                │
│  ─────────────────────────────────────                              │
│  interp_band_gain(rf, r)                                            │
│                                                                     │
│  Bước 1.4: Trộn phổ                                                 │
│  ───────────────────                                                 │
│  VỚI MỖI BIN b (0..480):                                            │
│    X_new[b] = delayed_X[b] + rf[b] * delayed_P[b]                  │
│                                                                     │
│  Ví dụ với Bin 4 (f₀ region):                                        │
│    delayed_X[4] = 0.834∠156.4°                                     │
│    rf[4]       = 0.956                                              │
│    delayed_P[4] = 0.756∠89.2°                                       │
│    X_new[4]    = 0.834∠156.4° + 0.956×0.756∠89.2°                  │
│               = 0.834 + 0.723∠              (tăng cường!)          │
│                                                                     │
│  Bước 1.5: Normalize energy sau khi trộn                            │
│  ─────────────────────────────────────                              │
│  newE = Tính năng lượng band của X_new                              │
│  norm[i] = sqrt(Ex[i] / (newE[i] + ε))                             │
│  X_new *= normf                                                     │
│  → Đảm bảo energy không đổi sau filter                              │
└─────────────────────────────────────────────────────────────────────┘
```

**Ý nghĩa vật lý:**
```
Trước pitch filter:
  ★ ★ ★ ★ ★           ← Tiếng nói harmonic (yếu vì bị noise che)
  ░░░░░░░░░░░░░░░░░   ← Noise

Sau pitch filter:
  ★★★★★★★★★★★        ← Harmonics được tăng cường (comb filter)
  ░░░░░░░░░░░░░░░░░   ← Noise không thay đổi

→ Tiếng nói nổi bật hơn trong noise
```

---

### BƯỚC 2: NỘI SUY 32 → 481 BINS

**Input:** `g[0..31]` (32 gain values từ RNN)
**Output:** `gf[0..480]` (481 gain values cho 481 FFT bins)

```
Band  0 (bins  0- 1):  g = 0.856  →  gf[0] = 0.856,  gf[1] = 0.856
Band  1 (bins  2- 3):  g = 0.812  →  gf[2] = 0.812,  gf[3] = 0.812
Band  2 (bins  4- 5):  g = 0.945  →  gf[4] = 0.945,  gf[5] = 0.945
Band  3 (bins  6- 7):  g = 0.923  →  gf[6] = 0.923,  gf[7] = 0.923
Band  4 (bins  8-11):  g = 0.956  →  gf[8]  = 0.956 ... gf[11] = 0.956
Band  5 (bins 12-14):  g = 0.934  →  gf[12] = 0.934 ... gf[14] = 0.934
Band  6 (bins 15-17):  g = 0.912  →  gf[15] = 0.912 ... gf[17] = 0.912
Band  7 (bins 18-20):  g = 0.967  →  gf[18] = 0.967 ... gf[20] = 0.967
...
Band 30 (bins 282-316): g = 0.056 →  gf[282] = 0.056 ... gf[316] = 0.056
Band 31 (bins 317-356): g = 0.023 →  gf[317] = 0.023 ... gf[356] = 0.023
Band 32 (bins 357-400): padding = 0.356

Tại ranh giới bands (weighted interpolation):
  Band 2 (g = 0.945) → Band 3 (g = 0.923)
  Bin 4: gf[4] = 0.945
  Bin 5: gf[5] = (0.945 + 0.923) / 2 = 0.934  ← linear interpolation
  Bin 6: gf[6] = 0.923
```

**Gain Mask (ASCII visualization):**

```
gf (Gain Mask - Mặt nạ Gain)
  1.0 |████████████████████
      |████████████████████
  0.9 |████████████████████
      |████████████████████
  0.8 |████████████████████
      |████████████████████
  0.7 |████████████████████
      |████████████████████
  0.6 |████████████████████
      |████████████████████
  0.5 |████████████████████
      |████████████████████
  0.4 |███████████████████░
      |██████████████████░░
  0.3 |████████████████░░░
      |███████████████░░░░
  0.2 |█████████████░░░░░░
      |███████████░░░░░░░░
  0.1 |███████░░░░░░░░░░░░
      |████░░░░░░░░░░░░░░░
  0.0 |██░░░░░░░░░░░░░░░░░
      +------------------------------
        0   50  100  150  200  250  300  350  400  450  480  Bin
                  ↑
            Noise suppressed ở đây (gain thấp)
```

---

### BƯỚC 3: NHÂN PHỔ (X_p * G_f)

**Element-wise multiplication trong frequency domain**

```
PHÉP TOÁN:
  VỚI MỖI BIN b (0..480):
    delayed_X[b] = delayed_X[b] * gf[b]

VÍ DỤ CỤ THỂ:

┌────────────────────────────────────────────────────────────────────────────┐
│ BIN    FREQUENCY   X_p (trước)   gf (gain)   X_p * gf (sau)    GIẢM     │
├────────────────────────────────────────────────────────────────────────────┤
│   0        0 Hz        0.002        0.856        0.002        -14%       │
│   4      200 Hz        0.834        0.945        0.788         -6%       │
│   5      250 Hz        0.756        0.934        0.706         -7%       │
│  10      500 Hz        0.523        0.912        0.477         -9%       │
│  50    2500 Hz        0.298        0.234        0.070        -77%   ★★★  │
│ 100    5000 Hz        0.156        0.112        0.017        -89%   ★★★  │
│ 200   10000 Hz        0.045        0.089        0.004        -91%   ★★★  │
│ 300   15000 Hz        0.023        0.056        0.001        -96%   ★★★  │
│ 400   20000 Hz        0.012        0.023        0.000        -98%   ★★★  │
└────────────────────────────────────────────────────────────────────────────┘

★ = Tiếng ồn cao tần bị triệt tiêu mạnh (89-98%)
```

**Ý nghĩa vật lý:**
```
Frequency Domain Multiplication = Time Domain Convolution
  X_p * G_f trong frequency domain
  = Convolution với bộ lọc trong time domain

Gain mask hoạt động như "bộ lọc mềm" (soft mask):
  - Dải tần thấp (0-4 kHz): Gain cao (0.8-1.0) → Tai người nhạy cảm
  - Dải tần cao (> 8 kHz): Gain thấp (0.0-0.2) → Noise nhiều
```

---

### BƯỚC 4: IFFT + OVERLAP-ADD (frame_synthesis)

**4.1 IFFT (Inverse FFT):**

```
Input:  Phổ phức đã khử nhiễu (481 bins)
Output: Time-domain samples (960 samples)

Các bước:
  1. Symmetric extension:
     bins[0..480] → extended[0..959]
     extended[b] = bins[b]                         với b = 0..480
     extended[b] = conjugate(bins[960-b])         với b = 481..959

  2. FFT:
     extended[0..959] ──FFT──→ y[0..959]

  3. Trích real part (với normalization):
     out[b] = 960 * y[960-b].real    với b = 1..959
     out[0] = 960 * y[0].real
```

**4.2 Áp Hann Window:**

```
Sau IFFT, ta có 960 samples. Nhân với Hann window:

Hann Window (960 điểm):
  w[n] = 0.5 * (1 - cos(2πn / 959))

Window shape:
  1.0 |         __________
      |        /          \
      |       /            \
      |      /              \
      |     /                \
      |    /                  \
  0.0 |---                    ----
      0    120  240  360  480  600  720  840  960
             FRAME          OVERLAP

Nhân window:
  x_windowed[n] = x[n] * w[n]
```

**4.3 Overlap-Add:**

```
Overlapping frames (50% overlap):

Frame N-1 (đã xử lý ở bước trước):
  [    overlap (480)     |     output_N-1 (480)     ]
  sample 480────────────→sample 959

Frame N (bước hiện tại):
  [     input_N (480)   |    overlap_N+1 (480)      ]
  sample 0──────────────→sample 479

Overlap-Add:
  VỚI MỖI SAMPLE i (0..479):
    output_N[i] = overlap_N[i] + x_windowed[i]

  overlap_N[i] = x_windowed[480 + i]    ← Lưu cho frame sau

Ví dụ:
  x_windowed = [s0, s1, s2, ..., s479, s480, s481, ..., s959]
                 ←── output_N ───→  ←──── overlap_N+1 ────→

  output_N[0]   = overlap[0] + s0      = s0_from_prev + s0_current
  output_N[1]   = overlap[1] + s1
  ...
  output_N[479] = overlap[479] + s479

  overlap mới  = [s480, s481, ..., s959]  ← Lưu cho frame N+1
```

---

## ═══════════════════════════════════════════════════════════════
## TÓM TẮT: TRACKING DỮ LIỆU QUA TỪNG BƯỚC
## ═══════════════════════════════════════════════════════════════

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         TRACKING DỮ LIỆU MỘT VÍ DỤ                        │
└────────────────────────────────────────────────────────────────────────────┘

MICROPHONE INPUT (480 samples @ 48kHz = 10ms):
  in = [0.023, -0.015, 0.041, 0.002, ..., -0.031]
  │
  ▼ GIAI ĐOẠN 1
  │
  ├─ High-pass filter
  │    in_hp = [0.020, -0.018, 0.038, 0.000, ..., -0.028]
  │
  ├─ FFT (960-point)
  │    X = [0.002∠0°, 0.023∠45°, 0.089∠12°, ..., 0.001∠15°]
  │        (481 complex values)
  │
  ├─ Pitch detection
  │    pitch_period = 267 samples (f₀ = 180 Hz)
  │
  └─ Band energies (32 bands)
       Ex = [0.004, 0.023, 0.089, 0.156, 0.612, ..., 0.012]
       Ep = [0.002, 0.015, 0.078, 0.134, 0.534, ..., 0.008]
       Exp = [0.001, 0.456, 0.756, 0.689, 0.712, ..., 0.023]
       pitch_feature = -0.33

  features = [log10(Ex), DCT(Exp), pitch_feature]
           = [22 giá trị]
  │
  ▼ GIAI ĐOẠN 2
  │
  └─ RNN Inference
       g = [0.856, 0.812, 0.945, 0.923, 0.956, ..., 0.023]
       vad_prob = 0.923
  │
  ▼ GIAI ĐOẠN 3 ★
  │
  ├─ delayed buffers (từ frame trước)
  │    delayed_X = [spectrum của frame N-1]
  │    delayed_P = [pitch spectrum của frame N-1]
  │
  ├─ Pitch Filter
  │    X_p = delayed_X + rf * delayed_P
  │
  ├─ Interpolate Gain
  │    gf = interp(g) = [481 gain values]
  │
  ├─ Spectral Multiplication
  │    X_denoised = X_p * gf
  │
  ├─ IFFT (481 → 960)
  │    x_time = IFFT(X_denoised)
  │
  ├─ Window
  │    x_windowed = x_time * Hann
  │
  ├─ Overlap-Add
  │    out = overlap + x_windowed[0..479]
  │    overlap_new = x_windowed[480..959]
  │
  └─ OUTPUT
       out = [0.018, -0.009, 0.035, 0.003, ..., 0.005]
       (480 samples @ 48kHz = 10ms clean audio)

────────────────────────────────────────────────────────────────────────────

SO SÁNH INPUT vs OUTPUT:

                    INPUT (có noise)    OUTPUT (đã khử)
  RMS:                  0.234              0.198
  Noise floor:         -45 dB             -65 dB
  SNR:                   8 dB               15 dB

  Band 30-31 (noise):  0.134 (mạnh)        0.004 (yếu)
  Band 4-8 (speech):   0.756 (mạnh)        0.712 (mạnh)
```

---

## ═══════════════════════════════════════════════════════════════
## SƠ ĐỒ TỔNG HỢP: TOÀN BỘ PIPELINE
## ═══════════════════════════════════════════════════════════════

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                          PIPELINE TỔNG HỢP VỚI DATA                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  FRAME N-1: Đang được xử lý → Output đã ra loa                              ║
║  FRAME N:   Đang ở Giai đoạn 3 (DSP Synthesis)                              ║
║  FRAME N+1: Chờ trong buffer                                                 ║
║                                                                              ║
║  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                    ║
║  │ FRAME N-1   │    │ FRAME N      │    │ FRAME N+1    │                    ║
║  │ (xong rồi)  │    │ (đang xử lý)│    │ (chờ)        │                    ║
║  │             │    │              │    │              │                    ║
║  │ Output:     │    │ Input:       │    │ Input:        │                    ║
║  │ [480 samps] │    │ [480 samps]  │    │ [480 samps]   │                    ║
║  │ Clean audio │    │ + Noise      │    │               │                    ║
║  └──────────────┘    └──────────────┘    └──────────────┘                    ║
║         │                   │                   │                            ║
║         │            ┌──────┴──────┐            │                            ║
║         │            │             │            │                            ║
║         │     ┌──────▼──────┐      │            │                            ║
║         │     │ GIAI ĐOẠN 1 │      │            │                            ║
║         │     │ Analysis    │      │            │                            ║
║         │     │ • HP filter │      │            │                            ║
║         │     │ • FFT       │      │            │                            ║
║         │     │ • Pitch     │      │            │                            ║
║         │     └──────┬──────┘      │            │                            ║
║         │            │             │            │                            ║
║         │     ┌──────▼──────┐      │            │                            ║
║         │     │ GIAI ĐOẠN 2 │      │            │                            ║
║         │     │ Neural Net  │      │            │                            ║
║         │     │ RNN: 22→32  │      │            │                            ║
║         │     │ g[32] = ... │      │            │                            ║
║         │     └──────┬──────┘      │            │                            ║
║         │            │             │            │                            ║
║         │     ┌──────▼──────┐      │            │                            ║
║         │     │ GIAI ĐOẠN 3 ★│      │            │                            ║
║         │     │ DSP          │      │            │                            ║
║         │     │              │      │            │                            ║
║         │     │ delayed_X ───┼──→ Pitch Filter  │                            ║
║         │     │ delayed_P ───┤      │            │                            ║
║         │     │              │      │            │                            ║
║         │     │ g[32] ───────┼──→ Interp ───────┼──→ 481 bins               ║
║         │     │              │      │            │                            ║
║         │     │ X_p * G_f ───┼──→ Multiply      │                            ║
║         │     │              │      │            │                            ║
║         │     │ IFFT ────────┼──→ IFFT + OLA    │                            ║
║         │     │              │      │            │                            ║
║         │     │ out ─────────┼──→ Clean audio ──┼──→ LOA                     ║
║         │     │              │      │            │                            ║
║         │     │ Save delayed │      │            │                            ║
║         │     │ for next     │      │            │                            ║
║         │     └──────────────┘      │            │                            ║
║         │                           │            │                            ║
║         ▼                           ▼            ▼                            ║
║    Đã ra loa                  Đang xử lý     Buffer                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## ❓ CÂU HỎI THƯỜNG GẶP

### Q1: Tại sao cần 480 samples cho mỗi frame?
```
48,000 samples/giây
÷  100 lần/giây (10ms)
=    480 samples/frame

→ Đủ dài để phân tích pitch (cần ~1 pitch period)
→ Đủ ngắn để bắt thay đổi nhanh trong tiếng nói
```

### Q2: Tại sao dùng 50% overlap?
```
Frame 0: [    overlap    |   output 0    ]  ← Đã xử lý
Frame 1: [   output 0    |   overlap 1   ]  ← Đang xử lý
Frame 2: [   overlap 1   |   output 1    ]  ← Trong buffer

→ Không có "seam" (vách ngăn) giữa các frames
→ Tín hiệu liên tục, mượt mà
```

### Q3: Tại sao cần delay buffer?
```
RNN cần ~1 frame để xử lý features
→ FFT của frame N xong, phải chờ RNN
→ Trong khi chờ, xử lý frame N-1 với delayed data
→ Khi RNN trả kết quả, áp dụng vào delayed frame
→ Đồng bộ được tín hiệu vào/ra
```

### Q4: Pitch filter làm gì?
```
Tiếng nói = Tổng của nhiều sine waves (harmonics)
  f₀ = 180 Hz (fundamental)
  2f₀ = 360 Hz (2nd harmonic)
  3f₀ = 540 Hz (3rd harmonic)
  ...

Comb filter cộng thêm năng lượng vào các harmonics này
→ Tiếng nói rõ ràng, trong hơn
→ Giảm noise giữa các harmonics
```

### Q5: RNNoise có chạy được real-time không?
```
Target: 10ms latency (1 frame)
Performance requirements:
  - FFT: ~0.1ms
  - RNN inference: ~2-3ms
  - Pitch filter: ~0.5ms
  - IFFT: ~0.1ms
  ─────────────────────
  Tổng: ~3-4ms (trong 10ms budget) ✓

→ Có thể chạy real-time trên:
  - Desktop: Core i5/i7 (100x faster)
  - Mobile: ARM Cortex-A series
  - Embedded: ESP32, Raspberry Pi
```

---

## 📁 FILE SOURCE CODE LIÊN QUAN

| File | Mô tả |
|------|-------|
| `src-c/denoise.c` | Code chính: 3 giai đoạn |
| `src-c/pitch.c` | Pitch detection |
| `src-c/rnn.h` | RNN inference |
| `src-c/kiss_fft.h` | FFT/IFFT implementation |

---

**Tổng kết:**
- **Input:** 480 samples (10ms audio @ 48kHz)
- **Output:** 480 samples (10ms clean audio)
- **Latency:** 10ms (real-time)
- **Noise reduction:** ~10-20 dB tùy loại noise
