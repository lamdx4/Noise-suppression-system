/* Copyright (c) 2024 Jean-Marc Valin
 * Copyright (c) 2018 Gregor Richards
 * Copyright (c) 2017 Mozilla */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "kiss_fft.h"
#include "common.h"
#include "denoise.h"
#include <math.h>
#include "rnnoise.h"
#include "pitch.h"
#include "arch.h"
#include "rnn.h"
#include "cpu_support.h"

#define SQUARE(x) ((x)*(x))


#ifndef TRAINING
#define TRAINING 0
#endif


/* ============================================================================
 * BẢNG RANH GIỚI BAND THEO THANG ERB (Equivalent Rectangular Bandwidth)
 * ============================================================================
 *
 * ERB là đơn vị đo "độ rộng băng tần tương đương" - mô phỏng cách tai người
 * phân biệt tần số. Các giá trị dưới đây là chỉ số FFT bin tương ứng với
 * ranh giới của mỗi band (tính ở sample rate 48kHz).
 *
 * Ví dụ: Band 0 chứa bins 0-1 (0-100 Hz)
 *         Band 1 chứa bins 2-3 (100-200 Hz)
 *         Band 7 chứa bins 12-14 (600-750 Hz)
 *         ...
 *         Band 31 chứa bins 317-355 (15.9-17.8 kHz)
 *
 * Tại sao dùng ERB?
 * - Tai người phân giải tần số kém ở dải cao → cần ít band hơn
 * - Tai người phân giải tốt ở dải trung (1-4 kHz) → cần nhiều band hơn
 * - RNNoise dùng 32 bands → phù hợp cho neural network xử lý
 *
 * Công thức tính ERB:
 *   B(1)=400;
 *   for k=2:35
 *     B(k) = B(k-1) - max(2, round(24.7*(4.37*B(k-1)/20+1)/50));
 *   end
 */
const int eband20ms[NB_BANDS+2] = {
/*0 100 200 300 400 500 600 750 900 1.1 1.2 1.4 1.6 1.8 2.1 2.4 2.7 3.0 3.4 3.9 4.4 4.9  5.5  6.2  7.0  7.9  8.8  9.9 11.2 12.6 14.1 15.9 17.8 20.0
  |        Dải thấp       |  Dải trung (lời nói)  |              Dải cao             |  */
  0, 2,  4,  6,  8,  10, 12, 15, 18, 21, 24, 28, 32, 36, 41, 47, 53, 60, 68, 77, 87, 98, 110, 124, 140, 157, 176, 198, 223, 251, 282, 317, 356, 400};


/* ============================================================================
 * CẤU TRÚC TRẠNG THÁI DENOISE - Lưu trữ giữa các frame xử lý
 * ============================================================================
 *
 * RNNoise xử lý audio theo từng frame 10ms (480 samples @ 48kHz).
 * Cấu trúc này lưu trữ trạng thái giữa các frame để đảm bảo:
 * 1. Continuity: Các frame liên tiếp được xử lý nhất quán
 * 2. Pitch tracking: Theo dõi pitch period qua nhiều frames
 * 3. Signal smoothing: Tránh artifacts ở ranh giới frame
 */
struct DenoiseState {
  RNNoise model;                    /* Trọng số neural network */
#if !TRAINING
  int arch;                         /* Kiến trúc CPU (ARM, x86, SIMD...) */
#endif

  /* ----- BỘ NHỚ FFT Analysis/Synthesis ----- */
  float analysis_mem[FRAME_SIZE];   /* Đệm cho overlap-add (phân tích) */
  int memid;                        /* ID bộ nhớ (debug) */
  float synthesis_mem[FRAME_SIZE];  /* Đệm overlap-add (tổng hợp) */

  /* ----- PITCH TRACKING ----- */
  float pitch_buf[PITCH_BUF_SIZE];  /* Buffer lưu audio cho pitch detection
                                       PITCH_BUF_SIZE = 768 + 960 = 1728 samples
                                       Đủ để tìm pitch period trong khoảng 60-768 samples */
  float pitch_enh_buf[PITCH_BUF_SIZE]; /* Buffer cho pitch enhancement */
  float last_gain;                  /* Gain pitch của frame trước */
  int last_period;                  /* Pitch period (samples) của frame trước */

  /* ----- HIGH-PASS FILTER MEMORY ----- */
  float mem_hp_x[2];               /* Bộ nhớ bộ lọc high-pass để loại bỏ DC offset */

  /* ----- GAIN SMOOTHING ----- */
  float lastg[NB_BANDS];           /* Gain của frame trước (cho smoothing) */

  RNNState rnn;                    /* Trạng thái hidden layer của RNN */

  /* =========================================================================
   * ĐỆM TRỄ (DELAY BUFFER) - QUAN TRỌNG CHO GIAI ĐOẠN 3: DSP SYNTHESIS
   * =========================================================================
   *
   * Tại sao cần đệm trễ?
   * - RNN cần ~1 frame để xử lý features
   * - Pitch filter cần áp dụng lên frame đúng (không phải frame mới nhất)
   * - Đệm trễ đảm bảo tín hiệu vào/ra được đồng bộ
   *
   * Signal flow:
   *   Frame N-1 ──được lưu──> delayed_X/delayed_P
   *                         │
   *   Frame N ────FFT─────> X[N], P[N]
   *                         │
   *                    DSP Synthesis
   *                         │
   *                         ▼
   *              Áp dụng pitch filter + gain
   *              vào delayed_X (frame N-1)
   *                         │
   *                         ▼
   *                    IFFT + Overlap-Add
   *                         │
   *                         ▼
   *                   Clean Audio Output
   */
  kiss_fft_cpx delayed_X[FREQ_SIZE];  /* Phổ tín hiệu đã trễ (481 bins) */
  kiss_fft_cpx delayed_P[FREQ_SIZE];  /* Phổ pitch đã trễ (481 bins) */
  float delayed_Ex[NB_BANDS];          /* Band energies của X đã trễ (32 bands) */
  float delayed_Ep[NB_BANDS];          /* Band energies của P đã trễ (32 bands) */
  float delayed_Exp[NB_BANDS];         /* Band correlations đã trễ (32 bands) */
};

/* ============================================================================
 * TÍNH NĂNG LƯỢNG BAND - Cung cấp 32 biên độ lọc cho Giai đoạn 3
 * ============================================================================
 *
 * Input:  X - Phổ phức từ FFT (kích thước FREQ_SIZE = 481 bins)
 * Output: bandE - Mảng 32 giá trị năng lượng, mỗi giá trị = tổng bình phương
 *         magnitude của tất cả FFT bins trong band tương ứng
 *
 * Thuật toán:
 * 1. Duyệt qua 32 bands (theo bảng eband20ms)
 * 2. Với mỗi bin trong band, tính magnitude² = real² + imag²
 * 3. Cộng dồn vào sum của band (có weighted interpolation giữa 2 bands)
 *
 * Tại sao cần "weighted interpolation"?
 * - FFT bins không khớp chính xác với ERB band boundaries
 * - Chia weight theo vị trí: bin gần boundary sẽ contribute vào cả 2 bands
 *
 * Ví dụ: Band i chứa bins từ eband20ms[i] đến eband20ms[i+1]-1
 *         Bin j trong band có vị trí frac = j/band_size
 *         → (1-frac)*tmp cộng vào sum[i]
 *         → frac*tmp cộng vào sum[i+1]
 */
static void IRAM_ATTR compute_band_energy(float *bandE, const kiss_fft_cpx *X) {
  int i;
  float sum[NB_BANDS+2] = {0};
  for (i=0;i<NB_BANDS+1;i++)
  {
    int j;
    int band_size;
    band_size = eband20ms[i+1]-eband20ms[i];
    for (j=0;j<band_size;j++) {
      float tmp;
      float frac = (float)j/band_size;
      tmp = SQUARE(X[eband20ms[i] + j].r);
      tmp += SQUARE(X[eband20ms[i] + j].i);
      sum[i] += (1-frac)*tmp;
      sum[i+1] += frac*tmp;
    }
  }
  sum[1] = (sum[0]+sum[1])*2/3;
  sum[NB_BANDS] = (sum[NB_BANDS]+sum[NB_BANDS+1])*2/3;
  for (i=0;i<NB_BANDS;i++)
  {
    bandE[i] = sum[i+1];
  }
}

/* ============================================================================
 * TÍNH TƯƠNG QUAN BAND - Đo độ giống nhau giữa tín hiệu gốc và pitch
 * ============================================================================
 *
 * Input:  X - Phổ tín hiệu gốc
 *         P - Phổ pitch (đã chuẩn hóa theo pitch period)
 * Output: bandE - Mảng 32 giá trị tương quan (correlation)
 *
 * Công thức: correlation = Σ (X_real * P_real + X_imag * P_imag)
 *            = Dot product của 2 vectors trong không gian phức
 *
 * Ý nghĩa:
 * - Exp[i] cao → Tín hiệu ở band i chứa nhiều thành phần pitch (giọng nói)
 * - Exp[i] thấp → Band i chủ yếu là noise
 *
 * Tại sao cần correlation thay vì chỉ dùng energy?
 * - Energy chỉ cho biết "độ mạnh" của tín hiệu
 * - Correlation cho biết "tín hiệu có periodic (lặp lại) hay không"
 * - Periodic = có pitch = có giọng nói
 */
static void compute_band_corr(float *bandE, const kiss_fft_cpx *X, const kiss_fft_cpx *P) {
  int i;
  float sum[NB_BANDS+2] = {0};
  for (i=0;i<NB_BANDS+1;i++)
  {
    int j;
    int band_size;
    band_size = eband20ms[i+1]-eband20ms[i];
    for (j=0;j<band_size;j++) {
      float tmp;
      float frac = (float)j/band_size;
      tmp = X[eband20ms[i] + j].r * P[eband20ms[i] + j].r;
      tmp += X[eband20ms[i] + j].i * P[eband20ms[i] + j].i;
      sum[i] += (1-frac)*tmp;
      sum[i+1] += frac*tmp;
    }
  }
  sum[1] = (sum[0]+sum[1])*2/3;
  sum[NB_BANDS] = (sum[NB_BANDS]+sum[NB_BANDS+1])*2/3;
  for (i=0;i<NB_BANDS;i++)
  {
    bandE[i] = sum[i+1];
  }
}

/* ============================================================================
 * NỘI SUY GAIN TỪ 32 BANDS LÊN 481 BINS - "Nội suy 481 bins" trong sơ đồ
 * ============================================================================
 *
 * Input:  bandE - Mảng 32 giá trị gain (1 giá trị cho mỗi ERB band)
 * Output: g - Mảng 481 giá trị gain (1 giá trị cho mỗi FFT bin)
 *
 * Thuật toán: Linear interpolation giữa các band boundaries
 *
 * Ví dụ:
 *   Band 0: bins 0-1, value = bandE[0]
 *   Band 1: bins 2-3, value = bandE[1]
 *   Band 2: bins 4-5, value = bandE[2]
 *   ...
 *   Band 7: bins 12-14, value = bandE[7]
 *
 *   Bin 12 (ranh giới) → g[12] = (bandE[6] + bandE[7]) / 2
 *   Bin 13 → g[13] = bandE[7]
 *   ...
 *
 * Tại sao cần nội suy?
 * - RNN chỉ output 32 giá trị (1 giá trị cho mỗi band)
 * - Để apply gain lên phổ 481 bins, cần "fill in" các giá trị trung gian
 * - Linear interpolation là cách đơn giản và hiệu quả
 *
 * Trong sơ đồ Giai đoạn 3:
 *   "Cung cấp 32 biên độ lọc" ──interp_band_gain──> "Mặt nạ Gain G_f mềm" (481 bins)
 */
static void interp_band_gain(float *g, const float *bandE) {
  int i,j;
  memset(g, 0, FREQ_SIZE);
  for (i=1;i<NB_BANDS;i++)
  {
    int band_size;
    band_size = eband20ms[i+1]-eband20ms[i];
    for (j=0;j<band_size;j++) {
      float frac = (float)j/band_size;
      g[eband20ms[i] + j] = (1-frac)*bandE[i-1] + frac*bandE[i];
    }
  }
  for (j=0;j<eband20ms[1];j++) g[j] = bandE[0];
  for (j=eband20ms[NB_BANDS];j<eband20ms[NB_BANDS+1];j++) g[j] = bandE[NB_BANDS-1];
}

extern const float rnn_dct_table[];
extern const kiss_fft_state rnn_kfft;
extern const float rnn_half_window[];

static void IRAM_ATTR dct(float *out, const float *in) {
  int i;
  for (i=0;i<NB_BANDS;i++) {
    int j;
    float sum = 0;
    for (j=0;j<NB_BANDS;j++) {
      sum += in[j] * rnn_dct_table[j*NB_BANDS + i];
    }
    out[i] = sum*sqrt(2./22);
  }
}

#if 0
static void idct(float *out, const float *in) {
  int i;
  for (i=0;i<NB_BANDS;i++) {
    int j;
    float sum = 0;
    for (j=0;j<NB_BANDS;j++) {
      sum += in[j] * rnn_dct_table[i*NB_BANDS + j];
    }
    out[i] = sum*sqrt(2./22);
  }
}
#endif

static void forward_transform(kiss_fft_cpx *out, const float *in) {
  int i;
  kiss_fft_cpx x[WINDOW_SIZE];
  kiss_fft_cpx y[WINDOW_SIZE];
  for (i=0;i<WINDOW_SIZE;i++) {
    x[i].r = in[i];
    x[i].i = 0;
  }
  rnn_fft(&rnn_kfft, x, y, 0);
  for (i=0;i<FREQ_SIZE;i++) {
    out[i] = y[i];
  }
}

/* ============================================================================
 * IFFT (INVERSE FFT) - Phần của "IEFT" trong sơ đồ
 * ============================================================================
 *
 * Chuyển tín hiệu từ miền tần số (frequency domain) về miền thời gian
 * (time domain).
 *
 * Input:  in - Phổ phức kiss_fft_cpx (481 bins)
 * Output: out - Time-domain samples (960 samples = WINDOW_SIZE)
 *
 * Thuật toán:
 * 1. Symmetric extension: FFT chỉ lưu nửa phổ (0 → Nyquist)
 *    Cần reflect để có đủ WINDOW_SIZE bins:
 *    - bins 0..480 giữ nguyên
 *    - bins 481..959 là conjugate mirror của bins 1..479
 *
 * 2. FFT: Dùng cùng thuật toán FFT với conjugate symmetry
 *
 * 3. Trích real part: Vì input là symmetric, output chỉ cần lấy phần thực
 *    và nhân với WINDOW_SIZE (normalization factor)
 *
 * Tại sao dùng "reverse order" (line 350)?
 * - FFT/IFFT có tính chất: IFFT(FFT(x)) = N * x
 * - Để inverse đúng, cần đảo thứ tự bins trước khi FFT
 * - out[i] = y[WINDOW_SIZE - i] (với i > 0)
 */
static void inverse_transform(float *out, const kiss_fft_cpx *in) {
  int i;
  kiss_fft_cpx x[WINDOW_SIZE];
  kiss_fft_cpx y[WINDOW_SIZE];
  for (i=0;i<FREQ_SIZE;i++) {
    x[i] = in[i];
  }
  for (;i<WINDOW_SIZE;i++) {
    x[i].r = x[WINDOW_SIZE - i].r;
    x[i].i = -x[WINDOW_SIZE - i].i;
  }
  rnn_fft(&rnn_kfft, x, y, 0);
  /* output in reverse order for IFFT. */
  out[0] = WINDOW_SIZE*y[0].r;
  for (i=1;i<WINDOW_SIZE;i++) {
    out[i] = WINDOW_SIZE*y[WINDOW_SIZE - i].r;
  }
}

/* ============================================================================
 * ÁP CỬA SỔ (WINDOWING) - Dùng trong cả Analysis và Synthesis
 * ============================================================================
 *
 * Nhân tín hiệu với Hann window để smooth ở 2 đầu frame.
 *
 * Input/Output: x - Buffer kích thước WINDOW_SIZE (960 samples)
 *               Đây là IN-PLACE operation
 *
 * Hann Window:
 *   w[n] = 0.5 * (1 - cos(2πn / (N-1)))
 *   Với N = FRAME_SIZE = 480
 *
 * Window shape:
 *   Đầu frame (n=0..479):    w[n] tăng từ 0 → 1
 *   Cuối frame (n=480..959): w[n] giảm từ 1 → 0
 *
 * Tại sao cần windowing?
 * 1. STFT (Short-Time Fourier Transform) yêu cầu localized analysis
 * 2. Smooth transitions giữa các frames
 * 3. Tránh spectral leakage (năng lượng "tràn" sang bins lân cận)
 *
 * Trong Overlap-Add:
 *   - Frame N: window tail = 1 → window head = 0
 *   - Frame N+1: window tail = 0 → window head = 1
 *   - Khi cộng lại: 1 + 0 = 1 (hoặc 0 + 1 = 1) → tín hiệu liên tục
 *
 * Ví dụ với 50% overlap:
 *   Frame 0: [==== overlap ====][  output_0  ]  (w = 0..1)
 *   Frame 1:                          [  output_1  ][==== overlap ====]
 *                                            = 1 + 1 = 2
 *
 *   Cần normalize bằng cách nhân với 2 khi overlap = 50%
 *   (RNNoise dùng 50% overlap → mỗi sample được cộng 2 lần)
 */
static void apply_window(float *x) {
  int i;
  for (i=0;i<FRAME_SIZE;i++) {
    x[i] *= rnn_half_window[i];
    x[WINDOW_SIZE - 1 - i] *= rnn_half_window[i];
  }
}

struct RNNModel {
  /* Set either blob or const_blob. */
  const void *const_blob;
  void *blob;
  int blob_len;
  FILE *file;
};

RNNModel *rnnoise_model_from_buffer(const void *ptr, int len) {
  RNNModel *model;
  model = malloc(sizeof(*model));
  model->blob = NULL;
  model->const_blob = ptr;
  model->blob_len = len;
  return model;
}

RNNModel *rnnoise_model_from_filename(const char *filename) {
  RNNModel *model;
  FILE *f = fopen(filename, "rb");
  model = rnnoise_model_from_file(f);
  model->file = f;
  return model;
}

RNNModel *rnnoise_model_from_file(FILE *f) {
  RNNModel *model;
  model = malloc(sizeof(*model));
  model->file = NULL;

  fseek(f, 0, SEEK_END);
  model->blob_len = ftell(f);
  fseek(f, 0, SEEK_SET);

  model->const_blob = NULL;
  model->blob = malloc(model->blob_len);
  if (fread(model->blob, model->blob_len, 1, f) != 1)
  {
    rnnoise_model_free(model);
    return NULL;
  }
  return model;
}

void rnnoise_model_free(RNNModel *model) {
  if (model->file != NULL) fclose(model->file);
  if (model->blob != NULL) free(model->blob);
  free(model);
}

int rnnoise_get_size(void) {
  return sizeof(DenoiseState);
}

int rnnoise_get_frame_size(void) {
  return FRAME_SIZE;
}

int rnnoise_init(DenoiseState *st, RNNModel *model) {
  memset(st, 0, sizeof(*st));
#if !TRAINING
  if (model != NULL) {
    WeightArray *list;
    int ret = 1;
    parse_weights(&list, model->blob ? model->blob : model->const_blob, model->blob_len);
    if (list != NULL) {
      ret = init_rnnoise(&st->model, list);
      opus_free(list);
    }
    if (ret != 0) return -1;
  }
#ifndef USE_WEIGHTS_FILE
  else {
    int ret = init_rnnoise(&st->model, rnnoise_arrays);
    if (ret != 0) return -1;
  }
#endif
  st->arch = rnn_select_arch();
#else
  (void)model;
#endif
  return 0;
}

DenoiseState *rnnoise_create(RNNModel *model) {
  int ret;
  DenoiseState *st;
  st = malloc(rnnoise_get_size());
  ret = rnnoise_init(st, model);
  if (ret != 0) {
    free(st);
    return NULL;
  }
  return st;
}

void rnnoise_destroy(DenoiseState *st) {
  free(st);
}

#if TRAINING
extern int lowpass;
extern int band_lp;
#endif

void rnn_frame_analysis(DenoiseState *st, kiss_fft_cpx *X, float *Ex, const float *in) {
  int i;
  float x[WINDOW_SIZE];
  RNN_COPY(x, st->analysis_mem, FRAME_SIZE);
  for (i=0;i<FRAME_SIZE;i++) x[FRAME_SIZE + i] = in[i];
  RNN_COPY(st->analysis_mem, in, FRAME_SIZE);
  apply_window(x);
  forward_transform(X, x);
#if TRAINING
  for (i=lowpass;i<FREQ_SIZE;i++)
    X[i].r = X[i].i = 0;
#endif
  compute_band_energy(Ex, X);
}

int rnn_compute_frame_features(DenoiseState *st, kiss_fft_cpx *X, kiss_fft_cpx *P,
                                  float *Ex, float *Ep, float *Exp, float *features, const float *in) {
  int i;
  float E = 0;
  float Ly[NB_BANDS];
  float p[WINDOW_SIZE];
  float pitch_buf[PITCH_BUF_SIZE>>1];
  int pitch_index;
  float gain;
  float *(pre[1]);
  float follow, logMax;
  rnn_frame_analysis(st, X, Ex, in);
  RNN_MOVE(st->pitch_buf, &st->pitch_buf[FRAME_SIZE], PITCH_BUF_SIZE-FRAME_SIZE);
  RNN_COPY(&st->pitch_buf[PITCH_BUF_SIZE-FRAME_SIZE], in, FRAME_SIZE);
  pre[0] = &st->pitch_buf[0];
  rnn_pitch_downsample(pre, pitch_buf, PITCH_BUF_SIZE, 1);
  rnn_pitch_search(pitch_buf+(PITCH_MAX_PERIOD>>1), pitch_buf, PITCH_FRAME_SIZE,
               PITCH_MAX_PERIOD-3*PITCH_MIN_PERIOD, &pitch_index);
  pitch_index = PITCH_MAX_PERIOD-pitch_index;

  gain = rnn_remove_doubling(pitch_buf, PITCH_MAX_PERIOD, PITCH_MIN_PERIOD,
          PITCH_FRAME_SIZE, &pitch_index, st->last_period, st->last_gain);
  st->last_period = pitch_index;
  st->last_gain = gain;
  for (i=0;i<WINDOW_SIZE;i++)
    p[i] = st->pitch_buf[PITCH_BUF_SIZE-WINDOW_SIZE-pitch_index+i];
  apply_window(p);
  forward_transform(P, p);
  compute_band_energy(Ep, P);
  compute_band_corr(Exp, X, P);
  for (i=0;i<NB_BANDS;i++) Exp[i] = Exp[i]/sqrt(.001+Ex[i]*Ep[i]);
  dct(&features[NB_BANDS], Exp);
  features[2*NB_BANDS] = .01*(pitch_index-300);
  logMax = -2;
  follow = -2;
  for (i=0;i<NB_BANDS;i++) {
    Ly[i] = log10(1e-2+Ex[i]);
    Ly[i] = MAX16(logMax-7, MAX16(follow-1.5, Ly[i]));
    logMax = MAX16(logMax, Ly[i]);
    follow = MAX16(follow-1.5, Ly[i]);
    E += Ex[i];
  }
  if (!TRAINING && E < 0.04) {
    /* If there's no audio, avoid messing up the state. */
    RNN_CLEAR(features, NB_FEATURES);
    return 1;
  }
  dct(features, Ly);
  features[0] -= 12;
  features[1] -= 4;
  return TRAINING && E < 0.1;
}

/* ============================================================================
 * TỔNG HỢP KHUNG ÂM (FRAME SYNTHESIS) - Tương ứng với "IEFT" trong sơ đồ
 * ============================================================================
 *
 * Chuyển đổi tín hiệu từ miền tần số (frequency domain) về miền thời gian
 * (time domain) bằng IFFT, sau đó áp dụng Overlap-Add để nối các frames.
 *
 * Input:
 *   st - Trạng thái denoiser (chứa synthesis_mem)
 *   y  - Phổ phức trong frequency domain (481 bins)
 *   out - Output buffer (480 samples = 1 frame = 10ms @ 48kHz)
 *
 * Thuật toán:
 * ============================================================================
 *
 * BƯỚC 1: IFFT (Inverse FFT)
 * ----------------------------------------------------------------------------
 *   inverse_transform(x, y)
 *   Chuyển từ 481 frequency bins về 960 time-domain samples (WINDOW_SIZE)
 *
 * BƯỚC 2: ÁP CỬA SỔ (Windowing)
 * ----------------------------------------------------------------------------
 *   apply_window(x)
 *   Nhân với Hann window để smooth ở 2 đầu frame
 *   Tránh artifacts ở ranh giới frame
 *
 * BƯỚC 3: OVERLAP-ADD
 * ----------------------------------------------------------------------------
 *   overlap-add = frame_mới + overlap_của_frame_trước
 *   output[i] = x[i] + synthesis_mem[i]
 *
 *   Vì mỗi frame chỉ output 480 samples, nhưng window size = 960
 *   Nên 480 samples "đầu" của window chứa overlap từ frame trước
 *
 * BƯỚC 4: LƯU OVERLAP
 * ----------------------------------------------------------------------------
 *   synthesis_mem = x[FRAME_SIZE..WINDOW_SIZE-1]
 *   480 samples "cuối" được lưu để overlap với frame sau
 *
 * ============================================================================
 * Ví dụ với 3 frames:
 *
 * Frame 0:
 *   [  overlap_0  |  output_0  ]     output_0 = frame_0 + mem (mem = 0 ban đầu)
 *   [      480     |     480     ]
 *                                ^
 *                           Lưu overlap_1
 *
 * Frame 1:
 *   [  overlap_1  |  output_1  ]     output_1 = frame_1 + overlap_1
 *   [      480     |     480     ]
 *                                ^
 *                           Lưu overlap_2
 *
 * Frame 2:
 *   [  overlap_2  |  output_2  ]     output_2 = frame_2 + overlap_2
 *   ...
 *
 * Kết quả: Tín hiệu liên tục, không có "seam" ở ranh giới frame
 */
static void frame_synthesis(DenoiseState *st, float *out, const kiss_fft_cpx *y) {
  float x[WINDOW_SIZE];
  int i;
  inverse_transform(x, y);
  apply_window(x);
  for (i=0;i<FRAME_SIZE;i++) out[i] = x[i] + st->synthesis_mem[i];
  RNN_COPY(st->synthesis_mem, &x[FRAME_SIZE], FRAME_SIZE);
}

void rnn_biquad(float *y, float mem[2], const float *x, const float *b, const float *a, int N) {
  int i;
  for (i=0;i<N;i++) {
    float xi, yi;
    xi = x[i];
    yi = x[i] + mem[0];
    mem[0] = mem[1] + (b[0]*(double)xi - a[0]*(double)yi);
    mem[1] = (b[1]*(double)xi - a[1]*(double)yi);
    y[i] = yi;
  }
}

/* ============================================================================
 * LỌC RĂNG LƯỢC PITCH - "Lọc răng lược Pitch" trong sơ đồ Giai đoạn 3
 * ============================================================================
 *
 * Mục đích: Tăng cường thành phần harmonic của giọng nói bằng cách kết hợp
 *           phổ tín hiệu gốc với phổ pitch đã chuẩn hóa.
 *
 * Input:
 *   X   - Phổ tín hiệu đầu vào (cần filter)
 *   P   - Phổ pitch (chứa thành phần periodic của tín hiệu)
 *   Ex  - Năng lượng band của X
 *   Ep  - Năng lượng band của P
 *   Exp - Tương quan band giữa X và P
 *   g   - Gain mask từ RNN (32 bands)
 *
 * Output: X được modify (thêm thành phần pitch vào)
 *
 * Thuật toán:
 * ============================================================================
 *
 * BƯỚC 1: Tính hệ số lọc răng lược r[i] cho mỗi band
 * ----------------------------------------------------------------------------
 * Công thức:
 *   Nếu Exp[i] > g[i]  → r[i] = 1 (tín hiệu có pitch rõ, giữ nguyên)
 *   Ngược lại          → r[i] = f(Exp[i], g[i])
 *
 * Trong đó:
 *   Exp[i] = correlation(X, P) = mức độ "có pitch" của tín hiệu
 *   g[i]   = gain từ RNN       = mức độ "có tiếng nói" theo neural network
 *
 * Logic:
 *   - Exp cao + g thấp → có pitch nhưng bị suppressed → restore
 *   - Exp thấp        → không có pitch (noise) → giữ nguyên
 *   - g cao           → neural network muốn giữ → ưu tiên
 *
 * BƯỚC 2: Nội suy r[i] từ 32 bands lên 481 bins
 * ----------------------------------------------------------------------------
 *   interp_band_gain(rf, r) → rf[0..480]
 *
 * BƯỚC 3: Trộn phổ X với P theo hệ số rf
 * ----------------------------------------------------------------------------
 *   X_new = X + rf * P
 *   (Thêm thành phần pitch vào tín hiệu gốc)
 *
 * BƯỚC 4: Normalize energy
 * ----------------------------------------------------------------------------
 *   Tính newE = energy(X_new)
 *   norm[i] = sqrt(Ex[i] / (newE[i] + epsilon))
 *   X_new *= normf (để đảm bảo energy không đổi sau filter)
 *
 * ============================================================================
 * Ý nghĩa vật lý:
 * ============================================================================
 * Giọng nói có cấu trúc harmonic (f0, 2f0, 3f0, 4f0...)
 * Bộ lọc răng lược (comb filter) tăng cường các thành phần này
 * → Giọng nói trong suốt và rõ ràng hơn
 * → Giảm artifacts từ noise suppression
 *
 * Trong sơ đồ:
 *   X_d, P_d ──> "Lọc răng lược Pitch" ──> "Phổ X_p đã tỉa"
 */
void rnn_pitch_filter(kiss_fft_cpx *X, const kiss_fft_cpx *P, const float *Ex, const float *Ep,
                  const float *Exp, const float *g) {
  int i;
  float r[NB_BANDS];
  float rf[FREQ_SIZE] = {0};
  float newE[NB_BANDS];
  float norm[NB_BANDS];
  float normf[FREQ_SIZE]={0};
  for (i=0;i<NB_BANDS;i++) {
#if 0
    if (Exp[i]>g[i]) r[i] = 1;
    else r[i] = Exp[i]*(1-g[i])/(.001 + g[i]*(1-Exp[i]));
    r[i] = MIN16(1, MAX16(0, r[i]));
#else
    if (Exp[i]>g[i]) r[i] = 1;
    else r[i] = SQUARE(Exp[i])*(1-SQUARE(g[i]))/(.001 + SQUARE(g[i])*(1-SQUARE(Exp[i])));
    r[i] = sqrt(MIN16(1, MAX16(0, r[i])));
#endif
    r[i] *= sqrt(Ex[i]/(1e-8+Ep[i]));
  }
  interp_band_gain(rf, r);
  for (i=0;i<FREQ_SIZE;i++) {
    X[i].r += rf[i]*P[i].r;
    X[i].i += rf[i]*P[i].i;
  }
  compute_band_energy(newE, X);
  for (i=0;i<NB_BANDS;i++) {
    norm[i] = sqrt(Ex[i]/(1e-8+newE[i]));
  }
  interp_band_gain(normf, norm);
  for (i=0;i<FREQ_SIZE;i++) {
    X[i].r *= normf[i];
    X[i].i *= normf[i];
  }
}

/* ============================================================================
 * HÀM CHÍNH: XỬ LÝ 1 FRAME - Điều phối toàn bộ pipeline RNNoise
 * ============================================================================
 *
 * Đây là hàm public API duy nhất cần gọi từ bên ngoài.
 * Mỗi lần gọi xử lý 1 frame audio (480 samples = 10ms @ 48kHz)
 *
 * Input:
 *   st  - Trạng thái denoiser (đã được khởi tạo bằng rnnoise_create)
 *   in  - Buffer 480 samples audio đầu vào (float)
 *   out - Buffer 480 samples audio đã khử nhiễu (output)
 *
 * Return: Xác suất có tiếng nói (VAD probability) [0..1]
 *         > 0.5 → có tiếng nói, < 0.5 → silence/noise
 *
 * ============================================================================
 * PIPELINE TỔNG QUAN (3 GIAI ĐOẠN):
 * ============================================================================
 *
 * GIAI ĐOẠN 1: PHÂN TÍCH (Analysis)
 *   ├── High-pass filter (loại bỏ DC offset)
 *   ├── FFT analysis (time → frequency)
 *   ├── Pitch detection (tìm pitch period)
 *   └── Tính features (band energies, correlations)
 *
 * GIAI ĐOẠN 2: NEURAL NETWORK
 *   └── RNN inference: features → gain mask (32 bands)
 *
 * GIAI ĐOẠN 3: PHỤC HỒI VÀ CHỐNG MÉO (DSP Synthesis) ★
 *   ├── Lọc răng lược Pitch (pitch filtering)
 *   ├── Nội suy gain 32 → 481 bins
 *   ├── Nhân phổ (spectral masking)
 *   └── IFFT + Overlap-Add (→ clean audio)
 *
 * ============================================================================
 * CHI TIẾT GIAI ĐOẠN 3 (DSP SYNTHESIS) - Mapping với sơ đồ:
 * ============================================================================
 *
 *   ┌─────────────────────────────────────────────────────────────────────┐
 *   │  ĐỆM TRỄ 1 FRAME                                                   │
 *   │  delayed_X, delayed_P (từ frame trước)                              │
 *   │                                                                     │
 *   │    X_d, P_d ──→ ┌─────────────────────────┐                         │
 *   │                │ LỌC RĂNG LƯỢC PITCH   │ ──→ Phổ X_p đã tỉa     │
 *   │  32 bands ────→ │ (rnn_pitch_filter)     │                         │
 *   │  (Ex,Ep,Exp)    └─────────────────────────┘                         │
 *   │                                                                     │
 *   │  32 bands (g) ──→ ┌─────────────────────────┐                       │
 *   │                  │ NỘI SUY 481 BINS        │ ──→ Gain mask G_f     │
 *   │                  │ (interp_band_gain)       │                       │
 *   │                  └─────────────────────────┘                       │
 *   │                                                                     │
 *   │    X_p * G_f ──→ ┌─────────────────────────┐                       │
 *   │                  │ NHÂN PHỔ               │ ──→ Phổ đã khử nhiễu   │
 *   │                  │ delayed_X *= gf        │                         │
 *   │                  └─────────────────────────┘                       │
 *   │                            │                                       │
 *   │                            ▼                                       │
 *   │                  ┌─────────────────────────┐                       │
 *   │                  │ IFFT + OVERLAP-ADD     │ ──→ Âm thanh sạch      │
 *   │                  │ (frame_synthesis)       │     (480 samples)     │
 *   │                  └─────────────────────────┘                       │
 *   └─────────────────────────────────────────────────────────────────────┘
 *
 * ============================================================================
 * CÁC BƯỚC TRONG HÀM NÀY:
 * ============================================================================
 *
 * BƯỚC 1: High-pass filter (line 727)
 *   Loại bỏ DC offset và low-frequency noise (< ~80 Hz)
 *   Dùng bộ lọc IIR biquad
 *
 * BƯỚC 2: Tính features (line 728)
 *   FFT → Band energies → Features cho RNN
 *   (Chi tiết ở hàm rnn_compute_frame_features)
 *
 * BƯỚC 3: RNN Inference (line 732)
 *   Neural network tính gain mask g[32]
 *
 * BƯỚC 4: PITCH FILTERING (line 734) ★ GIAI ĐOẠN 3
 *   rnn_pitch_filter(delayed_X, delayed_P, ...)
 *   Tăng cường thành phần harmonic bằng comb filtering
 *
 * BƯỚC 5: GAIN SMOOTHING (lines 735-743)
 *   Giới hạn tốc độ thay đổi gain để tránh artifacts
 *   - G[band] = max(G[band], 0.6 * lastG[band])
 *   - lastG[band] = min(1, G[band] * energy_ratio)
 *
 * BƯỚC 6: NỘI SUY GAIN (line 744)
 *   interp_band_gain(gf, g)
 *   Chuyển gain từ 32 bands → 481 FFT bins
 *
 * BƯỚC 7: NHÂN PHỔ (lines 746-750)
 *   delayed_X *= gf (element-wise multiplication)
 *   → Phổ đã khử nhiễu (trong frequency domain)
 *
 * BƯỚC 8: IFFT + OVERLAP-ADD (line 752)
 *   frame_synthesis(out, delayed_X)
 *   Chuyển về time domain + nối frames
 *
 * BƯỚC 9: CẬP NHẬT DELAYED BUFFERS (lines 754-758)
 *   Lưu frame hiện tại vào delayed buffers
 *   → Frame sau sẽ dùng frame này để pitch filter
 */
float rnnoise_process_frame(DenoiseState *st, float *out, const float *in) {
  int i;
  kiss_fft_cpx X[FREQ_SIZE];
  kiss_fft_cpx P[FREQ_SIZE];
  float x[FRAME_SIZE];
  float Ex[NB_BANDS], Ep[NB_BANDS];
  float Exp[NB_BANDS];
  float features[NB_FEATURES];
  float g[NB_BANDS];
  float gf[FREQ_SIZE]={1};
  float vad_prob = 0;
  int silence;
  static const float a_hp[2] = {-1.99599, 0.99600};
  static const float b_hp[2] = {-2, 1};
  rnn_biquad(x, st->mem_hp_x, in, b_hp, a_hp, FRAME_SIZE);
  silence = rnn_compute_frame_features(st, X, P, Ex, Ep, Exp, features, x);

  if (!silence) {
#if !TRAINING
    compute_rnn(&st->model, &st->rnn, g, &vad_prob, features, st->arch);
#endif
    rnn_pitch_filter(st->delayed_X, st->delayed_P, st->delayed_Ex, st->delayed_Ep, st->delayed_Exp, g);

    /* =========================================================================
     * GAIN SMOOTHING - Làm mượt gain để tránh artifacts
     * =========================================================================
     *
     * Vấn đề: RNN có thể thay đổi gain đột ngột giữa các frames
     * → Gây ra "musical noise" (tiếng rít, bíp bíp)
     *
     * Giải pháp: Giới hạn tốc độ decay của gain
     *
     * Công thức 1: g[i] = max(g[i], 0.6 * lastg[i])
     *   - Gain không được giảm quá 40% mỗi frame
     *   - 0.6^10 ≈ 0.006 → sau 100ms gain có thể giảm 99.4%
     *   - RT60 ( Reverberation Time) ≈ 135ms
     *
     * Công thức 2: lastg[i] = min(1, g[i] * delayed_Ex[i] / Ex[i])
     *   - Compensate cho thay đổi energy
     *   - Nếu frame hiện tại mạnh hơn frame trước → cho phép gain cao hơn
     *   - Tránh "leaking noise" khi có transient (tiếng động đột ngột)
     */
    for (i=0;i<NB_BANDS;i++) {
      float alpha = .6f;
      /* Giới hạn tốc độ decay: gain ≥ alpha * last_gain */
      g[i] = MAX16(g[i], alpha*st->lastg[i]);
      /* Compensate thay đổi energy và lưu cho frame sau */
      st->lastg[i] = MIN16(1.f, g[i]*(st->delayed_Ex[i]+1e-3)/(Ex[i]+1e-3));
    }

    /* =========================================================================
     * NỘI SUY GAIN TỪ 32 BANDS → 481 FFT BINS
     * =========================================================================
     *
     * "Mặt nạ Gain G_f mềm" trong sơ đồ
     * RNN output: g[0..31] (32 giá trị)
     * Cần apply: gf[0..480] (481 giá trị)
     */
    interp_band_gain(gf, g);

    /* =========================================================================
     * NHÂN PHỔ - "Nhân phổ X_p * G_f" trong sơ đồ ★
     * =========================================================================
     *
     * Element-wise multiplication trong frequency domain
     *
     * Phép toán: delayed_X[i] *= gf[i]  (với i = 0..480)
     *   delayed_X = Phổ đã pitch-filter (X_p)
     *   gf        = Gain mask từ RNN (G_f)
     *   Result    = Phổ đã khử nhiễu
     *
     * Ý nghĩa vật lý:
     * - Nhân phổ = convolution trong time domain
     * - Gain mask = bộ lọc "mềm" (soft mask)
     * - Band nào có gain cao → giữ nguyên
     * - Band nào có gain thấp → suppressed (giảm amplitude)
     *
     * Tại sao dùng delayed_X thay vì X?
     * - delayed_X là frame N-1 (đã được FFT ở frame trước)
     * - X là frame N (FFT vừa được tính)
     * - Pitch filter và gain phải áp dụng vào frame đúng với features
     */
#if 1
    for (i=0;i<FREQ_SIZE;i++) {
      st->delayed_X[i].r *= gf[i];
      st->delayed_X[i].i *= gf[i];
    }
#endif
  }
  frame_synthesis(st, out, st->delayed_X);

  RNN_COPY(st->delayed_X, X, FREQ_SIZE);
  RNN_COPY(st->delayed_P, P, FREQ_SIZE);
  RNN_COPY(st->delayed_Ex, Ex, NB_BANDS);
  RNN_COPY(st->delayed_Ep, Ep, NB_BANDS);
  RNN_COPY(st->delayed_Exp, Exp, NB_BANDS);
  return vad_prob;
}

