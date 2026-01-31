/*
 * pitch_opt_pie.c
 * Optimized Pitch Detection for ESP32-S3 using PIE SIMD
 *
 * PRODUCTION VERSION (Hybrid SIMD-Scalar)
 */

#include "pitch_opt_pie.h"
#include "esp_attr.h"
#include "esp_log.h"
#include <math.h>
#include <stdint.h>

#define SCRATCH_SIZE 2048

static int16_t x_int_buf[SCRATCH_SIZE] __attribute__((aligned(16)));
static int16_t y_int_buf[SCRATCH_SIZE * 2] __attribute__((aligned(16)));

// Assembly implemented in pitch_opt_pie_asm.S
extern int64_t xcorr_dot_product_pie_asm(const int16_t *x, const int16_t *y,
                                         int len);

static inline float vec_max_abs(const float *x, int len) {
  float max_val = 0.0f;
  for (int i = 0; i < len; i++) {
    float v = fabsf(x[i]);
    if (v > max_val)
      max_val = v;
  }
  return max_val;
}

static void vec_float_to_int16(int16_t *out, const float *in, int len,
                               float scale) {
  for (int i = 0; i < len; i++) {
    out[i] = (int16_t)lrintf(in[i] * scale);
  }
  int aligned = (len + 7) & ~7;
  for (int i = len; i < aligned; i++)
    out[i] = 0;
}

void compute_pitch_xcorr_pie_magic(const opus_val16 *_x, const opus_val16 *_y,
                                   opus_val32 *xcorr, int len, int max_pitch) {
  if (len + max_pitch > (SCRATCH_SIZE * 2) - 8)
    return;

  float max_all = fmaxf(vec_max_abs(_x, len), vec_max_abs(_y, len + max_pitch));
  float scale = 1.0f;
  if (max_all > 1e-9f) {
    scale = 5000.0f / max_all;
    if (scale > 30000.0f)
      scale = 30000.0f;
  }

  vec_float_to_int16(x_int_buf, _x, len, scale);
  vec_float_to_int16(y_int_buf, _y, len + max_pitch, scale);

  double scale_sq_inv = 1.0 / ((double)scale * (double)scale);

  for (int lag = 0; lag < max_pitch; lag++) {
    // 1. SIMD PATH (Scaled Int16) for multiples of 8
    int64_t sum_int =
        xcorr_dot_product_pie_asm(x_int_buf, y_int_buf + lag, len & ~7);

    // 2. SCALAR PATH for the remaining 1-7 samples
    for (int i = (len & ~7); i < len; i++) {
      sum_int += (int64_t)x_int_buf[i] * (int64_t)y_int_buf[i + lag];
    }

    xcorr[lag] = (opus_val32)((double)sum_int * scale_sq_inv);
  }
}
