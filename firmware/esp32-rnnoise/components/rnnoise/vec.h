#ifndef VEC_H
#define VEC_H

#include <esp_attr.h>
#include "opus_types.h"
#include <math.h>
#include "arch.h"
// ESP32 PORT: Hardcoded x86 include removed
// #include "x86/x86_arch_macros.h"

// ESP32 PORT: Force generic C implementation, disable x86/ARM SIMD
#if 0 // was: defined(__AVX__) || defined(__SSE2__)
#include "vec_avx.h"
#elif 0 // was: (defined(__ARM_NEON__) || defined(__ARM_NEON)) && !defined(DISABLE_NEON)
#include "vec_neon.h"
#else

#include "os_support.h"

#define MAX_INPUTS (2048)


static inline IRAM_ATTR void sgemv16x1(float *out, const float *weights, int rows, int cols, int col_stride, const float *x)
{
   int i, j;
   OPUS_CLEAR(out, rows);
   for (i = 0; i < rows; i += 16)
   {
      for (j = 0; j < cols; j++)
      {
         const float *restrict w;
         float *restrict y;
         float xj;
         w = &weights[j * col_stride + i];
         xj = x[j];
         y = &out[i];
         y[0] += w[0] * xj;
         y[1] += w[1] * xj;
         y[2] += w[2] * xj;
         y[3] += w[3] * xj;
         y[4] += w[4] * xj;
         y[5] += w[5] * xj;
         y[6] += w[6] * xj;
         y[7] += w[7] * xj;
         y[8] += w[8] * xj;
         y[9] += w[9] * xj;
         y[10] += w[10] * xj;
         y[11] += w[11] * xj;
         y[12] += w[12] * xj;
         y[13] += w[13] * xj;
         y[14] += w[14] * xj;
         y[15] += w[15] * xj;
      }
   }
}

static inline IRAM_ATTR void sgemv8x1(float *out, const float *weights, int rows, int cols, int col_stride, const float *x)
{
   int i, j;
   OPUS_CLEAR(out, rows);
   for (i = 0; i < rows; i += 8)
   {
      for (j = 0; j < cols; j++)
      {
         const float *restrict w;
         float *restrict y;
         float xj;
         w = &weights[j * col_stride + i];
         xj = x[j];
         y = &out[i];
         y[0] += w[0] * xj;
         y[1] += w[1] * xj;
         y[2] += w[2] * xj;
         y[3] += w[3] * xj;
         y[4] += w[4] * xj;
         y[5] += w[5] * xj;
         y[6] += w[6] * xj;
         y[7] += w[7] * xj;
      }
   }
}

static inline IRAM_ATTR void sgemv(float *out, const float *weights, int rows, int cols, int col_stride, const float *x)
{
   if ((rows & 0xf) == 0)
      sgemv16x1(out, weights, rows, cols, col_stride, x);
   else if ((rows & 0x7) == 0)
      sgemv8x1(out, weights, rows, cols, col_stride, x);
   else
   {
      int i, j;
      for (i = 0; i < rows; i++)
      {
         out[i] = 0;
         for (j = 0; j < cols; j++)
            out[i] += weights[j * col_stride + i] * x[j];
      }
   }
}

static inline IRAM_ATTR void sparse_sgemv8x4(float *out, const float *w, const int *idx, int rows, const float *x)
{
   int i, j;
   OPUS_CLEAR(out, rows);
   for (i = 0; i < rows; i += 8)
   {
      int cols;
      cols = *idx++;
      for (j = 0; j < cols; j++)
      {
         int pos;
         float *restrict y;
         float xj0, xj1, xj2, xj3;
         pos = (*idx++);
         xj0 = x[pos + 0];
         xj1 = x[pos + 1];
         xj2 = x[pos + 2];
         xj3 = x[pos + 3];
         y = &out[i];
         y[0] += w[0] * xj0;
         y[1] += w[1] * xj0;
         y[2] += w[2] * xj0;
         y[3] += w[3] * xj0;
         y[4] += w[4] * xj0;
         y[5] += w[5] * xj0;
         y[6] += w[6] * xj0;
         y[7] += w[7] * xj0;

         y[0] += w[8] * xj1;
         y[1] += w[9] * xj1;
         y[2] += w[10] * xj1;
         y[3] += w[11] * xj1;
         y[4] += w[12] * xj1;
         y[5] += w[13] * xj1;
         y[6] += w[14] * xj1;
         y[7] += w[15] * xj1;

         y[0] += w[16] * xj2;
         y[1] += w[17] * xj2;
         y[2] += w[18] * xj2;
         y[3] += w[19] * xj2;
         y[4] += w[20] * xj2;
         y[5] += w[21] * xj2;
         y[6] += w[22] * xj2;
         y[7] += w[23] * xj2;

         y[0] += w[24] * xj3;
         y[1] += w[25] * xj3;
         y[2] += w[26] * xj3;
         y[3] += w[27] * xj3;
         y[4] += w[28] * xj3;
         y[5] += w[29] * xj3;
         y[6] += w[30] * xj3;
         y[7] += w[31] * xj3;
         w += 32;
      }
   }
}

#define FORCE_INLINE __attribute__((always_inline)) inline
#define ALIGNED_16 __attribute__((aligned(16)))
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

// Shared static cache for quantization
static ALIGNED_16 uint8_t x_cache[MAX_INPUTS];

// KERNEL: Optimized with 32-bit loads and register blocking
static IRAM_ATTR FORCE_INLINE void compute_sparse_8x4_kernel(
    int32_t acc[8],
    const int8_t *weights,
    const uint8_t *inputs)
{
    // 1. LOAD INPUT (4 bytes at once)
    uint32_t x_pack;
    memcpy(&x_pack, inputs, 4);

    // Unpack to 32-bit registers (zero-extended since inputs are uint8)
    uint32_t x0 = (uint32_t)(uint8_t)(x_pack);
    uint32_t x1 = (uint32_t)(uint8_t)(x_pack >> 8);
    uint32_t x2 = (uint32_t)(uint8_t)(x_pack >> 16);
    uint32_t x3 = (uint32_t)(uint8_t)(x_pack >> 24);

    // 2. LOAD WEIGHTS (Assuming 4-byte alignment)
    const uint32_t *w32 = (const uint32_t *)weights;
    uint32_t w0 = w32[0], w1 = w32[1], w2 = w32[2], w3 = w32[3];
    uint32_t w4 = w32[4], w5 = w32[5], w6 = w32[6], w7 = w32[7];

    // 3. COMPUTE ALL 8 ROWS (Unrolled)
    // acc = signed int8 weight * unsigned uint8 input
    acc[0] += (int8_t)(w0) * x0 + (int8_t)(w0 >> 8) * x1 + (int8_t)(w0 >> 16) * x2 + (int8_t)(w0 >> 24) * x3;
    acc[1] += (int8_t)(w1) * x0 + (int8_t)(w1 >> 8) * x1 + (int8_t)(w1 >> 16) * x2 + (int8_t)(w1 >> 24) * x3;
    acc[2] += (int8_t)(w2) * x0 + (int8_t)(w2 >> 8) * x1 + (int8_t)(w2 >> 16) * x2 + (int8_t)(w2 >> 24) * x3;
    acc[3] += (int8_t)(w3) * x0 + (int8_t)(w3 >> 8) * x1 + (int8_t)(w3 >> 16) * x2 + (int8_t)(w3 >> 24) * x3;
    acc[4] += (int8_t)(w4) * x0 + (int8_t)(w4 >> 8) * x1 + (int8_t)(w4 >> 16) * x2 + (int8_t)(w4 >> 24) * x3;
    acc[5] += (int8_t)(w5) * x0 + (int8_t)(w5 >> 8) * x1 + (int8_t)(w5 >> 16) * x2 + (int8_t)(w5 >> 24) * x3;
    acc[6] += (int8_t)(w6) * x0 + (int8_t)(w6 >> 8) * x1 + (int8_t)(w6 >> 16) * x2 + (int8_t)(w6 >> 24) * x3;
    acc[7] += (int8_t)(w7) * x0 + (int8_t)(w7 >> 8) * x1 + (int8_t)(w7 >> 16) * x2 + (int8_t)(w7 >> 24) * x3;
}

#ifdef USE_SU_BIAS
// Optimized quantization with 127-offset for USE_SU_BIAS
static IRAM_ATTR FORCE_INLINE uint8_t quantize_for_subias(float x) {
    int32_t val = 127 + lrintf(127.0f * x);
    if (val > 255) val = 255;
    else if (val < 0) val = 0;
    return (uint8_t)val;
}

static inline IRAM_ATTR void sparse_cgemv8x4(float *out, const opus_int8 *w, const int *idx, const float *scale, int rows, int cols, const float *_x)
{
    int i = 0, j;
    int cols_aligned8 = cols & ~7;
    
    for (; i < cols_aligned8; i += 8) {
        x_cache[i+0] = quantize_for_subias(_x[i+0]);
        x_cache[i+1] = quantize_for_subias(_x[i+1]);
        x_cache[i+2] = quantize_for_subias(_x[i+2]);
        x_cache[i+3] = quantize_for_subias(_x[i+3]);
        x_cache[i+4] = quantize_for_subias(_x[i+4]);
        x_cache[i+5] = quantize_for_subias(_x[i+5]);
        x_cache[i+6] = quantize_for_subias(_x[i+6]);
        x_cache[i+7] = quantize_for_subias(_x[i+7]);
    }
    for (; i < cols; i++) x_cache[i] = quantize_for_subias(_x[i]);
    
    for (i = 0; i < rows; i += 8) {
        int colblocks = *idx++;
        int32_t acc[8] = {0};
        
        if (UNLIKELY(colblocks > 4)) __builtin_prefetch(w + 128, 0, 0); 
        
        for (j = 0; j < colblocks; j++) {
            int pos = *idx++;
            compute_sparse_8x4_kernel(acc, w, &x_cache[pos]);
            w += 32; 
        }
        for (j = 0; j < 8; j++) out[i+j] = (float)acc[j] * scale[i+j];
    }
}

static inline IRAM_ATTR void cgemv8x4(float *out, const opus_int8 *w, const float *scale, int rows, int cols, const float *_x)
{
    int i = 0, j;
    int cols_aligned8 = cols & ~7;
    for (; i < cols_aligned8; i += 8) {
        x_cache[i+0] = quantize_for_subias(_x[i+0]);
        x_cache[i+1] = quantize_for_subias(_x[i+1]);
        x_cache[i+2] = quantize_for_subias(_x[i+2]);
        x_cache[i+3] = quantize_for_subias(_x[i+3]);
        x_cache[i+4] = quantize_for_subias(_x[i+4]);
        x_cache[i+5] = quantize_for_subias(_x[i+5]);
        x_cache[i+6] = quantize_for_subias(_x[i+6]);
        x_cache[i+7] = quantize_for_subias(_x[i+7]);
    }
    for (; i < cols; i++) x_cache[i] = quantize_for_subias(_x[i]);
    
    for (i = 0; i < rows; i += 8) {
        int32_t acc[8] = {0};
        for (j = 0; j < cols; j += 4) {
            compute_sparse_8x4_kernel(acc, w, &x_cache[j]);
            w += 32;
        }
        for (j = 0; j < 8; j++) out[i+j] = (float)acc[j] * scale[i+j];
    }
}
#else
// Standard version (no offset)
static IRAM_ATTR FORCE_INLINE uint8_t quantize_for_standard(float x) {
    int32_t val = lrintf(127.0f * x);
    if (val > 127) val = 127;
    else if (val < -128) val = -128;
    // Map [-128, 127] to [0, 255] for uint8_t storage
    return (uint8_t)(val + 128);
}

static inline IRAM_ATTR void sparse_cgemv8x4(float *out, const opus_int8 *w, const int *idx, const float *scale, int rows, int cols, const float *_x)
{
    int i = 0, j;
    int cols_aligned8 = cols & ~7;

    for (; i < cols_aligned8; i += 8) {
        x_cache[i+0] = quantize_for_standard(_x[i+0]);
        x_cache[i+1] = quantize_for_standard(_x[i+1]);
        x_cache[i+2] = quantize_for_standard(_x[i+2]);
        x_cache[i+3] = quantize_for_standard(_x[i+3]);
        x_cache[i+4] = quantize_for_standard(_x[i+4]);
        x_cache[i+5] = quantize_for_standard(_x[i+5]);
        x_cache[i+6] = quantize_for_standard(_x[i+6]);
        x_cache[i+7] = quantize_for_standard(_x[i+7]);
    }
    for (; i < cols; i++) x_cache[i] = quantize_for_standard(_x[i]);

    for (i = 0; i < rows; i += 8) {
        int colblocks = *idx++;
        int32_t acc[8] = {0};
        
        if (UNLIKELY(colblocks > 4)) __builtin_prefetch(w + 128, 0, 0); 
        
        for (j = 0; j < colblocks; j++) {
            int pos = *idx++;
            compute_sparse_8x4_kernel(acc, w, &x_cache[pos]);
            w += 32; 
        }
        // Adjust for the 128 offset applied during quantization
        for (j = 0; j < 8; j++) out[i+j] = (float)(acc[j] - (128 * colblocks * 4)) * scale[i+j];
    }
}

static inline IRAM_ATTR void cgemv8x4(float *out, const opus_int8 *w, const float *scale, int rows, int cols, const float *_x)
{
    int i = 0, j;
    int cols_aligned8 = cols & ~7;
    for (; i < cols_aligned8; i += 8) {
        x_cache[i+0] = quantize_for_standard(_x[i+0]);
        x_cache[i+1] = quantize_for_standard(_x[i+1]);
        x_cache[i+2] = quantize_for_standard(_x[i+2]);
        x_cache[i+3] = quantize_for_standard(_x[i+3]);
        x_cache[i+4] = quantize_for_standard(_x[i+4]);
        x_cache[i+5] = quantize_for_standard(_x[i+5]);
        x_cache[i+6] = quantize_for_standard(_x[i+6]);
        x_cache[i+7] = quantize_for_standard(_x[i+7]);
    }
    for (; i < cols; i++) x_cache[i] = quantize_for_standard(_x[i]);
    
    for (i = 0; i < rows; i += 8) {
        int32_t acc[8] = {0};
        int num_blocks = cols / 4;
        for (j = 0; j < cols; j += 4) {
            compute_sparse_8x4_kernel(acc, w, &x_cache[j]);
            w += 32;
        }
        // Adjust for the 128 offset applied during quantization
        for (j = 0; j < 8; j++) out[i+j] = (float)(acc[j] - (128 * num_blocks * 4)) * scale[i+j];
    }
}
#endif /* USE_SU_BIAS */

#ifndef LPCNET_TEST
static IRAM_ATTR FORCE_INLINE float lpcnet_exp2(float x)
{
   int integer;
   float frac;
   union
   {
      float f;
      opus_uint32 i;
   } res;
   integer = floor(x);
   if (integer < -50)
      return 0;
   frac = x - integer;
   /* K0 = 1, K1 = log(2), K2 = 3-4*log(2), K3 = 3*log(2) - 2 */
   res.f = 0.99992522f + frac * (0.69583354f + frac * (0.22606716f + 0.078024523f * frac));
   res.i = (res.i + (integer << 23)) & 0x7fffffff;
   return res.f;
}
#define lpcnet_exp(x) lpcnet_exp2((x) * 1.44269504f)

#define fmadd(a, b, c) ((a) * (b) + (c))
static IRAM_ATTR FORCE_INLINE float tanh_approx(float x)
{
   const float N0 = 952.52801514f;
   const float N1 = 96.39235687f;
   const float N2 = 0.60863042f;
   const float D0 = 952.72399902f;
   const float D1 = 413.36801147f;
   const float D2 = 11.88600922f;
   float X2, num, den;
   X2 = x * x;
   num = fmadd(fmadd(N2, X2, N1), X2, N0);
   den = fmadd(fmadd(D2, X2, D1), X2, D0);
   num = num * x / den;
   return MAX32(-1.f, MIN32(1.f, num));
}

static IRAM_ATTR FORCE_INLINE float sigmoid_approx(float x)
{
   return .5f + .5f * tanh_approx(.5f * x);
}

static inline IRAM_ATTR void softmax(float *y, const float *x, int N)
{
   int i;
   for (i = 0; i < N; i++)
      y[i] = lpcnet_exp(x[i]);
}

static inline IRAM_ATTR void vec_tanh(float *y, const float *x, int N)
{
   int i;
   for (i = 0; i < N; i++)
   {
      y[i] = tanh_approx(x[i]);
   }
}

static inline IRAM_ATTR void vec_sigmoid(float *y, const float *x, int N)
{
   int i;
   for (i = 0; i < N; i++)
   {
      y[i] = sigmoid_approx(x[i]);
   }
}
#endif /* LPCNET_TEST */

#define SCALE (128.f * 127.f)
#define SCALE_1 (1.f / 128.f / 127.f)

#endif /*no optimizations*/
#endif /*VEC_H*/
