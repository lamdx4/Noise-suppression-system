/* Copyright (c) 2018 Mozilla
                 2008-2011 Octasic Inc.
                 2012-2017 Jean-Marc Valin */
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

#ifndef VEC_H
#define VEC_H

#include "opus_types.h"
#include <math.h>
#include "arch.h"
// ESP32 PORT: Hardcoded x86 include removed
// #include "x86/x86_arch_macros.h"

// ESP32 PORT: Force generic C implementation, disable x86/ARM SIMD
#if 0  // was: defined(__AVX__) || defined(__SSE2__)
#include "vec_avx.h"
#elif 0  // was: (defined(__ARM_NEON__) || defined(__ARM_NEON)) && !defined(DISABLE_NEON)
#include "vec_neon.h"
#else

#include "os_support.h"

#define MAX_INPUTS (2048)

#define NO_OPTIMIZATIONS

static inline void sgemv16x1(float *out, const float *weights, int rows, int cols, int col_stride, const float *x)
{
   int i, j;
   OPUS_CLEAR(out, rows);
   for (i=0;i<rows;i+=16)
   {
      for (j=0;j<cols;j++)
      {
         const float * restrict w;
         float * restrict y;
         float xj;
         w = &weights[j*col_stride + i];
         xj = x[j];
         y = &out[i];
         y[0] += w[0]*xj;
         y[1] += w[1]*xj;
         y[2] += w[2]*xj;
         y[3] += w[3]*xj;
         y[4] += w[4]*xj;
         y[5] += w[5]*xj;
         y[6] += w[6]*xj;
         y[7] += w[7]*xj;
         y[8] += w[8]*xj;
         y[9] += w[9]*xj;
         y[10] += w[10]*xj;
         y[11] += w[11]*xj;
         y[12] += w[12]*xj;
         y[13] += w[13]*xj;
         y[14] += w[14]*xj;
         y[15] += w[15]*xj;
      }
   }
}

static inline void sgemv8x1(float *out, const float *weights, int rows, int cols, int col_stride, const float *x)
{
   int i, j;
   OPUS_CLEAR(out, rows);
   for (i=0;i<rows;i+=8)
   {
      for (j=0;j<cols;j++)
      {
         const float * restrict w;
         float * restrict y;
         float xj;
         w = &weights[j*col_stride + i];
         xj = x[j];
         y = &out[i];
         y[0] += w[0]*xj;
         y[1] += w[1]*xj;
         y[2] += w[2]*xj;
         y[3] += w[3]*xj;
         y[4] += w[4]*xj;
         y[5] += w[5]*xj;
         y[6] += w[6]*xj;
         y[7] += w[7]*xj;
      }
   }
}

static inline void sgemv(float *out, const float *weights, int rows, int cols, int col_stride, const float *x)
{
   if ((rows&0xf) == 0) sgemv16x1(out, weights, rows, cols, col_stride, x);
   else if ((rows&0x7) == 0) sgemv8x1(out, weights, rows, cols, col_stride, x);
   else {
      int i, j;
      for (i=0;i<rows;i++)
      {
         out[i] = 0;
         for (j=0;j<cols;j++) out[i] += weights[j*col_stride + i]*x[j];
      }
   }
}

static inline void sparse_sgemv8x4(float *out, const float *w, const int *idx, int rows, const float *x)
{
   int i, j;
   OPUS_CLEAR(out, rows);
   for (i=0;i<rows;i+=8)
   {
      int cols;
      cols = *idx++;
      for (j=0;j<cols;j++)
      {
         int pos;
         float * restrict y;
         float xj0, xj1, xj2, xj3;
         pos = (*idx++);
         xj0 = x[pos+0];
         xj1 = x[pos+1];
         xj2 = x[pos+2];
         xj3 = x[pos+3];
         y = &out[i];
         y[0] += w[0]*xj0;
         y[1] += w[1]*xj0;
         y[2] += w[2]*xj0;
         y[3] += w[3]*xj0;
         y[4] += w[4]*xj0;
         y[5] += w[5]*xj0;
         y[6] += w[6]*xj0;
         y[7] += w[7]*xj0;

         y[0] += w[8]*xj1;
         y[1] += w[9]*xj1;
         y[2] += w[10]*xj1;
         y[3] += w[11]*xj1;
         y[4] += w[12]*xj1;
         y[5] += w[13]*xj1;
         y[6] += w[14]*xj1;
         y[7] += w[15]*xj1;

         y[0] += w[16]*xj2;
         y[1] += w[17]*xj2;
         y[2] += w[18]*xj2;
         y[3] += w[19]*xj2;
         y[4] += w[20]*xj2;
         y[5] += w[21]*xj2;
         y[6] += w[22]*xj2;
         y[7] += w[23]*xj2;

         y[0] += w[24]*xj3;
         y[1] += w[25]*xj3;
         y[2] += w[26]*xj3;
         y[3] += w[27]*xj3;
         y[4] += w[28]*xj3;
         y[5] += w[29]*xj3;
         y[6] += w[30]*xj3;
         y[7] += w[31]*xj3;
         w += 32;
      }
   }
}

#ifdef USE_SU_BIAS

static inline void sparse_cgemv8x4(float *out, const opus_int8 *w, const int *idx, const float *scale, int rows, int cols, const float *_x)
{
   int i, j;
   unsigned char x[MAX_INPUTS];
   for (i=0;i<rows;i++) out[i] = 0;
   
   // FIX: Use lrintf for hardware-accelerated rounding (Xtensa FPU)
   // Step 3: Unroll quantization loop (4x)
   int remainder = cols & 3;
   for (i=0; i<cols-3; i+=4) {
      x[i+0] = (unsigned char)(127 + lrintf(127.0f * _x[i+0]));
      x[i+1] = (unsigned char)(127 + lrintf(127.0f * _x[i+1]));
      x[i+2] = (unsigned char)(127 + lrintf(127.0f * _x[i+2]));
      x[i+3] = (unsigned char)(127 + lrintf(127.0f * _x[i+3]));
   }
   for (; i<cols; i++) {
       x[i] = (unsigned char)(127 + lrintf(127.0f * _x[i]));
   }
   
   for (i=0;i<rows;i+=8)
   {
      int colblocks;
      colblocks = *idx++;
      for (j=0;j<colblocks;j++)
      {
         int pos;
         float * restrict y;
         int xj0, xj1, xj2, xj3;
         pos = (*idx++);
         xj0 = x[pos+0];
         xj1 = x[pos+1];
         xj2 = x[pos+2];
         xj3 = x[pos+3];
         y = &out[i];
         y[0] += (w[0]*xj0+w[1]*xj1+w[2]*xj2+w[3]*xj3);
         y[1] += (w[4]*xj0+w[5]*xj1+w[6]*xj2+w[7]*xj3);
         y[2] += (w[8]*xj0+w[9]*xj1+w[10]*xj2+w[11]*xj3);
         y[3] += (w[12]*xj0+w[13]*xj1+w[14]*xj2+w[15]*xj3);
         y[4] += (w[16]*xj0+w[17]*xj1+w[18]*xj2+w[19]*xj3);
         y[5] += (w[20]*xj0+w[21]*xj1+w[22]*xj2+w[23]*xj3);
         y[6] += (w[24]*xj0+w[25]*xj1+w[26]*xj2+w[27]*xj3);
         y[7] += (w[28]*xj0+w[29]*xj1+w[30]*xj2+w[31]*xj3);
         w += 32;
      }
   }
   for (i=0;i<rows;i++) out[i] *= scale[i];
}
static inline void cgemv8x4(float *out, const opus_int8 *w, const float *scale, int rows, int cols, const float *_x)
{
   int i, j;
   unsigned char x[MAX_INPUTS];
   for (i=0;i<rows;i++) out[i] = 0;
   for (i=0;i<cols;i++) x[i] = 127+(int)floor(.5+127*_x[i]);
   for (i=0;i<rows;i+=8)
   {
      for (j=0;j<cols;j+=4)
      {
         float *y;
         float xj0, xj1, xj2, xj3;
         xj0 = x[j+0];
         xj1 = x[j+1];
         xj2 = x[j+2];
         xj3 = x[j+3];
         y = &out[i];
         y[0] += (w[0]*xj0+w[1]*xj1+w[2]*xj2+w[3]*xj3);
         y[1] += (w[4]*xj0+w[5]*xj1+w[6]*xj2+w[7]*xj3);
         y[2] += (w[8]*xj0+w[9]*xj1+w[10]*xj2+w[11]*xj3);
         y[3] += (w[12]*xj0+w[13]*xj1+w[14]*xj2+w[15]*xj3);
         y[4] += (w[16]*xj0+w[17]*xj1+w[18]*xj2+w[19]*xj3);
         y[5] += (w[20]*xj0+w[21]*xj1+w[22]*xj2+w[23]*xj3);
         y[6] += (w[24]*xj0+w[25]*xj1+w[26]*xj2+w[27]*xj3);
         y[7] += (w[28]*xj0+w[29]*xj1+w[30]*xj2+w[31]*xj3);
         w += 32;
      }
   }
   for (i=0;i<rows;i++) out[i] *= scale[i];
}
#else
static inline void sparse_cgemv8x4(float *out, const opus_int8 *w, const int *idx, const float *scale, int rows, int cols, const float *_x)
{
   int i, j;
   opus_int8 x[MAX_INPUTS];
   
   // Quantization (Optimize with lrintf)
   for (i=0;i<cols;i++) x[i] = (opus_int8)lrintf(127.0f * _x[i]);
   
   for (i=0;i<rows;i+=8)
   {
      int colblocks;
      int32_t acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      
      colblocks = *idx++;
      for (j=0;j<colblocks;j++)
      {
         int pos;
         int xj0, xj1, xj2, xj3;
         pos = (*idx++);
         xj0 = x[pos+0];
         xj1 = x[pos+1];
         xj2 = x[pos+2];
         xj3 = x[pos+3];
         
         acc[0] += (w[0]*xj0+w[1]*xj1+w[2]*xj2+w[3]*xj3);
         acc[1] += (w[4]*xj0+w[5]*xj1+w[6]*xj2+w[7]*xj3);
         acc[2] += (w[8]*xj0+w[9]*xj1+w[10]*xj2+w[11]*xj3);
         acc[3] += (w[12]*xj0+w[13]*xj1+w[14]*xj2+w[15]*xj3);
         acc[4] += (w[16]*xj0+w[17]*xj1+w[18]*xj2+w[19]*xj3);
         acc[5] += (w[20]*xj0+w[21]*xj1+w[22]*xj2+w[23]*xj3);
         acc[6] += (w[24]*xj0+w[25]*xj1+w[26]*xj2+w[27]*xj3);
         acc[7] += (w[28]*xj0+w[29]*xj1+w[30]*xj2+w[31]*xj3);
         w += 32;
      }
      out[i+0] = (float)acc[0] * scale[i+0];
      out[i+1] = (float)acc[1] * scale[i+1];
      out[i+2] = (float)acc[2] * scale[i+2];
      out[i+3] = (float)acc[3] * scale[i+3];
      out[i+4] = (float)acc[4] * scale[i+4];
      out[i+5] = (float)acc[5] * scale[i+5];
      out[i+6] = (float)acc[6] * scale[i+6];
      out[i+7] = (float)acc[7] * scale[i+7];
   }
}
static inline void cgemv8x4(float *out, const opus_int8 *w, const float *scale, int rows, int cols, const float *_x)
{
   int i, j;
   opus_int8 x[MAX_INPUTS];
   
   for (i=0; i<rows; i++) out[i] = 0;
   
   // Quantization
   for (i=0; i<cols; i++) {
      float tmp = _x[i] * 127.0f;
      x[i] = (opus_int8)(tmp + (tmp >= 0 ? 0.5f : -0.5f));
   }
   
   // INT32 accumulation
   for (i=0; i<rows; i+=8)
   {
      int32_t acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      
      for (j=0; j<cols; j+=4)
      {
         int xj0 = x[j+0];
         int xj1 = x[j+1];
         int xj2 = x[j+2];
         int xj3 = x[j+3];
         
         acc[0] += w[0]*xj0 + w[1]*xj1 + w[2]*xj2 + w[3]*xj3;
         acc[1] += w[4]*xj0 + w[5]*xj1 + w[6]*xj2 + w[7]*xj3;
         acc[2] += w[8]*xj0 + w[9]*xj1 + w[10]*xj2 + w[11]*xj3;
         acc[3] += w[12]*xj0 + w[13]*xj1 + w[14]*xj2 + w[15]*xj3;
         acc[4] += w[16]*xj0 + w[17]*xj1 + w[18]*xj2 + w[19]*xj3;
         acc[5] += w[20]*xj0 + w[21]*xj1 + w[22]*xj2 + w[23]*xj3;
         acc[6] += w[24]*xj0 + w[25]*xj1 + w[26]*xj2 + w[27]*xj3;
         acc[7] += w[28]*xj0 + w[29]*xj1 + w[30]*xj2 + w[31]*xj3;
         w += 32;
      }
      
      // ✅ ĐÚNG: Chỉ nhân scale (đã bao gồm /127)
      for (j=0; j<8; j++) {
         out[i+j] = (float)acc[j] * scale[i+j];
      }
   }
}
#endif

/* No AVX2/FMA support */
#ifndef LPCNET_TEST
static inline float lpcnet_exp2(float x)
{
   int integer;
   float frac;
   union {
      float f;
      opus_uint32 i;
   } res;
   integer = floor(x);
   if (integer < -50)
      return 0;
   frac = x-integer;
   /* K0 = 1, K1 = log(2), K2 = 3-4*log(2), K3 = 3*log(2) - 2 */
   res.f = 0.99992522f + frac * (0.69583354f
           + frac * (0.22606716f + 0.078024523f*frac));
   res.i = (res.i + (integer<<23)) & 0x7fffffff;
   return res.f;
}
#define lpcnet_exp(x) lpcnet_exp2((x)*1.44269504f)

#define fmadd(a, b, c) ((a)*(b)+(c))
static OPUS_INLINE float tanh_approx(float x)
{
    const float N0 = 952.52801514f;
    const float N1 = 96.39235687f;
    const float N2 = 0.60863042f;
    const float D0 = 952.72399902f;
    const float D1 = 413.36801147f;
    const float D2 = 11.88600922f;
    float X2, num, den;
    X2 = x*x;
    num = fmadd(fmadd(N2, X2, N1), X2, N0);
    den = fmadd(fmadd(D2, X2, D1), X2, D0);
    num = num*x/den;
    return MAX32(-1.f, MIN32(1.f, num));
}

static inline float sigmoid_approx(float x)
{
   return .5f + .5f*tanh_approx(.5f*x);
}

static inline void softmax(float *y, const float *x, int N)
{
    int i;
    for (i=0;i<N;i++)
        y[i] = lpcnet_exp(x[i]);
}

static inline void vec_tanh(float *y, const float *x, int N)
{
    int i;
    for (i=0;i<N;i++)
    {
        y[i] = tanh_approx(x[i]);
    }
}

static inline void vec_sigmoid(float *y, const float *x, int N)
{
    int i;
    for (i=0;i<N;i++)
    {
        y[i] = sigmoid_approx(x[i]);
    }
}
#endif

#define SCALE (128.f*127.f)
#define SCALE_1 (1.f/128.f/127.f)

#endif /*no optimizations*/
#endif /*VEC_H*/
