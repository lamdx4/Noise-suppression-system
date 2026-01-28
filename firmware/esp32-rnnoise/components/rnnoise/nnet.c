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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "arch.h"
#include "common.h"
#include "nnet.h"
#include "opus_types.h"
#include "vec.h"
#include <esp_attr.h>
#include <math.h>
#include <stdlib.h>

#ifdef ENABLE_OSCE
#include "osce.h"
#endif

#ifdef NO_OPTIMIZATIONS
#if defined(_MSC_VER)
#pragma message(                                                               \
    "Compiling without any vectorization. This code will be very slow")
#else
#warning Compiling without any vectorization. This code will be very slow
#endif
#endif

#define SOFTMAX_HACK

void IRAM_ATTR compute_generic_dense(const LinearLayer *layer, float *output,
                                     const float *input, int activation,
                                     int arch) {
  compute_linear(layer, output, input, arch);
  compute_activation(output, output, layer->nb_outputs, activation, arch);
}

// ...

void IRAM_ATTR compute_generic_gru(const LinearLayer *input_weights,
                                   const LinearLayer *recurrent_weights,
                                   float *state, const float *in, int arch) {
  // ... implementation
}

void IRAM_ATTR compute_glu(const LinearLayer *layer, float *output,
                           const float *input, int arch) {
  // ... implementation
}

void IRAM_ATTR compute_generic_conv1d(const LinearLayer *layer, float *output,
                                      float *mem, const float *input,
                                      int input_size, int activation,
                                      int arch) {
  // ... implementation
}
