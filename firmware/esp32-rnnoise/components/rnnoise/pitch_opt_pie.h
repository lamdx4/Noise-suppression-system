#ifndef PITCH_OPT_PIE_H
#define PITCH_OPT_PIE_H

#include <stdint.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations for RNNoise types
#ifndef OPUS_TYPES_H
typedef float opus_val16;
typedef float opus_val32;
#endif

/**
 * Optimized Pitch Correlation using ESP32-S3 PIE SIMD.
 * This version uses a unique "magic" name to guarantee no link collisions.
 */
void compute_pitch_xcorr_pie_magic(const opus_val16 *_x, const opus_val16 *_y,
                                  opus_val32 *xcorr, int len, int max_pitch);

#ifdef __cplusplus
}
#endif

#endif // PITCH_OPT_PIE_H
