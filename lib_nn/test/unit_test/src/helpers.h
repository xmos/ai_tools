#ifndef HELPERS_H
#define HELPERS_H

#include "nn_operator.h"

extern const int post_vlmul_shr;

void larq_ref_bconv2d_bin_out(const nn_image_params_t* x, const nn_image_params_t* y,
                      const nn_window_params_t* k,
                      const int32_t* packed_input_data,
                      const int32_t* packed_filter_data,
                      int32_t* packed_output_data, const int32_t* thresholds);

void larq_ref_bconv2d_int8_out(const nn_image_params_t* x, const nn_image_params_t* y,
                      const nn_window_params_t* k,
                      const int32_t* packed_input_data,
                      const int32_t* packed_filter_data,
                      int8_t* output_data,
                      const float* post_activation_multiplier, 
                      const float* post_activation_bias );
int pseudo_rand(int *seed);
void pick_post_activation_params(float * post_activation_multiplier, float * post_activation_bias, 
  unsigned chans_out, unsigned receptive_volume, int * seed);

#define DIV_BY_AND_ROUND_UP(x, y) (((x) + (y) - 1) / (y))

#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

int clrsb(int x);
int clrsbll(long long x);

#endif //HELPERS_H