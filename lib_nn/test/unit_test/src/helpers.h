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
int32_t ashr(int32_t x, int shr);
int32_t mul(int32_t x, int32_t m);

typedef struct {
  int output_error[256];
  unsigned abs_output_error[256];
  unsigned error_counter[256];
} error_stats_t;

int get_accumulator_ashr(int32_t max_accu_post_clamp, int32_t min_accu_post_clamp, 
  int16_t max_quantised_pam, int16_t min_quantised_pam, unsigned post_vlmul_shr);
  
int get_pam_exponent(float* post_activation_multiplier, unsigned chans_out);

void quantise_activation(
               int16_t * post_activation_multiplier_q,
               int16_t* post_activation_bias_q,
               float* post_activation_multiplier,
               float* post_activation_bias, 
               unsigned chans_out,
               int32_t clamp_low,
               int32_t clamp_high,
               int *accu_shr,
               int *final_shr, 
               int32_t receptive_field, int * chan_overlaps);

void measure_quantisation(
               int16_t * post_activation_multiplier_q,
               int16_t* post_activation_bias_q,
               float* post_activation_multiplier,
               float* post_activation_bias, 
               unsigned chans_out,
               int32_t clamp_low,
               int32_t clamp_high,
               int accu_shr,
               int final_shr,
               int32_t receptive_field,
               error_stats_t * results);
#endif //HELPERS_H