
#include "helpers.h"
#include <float.h>
#include <limits.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

const int post_vlmul_shr = 14; //defined by the HW

int clrsb(int x){
  #if defined(__XS3A__)
  for (unsigned i=0;i<32;i++){
    int y = (x<<i)>>i;
    if (y != x)
      return (i-1);
  }
  return 32;
  #else
  return __builtin_clrsb(x);
  #endif
}
int clrsbll(long long x){
  #if defined(__XS3A__)
  for (unsigned i=0;i<64;i++){
    long long y = (x<<i)>>i;
    if (y != x)
    return (i-1);
  }
  return 64;
  #else
  return __builtin_clrsbll(x);
  #endif
}

int32_t ashr(int32_t x, int shr){
  if (shr == 0)
    return x;

  if (shr > 0){
    int32_t rounding = (1 << (shr-1));
    return (x + rounding) >> shr;
  } else
    return x << (-shr);
}

int32_t mul(int32_t x, int32_t m){
  int64_t t = (int64_t)x * (int64_t)m;
  if(t > INT32_MAX) t = INT32_MAX;
  if(t < INT32_MIN) t = INT32_MIN;
  return (int32_t)t; 
}

//TODO rename this make post activations
//TODO pass the clamps
void make_thresholds(float * post_activation_multiplier, float * post_activation_bias, 
  unsigned chans_out, unsigned receptive_volume, int seed){

/*
  std::int8_t Run(const std::int32_t accum, int out_channel) const {
    // Clamping is done in int32
    std::int32_t x = accum << 1;
    x = std::max<std::int32_t>(std::min<std::int32_t>(x, clamp_max), clamp_min);
    // The linear transformation is done in float
    float y =
        static_cast<float>(x) * multiplier[out_channel] + bias[out_channel];
    // And then we round back to int32 and clamp to the int8 range
    return saturate(round(y));
  }
*/
  srand(seed);

  //The input range is from 0 to the receptive_volume (xor_popcount)
  float accu_min = 0; 
  float accu_max = receptive_volume*2; //the times 2 is due to the left shift in the output transform. 

  float input_range = accu_max - accu_min;

  float output_min = (float)INT8_MIN; 
  float output_max = (float)INT8_MAX; 

  for (unsigned ch = 0; ch < chans_out; ch++){

    unsigned range = rand()%receptive_volume;

    // Scale the input to extend beyond the range of the output such that we get some 
    // outputs that will saturate.
    float output_overscale = 0.5 + (float)rand()/(float)RAND_MAX;

    float output_range = (output_max - output_min)*output_overscale;

    // This offset allows the output range to completly miss the int8 output range sometimes
    float offset = 3.1 * output_range * (float)rand()/(float)RAND_MAX;

    post_activation_multiplier[ch] = output_range / input_range;
    post_activation_bias[ch] = output_min*output_overscale - accu_min* output_range / input_range + offset;
  }
}

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
               error_stats_t * results){

  int test_error_sum = 0;
  unsigned test_abs_error_sum = 0;
  unsigned count = 0;
  memset(results, 0, sizeof(error_stats_t));

  for (unsigned ch=0;ch<chans_out;ch++){

    float PAM = post_activation_multiplier[ch];
    float Bias = post_activation_bias[ch];

    for (int32_t accu_output = 0; accu_output <= receptive_field; accu_output++){

      int32_t vpu_output = (receptive_field - (2*accu_output))/2;

      //This is how the reference is defined
      float clamped_accu = min(max(accu_output * 2., clamp_low), clamp_high);
      int R = round((clamped_accu * PAM) + Bias); 
      R = max(min(R, INT8_MAX), INT8_MIN);

      //pretend clamping
      int32_t r = ashr(vpu_output, accu_shr);
      r = mul(r, post_activation_multiplier_q[ch]);

      r = ashr(r, post_vlmul_shr);
      assert (clrsb(r) >= 16);
      r += post_activation_bias_q[ch];

      r = ashr(r, final_shr);
      r = r&0xffffff00;
      r = r>>8;
      r = max(min(r, INT8_MAX), INT8_MIN);

      int error = r - R;
      unsigned abs_error = abs(error);

      results->output_error[R - INT8_MIN] += error;
      results->abs_output_error[R - INT8_MIN] += abs_error;
      results->error_counter[R - INT8_MIN] += 1;

      assert(abs_error <= 1);

      test_error_sum += error;
      test_abs_error_sum += abs_error;
      count += 1;
    }
  }
}
