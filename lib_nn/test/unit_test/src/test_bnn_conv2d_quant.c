#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "tst_common.h"
#include "unity.h"

#include "helpers.h"

static int64_t vpu_saturate(
    const int64_t input,
    const unsigned bits)
{
    const int64_t max_val = (((int64_t)1)<<(bits-1))-1;
    const int64_t min_val = -max_val;
    
    return (input > max_val)?  max_val : (input < min_val)? min_val : input;
}

static int64_t ashr(int64_t value, int shr){
  if(shr > 0){
    return (value + (1 << (shr-1))) >> shr;
  } else {
    return value << (-shr);
  }
}

static void measure_quantisation(
               int16_t * post_activation_multiplier_q,
               int16_t* post_activation_bias_q,

               float* post_activation_multiplier,
               float* post_activation_bias, 

               unsigned chans_out,

               int32_t clamp_low,
               int32_t clamp_high,

               int accu_shr,
               int16_t bias_multipler,
               int final_shr,

               int32_t receptive_volume, 

               float * error_sum,
               float * abs_error_sum,
               unsigned * sum_count 
){

  for(unsigned ch=0;ch < chans_out;ch++){

    //Iterate over all possible VPU accumulator outputs 
    for(int32_t vpu_acc=0; vpu_acc<receptive_volume; vpu_acc++){

      //convert to larq accu space
      float larq_accu = -(float)(vpu_acc) + (float)receptive_volume/2.0;
      float r = post_activation_multiplier[ch] * 2.0 * larq_accu + post_activation_bias[ch];
      float ref_output = fmin(fmax(r, (double)INT8_MIN), (double)INT8_MAX);

      //asm implementation
      int64_t scaled_accu = vpu_saturate(ashr(vpu_acc, accu_shr), 16);
      int64_t bias = vpu_saturate((int64_t)post_activation_bias_q[ch] * (int64_t)bias_multipler, 32);
      int64_t product = vpu_saturate((int64_t)post_activation_multiplier_q[ch] * (int64_t)scaled_accu + bias, 32);
      int64_t product_shr = vpu_saturate(ashr(product, final_shr), 16);
      int8_t output = vpu_saturate(ashr(product_shr, 8), 8);

      float error = ref_output - (float)output;
      *error_sum += error;
      *abs_error_sum += fabs(error);
      *sum_count += 1;

      TEST_ASSERT_TRUE(fabs(error) <= 1.0);
    }
  }
}


void test_quantisation(){

  float error_sum = 0.0;
  float abs_error_sum = 0.0;
  unsigned sum_count = 0;

  int seed = 0;
  for(unsigned k_dim=1;k_dim < 7;k_dim += 2){
    for(unsigned chans_in=32; chans_in< 32*8; chans_in+=32){
      for(unsigned chans_out=4; chans_out < 1024; chans_out+=4){

        unsigned receptive_volume = k_dim*chans_in; 

        int32_t clamp_low = 0;
        int32_t clamp_high = receptive_volume*2;
        
        int16_t *post_activation_multiplier_q = (int16_t *)malloc(sizeof(int16_t)*(chans_out+(16 - chans_out%16)));
        int16_t *post_activation_bias_q = (int16_t *)malloc(sizeof(int16_t)*(chans_out+(16 - chans_out%16)));

        float * post_activation_multiplier = (float *)malloc(sizeof(float)*chans_out);
        float * post_activation_bias = (float *)malloc(sizeof(float)*chans_out);
        int * chan_overlaps = (int *)malloc(sizeof(int)*(chans_out));
        memset(chan_overlaps, 0, sizeof(int)*(chans_out));

        pick_post_activation_params(post_activation_multiplier, post_activation_bias, chans_out, receptive_volume, &seed);

        int16_t bias_multipler;
        int accu_shr, final_shr;

        bnn_quantise_activation(
            post_activation_multiplier_q,
            post_activation_bias_q,

            post_activation_multiplier,
            post_activation_bias, 

            chans_out,

            clamp_low, clamp_high,
            &accu_shr, &bias_multipler, &final_shr, receptive_volume, chan_overlaps
        );

        measure_quantisation(
            post_activation_multiplier_q,
            post_activation_bias_q,

            post_activation_multiplier,
            post_activation_bias, 

            chans_out,

            clamp_low, clamp_high,
            accu_shr, bias_multipler, final_shr, receptive_volume, 

            &error_sum, &abs_error_sum, &sum_count);
      }
    }
  }
  printf("%f %f %d\n", error_sum / sum_count, abs_error_sum / sum_count, sum_count); 
}

void test_bnn_conv2d_quant() {
  UNITY_SET_FILE();
  RUN_TEST(test_quantisation);
}
