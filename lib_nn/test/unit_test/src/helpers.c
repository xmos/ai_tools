
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


int get_accumulator_ashr(int32_t max_accu_post_clamp, int32_t min_accu_post_clamp, 
  int16_t max_quantised_pam, int16_t min_quantised_pam, unsigned post_vlmul_shr){

 // If the abs max accu multiplied by the abs max pam has a bit of headroom then remove it from the accu

  int64_t max_max = (int64_t)max_accu_post_clamp * (int64_t)max_quantised_pam;
  int64_t max_min = (int64_t)max_accu_post_clamp * (int64_t)min_quantised_pam;
  int64_t min_max = (int64_t)min_accu_post_clamp * (int64_t)max_quantised_pam;
  int64_t min_min = (int64_t)min_accu_post_clamp * (int64_t)min_quantised_pam;

  int max_max_rsb = clrsbll(max_max);
  int max_min_rsb = clrsbll(max_min);
  int min_max_rsb = clrsbll(min_max);
  int min_min_rsb = clrsbll(min_min);

  int min_rsb = min(max_max_rsb, min(max_min_rsb, min(min_max_rsb, min_min_rsb)));

  // This defines the maximum amount we are alowed to shift the accu left by and keep it within the 16 bit register
  int max_accu_post_clamp_rsb = clrsb(max_accu_post_clamp) - 16;
  int min_accu_post_clamp_rsb = clrsb(min_accu_post_clamp) - 16;

  int max_shl = min(max_accu_post_clamp_rsb, min_accu_post_clamp_rsb);

  int t = min_rsb - 32 - (16 - post_vlmul_shr);
  int accu_shr = -min(t, max_shl);

  //test for rounding overflow
  int32_t max_accu_shr = ashr(max_accu_post_clamp, accu_shr);
  int32_t min_accu_shr = ashr(min_accu_post_clamp, accu_shr);

  if (clrsb(ashr(max_accu_shr * max_quantised_pam, post_vlmul_shr)) < 16)
    accu_shr += 1;  
  if (clrsb(ashr(max_accu_shr * min_quantised_pam, post_vlmul_shr)) < 16)
    accu_shr += 1;  
  if (clrsb(ashr(min_accu_shr * max_quantised_pam, post_vlmul_shr)) < 16)
    accu_shr += 1;  
  if (clrsb(ashr(min_accu_shr * min_quantised_pam, post_vlmul_shr)) < 16)
    accu_shr += 1;  

  assert(clrsb(ashr(max_accu_post_clamp, accu_shr)) >=16);
  assert(clrsb(ashr(min_accu_post_clamp, accu_shr)) >=16);
  return accu_shr;
}

int get_pam_exponent(float* post_activation_multiplier, unsigned chans_out){
  float abs_max_pam = -FLT_MAX;
  float min_pam = FLT_MAX;
  float max_pam = -FLT_MAX;
  for (unsigned ch=0; ch < chans_out; ch++){
    abs_max_pam = max(abs_max_pam, fabs(post_activation_multiplier[ch]));
    min_pam = min(min_pam, post_activation_multiplier[ch]);
    max_pam = max(max_pam, post_activation_multiplier[ch]);
  }

  int abs_max_pam_exp;
  frexp(abs_max_pam, &abs_max_pam_exp);

  // Raise any multiplier to b to get them as big as possible - 
  // this should be possible without the loop
  int M = 15 - abs_max_pam_exp;

  while ( (((int32_t)round(ldexp(min_pam, M))) > INT16_MAX) || 
          (((int32_t)round(ldexp(min_pam, M))) < INT16_MIN) || 
          (((int32_t)round(ldexp(max_pam, M))) > INT16_MAX) || 
          (((int32_t)round(ldexp(max_pam, M))) < INT16_MIN)
          ){
    M -= 1;
  }
  
  return M;
}

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
               int32_t receptive_field
               ){

  int32_t vpu_clamp_low = max(((2*clamp_low)-receptive_field)/2, ((2*0)-receptive_field)/2);
  int32_t vpu_clamp_high = min(((2*clamp_high)-receptive_field)/2, ((2*receptive_field)-receptive_field)/2);

  float * pam = (float *)malloc(sizeof(float) * chans_out);
  float * pab = (float *)malloc(sizeof(float) * chans_out);

  for (unsigned ch=0;ch<chans_out;ch++){
    pam[ch] = post_activation_multiplier[ch] * -2;
  }
  for (unsigned ch=0;ch<chans_out;ch++){
    pab[ch] = post_activation_bias[ch] + post_activation_multiplier[ch]*(float)receptive_field;
  }

  int M = get_pam_exponent(pam, chans_out);

  int16_t max_quantised_pam = INT16_MIN;
  int16_t min_quantised_pam = INT16_MAX;
  int min_pam_rsb = INT_MAX;
  for (unsigned ch=0;ch<chans_out;ch++){
    int16_t pa_mul = (int16_t)round(ldexp(pam[ch], M));

    post_activation_multiplier_q[ch] = pa_mul;
    
    max_quantised_pam = max(max_quantised_pam, pa_mul);
    min_quantised_pam = min(min_quantised_pam, pa_mul);

    //Check that there is the required amount of headroom in the pa_multipliers
    int rsb = clrsb(post_activation_multiplier_q[ch]) - 16;
    assert (rsb >= 0);
    if(rsb < min_pam_rsb)
      min_pam_rsb = rsb;
  }
  
  //There should be at least one multiplier that has zero headroom.
  // assert(min_pam_rsb == 0);
  //This can fail in the case of pam*exp = -32769 and pam*(exp-1) =  -16384

  float min_pab = FLT_MAX;
  float max_pab = -FLT_MAX;
  for (unsigned ch=0;ch<chans_out;ch++){
    min_pab = min(min_pab, pab[ch]);
    max_pab = max(min_pab, pab[ch]);
  }

  // Now find the accu shift (the value that the accu is shifted by to get the most resolution out of the pam). 
  // This will get the accu to occupy the bottom 15 or 16 bits of a 16 bit register.
  *accu_shr = get_accumulator_ashr(vpu_clamp_low, vpu_clamp_high, 
    max_quantised_pam, min_quantised_pam, post_vlmul_shr);

  *final_shr = (-*accu_shr + M - post_vlmul_shr);

  int success = 0;

  while(success == 0){
    success = 1;
    //Now quantise the biases
    for (unsigned ch=0;ch<chans_out;ch++){

      int32_t pa_bias = (int32_t)round(ldexp(pab[ch], *final_shr));
      
      pa_bias += (1 << (*final_shr - 1));
      //This bit is a hack to account for the bias causing overflow
      if((pa_bias > INT16_MAX) || (pa_bias < INT16_MIN)){
        success = 0;
        *accu_shr = *accu_shr + 1;
        *final_shr = (-*accu_shr + M - post_vlmul_shr);
        break;
      }

      post_activation_bias_q[ch] = (int16_t)pa_bias;
    }
  }
  free(pam);
  free(pab);
  //adjust it to reflect we are going to shift up to the upper half word( top 8 bits of 16)
  *final_shr = *final_shr-8;
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
