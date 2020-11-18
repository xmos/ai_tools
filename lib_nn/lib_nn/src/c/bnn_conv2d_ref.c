
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>

#include "nn_operator.h"
#include "../nn_op_helper.h"

#include <limits.h>
#include <math.h>

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
    return (unsigned)value << (-shr);
  }
}

int8_t bnn_post_activation_reference(
              const int32_t vpu_acc,
              const unsigned ch,
              const int16_t * post_activation_multiplier_q,
              const int16_t* post_activation_bias_q,
              const int accu_shr,
              const int16_t bias_multipler,
              const int final_shr){

  int64_t scaled_accu = vpu_saturate(ashr(vpu_acc, accu_shr), 16);
  int64_t bias = vpu_saturate((int64_t)post_activation_bias_q[ch] * (int64_t)bias_multipler, 32);
  int64_t product = vpu_saturate((int64_t)post_activation_multiplier_q[ch] * (int64_t)scaled_accu + bias, 32);
  int64_t product_shr = vpu_saturate(ashr(product, final_shr), 16);

  int64_t output = ashr(product_shr, 8);
  if (output < INT8_MIN) output = INT8_MIN;
  if (output > INT8_MAX) output = INT8_MAX;

  return (int8_t) output;
}

static int clrsb(int x){
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

static void solve_constraint(
    int *B_res, int *A_res, int *M_res,
    float* post_activation_multiplier,
    float* post_activation_bias, 
    unsigned chans_out,
    int max_accu, int min_accu){

  int max_pab_exp = INT_MIN;
  int max_pam_exp = INT_MIN;
  int max_accu_exp = INT_MIN;

  for (unsigned ch=0;ch<chans_out; ch++){
    int exponent;
    frexp(post_activation_bias[ch], &exponent);
    if(exponent > max_pab_exp)
      max_pab_exp = exponent;
    frexp(post_activation_multiplier[ch], &exponent);
    if(exponent > max_pam_exp)
      max_pam_exp = exponent;
  }

  int exponent;
  frexp((float)max_accu, &max_accu_exp);
  frexp((float)min_accu, &exponent);
  if(exponent > max_accu_exp)
    max_accu_exp = exponent;

  //pab_hat = (pab*2**B)
  const int pab_hat_bits = 31;
  const int accu_hat_bits = 15;
  const int pam_hat_bits = 15;

  int B_max = (pab_hat_bits - 1) - max_pab_exp;
  int B_min = -max_pab_exp;

  int A_max = (accu_hat_bits - 1) - max_accu_exp;
  int A_min = -max_accu_exp;

  int M_max = (pam_hat_bits - 1) - max_pam_exp;
  int M_min = -max_pam_exp;

  //b_hat = pab*2**B must be a 32 bit value
  //b_hat is stored in 16 bits
  //b_hat is shifted using a VLMACC by 1<<Bs
  //Bs can be from 0 to 14 i.e. 1<<0 to 1<<14
  //If Bs needs to be negative then b_hat must be shifted right

  for (int B=B_max; B >= B_min; B--){

      // Check that pab*2**B will fit in pab_hat_bits bits
      for (int A=A_max; A >= A_min; A--){
        int M = B - A;

        if ((M <= M_max)&& (M >= M_min)){
            //int pam_bits = max_pab_exp + M;
            // int accu_bits = max_accu_exp;
            // if (A < 0)
            //   accu_bits += A;

            // int product_bits = pam_bits + accu_bits;
            *B_res = B;
            *A_res = A;
            *M_res = M;
            // printf("found B:%d A:%d M:%d\n", B, A, M);
            return;

        }

      }
  }
  assert(0);
}

void bnn_quantise_activation(
               int16_t * post_activation_multiplier_q,
               int16_t* post_activation_bias_q,

               float* post_activation_multiplier,
               float* post_activation_bias, 

               unsigned chans_out,

               int32_t clamp_low,
               int32_t clamp_high,

               int *accu_shr,
               int16_t *bias_multipler,
               int *final_shr,

               int32_t receptive_volume, 
               int * chan_overlaps
){

  // The max vlue that the actual VPU can output
  int max_accu = receptive_volume/2;
  int min_accu = -receptive_volume/2;

  int B, A, M;

  float * pam = (float *)malloc(sizeof(float) * chans_out);
  float * pab = (float *)malloc(sizeof(float) * chans_out);

  for (unsigned ch=0;ch<chans_out;ch++)
    pam[ch] = post_activation_multiplier[ch] * -2.;

  for (unsigned ch=0;ch<chans_out;ch++){
    pab[ch] = post_activation_bias[ch] + post_activation_multiplier[ch]*(float)receptive_volume;
    if(chan_overlaps)
      pab[ch] +=  (float)chan_overlaps[ch]*2.*post_activation_multiplier[ch];
  }

  solve_constraint(
    &B, &A, &M,
    pam,
    pab, 
    chans_out,
    max_accu, min_accu);

  //TODO make this into a function
  int max_pab_exp = INT_MIN;
  unsigned min_rsb = UINT_MAX;
  for (unsigned ch=0;ch<chans_out; ch++){
    int exp;
    frexp(pab[ch], &exp);
    if(exp > max_pab_exp)
      max_pab_exp = exp;
    unsigned rsb = clrsb((int)pab[ch]) - 16;
    if(rsb < min_rsb)
      min_rsb = rsb;
  }

  // We want to multiply pab by 2**B then quantise it, however, we can only store 16 bits
  // so what we will do is
  //if B > 0 make a 16 bit quantised bias and a 16 bit bias_multipler
  //if B < 0 bias_multipler = 1, bias = pam * 2 **B

  int bias_exp_adjust ;
  if (B > 0){
    bias_exp_adjust = 15 - max_pab_exp;

    //todo deal with the case that the bias_multipler wont fit in a 16 bit value
    *bias_multipler = (1<<(B - bias_exp_adjust)); //this is not so simple

  } else {
    *bias_multipler = 1;
    bias_exp_adjust = B;
  }

  // Now quantise the tensors
  for (unsigned ch = 0; ch < chans_out; ch++){

    int32_t pa_mul = (int32_t)round(ldexp(pam[ch], M));
    
    assert(clrsb(pa_mul) - 16 >= 0); // make sure there is no overflow
    post_activation_multiplier_q[ch] = (int16_t)pa_mul;

    int32_t pa_bias = (int32_t)round(ldexp(pab[ch], bias_exp_adjust));

    if (pa_bias == (1<<15)) //TODO think about this
      pa_bias  = INT16_MAX;

    // assert(clrsb(pa_bias) - 16 >= 0); // make sure there is no overflow
    post_activation_bias_q[ch] = (int16_t)pa_bias;

  }

  //todo check that post_activation_bias_q * bias_exp_adjust is ldexp(post_activation_bias, B)
  *accu_shr = -A;

  // The -8 is here to leave the result in a 16 bit form so that the quantisation to 8 bit 
  // can deal with the asymertic rounding.
  *final_shr = B - 8; 

  free(pam);
  free(pab);
}

void bnn_reorder_threshold_tensor(int32_t* thresh_boggled,
                                  const int32_t* thresholds_ref,
                                  const unsigned chans_out,
                                  const unsigned receptive_volume,
                                  int *chan_overlaps) {
  int16_t* thresholds = (int16_t*)thresh_boggled;

  for (unsigned i = 0; i < chans_out; i++) {

    unsigned bank = i / VPU_INT16_ACC_PERIOD;

    int32_t t = thresholds_ref[i] - (int32_t)(receptive_volume) / 2;

    if(chan_overlaps)
       t -= chan_overlaps[i];

    unsigned idx = bank * 2 * VPU_INT16_ACC_PERIOD + i % VPU_INT16_ACC_PERIOD;
    thresholds[idx] = t;
    thresholds[idx + VPU_INT16_ACC_PERIOD] = (t >> 16);
  }
}

static unsigned xor_pop_32(bnn_b32_t a, bnn_b32_t b) {
  unsigned c = 0;
  bnn_b32_t v = a ^ b;
 #if defined(__XS3A__)
    unsigned t = sizeof(bnn_b32_t);
    v = ~v;
    for (unsigned i = 0; i < t * 8; i++) {
      c += (v & 1);
      v >>= 1;
    }
  #else
    c += __builtin_popcount(~v);
  #endif
  return c;
}

//TODO consolidate these two functions
void bnn_reorder_kernel_tensor(bnn_b32_t* K_p, const bnn_b32_t* K_ref_p,
                               const unsigned k_height, const unsigned k_width,
                               const unsigned chans_in,
                               const unsigned chans_out, 
                               int * chan_overlaps) 
{                       
  //This is the count of full vector words that can be applied to the data    
  unsigned complete_256_bit_groups = ((chans_in*k_height*k_width) / XS3_VPU_VREG_WIDTH_BITS);

  // unsigned remainder_32_word_groups = ((chans_in*k_height*k_width) - complete_256_bit_groups*XS3_VPU_VREG_WIDTH_BITS) / 32;

  const unsigned inputs_per_b32 = 32;
  unsigned chan_b32_in = (chans_in + inputs_per_b32 - 1) / inputs_per_b32;

  bnn_b32_t(*K_ref)[k_height*k_width*chan_b32_in] =
      (bnn_b32_t(*)[k_height*k_width*chan_b32_in])K_ref_p;

  //the nuber of VPU_INT16_ACC_PERIOD groups there will be
  unsigned output_channel_groups = chans_out / VPU_INT16_ACC_PERIOD;

  unsigned remaining_input_channels = ((chans_in*k_height*k_width) % XS3_VPU_VREG_WIDTH_BITS)/32;

  bnn_b32_t(*K)[VPU_INT16_ACC_PERIOD][(8*complete_256_bit_groups) + remaining_input_channels] =
      (bnn_b32_t(*)[VPU_INT16_ACC_PERIOD][(8*complete_256_bit_groups) + remaining_input_channels])K_p;

  //This loops across groups of VPU_INT16_ACC_PERIOD output channels
  for (unsigned output_chan_group = 0; output_chan_group < output_channel_groups; 
      output_chan_group++) {

    bnn_b32_t * p = (bnn_b32_t *)&K[output_chan_group];

    //copy the groups of 256 input channels
    for (unsigned ic_group=0;ic_group < complete_256_bit_groups; ic_group++){
      //each group is of VPU_INT16_ACC_PERIOD channels 
      for (unsigned sub_grp_idx = 0; sub_grp_idx < VPU_INT16_ACC_PERIOD; sub_grp_idx++) {

        unsigned reversed_channel_order  = VPU_INT16_ACC_PERIOD - 1 - sub_grp_idx;

        memcpy(p,
          &K_ref[output_chan_group * VPU_INT16_ACC_PERIOD + reversed_channel_order][8*ic_group],
          sizeof(bnn_b32_t) * 8);
        p += (8);
      }
    }
    if (remaining_input_channels){
      for (unsigned sub_grp_idx = 0; sub_grp_idx < VPU_INT16_ACC_PERIOD; sub_grp_idx++) {

        unsigned reversed_channel_order  = VPU_INT16_ACC_PERIOD - 1 - sub_grp_idx;
        memcpy(p,
            &K_ref[output_chan_group * VPU_INT16_ACC_PERIOD + reversed_channel_order][8*complete_256_bit_groups],
                sizeof(bnn_b32_t)*remaining_input_channels);
        p += remaining_input_channels;
      }   
    }
    assert(p ==  (bnn_b32_t *)&(K[output_chan_group+1]));
  }
  //This is for the case of no overlap in the kernels
  if(chan_overlaps == 0)
    return;

  //Code only gets here if there is no overlap and hence no need to insert padding.

  //The filler value could be anything it just needs to be a known value
  char filler = 0x55;
  memset(&(K[output_channel_groups]), filler, sizeof(bnn_b32_t)*NN_BCONV2D_KERNEL_OVERRUN_WORDS);
  
  for (unsigned output_chan_group = 0; output_chan_group < output_channel_groups; 
      output_chan_group++) {

    bnn_b32_t * p = (bnn_b32_t *)&(K[output_chan_group]);

    p += (8*VPU_INT16_ACC_PERIOD*complete_256_bit_groups);
    
    if (remaining_input_channels){
      for (unsigned sub_grp_idx = 0; sub_grp_idx < VPU_INT16_ACC_PERIOD; sub_grp_idx++) {

        unsigned reversed_channel_order  = VPU_INT16_ACC_PERIOD - 1 - sub_grp_idx;

        bnn_b32_t zeros = 0x00000000;
        int total_xor_popcount = 0;
        for(unsigned o = remaining_input_channels; o < 8; o++){ //8 is 32 bit words per vpu load
          total_xor_popcount += (int)xor_pop_32(p[o], zeros) - 16;
        }
        chan_overlaps[ output_chan_group * VPU_INT16_ACC_PERIOD + reversed_channel_order] =  total_xor_popcount;

        p += remaining_input_channels;
      }   
    } else {
      // This code is here for the case where the overlap is being used with multiples 
      // 256 input channels.
      for (unsigned sub_grp_idx = 0; sub_grp_idx < VPU_INT16_ACC_PERIOD; sub_grp_idx++) {
        chan_overlaps[ output_chan_group * VPU_INT16_ACC_PERIOD + sub_grp_idx] =  0;
      }   
    }
  }
}

void bnn_reorder_int8_kernel_tensor(bnn_b32_t* K_p, const bnn_b32_t* K_ref_p,
                               const unsigned k_height, const unsigned k_width,
                               const unsigned chans_in,
                               const unsigned chans_out, 
                               int * chan_overlaps) {
                     
  //This is the count of full vector words that can be applied to the data    
  unsigned receptive_volume = chans_in*k_height*k_width;

  //The number of full XS3_VPU_VREG_WIDTH_BITS bit loads a single channel can process
  unsigned complete_256_bit_groups = receptive_volume / XS3_VPU_VREG_WIDTH_BITS;

  //This is the number of words remaining after complete_256_bit_groups*XS3_VPU_VREG_WIDTH_BITS bits
  //have been processed
  unsigned remaining_input_words = (receptive_volume % XS3_VPU_VREG_WIDTH_BITS)/32;

  const unsigned inputs_per_b32 = 32;
  assert(receptive_volume%inputs_per_b32 == 0);
  
  bnn_b32_t(*K_ref)[receptive_volume / inputs_per_b32] =
      (bnn_b32_t(*)[receptive_volume / inputs_per_b32])K_ref_p;

  //the nuber of VPU_INT16_ACC_PERIOD groups there will be
  unsigned output_chan_groups_of_accu_period = chans_out / VPU_INT16_ACC_PERIOD;
  unsigned output_chans_reamining = chans_out % VPU_INT16_ACC_PERIOD;

  bnn_b32_t * p = (bnn_b32_t *)K_p;
  //This loops across groups of VPU_INT16_ACC_PERIOD output channels
  for (unsigned output_chan_group = 0; output_chan_group < output_chan_groups_of_accu_period; 
      output_chan_group++) {

    //copy the groups of 256 input channels
    for (unsigned ic_group=0;ic_group < complete_256_bit_groups; ic_group++){
      //each group is of VPU_INT16_ACC_PERIOD channels 
      for (unsigned sub_grp_idx = 0; sub_grp_idx < VPU_INT16_ACC_PERIOD; sub_grp_idx++) {

        unsigned reversed_channel_order  = VPU_INT16_ACC_PERIOD - 1 - sub_grp_idx;
        memcpy(p,
          &K_ref[output_chan_group * VPU_INT16_ACC_PERIOD + reversed_channel_order][8*ic_group],
          sizeof(bnn_b32_t) * 8);
        p += 8;
      }
    }

    if (remaining_input_words){
      for (unsigned sub_grp_idx = 0; sub_grp_idx < VPU_INT16_ACC_PERIOD; sub_grp_idx++) {

        unsigned reversed_channel_order  = VPU_INT16_ACC_PERIOD - 1 - sub_grp_idx;
        memcpy(p,
            &K_ref[output_chan_group * VPU_INT16_ACC_PERIOD + reversed_channel_order][8*complete_256_bit_groups],
                sizeof(bnn_b32_t)*remaining_input_words);
        p += remaining_input_words;
      }   
    }
  }

  //If there are remaining input channels deal with there here
  if (output_chans_reamining){

    //copy the groups of 256 input channels
    for (unsigned ic_group=0;ic_group < complete_256_bit_groups; ic_group++){
      //each group is of VPU_INT16_ACC_PERIOD channels 
      for (unsigned sub_grp_idx = 0; sub_grp_idx < output_chans_reamining; sub_grp_idx++) {

        unsigned reversed_channel_order  = output_chans_reamining - 1 - sub_grp_idx;
        memcpy(p,
          &K_ref[output_chan_groups_of_accu_period * VPU_INT16_ACC_PERIOD + reversed_channel_order][8*ic_group],
          sizeof(bnn_b32_t) * 8);
        p += 8;
      }
    }

    if (remaining_input_words){
      for (unsigned sub_grp_idx = 0; sub_grp_idx < output_chans_reamining; sub_grp_idx++) {

        unsigned reversed_channel_order  = output_chans_reamining - 1 - sub_grp_idx;
        memcpy(p,
            &K_ref[output_chan_groups_of_accu_period * VPU_INT16_ACC_PERIOD + reversed_channel_order][8*complete_256_bit_groups],
                sizeof(bnn_b32_t)*remaining_input_words);
        p += remaining_input_words;
      }   
    }
  }

  //This is for the case of no overlap in the kernels
  if(chan_overlaps == 0)
    return;

  memset(chan_overlaps, 0, sizeof(int) * chans_out);
  //Code only gets here if there is no overlap and hence no need to insert padding.

  //The filler value could be anything it just needs to be a known value
  char filler = 0x55;
  memset(p, filler, sizeof(bnn_b32_t)*NN_BCONV2D_KERNEL_OVERRUN_WORDS); //TODO minimise this
  
  //Reset the pointer for another pass to get the overlaps now that the memory if laied out correctly
  p = (bnn_b32_t *)K_p;

  for (unsigned output_chan_group = 0; output_chan_group < output_chan_groups_of_accu_period; 
      output_chan_group++) {

    p += (8*VPU_INT16_ACC_PERIOD*complete_256_bit_groups);
    
    // printf("remaining_input_words %u\n", remaining_input_words);
    if (remaining_input_words){
      for (unsigned sub_grp_idx = 0; sub_grp_idx < VPU_INT16_ACC_PERIOD; sub_grp_idx++) {

        unsigned reversed_channel_order  = VPU_INT16_ACC_PERIOD - 1 - sub_grp_idx;

        bnn_b32_t zeros = 0x00000000;
        int total_xor_popcount = 0;
        for(unsigned o = remaining_input_words; o < 8; o++){ //8 is 32 bit words per vpu load
          total_xor_popcount += (int)xor_pop_32(p[o], zeros) - 16;
        }
        chan_overlaps[ output_chan_group * VPU_INT16_ACC_PERIOD + reversed_channel_order] =  total_xor_popcount;
        p += remaining_input_words;
      }   

    } else {
      // This code is here for the case where the overlap is being used with multiples 
      // 256 input channels.
      for (unsigned sub_grp_idx = 0; sub_grp_idx < VPU_INT16_ACC_PERIOD; sub_grp_idx++) {
        chan_overlaps[ output_chan_group * VPU_INT16_ACC_PERIOD + sub_grp_idx] =  0;
        
      }   
    }
  }
  if (output_chans_reamining){

    p += (8*output_chans_reamining*complete_256_bit_groups);

    if (remaining_input_words){
      for (unsigned sub_grp_idx = 0; sub_grp_idx < output_chans_reamining; sub_grp_idx++) {

        unsigned reversed_channel_order  = output_chans_reamining - 1 - sub_grp_idx;

        bnn_b32_t zeros = 0x00000000;
        int total_xor_popcount = 0;
        for(unsigned o = remaining_input_words; o < 8; o++){ //8 is 32 bit words per vpu load
          total_xor_popcount += (int)xor_pop_32(p[o], zeros) - 16;
        }
        chan_overlaps[ output_chan_groups_of_accu_period * VPU_INT16_ACC_PERIOD + reversed_channel_order] =  total_xor_popcount;
        p += remaining_input_words;
      }   
    }
  }
}
