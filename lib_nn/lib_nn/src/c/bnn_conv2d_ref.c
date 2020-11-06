
#include <stdint.h>
#include <string.h>
#include <assert.h>

#include "nn_operator.h"
#include "../nn_op_helper.h"

#include "xs3_vpu.h"
#include "vpu_sim.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

//TODO get these from headers
void bnn_conv2d_int8_out_asm_prepare(
    nn_bnn_conv2d_int8_out_asm_plan_t* plan, int8_t* Y_p,
    const bnn_b256_t* X_p, const bnn_b256_t* K_p, 
    
    const int16_t* post_activation_multiplier_q, 
    const int16_t* post_activation_bias_q,
    const int accu_shr,
    const int final_shr,

    const nn_image_params_t* x, 
    const nn_image_params_t* y,
    const nn_window_params_t* k, 
    const unsigned y_loc_x, const unsigned y_loc_y,
    const unsigned y_sub_width, const unsigned y_sub_height,
    const unsigned x_loc_x, const unsigned x_loc_y, 
    const unsigned k_loc_x, const unsigned k_loc_y, 
    const unsigned k_sub_width, const unsigned k_sub_height) ;

void bnn_conv2d_int8_out_SISO_asm_prepare(
    nn_bnn_conv2d_int8_out_SISO_asm_plan_t* plan, int8_t* Y_p,
    const bnn_b32_t* X_p, const bnn_b32_t* K_p, bnn_b32_t * data_scratch,
    
    const int16_t* post_activation_multiplier_q, 
    const int16_t* post_activation_bias_q,
    const int accu_shr,
    const int final_shr,

    const nn_image_params_t* x, 
    const nn_image_params_t* y,
    const nn_window_params_t* k, 
    const unsigned y_loc_x, const unsigned y_loc_y,
    const unsigned y_sub_width, const unsigned y_sub_height,
    const unsigned x_loc_x, const unsigned x_loc_y, 
    const unsigned k_loc_x, const unsigned k_loc_y, 
    const unsigned k_sub_width, const unsigned k_sub_height) ;


//This is the amount that the VLMUL instruction shifts the product of C and R by.
static const unsigned post_vlmul_shr = 14;

unsigned xor_pop_32(bnn_b32_t* a, bnn_b32_t* b) {
  unsigned c = 0;
  unsigned t = sizeof(bnn_b32_t);
  bnn_b32_t v = (*a) ^ (*b);
 #if defined(__XS3A__)
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

unsigned xor_pop_256(bnn_b256_t* a, bnn_b256_t* b) {

  unsigned elements = sizeof(((bnn_b256_t*)0)->d) /
    sizeof(((bnn_b256_t*)0)->d[0]);

  unsigned c = 0;
  for (unsigned e = 0; e < elements; e++) 
    c +=xor_pop_32(&(a->d[e]), &(b->d[e]));

  return c;
}

void bnn_reorder_threshold_tensor(int32_t* thresh_boggled,
                                  const int32_t* thresholds_ref,
                                  const unsigned chans_out,
                                  const unsigned receptive_field,
                                  int *chan_overlaps) {
  int16_t* thresholds = (int16_t*)thresh_boggled;

  for (unsigned i = 0; i < chans_out; i++) {
    unsigned bank = i / VPU_INT16_ACC_PERIOD;

    int32_t t = thresholds_ref[i] - ((int32_t)(receptive_field) / 2);

    if(chan_overlaps)
       t -= chan_overlaps[i];

    thresholds[(bank * (2*VPU_INT16_ACC_PERIOD)) + (i % VPU_INT16_ACC_PERIOD)] = t&0xffff;
    thresholds[(bank * (2*VPU_INT16_ACC_PERIOD)) + (i % VPU_INT16_ACC_PERIOD) + VPU_INT16_ACC_PERIOD] = (t >> 16)&0xffff;
  }
}

void bnn_reorder_kernel_tensor(bnn_b32_t* K_p, const bnn_b32_t* K_ref_p,
                               const unsigned k_height, const unsigned k_width,
                               const unsigned chans_in,
                               const unsigned chans_out, 
                               int * chan_overlaps) 
{                       
  //This is the count of full vector words that can be applied to the data    
  unsigned complete_256_bit_groups = ((chans_in*k_height*k_width) / XS3_VPU_VREG_WIDTH_BITS);

  unsigned remainder_32_word_groups = ((chans_in*k_height*k_width) - complete_256_bit_groups*XS3_VPU_VREG_WIDTH_BITS) / 32;

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
          total_xor_popcount += (int)xor_pop_32(&(p[o]), &zeros) - 16;
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
          total_xor_popcount += (int)xor_pop_32(&(p[o]), &zeros) - 16;
        }
        chan_overlaps[ output_chan_group * VPU_INT16_ACC_PERIOD + reversed_channel_order] =  total_xor_popcount;
        // printf("chan_overlaps[%u] %d\n", output_chan_group * VPU_INT16_ACC_PERIOD + reversed_channel_order, total_xor_popcount);
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

    // printf("remaining_input_words %u\n", remaining_input_words);
    if (remaining_input_words){
      for (unsigned sub_grp_idx = 0; sub_grp_idx < output_chans_reamining; sub_grp_idx++) {

        unsigned reversed_channel_order  = output_chans_reamining - 1 - sub_grp_idx;

        bnn_b32_t zeros = 0x00000000;
        int total_xor_popcount = 0;
        for(unsigned o = remaining_input_words; o < 8; o++){ //8 is 32 bit words per vpu load
          total_xor_popcount += (int)xor_pop_32(&(p[o]), &zeros) - 16;
        }
        chan_overlaps[ output_chan_groups_of_accu_period * VPU_INT16_ACC_PERIOD + reversed_channel_order] =  total_xor_popcount;
        // printf("chan_overlaps[%u] %d\n", output_chan_groups_of_accu_period * VPU_INT16_ACC_PERIOD + reversed_channel_order, total_xor_popcount);

        p += remaining_input_words;
      }   
    }
  }
}


static int32_t ashr(int32_t x, int shr){
  if (shr > 0)
    return (x + (1 << (shr-1))) >> shr;
  else
    return x << (-shr);
}

static int64_t saturate_non_sym(
    const int64_t input,
    const unsigned bits)
{
    const int64_t max_val = (((int64_t)1)<<(bits-1))-1;
    const int64_t min_val = -max_val - 1;
    
    return (input > max_val)?  max_val : (input < min_val)? min_val : input;
}
void VDEPTH8_FIXED(xs3_vpu* vpu){

    vpu_vector_t vec_tmp;
    memcpy(&vec_tmp, &(vpu->vR), sizeof(vpu_vector_t));
    memset(&(vpu->vR), 0, sizeof(vpu_vector_t));
    
    for(int i = 0; i < VPU_INT16_EPV; i++){
        int32_t elm = ((int32_t)vec_tmp.s16[i]) + (1 << 7);
        vpu->vR.s8[i] = saturate_non_sym(elm >> 8, 8);
    }
}

WEAK_FUNC
void bnn_conv2d_int8_out_asm(nn_bnn_conv2d_int8_out_asm_plan_t * plan){

  xs3_vpu vpu_data;
  xs3_vpu * vpu = &vpu_data;

  vpu_vector_t sat_mem;
  vpu_vector_t temp_mem;
  VSETC(vpu, MODE_S16);

  for(unsigned i=0;i<VPU_INT16_EPV;i++)
    sat_mem.s16[i] = plan->vlsat;

  void * X_p = plan->X;
  void * Y_p = plan->Y;

  for (int xh = plan->x_height_loop_counter; xh > 0 ; xh-- ) {
    for (int xv = plan->x_width_loop_counter; xv >= 0 ; xv-- ) {

      void * cur_post_activation_mul = plan->post_activation_mul;
      void * cur_post_activation_bias = plan->post_activation_bias;
      void * K_p = plan->K;
      for (int oc = plan->output_channel_loop_counter; oc >= 0 ; oc-- ) {

        void * X_cur_p = X_p;
        VCLRDR(vpu);

        for (int kh = plan->k_height_loop_counter; kh >= 0 ; kh-- )  {
          for (int kw = plan->k_width_loop_counter; kw >= 0 ; kw-- )  {
            for (int ic = plan->input_channel_loop_counter; ic >= 0 ; ic-- ) {
              VLDC(vpu, X_cur_p);
              X_cur_p += 32;

              for(unsigned l=0; l<16; l++){
                VLMACCR1(vpu, K_p);
                K_p += 32;
              }
            }
            X_cur_p += plan->inner_x_h_step;
            K_p += plan->k_h_step;
          }
          X_cur_p += plan->inner_x_v_step;
          K_p += plan->k_v_step;
        }

        VLSAT(vpu, &sat_mem);
        VSTR(vpu, &temp_mem);
        VLASHR(vpu, &temp_mem, plan->ashr);
        VLMUL(vpu, cur_post_activation_mul);
        VLADD(vpu, cur_post_activation_bias);

        VSTR(vpu, &temp_mem);
        VLASHR(vpu, &temp_mem, plan->final_shr);
        VDEPTH8_FIXED(vpu);
        VSTRPV(vpu, Y_p, 0xffff);
        Y_p += 16;

        cur_post_activation_mul += 32;
        cur_post_activation_bias += 32;
      }
      X_p += plan->outer_x_h_step;
    }
    X_p += plan->outer_x_v_step;
    Y_p += plan->y_v_step;
  }
}















static void make_patch(xs3_vpu * vpu, nn_bnn_conv2d_int8_out_SISO_asm_plan_t * plan, void * X_p){

    void * X_cur_p = X_p;
    void * D_p = plan->data_scratch;

    for (int kh = plan->k_height_loop_counter; kh >= 0 ; kh-- )  {
        for (int kw = plan->k_width_loop_counter; kw >= 0 ; kw-- )  {
            for (int ic = plan->input_channel_loop_counter; ic >= 0 ; ic-- ) {
                VLDD(vpu, X_cur_p);
                X_cur_p += 32;
                VSTD(vpu, D_p);
                D_p += 32;
            }
            X_cur_p += plan->inner_x_h_step;
            D_p += plan->data_scratch_adjust;
        }
        X_cur_p += plan->inner_x_v_step;
    }
    VCLRDR(vpu);
    VSTD(vpu, D_p);
}

void compute_patch(nn_bnn_conv2d_int8_out_SISO_asm_plan_t *plan, 
  void ** K_p, int step, xs3_vpu * vpu, vpu_vector_t *sat_mem, 
  void * cur_post_activation_mul, 
  void * cur_post_activation_bias){

  VCLRDR(vpu);
  void * D_p = plan->data_scratch;
  for (unsigned p = plan->patch_loop_counter; p > 0; p--){
    VLDC(vpu, D_p);
    D_p += 32;
    for(unsigned l=0; l<15; l++){
      VLMACCR1(vpu, *K_p);
      *K_p += 32;
    }
    VLMACCR1(vpu, *K_p);
    *K_p += step;
  }

  VLDC(vpu, D_p);
  
  unsigned loops;
  switch(step){
    case 32: {loops=16; break;}
    case -96: {loops=12; break;}
    case -224: {loops=8; break;}
    case -352: {loops=4; break;}
  } 
  // printf("loops: %u plan->k_p_adjust:%d\n", loops, plan->k_p_adjust);
  for(unsigned l=0; l<loops; l++){
    VLMACCR1(vpu, *K_p);
    *K_p += plan->k_p_adjust;
  }

  vpu_vector_t temp_mem;
  memset(&temp_mem, 0, sizeof(temp_mem));

  VLSAT(vpu, sat_mem);
  VSTR(vpu, &temp_mem);
  VLASHR(vpu, &temp_mem, plan->ashr);
  
  VLMUL(vpu, cur_post_activation_mul);
  VLADD(vpu, cur_post_activation_bias);


  /////
  // VSTR(vpu, &temp_mem);
  // VCLRDR(vpu);
  // VLDC(vpu, cur_post_activation_bias);
  // VLMACC(vpu, bias_shift);
  // VLDC(vpu, temp_mem);
  // VLMACC(vpu, cur_post_activation_mul);
  // VDEPTH16(vpu);
  /////

  VSTR(vpu, &temp_mem);
  VLASHR(vpu, &temp_mem, plan->final_shr);
  VDEPTH8_FIXED(vpu);

}

WEAK_FUNC
void bnn_conv2d_int8_out_SISO_asm(nn_bnn_conv2d_int8_out_SISO_asm_plan_t *plan){

  xs3_vpu vpu_data;
  memset(&vpu_data, 0, sizeof(vpu_data));
  xs3_vpu * vpu = &vpu_data;

  vpu_vector_t sat_mem;
  VSETC(vpu, MODE_S16);

  for(unsigned i=0;i<VPU_INT16_EPV;i++)
    sat_mem.s16[i] = plan->vlsat;

  void * X_p = plan->X;
  void * Y_p = plan->Y;

  for (int xh = plan->x_height_loop_counter; xh > 0 ; xh-- ) {
    for (int xv = plan->x_width_loop_counter; xv >= 0 ; xv-- ) {

      make_patch(vpu, plan, X_p);

      void * cur_post_activation_mul = plan->post_activation_mul;
      void * cur_post_activation_bias = plan->post_activation_bias;
      void * K_p = plan->K;
      for (int oc = plan->output_channel_loop_counter; oc > 0 ; oc-- ) {

        compute_patch(plan, &K_p, 32, vpu, &sat_mem, 
          cur_post_activation_mul, cur_post_activation_bias);

        VSTRPV(vpu, Y_p, 0xffff);
        Y_p += 16;

        cur_post_activation_mul += 32;
        cur_post_activation_bias += 32;
      }
      
      compute_patch(plan, &K_p, plan->k_p_rewind, vpu, &sat_mem, 
        cur_post_activation_mul, cur_post_activation_bias);

      VSTRPV(vpu, Y_p, plan->final_channels_mask);

      Y_p += plan->final_channels_bytes;
      X_p += plan->outer_x_h_step;
    }
    X_p += plan->outer_x_v_step;
    Y_p += plan->y_v_step;
  }


}

void bnn_conv2d_int8_out(int8_t* Y_p,
    const bnn_b256_t* X_p, const bnn_b256_t* K_p, 
    
    const int16_t* post_activation_multiplier_q, 
    const int16_t* post_activation_bias_q,
    const int accu_shr,
    const int final_shr,
    
    const nn_image_params_t* x, //The full image of x
    const nn_image_params_t* y, // the full image of y
    const nn_window_params_t* k, //the full kernel k
    
    const unsigned y_loc_x, const unsigned y_loc_y,
    const unsigned y_sub_width, const unsigned y_sub_height,

    const unsigned x_loc_x, const unsigned x_loc_y, 
    
    const unsigned k_loc_x, const unsigned k_loc_y, 
    const unsigned k_sub_width, const unsigned k_sub_height
){
  nn_bnn_conv2d_int8_out_asm_plan_t plan;


    bnn_conv2d_int8_out_asm_prepare(&plan, Y_p,
        X_p,  K_p,
        post_activation_multiplier_q, 
        post_activation_bias_q,
        accu_shr,
        final_shr,
        x, y, k, 
        y_loc_x, y_loc_y, y_sub_width, y_sub_height,
        x_loc_x, x_loc_y, 
        k_loc_x, k_loc_y, k_sub_width, k_sub_height);


  bnn_conv2d_int8_out_asm(&plan);
}

void bnn_conv2d_int8_out_SISO(int8_t* Y_p,
    const bnn_b32_t* X_p, const bnn_b32_t* K_p, 
    
    const int16_t* post_activation_multiplier_q, 
    const int16_t* post_activation_bias_q,
    const int accu_shr,
    const int final_shr,

    bnn_b32_t * data_scratch,
    
    const nn_image_params_t* x, //The full image of x
    const nn_image_params_t* y, // the full image of y
    const nn_window_params_t* k, //the full kernel k
    
    const unsigned y_loc_x, const unsigned y_loc_y,
    const unsigned y_sub_width, const unsigned y_sub_height,

    const unsigned x_loc_x, const unsigned x_loc_y, 
    
    const unsigned k_loc_x, const unsigned k_loc_y, 
    const unsigned k_sub_width, const unsigned k_sub_height
) {
    nn_bnn_conv2d_int8_out_SISO_asm_plan_t plan;

    bnn_conv2d_int8_out_SISO_asm_prepare(&plan, Y_p,
        X_p,  K_p, data_scratch,
        post_activation_multiplier_q, 
        post_activation_bias_q,
        accu_shr,
        final_shr,
        x, y, k, 
        y_loc_x, y_loc_y, y_sub_width, y_sub_height,
        x_loc_x, x_loc_y, 
        k_loc_x, k_loc_y, k_sub_width, k_sub_height);

    bnn_conv2d_int8_out_SISO_asm(&plan);
}
