
#include <stdint.h>
#include <string.h>
#include <assert.h>

#include "nn_operator.h"
#include "../nn_op_helper.h"

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

    thresholds[(bank * (2*VPU_INT16_ACC_PERIOD)) + (i % VPU_INT16_ACC_PERIOD)] = t;
    thresholds[(bank * (2*VPU_INT16_ACC_PERIOD)) + (i % VPU_INT16_ACC_PERIOD) + VPU_INT16_ACC_PERIOD] = (t >> VPU_INT16_ACC_PERIOD);
  }
}

void bnn_reorder_multiplier_and_bias_tensors(
                                  int16_t* post_activation_multiplier_q_reordered,
                                  const int16_t* post_activation_multiplier_q,
                                  int16_t* post_activation_bias_q_reordered,
                                  const int16_t* post_activation_bias_q,
                                  const unsigned chans_out) {

  for (unsigned b=0;b < chans_out/16;b++){
    for(unsigned i=0;i<VPU_INT16_ACC_PERIOD;i++){

      unsigned interleaved_oc;
      if (i<(VPU_INT16_ACC_PERIOD/2)){
        interleaved_oc = (2*i) + 1;
      } else{
        interleaved_oc = 2*(i-(VPU_INT16_ACC_PERIOD/2));
      }

      post_activation_multiplier_q_reordered[b*VPU_INT16_ACC_PERIOD + i] = 
        post_activation_multiplier_q[b*VPU_INT16_ACC_PERIOD + interleaved_oc];
      post_activation_bias_q_reordered[b*VPU_INT16_ACC_PERIOD + i] = 
        post_activation_bias_q[b*VPU_INT16_ACC_PERIOD + interleaved_oc];
    }

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

  const unsigned outputs_per_b32 = 32;
  unsigned chan_b32_in = (chans_in + outputs_per_b32 - 1) / outputs_per_b32;

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

void bnn_reorder_int8_kernel_tensor(bnn_b256_t* K_p, const bnn_b256_t* K_ref_p,
                               const unsigned k_height, const unsigned k_width,
                               const unsigned chans_in,
                               const unsigned chans_out) {
  unsigned chan_b256_in =
      (chans_in + XS3_VPU_VREG_WIDTH_BITS - 1) / XS3_VPU_VREG_WIDTH_BITS;

  bnn_b256_t(*K_ref)[k_height][k_width][chan_b256_in] =
      (bnn_b256_t(*)[k_height][k_width][chan_b256_in])K_ref_p;

  bnn_b256_t(*K)[k_height][k_width][chan_b256_in][VPU_INT16_ACC_PERIOD] =
      (bnn_b256_t(*)[k_height][k_width][chan_b256_in][VPU_INT16_ACC_PERIOD])K_p;

  for (unsigned output_chan_group = 0; output_chan_group < chans_out / VPU_INT16_ACC_PERIOD; 
      output_chan_group++) {
    for (unsigned h = 0; h < k_height; h++) {
      for (unsigned w = 0; w < k_width; w++) {
        for (unsigned ic = 0; ic < chan_b256_in; ic++) {
          for (unsigned sub_grp_idx = 0; sub_grp_idx < VPU_INT16_ACC_PERIOD; sub_grp_idx++) {

            //This is to compensate for the way the asm interleaves the 
            //upper and lower 8 outputs.
            unsigned interleaved_oc;
            if (sub_grp_idx < (VPU_INT16_ACC_PERIOD/2)) {
              interleaved_oc = (2*sub_grp_idx) + 1;
            } else{
              interleaved_oc = 2*(sub_grp_idx - (VPU_INT16_ACC_PERIOD/2));
            }
            memcpy(& K[output_chan_group][h][w][ic][VPU_INT16_ACC_PERIOD - 1 - sub_grp_idx], 
              & K_ref[output_chan_group * VPU_INT16_ACC_PERIOD + interleaved_oc][h][w][ic], 
              sizeof(bnn_b256_t));
          }
        }
      }
    }
  }
}


WEAK_FUNC
void bnn_conv2d_bin_out_SISO(bnn_b32_t* Y_p,
    const bnn_b32_t* X_p, const bnn_b32_t* K_p, const int32_t* thresholds,
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

  const unsigned channels_per_output_write = 32;
  const unsigned channels_per_input_word = 32;
  const unsigned chan_b32_in = (x->channels + channels_per_input_word - 1) / channels_per_input_word;
  const unsigned chan_b32_out = (y->channels + channels_per_output_write - 1) / channels_per_output_write;

  const unsigned h_stride = k->stride.horizontal;
  const unsigned v_stride = k->stride.vertical;  
  const unsigned h_dilation = k->dilation.horizontal;
  const unsigned v_dilation = k->dilation.vertical;  

  bnn_b32_t(*Y)[y->width][chan_b32_out] =
      (bnn_b32_t(*)[y->width][chan_b32_out])Y_p;

  bnn_b32_t(*X)[x->width][chan_b32_in] =
      (bnn_b32_t(*)[x->width][chan_b32_in])X_p;

  bnn_b32_t(*K)[k->shape.height][k->shape.width][chan_b32_in] =
      (bnn_b32_t(*)[k->shape.height][k->shape.width][chan_b32_in])K_p;

  unsigned x_sub_height = CONV2D_INPUT_LENGTH(y_sub_height, k_sub_height, v_dilation, v_stride );
  unsigned x_sub_width = CONV2D_INPUT_LENGTH(y_sub_width, k_sub_width, h_dilation, h_stride );

  for (unsigned h = x_loc_y; h < (x_loc_y + x_sub_height) - k_sub_height + 1; h += v_stride) {
    for (unsigned w = x_loc_x; w < (x_loc_x + x_sub_width) - k_sub_width + 1; w += h_stride) {
      for (unsigned oc_word = 0; oc_word < chan_b32_out; oc_word += 1) {
        bnn_b32_t bitpacked_column = 0;

        for (unsigned oc_bit = 0; oc_bit < channels_per_output_write; oc_bit += 1) {
          unsigned oc = oc_bit + (channels_per_output_write * oc_word);
          int32_t sum = 0;
          for (unsigned kh = k_loc_y; kh < k_loc_y + k_sub_height; kh += 1) {
            for (unsigned kw = k_loc_x; kw < k_loc_x + k_sub_width; kw += 1) {
              for (unsigned ic = 0; ic < chan_b32_in; ic += 1) {
                sum += xor_pop_32(&(X[h + kh][w + kw][ic]), &(K[oc][kh][kw][ic]));
              }
            }
          }

          sum = (k->shape.height * k->shape.width * chan_b32_in * channels_per_input_word) - sum;

          unsigned bit = sum > thresholds[oc];
          if (bit) bitpacked_column |= 1ULL << oc_bit;
        }
        Y[y_loc_y + ((h-x_loc_y) / v_stride)][y_loc_x + ((w-x_loc_x) / h_stride)][oc_word] = bitpacked_column;
      }
    }
  }


}

WEAK_FUNC
void bnn_conv2d_bin_out(bnn_b32_t* Y_p,
    const bnn_b256_t* X_p, const bnn_b256_t* K_p, const int32_t* thresholds,
    
    const nn_image_params_t* x, //The full image of x
    const nn_image_params_t* y, // the full image of y
    const nn_window_params_t* k, //the full kernel k
    
    const unsigned y_loc_x, const unsigned y_loc_y,
    const unsigned y_sub_width, const unsigned y_sub_height,

    const unsigned x_loc_x, const unsigned x_loc_y, 
    
    const unsigned k_loc_x, const unsigned k_loc_y, 
    const unsigned k_sub_width, const unsigned k_sub_height
) {

  const unsigned channels_per_output_write = 32;
  const unsigned chan_b256_in = (x->channels + XS3_VPU_VREG_WIDTH_BITS - 1) / XS3_VPU_VREG_WIDTH_BITS;
  const unsigned chan_b32_out = (y->channels + channels_per_output_write - 1) / channels_per_output_write;

  const unsigned h_stride = k->stride.horizontal;
  const unsigned v_stride = k->stride.vertical;  
  const unsigned h_dilation = k->dilation.horizontal;
  const unsigned v_dilation = k->dilation.vertical;  

  bnn_b32_t(*Y)[y->width][chan_b32_out] =
      (bnn_b32_t(*)[y->width][chan_b32_out])Y_p;

  bnn_b256_t(*X)[x->width][chan_b256_in] =
      (bnn_b256_t(*)[x->width][chan_b256_in])X_p;

  bnn_b256_t(*K)[k->shape.height][k->shape.width][chan_b256_in] =
      (bnn_b256_t(*)[k->shape.height][k->shape.width][chan_b256_in])K_p;

  unsigned x_sub_height = CONV2D_INPUT_LENGTH(y_sub_height, k_sub_height, v_dilation, v_stride );
  unsigned x_sub_width = CONV2D_INPUT_LENGTH(y_sub_width, k_sub_width, h_dilation, h_stride );

  for (unsigned h = x_loc_y; h < (x_loc_y + x_sub_height) - k_sub_height + 1; h += v_stride) {
    for (unsigned w = x_loc_x; w < (x_loc_x + x_sub_width) - k_sub_width + 1; w += h_stride) {
      for (unsigned oc_word = 0; oc_word < chan_b32_out; oc_word += 1) {
        bnn_b32_t bitpacked_column = 0;

        for (unsigned oc_bit = 0; oc_bit < channels_per_output_write; oc_bit += 1) {
          unsigned oc = oc_bit + (channels_per_output_write * oc_word);
          int32_t sum = 0;
          for (unsigned kh = k_loc_y; kh < k_loc_y + k_sub_height; kh += 1) {
            for (unsigned kw = k_loc_x; kw < k_loc_x + k_sub_width; kw += 1) {
              for (unsigned ic = 0; ic < chan_b256_in; ic += 1) {
                sum += xor_pop_256(&(X[h + kh][w + kw][ic]), &(K[oc][kh][kw][ic]));
              }
            }
          }

          sum = (k->shape.height * k->shape.width * chan_b256_in * XS3_VPU_VREG_WIDTH_BITS) - sum;
          unsigned bit = sum > thresholds[oc];
          if (bit) bitpacked_column |= 1ULL << oc_bit;
        }
        Y[y_loc_y + ((h-x_loc_y) / v_stride)][y_loc_x + ((w-x_loc_x) / h_stride)][oc_word] = bitpacked_column;
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

WEAK_FUNC
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
) {

  const unsigned chan_b256_in = (x->channels + XS3_VPU_VREG_WIDTH_BITS - 1) / XS3_VPU_VREG_WIDTH_BITS;
  const unsigned chans_out = y->channels;

  const unsigned h_stride = k->stride.horizontal;
  const unsigned v_stride = k->stride.vertical;  
  const unsigned h_dilation = k->dilation.horizontal;
  const unsigned v_dilation = k->dilation.vertical;  

  int8_t(*Y)[y->width][chans_out] =
      (int8_t(*)[y->width][chans_out])Y_p;

  bnn_b256_t(*X)[x->width][chan_b256_in] =
      (bnn_b256_t(*)[x->width][chan_b256_in])X_p;

  bnn_b256_t(*K)[k->shape.height][k->shape.width][chan_b256_in] =
      (bnn_b256_t(*)[k->shape.height][k->shape.width][chan_b256_in])K_p;

  unsigned x_sub_height = CONV2D_INPUT_LENGTH(y_sub_height, k_sub_height, v_dilation, v_stride );
  unsigned x_sub_width = CONV2D_INPUT_LENGTH(y_sub_width, k_sub_width, h_dilation, h_stride );

  for (unsigned h = x_loc_y; h < (x_loc_y + x_sub_height) - k_sub_height + 1; h += v_stride) {
    for (unsigned w = x_loc_x; w < (x_loc_x + x_sub_width) - k_sub_width + 1; w += h_stride) {
      for (unsigned oc = 0; oc < chans_out; oc += 1) {

        int32_t sum = 0;
        for (unsigned kh = k_loc_y; kh < k_loc_y + k_sub_height; kh += 1) {
          for (unsigned kw = k_loc_x; kw < k_loc_x + k_sub_width; kw += 1) {
            for (unsigned ic = 0; ic < chan_b256_in; ic += 1) {
              sum += xor_pop_256(&(X[h + kh][w + kw][ic]), &(K[oc][kh][kw][ic]));
            }
          }
        }

        int32_t backtransform_add = (k->shape.height * k->shape.width * chan_b256_in * XS3_VPU_VREG_WIDTH_BITS);
        
        // This converts xor_popcount to macc format
        int32_t vpu_output = ((2*sum)-backtransform_add)/2;

        //not rounding has happened to the point
        int32_t r = ashr(vpu_output, accu_shr) ;
        
        r *= (int32_t) post_activation_multiplier_q[oc];

        r = ashr(r, post_vlmul_shr);

        r += post_activation_bias_q[oc];

        r = ashr(r, final_shr);

        r = r >> 8; //Use a store part word to extract bits 8-15

        if (r > INT8_MAX) r = INT8_MAX;
        if (r < INT8_MIN) r = INT8_MIN;

        Y[y_loc_y + ((h-x_loc_y) / v_stride)][y_loc_x + ((w-x_loc_x) / h_stride)][oc] = (int8_t)r;
        
      }
    }
  }
}
