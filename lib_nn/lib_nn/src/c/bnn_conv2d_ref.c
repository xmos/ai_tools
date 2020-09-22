
#include <stdint.h>
#include <string.h>
#include <assert.h>

#include "nn_operator.h"
#include "../nn_op_helper.h"
#include "nn_op_structs.h"

#include "xs3_vpu.h"
#define ACC_PERIOD (VPU_INT16_ACC_PERIOD)

//This is the amount that the VLMUL instruction shifts the product of C and R by.
static const unsigned post_vlmul_shr = 14;

void bnn_reorder_threshold_tensor(const int32_t* thresh_boggled,
                                  const int32_t* thresholds_ref,
                                  const unsigned chans_out,
                                  const unsigned receptive_field) {
  int16_t* thresholds = (int16_t*)thresh_boggled;

  for (unsigned i = 0; i < chans_out; i++) {
    unsigned bank = i / ACC_PERIOD;

    int32_t t = thresholds_ref[i] - ((receptive_field) / 2);
    thresholds[(bank * (2*ACC_PERIOD)) + (i % ACC_PERIOD)] = (t >> 0);
    thresholds[(bank * (2*ACC_PERIOD)) + (i % ACC_PERIOD) + ACC_PERIOD] = (t >> ACC_PERIOD);
  }
}

void bnn_reorder_multiplier_and_bias_tensors(
                                  int16_t* post_activation_multiplier_q_reordered,
                                  const int16_t* post_activation_multiplier_q,
                                  int16_t* post_activation_bias_q_reordered,
                                  const int16_t* post_activation_bias_q,
                                  const unsigned chans_out) {

  for (unsigned b=0;b < chans_out/16;b++){
    for(unsigned i=0;i<ACC_PERIOD;i++){

      unsigned interleaved_oc;
      if (i<(ACC_PERIOD/2)){
        interleaved_oc = (2*i) + 1;
      } else{
        interleaved_oc = 2*(i-(ACC_PERIOD/2));
      }

      post_activation_multiplier_q_reordered[b*ACC_PERIOD + i] = 
        post_activation_multiplier_q[b*ACC_PERIOD + interleaved_oc];
      post_activation_bias_q_reordered[b*ACC_PERIOD + i] = 
        post_activation_bias_q[b*ACC_PERIOD + interleaved_oc];
    }

  }
}

void bnn_reorder_kernel_tensor(bnn_b256_t* K_p, const bnn_b256_t* K_ref_p,
                               const unsigned k_height, const unsigned k_width,
                               const unsigned chans_in,
                               const unsigned chans_out) {
  unsigned chan_b256_in =
      (chans_in + XS3_VPU_VREG_WIDTH_BITS - 1) / XS3_VPU_VREG_WIDTH_BITS;

  bnn_b256_t(*K_ref)[k_height][k_width][chan_b256_in] =
      (bnn_b256_t(*)[k_height][k_width][chan_b256_in])K_ref_p;

  bnn_b256_t(*K)[k_height][k_width][chan_b256_in][ACC_PERIOD] =
      (bnn_b256_t(*)[k_height][k_width][chan_b256_in][ACC_PERIOD])K_p;

  for (unsigned output_chan_group = 0; output_chan_group < chans_out / ACC_PERIOD; 
      output_chan_group++) {
    for (unsigned h = 0; h < k_height; h++) {
      for (unsigned w = 0; w < k_width; w++) {
        for (unsigned ic = 0; ic < chan_b256_in; ic++) {
          for (unsigned sub_grp_idx = 0; sub_grp_idx < ACC_PERIOD; sub_grp_idx++) {

            memcpy(&K[output_chan_group][h][w][ic][ACC_PERIOD - 1 - sub_grp_idx],
              &K_ref[output_chan_group * ACC_PERIOD + sub_grp_idx][h][w][ic], 
              sizeof(bnn_b256_t));
          }
        }
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

  bnn_b256_t(*K)[k_height][k_width][chan_b256_in][ACC_PERIOD] =
      (bnn_b256_t(*)[k_height][k_width][chan_b256_in][ACC_PERIOD])K_p;

  for (unsigned output_chan_group = 0; output_chan_group < chans_out / ACC_PERIOD; 
      output_chan_group++) {
    for (unsigned h = 0; h < k_height; h++) {
      for (unsigned w = 0; w < k_width; w++) {
        for (unsigned ic = 0; ic < chan_b256_in; ic++) {
          for (unsigned sub_grp_idx = 0; sub_grp_idx < ACC_PERIOD; sub_grp_idx++) {

            //This is to compensate for the way the asm interleaves the 
            //upper and lower 8 outputs.
            unsigned interleaved_oc;
            if (sub_grp_idx < (ACC_PERIOD/2)) {
              interleaved_oc = (2*sub_grp_idx) + 1;
            } else{
              interleaved_oc = 2*(sub_grp_idx - (ACC_PERIOD/2));
            }
            memcpy(& K[output_chan_group][h][w][ic][ACC_PERIOD - 1 - sub_grp_idx], 
              & K_ref[output_chan_group * ACC_PERIOD + interleaved_oc][h][w][ic], 
              sizeof(bnn_b256_t));
          }
        }
      }
    }
  }
}

unsigned xor_pop(bnn_b256_t* a, bnn_b256_t* b) {
  unsigned t = sizeof(((bnn_b256_t*)0)->d[0]);
  unsigned elements = sizeof(((bnn_b256_t*)0)->d) / t;

  unsigned c = 0;
  for (unsigned e = 0; e < elements; e++) {
    uint32_t v = a->d[e] ^ b->d[e];
 #if defined(__XS3A__)
    v = ~v;
    for (unsigned i = 0; i < t * 8; i++) {
      c += (v & 1);
      v >>= 1;
    }
    #else
    c += __builtin_popcount(~v);
    #endif
  }
  return c;
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
                sum += xor_pop(&(X[h + kh][w + kw][ic]), &(K[oc][kh][kw][ic]));
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
              sum += xor_pop(&(X[h + kh][w + kw][ic]), &(K[oc][kh][kw][ic]));
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
