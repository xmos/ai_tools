

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

#include "nn_operator.h"
#include "../nn_op_helper.h"
#include "nn_op_structs.h"

#include "xs3_vpu.h"

void bnn_conv2d_bin_out_asm_init(nn_bnn_conv2d_bin_out_asm_plan_t* plan,
                                 const nn_image_params_t* x,
                                 const nn_image_params_t* y,
                                 const nn_window_params_t* k) {
  unsigned bytes_per_input_channel = x->channels / 8;
  unsigned bytes_per_output_channel = y->channels / 8;

  // This is 32 to make it easier and be more compatable with larq
  const unsigned out_chans_multiplier = 32;

  assert((x->channels % XS3_VPU_VREG_WIDTH_BITS) == 0);
  assert((y->channels % out_chans_multiplier) == 0);

  plan->k_height_loop_counter = k->shape.height - 1;
  plan->k_width_loop_counter = k->shape.width - 1;

  plan->input_channel_loop_counter =
      (x->channels / XS3_VPU_VREG_WIDTH_BITS) - 1;
  plan->output_channel_loop_counter = (y->channels / out_chans_multiplier) - 1;
  plan->x_height_loop_counter = x->height - (k->shape.height - 1);
  plan->x_width_loop_counter = x->width - 1 - (k->shape.width - 1);

  plan->y_v_step = 0;

  // This assumes that any slicing will be done horizontally.
  plan->inner_x_h_step = bytes_per_input_channel * (k->stride.horizontal - 1);
  plan->inner_x_v_step =
      (bytes_per_input_channel * (k->shape.width - 1)) - plan->inner_x_h_step;

  plan->outer_x_h_step = bytes_per_input_channel * (k->stride.horizontal);
  plan->outer_x_v_step =
      (bytes_per_input_channel * (k->shape.width)) - plan->outer_x_h_step;
}

unsigned xor_pop(bnn_b256_t* a, bnn_b256_t* b) {
  unsigned t = sizeof(((bnn_b256_t*)0)->d[0]);
  unsigned elements = sizeof(((bnn_b256_t*)0)->d) / t;

  unsigned c = 0;
  for (unsigned e = 0; e < elements; e++) {
    uint32_t v = a->d[e] ^ b->d[e];
    v = ~v;
    for (unsigned i = 0; i < t * 8; i++) {
      c += (v & 1);
      v >>= 1;
    }
  }
  return c;
}

void print_vector(bnn_b256_t b);

WEAK_FUNC
void bnn_conv2d_bin_out(bnn_b32_t* Y_p, const bnn_b256_t* X_p,
                        const bnn_b256_t* K_p,
                        int32_t* thresholds,  //[out_channel];
                        const nn_bnn_conv2d_bin_out_plan_t* plan) {
  const unsigned kernel_height = plan->k_dims[0];
  const unsigned kernel_width = plan->k_dims[1];
  const unsigned chan_b256_in = plan->x_dims[2];
  const unsigned chan_b32_out = plan->y_dims[2];
  const unsigned x_height = plan->x_dims[0];
  const unsigned x_width = plan->x_dims[1];
  const unsigned y_height = plan->y_dims[0];
  const unsigned y_width = plan->y_dims[1];

  bnn_b32_t(*Y)[y_width][chan_b32_out] =
      (bnn_b32_t(*)[y_width][chan_b32_out])Y_p;

  bnn_b256_t(*X)[x_width][chan_b256_in] =
      (bnn_b256_t(*)[x_width][chan_b256_in])X_p;

  bnn_b256_t(*K)[kernel_height][kernel_width][chan_b256_in] =
      (bnn_b256_t(*)[kernel_height][kernel_width][chan_b256_in])K_p;

  for (unsigned h = 0; h < x_height - kernel_height + 1; h += plan->stride[0]) {
    for (unsigned w = plan->start_loc[1]; w < x_width - kernel_width + 1;
         w += plan->stride[1]) {
      for (unsigned oc_word = 0; oc_word < chan_b32_out; oc_word += 1) {
        bnn_b32_t bitpacked_column = 0;

        for (unsigned oc_bit = 0; oc_bit < 32; oc_bit += 1) {
          unsigned oc = oc_bit + (32 * oc_word);
          int32_t sum = 0;
          for (unsigned kh = 0; kh < kernel_height; kh += 1) {
            for (unsigned kw = 0; kw < kernel_width; kw += 1) {
              for (unsigned ic = 0; ic < chan_b256_in; ic += 1) {
                if ((oc == 0) || (oc == 16)) {
                  print_vector(
                      X[h * plan->stride[0] + kh + plan->start_loc[0]]
                       [w * plan->stride[1] + kw + plan->start_loc[1]][ic]);
                  print_vector(K[oc][kh][kw][ic]);
                  printf("\n");
                }
                sum += xor_pop(
                    &(X[h * plan->stride[0] + kh + plan->start_loc[0]]
                       [w * plan->stride[1] + kw + plan->start_loc[1]][ic]),
                    &(K[oc][kh][kw][ic]));
              }
            }
          }

          sum = (kernel_height * kernel_width * chan_b256_in * 256) - sum;

          // printf("c %u %u %u %ld\n", h, w, oc, sum);
          unsigned bit = sum > thresholds[oc];

          if (bit) bitpacked_column |= 1ULL << oc_bit;
        }
        Y[h][w][oc_word] = bitpacked_column;
      }
    }
  }
}

void bnn_conv2d_bin_out_init(nn_bnn_conv2d_bin_out_plan_t* plan,
                             const nn_image_params_t* x,
                             const nn_image_params_t* y,
                             const nn_window_params_t* k) {
  // assert((y->channels % 16) == 0);
  // assert((x->channels % XS3_VPU_VREG_WIDTH_BITS) == 0);
  plan->k_dims[0] = k->shape.height;
  plan->k_dims[1] = k->shape.width;

  plan->y_dims[0] = y->height;
  plan->y_dims[1] = y->width;
  plan->y_dims[2] = (y->channels + 32 - 1) / 32;

  plan->x_dims[0] = x->height;
  plan->x_dims[1] = x->width;
  plan->x_dims[2] =
      (x->channels + XS3_VPU_VREG_WIDTH_BITS - 1) / XS3_VPU_VREG_WIDTH_BITS;
  plan->stride[0] = k->stride.vertical;
  plan->stride[1] = k->stride.horizontal;

  plan->start_loc[0] = k->start.column;
  plan->start_loc[1] = k->start.row;
}

WEAK_FUNC
void bnn_conv2d_bin_out_ref(bnn_bool_t* Y_p, const bnn_bool_t* X_p,
                            const bnn_bool_t* K_p,
                            int16_t* threshold,  //[out_channel];
                            const nn_bnn_conv2d_bin_out_ref_plan_t* plan) {
  const unsigned kernel_height = plan->k_dims[0];
  const unsigned kernel_width = plan->k_dims[1];
  const unsigned chans_in = plan->x_dims[2];
  const unsigned chans_out = plan->y_dims[2];
  const unsigned x_height = plan->x_dims[0];
  const unsigned x_width = plan->x_dims[1];
  const unsigned y_height = plan->y_dims[0];
  const unsigned y_width = plan->y_dims[1];

  //   bnn_t WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
  //   bnn_t WORD_ALIGNED X[X_HEIGHT][X_WIDTH][CHANS_IN];
  //   bnn_t WORD_ALIGNED K[CHANS_OUT][K_HEIGHT][K_WIDTH][CHANS_IN];

  bnn_bool_t(*Y)[y_width][chans_out] = (bnn_bool_t(*)[y_width][chans_out])Y_p;

  bnn_bool_t(*X)[x_width][chans_in] = (bnn_bool_t(*)[x_width][chans_in])X_p;

  bnn_bool_t(*K)[kernel_height][kernel_width][chans_in] =
      (bnn_bool_t(*)[kernel_height][kernel_width][chans_in])K_p;

  for (unsigned h = plan->start_loc[0]; h < x_height - kernel_height + 1;
       h += plan->stride[0]) {
    for (unsigned w = plan->start_loc[1]; w < x_width - kernel_width + 1;
         w += plan->stride[1]) {
      for (unsigned oc = 0; oc < chans_out; oc += 1) {
        int16_t sum = 0;
        for (unsigned kh = 0; kh < kernel_height; kh += 1) {
          for (unsigned kw = 0; kw < kernel_width; kw += 1) {
            for (unsigned ic = 0; ic < chans_in; ic += 1) {
              sum += (X[h + kh][w + kw][ic] != K[oc][kh][kw][ic]);
            }
          }
        }
        // Convert to pop count
        sum = ((int16_t)(chans_in * kernel_height * kernel_width) - sum) / 2;
        bnn_bool_t v = sum > threshold[oc];
        Y[h][w][oc] = 1 - (2 * v);
      }
    }
  }
}

void bnn_conv2d_bin_out_ref_init(nn_bnn_conv2d_bin_out_ref_plan_t* plan,
                                 const nn_image_params_t* x,
                                 const nn_image_params_t* y,
                                 const nn_window_params_t* k) {
  plan->k_dims[0] = k->shape.height;
  plan->k_dims[1] = k->shape.width;

  plan->y_dims[0] = y->height;
  plan->y_dims[1] = y->width;
  plan->y_dims[2] = y->channels;

  plan->x_dims[0] = x->height;
  plan->x_dims[1] = x->width;
  plan->x_dims[2] = x->channels;

  plan->stride[0] = k->stride.vertical;
  plan->stride[1] = k->stride.horizontal;

  plan->start_loc[0] = k->start.column;
  plan->start_loc[1] = k->start.row;
}
