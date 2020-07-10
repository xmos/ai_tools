

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

#include "nn_operator.h"
#include "../nn_op_helper.h"
#include "nn_op_structs.h"

#include "xs3_vpu.h"

unsigned vlmaccr1(bnn_b256_t* a, bnn_b256_t* b) {
  unsigned t = sizeof(((bnn_b256_t*)0)->d[0]);
  unsigned elements = sizeof(((bnn_b256_t*)0)->d) / t;

  unsigned c = 0;
  for (unsigned e = 0; e < elements; e++) {
    uint32_t v = a->d[e] ^ b->d[e];
    for (unsigned i = 0; i < t * 8; i++) {
      c += (v & 1);
      v >>= 1;
    }
  }
  return c;
}

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

  //   bnn_t WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
  //   bnn_t WORD_ALIGNED X[X_HEIGHT][X_WIDTH][CHAN_WORDS_IN];
  //   bnn_t WORD_ALIGNED K[CHANS_OUT][K_HEIGHT][K_WIDTH][CHAN_WORDS_IN];

  int32_t(*Y)[y_width][chan_b32_out] = (int32_t(*)[y_width][chan_b32_out])Y_p;

  bnn_b256_t(*X)[x_width][chan_b256_in] =
      (bnn_b256_t(*)[x_width][chan_b256_in])X_p;

  bnn_b256_t(*K)[kernel_height][kernel_width][chan_b256_in] =
      (bnn_b256_t(*)[kernel_height][kernel_width][chan_b256_in])K_p;

  for (unsigned h = 0; h < x_height - kernel_height + 1; h += plan->stride[0]) {
    for (unsigned w = plan->start_loc[1]; w < x_width - kernel_width + 1;
         w += plan->stride[1]) {
      for (unsigned oc_word = 0; oc_word < chan_b32_out; oc_word += 1) {
        int32_t bitpacked_column = 0;
        for (unsigned oc_bit = 0; oc_bit < 32; oc_bit += 1) {
          unsigned oc = oc_bit + 32 * oc_word;
          int32_t sum = 0;
          for (unsigned kh = 0; kh < kernel_height; kh += 1) {
            for (unsigned kw = 0; kw < kernel_width; kw += 1) {
              for (unsigned ic = 0; ic < chan_b256_in; ic += 1) {
                sum += vlmaccr1(
                    &(X[h * plan->stride[0] + kh + plan->start_loc[0]]
                       [w * plan->stride[1] + kw + plan->start_loc[1]][ic]),
                    &(K[oc][kh][kw][ic]));
              }
            }
          }
          // printf("sum %d\n", sum);
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
                            int32_t* threshold,  //[out_channel];
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
        int32_t sum = 0;
        for (unsigned kh = 0; kh < kernel_height; kh += 1) {
          for (unsigned kw = 0; kw < kernel_width; kw += 1) {
            for (unsigned ic = 0; ic < chans_in; ic += 1) {
              sum += (X[h + kh][w + kw][ic] * K[oc][kh][kw][ic]);
            }
          }
        }
        // Convert to pop count
        sum = ((int32_t)(chans_in * kernel_height * kernel_width) - sum) / 2;

        // printf("sum %d\n", sum);

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
