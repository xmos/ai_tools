
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#include "tst_common.h"
#include "nn_operator.h"
#include "nn_types.h"
#include "xs3_vpu.h"
#include "nn_bin_utils.h"

// #include "dsp_xs3_vector.h"
#include "unity.h"

void print_vector(bnn_b256_t b) {
  printf("0x");
  for (unsigned i = 0; i < 8; i++) printf("%08lx", b.d[i]);
  printf("\n");
}

#define OUTPUT_LENGTH(input_length, filter_size, dilation, stride)            \
  (((input_length - (filter_size + (filter_size - 1) * (dilation - 1)) + 1) + \
    stride - 1) /                                                             \
   stride)

void larq_ref_bconv2d(const nn_image_params_t* x, const nn_image_params_t* y,
                      const nn_window_params_t* k,
                      const int32_t* packed_input_data,
                      const int32_t* packed_filter_data,
                      int32_t* packed_output_data, const long* thresholds);

void boggle_threshold(int32_t* thresh_boggled, int32_t* thresholds_ref,
                      unsigned chans_out, unsigned receptive_field) {
  int16_t* thresholds = (int16_t*)thresh_boggled;

  // boggle the threshold(accum init)
  for (unsigned i = 0; i < chans_out; i++) {
    unsigned bank = i / 16;

    int32_t t = thresholds_ref[i] - ((receptive_field) / 2);
    thresholds[(bank * 32) + (i % 16)] = (t >> 0);
    thresholds[(bank * 32) + (i % 16) + 16] = (t >> 16);
  }
}

void boggle_kernel(bnn_b256_t* K_p, bnn_b256_t* K_ref_p, unsigned k_height,
                   unsigned k_width, unsigned chans_in, unsigned chans_out) {
  unsigned chan_b256_in =
      (chans_in + XS3_VPU_VREG_WIDTH_BITS - 1) / XS3_VPU_VREG_WIDTH_BITS;

  bnn_b256_t(*K_ref)[k_height][k_width][chan_b256_in] =
      (bnn_b256_t(*)[k_height][k_width][chan_b256_in])K_ref_p;

  bnn_b256_t(*K)[k_height][k_width][chan_b256_in][16] =
      (bnn_b256_t(*)[k_height][k_width][chan_b256_in][16])K_p;

  for (unsigned oc = 0; oc < chans_out / 16; oc++) {
    for (unsigned h = 0; h < k_height; h++) {
      for (unsigned w = 0; w < k_width; w++) {
        for (unsigned ic = 0; ic < chan_b256_in; ic++) {
          for (unsigned o = 0; o < 16; o++) {
            for (unsigned i = 0; i < 8; i++) {
              K[oc][h][w][ic][15 - o].d[i] = K_ref[oc * 16 + o][h][w][ic].d[i];
            }
          }
        }
      }
    }
  }
}

/*
X_ref and K_ref must be initialised before running this.
*/
int run_config(bnn_b32_t* Y_p, bnn_b32_t* Y_ref_p, bnn_b256_t* X_ref,
               bnn_b256_t* K_p, bnn_b256_t* K_ref_p, int32_t* thresholds_ref,
               int32_t* thresholds_p, unsigned x_height, unsigned x_width,
               unsigned k_height, unsigned k_width, unsigned chans_in,
               unsigned chans_out, unsigned h_stride, unsigned v_stride) {
  assert(Y_p != Y_ref_p);
  assert(K_p != K_ref_p);
  assert(thresholds_p != thresholds_ref);

  unsigned y_height = OUTPUT_LENGTH(x_height, k_height, 1, v_stride);
  unsigned y_width = OUTPUT_LENGTH(x_width, k_width, 1, h_stride);

  unsigned X_bytes = (x_height * x_width * chans_in) / 8;
  unsigned K_bytes = (k_width * k_height * chans_in * chans_out) / 8;
  unsigned Y_bytes = (y_width * y_height * chans_out) / 8;

  // This makes for a nice threshold for a random input
  for (unsigned i = 0; i < chans_out; i++)
    thresholds_ref[i] = i + ((chans_in * k_height * k_width - chans_out) / 2);

  boggle_threshold(thresholds_p, thresholds_ref, chans_out,
                   k_width * k_height * chans_in);

  boggle_kernel(K_p, K_ref_p, k_height, k_width, chans_in, chans_out);

  nn_image_params_t x;
  x.height = x_height;
  x.width = x_width;
  x.channels = chans_in;
  nn_image_params_t y;
  y.height = y_height;
  y.width = y_width;
  y.channels = chans_out;
  nn_window_params_t k;
  k.shape.height = k_height;
  k.shape.width = k_width;
  k.start.column = 0;
  k.start.row = 0;
  k.stride.horizontal = h_stride;
  k.stride.vertical = v_stride;

  larq_ref_bconv2d(&x, &y, &k, (int32_t*)X_ref, (int32_t*)K_ref_p,
                   (int32_t*)Y_ref_p, thresholds_ref);
#if 1
  nn_bnn_conv2d_bin_out_asm_plan_t plan;
  bnn_conv2d_bin_out_asm_init(&plan, &x, &y, &k);
  bnn_conv2d_bin_out_asm((bnn_b32_t*)Y_p, (bnn_b256_t*)X_ref, (bnn_b256_t*)K_p,
                         thresholds_p, &plan);
#else
  nn_bnn_conv2d_bin_out_plan_t plan;
  bnn_conv2d_bin_out_init(&plan, &x, &y, &k);
  bnn_conv2d_bin_out((bnn_b32_t*)Y_p, (bnn_b256_t*)X_ref, (bnn_b256_t*)K_ref,
                     thresholds_ref, &plan);
#endif

  unsigned chan_b32_out = (chans_out + 32 - 1) / 32;
  bnn_b32_t(*Y)[y_width][chan_b32_out] =
      (bnn_b32_t(*)[y_width][chan_b32_out])Y_p;

  bnn_b32_t(*Y_ref)[y_width][chan_b32_out] =
      (bnn_b32_t(*)[y_width][chan_b32_out])Y_ref_p;

  int all_equal = 1;
  for (unsigned h = 0; h < y_height; h++) {
    for (unsigned w = 0; w < y_width; w++) {
      for (unsigned c = 0; c < chan_b32_out; c++) {
        all_equal &= (Y_ref[h][w][c] == Y[h][w][c]);
      }
    }
  }

  if (1 - all_equal) {
    printf("\n");
    printf("x_height:%u\n", x_height);
    printf("x_width:%u\n", x_width);
    printf("k_height:%u\n", k_height);
    printf("k_width:%u\n", k_width);
    printf("chans_in:%u\n", chans_in);
    printf("chans_out:%u\n", chans_out);
    printf("h_stride:%u\n", h_stride);
    printf("v_stride:%u\n", v_stride);
    printf("y_height:%u\n", y_height);
    printf("y_width:%u\n", y_width);
  }
  return 1 - all_equal;
}

// https://github.com/tensorflow/tensorflow/blob/5912f51d580551e5cee2cfde4cb882594b4d3e60/tensorflow/python/keras/utils/conv_utils.py#L140
void test_bnn_conv2d_bin_out_pseudo_directed() {
#define X_V_DILATION 1
#define X_H_DILATION 1

#define X_HEIGHT 1
#define X_WIDTH 1
#define K_HEIGHT 1
#define K_WIDTH 1
#define CHANS_IN 512
#define CHANS_OUT 32
#define H_STRIDE 1
#define V_STRIDE 1
#define DILATED_FILTER_HEIGHT (K_HEIGHT + (K_HEIGHT - 1) * (X_V_DILATION - 1))
#define DILATED_FILTER_WIDTH (K_WIDTH + (K_WIDTH - 1) * (X_H_DILATION - 1))

#define Y_HEIGHT OUTPUT_LENGTH(X_HEIGHT, K_HEIGHT, X_V_DILATION, V_STRIDE)
#define Y_WIDTH OUTPUT_LENGTH(X_WIDTH, K_WIDTH, X_H_DILATION, H_STRIDE)

#define CHAN_WORDS_IN \
  ((CHANS_IN + XS3_VPU_VREG_WIDTH_BITS - 1) / XS3_VPU_VREG_WIDTH_BITS)
#define CHAN_WORDS_OUT ((CHANS_OUT + 32 - 1) / 32)

  bnn_b256_t WORD_ALIGNED K_ref[CHANS_OUT][K_HEIGHT][K_WIDTH][CHAN_WORDS_IN];
  bnn_b256_t WORD_ALIGNED
      K[CHANS_OUT / 16][K_HEIGHT][K_WIDTH][CHAN_WORDS_IN][16];

  bnn_b256_t WORD_ALIGNED X_ref[X_HEIGHT][X_WIDTH][CHAN_WORDS_IN];
  bnn_b32_t WORD_ALIGNED Y_ref[Y_HEIGHT][Y_WIDTH][CHAN_WORDS_OUT];
  bnn_b32_t WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHAN_WORDS_OUT];

  int32_t WORD_ALIGNED thresholds_ref[CHANS_OUT];
  int32_t WORD_ALIGNED thresholds[CHANS_OUT];

  srand(1);
  pseudo_rand_bytes((char*)X_ref, sizeof(X_ref));
  pseudo_rand_bytes((char*)K_ref, sizeof(K_ref));

  int failure =
      run_config((bnn_b32_t*)Y, (bnn_b32_t*)Y_ref, (bnn_b256_t*)X_ref,
                 (bnn_b256_t*)K, (bnn_b256_t*)K_ref, (int32_t*)thresholds_ref,
                 (int32_t*)thresholds, X_HEIGHT, X_WIDTH, K_HEIGHT, K_WIDTH,
                 CHANS_IN, CHANS_OUT, H_STRIDE, V_STRIDE);

  if (failure) {
    exit(1);
  }
  TEST_ASSERT_FALSE(failure);

#undef H_STRIDE
#undef V_STRIDE
#undef H_OFFSET
#undef V_OFFSET
#undef K_HEIGHT
#undef K_WIDTH
#undef CHANS_IN
#undef CHANS_OUT
#undef X_HEIGHT
#undef X_WIDTH
#undef Y_HEIGHT
#undef Y_WIDTH
}

void test_bnn_conv2d_bin_out_pseudo_random() {
#define MIN_H_STRIDE 1
#define MIN_V_STRIDE 1
#define MAX_H_STRIDE 4
#define MAX_V_STRIDE 4

#define MIN_K_HEIGHT 1
#define MIN_K_WIDTH 1
#define MAX_K_HEIGHT 5
#define MAX_K_WIDTH 5

#define MIN_CHANS_IN 256
#define MAX_CHANS_IN 512

#define MIN_CHANS_OUT 32
#define MAX_CHANS_OUT 64

#define MIN_X_HEIGHT MIN_K_HEIGHT
#define MIN_X_WIDTH MIN_K_WIDTH
#define MAX_X_HEIGHT 7
#define MAX_X_WIDTH 7

#define MAX_CHAN_WORDS_IN \
  ((MAX_CHANS_IN + XS3_VPU_VREG_WIDTH_BITS - 1) / XS3_VPU_VREG_WIDTH_BITS)
#define MAX_CHAN_WORDS_OUT ((MAX_CHANS_OUT + 32 - 1) / 32)

#define MAX_Y_HEIGHT (((MAX_X_HEIGHT - MIN_K_HEIGHT + 1) / MIN_V_STRIDE))
#define MAX_Y_WIDTH (((MAX_X_WIDTH - MIN_K_WIDTH + 1) / MIN_H_STRIDE))

  bnn_b256_t WORD_ALIGNED
      K_ref[MAX_CHANS_OUT][MAX_K_HEIGHT][MAX_K_WIDTH][MAX_CHAN_WORDS_IN];
  bnn_b256_t WORD_ALIGNED
      K[MAX_CHANS_OUT][MAX_K_HEIGHT][MAX_K_WIDTH][MAX_CHAN_WORDS_IN];

  bnn_b256_t WORD_ALIGNED X_ref[MAX_X_HEIGHT][MAX_X_WIDTH][MAX_CHAN_WORDS_IN];
  bnn_b32_t WORD_ALIGNED Y_ref[MAX_Y_HEIGHT][MAX_Y_WIDTH][MAX_CHAN_WORDS_OUT];
  bnn_b32_t WORD_ALIGNED Y[MAX_Y_HEIGHT][MAX_Y_WIDTH][MAX_CHAN_WORDS_OUT];

  int32_t WORD_ALIGNED thresholds_ref[MAX_CHANS_OUT];
  int32_t WORD_ALIGNED thresholds[MAX_CHANS_OUT];

  assert(((int)K & 0x3) == 0);
  assert(((int)K_ref & 0x3) == 0);
  assert(((int)X_ref & 0x3) == 0);
  assert(((int)Y & 0x3) == 0);
  assert(((int)Y_ref & 0x3) == 0);

  assert(((int)thresholds_ref & 0x3) == 0);
  assert(((int)thresholds & 0x3) == 0);

  printf("sizeof(X_ref): %u\n", sizeof(X_ref));
  printf("sizeof(K_ref): %u\n", sizeof(K_ref));
  printf("sizeof(K): %u\n", sizeof(K));
  printf("sizeof(Y_ref): %u\n", sizeof(Y_ref));
  printf("sizeof(Y): %u\n", sizeof(Y));
  printf("sizeof(thresholds_ref): %u\n", sizeof(thresholds_ref));
  printf("sizeof(thresholds): %u\n", sizeof(thresholds));

  pseudo_rand_bytes((char*)X_ref, sizeof(X_ref));
  pseudo_rand_bytes((char*)K_ref, sizeof(K_ref));

  srand(69);

  unsigned c = 0;
  for (unsigned h_stride = MIN_H_STRIDE; h_stride <= MAX_H_STRIDE; ++h_stride) {
    for (unsigned v_stride = MIN_V_STRIDE; v_stride <= MAX_V_STRIDE;
         ++v_stride) {
      for (unsigned k_height = MIN_K_HEIGHT; k_height <= MAX_K_HEIGHT;
           ++k_height) {
        for (unsigned k_width = MIN_K_WIDTH; k_width <= MAX_K_WIDTH;
             ++k_width) {
          for (unsigned x_height = k_height; x_height <= MAX_X_HEIGHT;
               ++x_height) {
            for (unsigned x_width = k_width; x_width <= MAX_X_WIDTH;
                 ++x_width) {
              for (unsigned chans_in = MIN_CHANS_IN; chans_in <= MAX_CHANS_IN;
                   chans_in += 256) {
                for (unsigned chans_out = MIN_CHANS_OUT;
                     chans_out <= MAX_CHANS_OUT; chans_out += 32) {
                  int r = run_config(
                      (bnn_b32_t*)Y, (bnn_b32_t*)Y_ref, (bnn_b256_t*)X_ref,
                      (bnn_b256_t*)K, (bnn_b256_t*)K_ref,
                      (int32_t*)thresholds_ref, (int32_t*)thresholds, x_height,
                      x_width, k_height, k_width, chans_in, chans_out, h_stride,
                      v_stride);
                  if ((c & 0x3ff) == 0) printf("%u\n", c);
                  c++;
                  TEST_ASSERT_FALSE(r);
                }
              }
            }
          }
        }
      }
    }
  }
}

void test_bnn_conv2d_bin_out_benchmark() {}

void test_bnn_conv2d() {
  UNITY_SET_FILE();
  RUN_TEST(test_bnn_conv2d_bin_out_pseudo_directed);
  RUN_TEST(test_bnn_conv2d_bin_out_pseudo_random);
  // RUN_TEST(test_bnn_conv2d_bin_out_benchmark);
}