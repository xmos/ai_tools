
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

void larq_ref_bconv2d(const nn_image_params_t* x, const nn_image_params_t* y,
                      const nn_window_params_t* k,
                      const int32_t* packed_input_data,
                      const int32_t* packed_filter_data,
                      int32_t* packed_output_data, const long* thresholds

);

void print_vector(bnn_b256_t b) {
  printf("0x");
  for (unsigned i = 0; i < 8; i++) printf("%08lx", b.d[i]);
  printf("\n");
}

void test_bnn_conv2d_bin_out_pseudo_random() {
#define H_STRIDE 1
#define V_STRIDE 1
#define H_OFFSET 0
#define V_OFFSET 0
#define K_HEIGHT 3
#define K_WIDTH 3
#define CHANS_IN 256
#define CHANS_OUT 32
#define X_HEIGHT 6
#define X_WIDTH 6
#define Y_HEIGHT (((X_HEIGHT - V_OFFSET) / V_STRIDE) - K_HEIGHT + 1)
#define Y_WIDTH (((X_WIDTH - H_OFFSET) / H_STRIDE) - K_WIDTH + 1)

#define CHAN_WORDS_IN \
  ((CHANS_IN + XS3_VPU_VREG_WIDTH_BITS - 1) / XS3_VPU_VREG_WIDTH_BITS)
#define CHAN_WORDS_OUT ((CHANS_OUT + 32 - 1) / 32)

  bnn_b256_t WORD_ALIGNED K_ref[CHANS_OUT][K_HEIGHT][K_WIDTH][CHAN_WORDS_IN];
  bnn_b256_t WORD_ALIGNED
      K[CHANS_OUT / 16][K_HEIGHT][K_WIDTH][CHAN_WORDS_IN][16];

  bnn_b256_t WORD_ALIGNED X_ref[X_HEIGHT][X_WIDTH][CHAN_WORDS_IN];
  bnn_b32_t WORD_ALIGNED Y_ref[Y_HEIGHT][Y_WIDTH][CHAN_WORDS_OUT];
  bnn_b32_t WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHAN_WORDS_OUT];

  memset(Y, 0, sizeof(Y));
  memset(Y_ref, 0, sizeof(Y_ref));

  memset(X_ref, 0, sizeof(X_ref));
  memset(K_ref, 0, sizeof(K_ref));

  srand(69);
  // void pseudo_rand_bytes(char* buffer, unsigned size)
  for (unsigned i = 0; i < sizeof(X_ref); i++) ((char*)&X_ref)[i] = rand();
  for (unsigned i = 0; i < sizeof(K_ref); i++) ((char*)&K_ref)[i] = rand();

  // memset(K_ref, 0xffffffff, sizeof(K_ref));

  int32_t thresholds_ref[CHANS_OUT];
  for (unsigned i = 0; i < CHANS_OUT; i++)
    thresholds_ref[i] = i + ((CHANS_IN * K_HEIGHT * K_WIDTH - CHANS_OUT) / 2);

  // boggle the kernel addresses
  for (unsigned oc = 0; oc < CHANS_OUT / 16; oc++) {
    for (unsigned h = 0; h < K_HEIGHT; h++) {
      for (unsigned w = 0; w < K_WIDTH; w++) {
        for (unsigned o = 0; o < 16; o++) {
          for (unsigned ic = 0; ic < CHAN_WORDS_OUT; ic++) {
            memcpy(&K[oc][h][w][ic][o], &K_ref[oc * 16 + o][h][w][ic],
                   sizeof(bnn_b256_t));
          }
        }
      }
    }
  }

  int16_t thresholds[CHANS_OUT];
  // boggle the threshold values
  for (unsigned i = 0; i < CHANS_OUT; i++) thresholds[i] = thresholds_ref[i];

  nn_image_params_t x;
  x.height = X_HEIGHT;
  x.width = X_WIDTH;
  x.channels = CHANS_IN;
  nn_image_params_t y;
  y.height = Y_HEIGHT;
  y.width = Y_WIDTH;
  y.channels = CHANS_OUT;
  nn_window_params_t k;
  k.shape.height = K_HEIGHT;
  k.shape.width = K_WIDTH;
  k.start.column = H_OFFSET;
  k.start.row = V_OFFSET;
  k.stride.horizontal = H_STRIDE;
  k.stride.vertical = V_STRIDE;

  larq_ref_bconv2d(&x, &y, &k, (int32_t*)X_ref, (int32_t*)K_ref,
                   (int32_t*)Y_ref, thresholds_ref);

  nn_bnn_conv2d_bin_out_asm_plan_t plan;
  bnn_conv2d_bin_out_asm_init(&plan, &x, &y, &k);
  bnn_conv2d_bin_out_asm((bnn_b32_t*)&Y, (bnn_b256_t*)&X_ref, (bnn_b256_t*)&K,
                         thresholds, &plan);

  // nn_bnn_conv2d_bin_out_plan_t plan_ref;
  // bnn_conv2d_bin_out_init(&plan_ref, &x, &y, &k);
  // bnn_conv2d_bin_out((bnn_b32_t*)&Y, (bnn_b256_t*)&X_ref,
  // (bnn_b256_t*)&K_ref,
  //                    thresholds, &plan_ref);

  for (unsigned h = 0; h < Y_HEIGHT; h++) {
    for (unsigned w = 0; w < Y_WIDTH; w++) {
      for (unsigned co = 0; co < CHANS_OUT; co++) {
        TEST_ASSERT_EQUAL_INT8(get_bit_b32(Y[h][w], co),
                               get_bit_b32(Y_ref[h][w], co));
      }
    }
  }
}

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

void test_bnn_conv2d_bin_out_directed() {
#define H_STRIDE 1
#define V_STRIDE 1
#define H_OFFSET 0
#define V_OFFSET 0
#define K_HEIGHT 2
#define K_WIDTH 2
#define CHANS_IN 256
#define CHANS_OUT 32
#define X_HEIGHT 5
#define X_WIDTH 5
#define Y_HEIGHT (((X_HEIGHT - H_OFFSET) / V_STRIDE) - K_HEIGHT + 1)
#define Y_WIDTH (((X_WIDTH - V_OFFSET) / H_STRIDE) - K_WIDTH + 1)

#define CHAN_WORDS_IN \
  ((CHANS_IN + XS3_VPU_VREG_WIDTH_BITS - 1) / XS3_VPU_VREG_WIDTH_BITS)
#define CHAN_WORDS_OUT ((CHANS_OUT + 32 - 1) / 32)

  bnn_bool_t WORD_ALIGNED K_ref[CHANS_OUT][K_HEIGHT][K_WIDTH][CHANS_IN];
  bnn_bool_t WORD_ALIGNED X_ref[X_HEIGHT][X_WIDTH][CHANS_IN];
  bnn_bool_t WORD_ALIGNED Y_ref[Y_HEIGHT][Y_WIDTH][CHANS_OUT];

  bnn_b256_t WORD_ALIGNED K[CHANS_OUT][K_HEIGHT][K_WIDTH][CHAN_WORDS_IN];
  bnn_b256_t WORD_ALIGNED X[X_HEIGHT][X_WIDTH][CHAN_WORDS_IN];
  bnn_b32_t WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHAN_WORDS_OUT];

  srand(42);

  memset(K, 0, sizeof(K));
  memset(X, 0, sizeof(X));
  memset(Y, 0, sizeof(Y));
  memset(K_ref, 0, sizeof(K_ref));
  memset(X_ref, 0, sizeof(X_ref));
  memset(Y_ref, 0, sizeof(Y_ref));

  for (unsigned co = 0; co < CHANS_OUT; co++) {
    for (unsigned h = 0; h < K_HEIGHT; h++) {
      for (unsigned w = 0; w < K_WIDTH; w++) {
        for (unsigned ci = 0; ci < CHANS_IN; ci++) {
          K_ref[co][h][w][ci] = ((rand() & 1) * 2) - 1;
        }
      }
    }
  }

  for (unsigned h = 0; h < X_HEIGHT; h++) {
    for (unsigned w = 0; w < X_WIDTH; w++) {
      for (unsigned ci = 0; ci < CHANS_IN; ci++) {
        X_ref[h][w][ci] = ((rand() & 1) * 2) - 1;
      }
    }
  }

  pack_bits_b256((bnn_bool_t*)K_ref, (bnn_b256_t*)K,
                 CHANS_OUT * K_HEIGHT * K_WIDTH, CHANS_IN);
  pack_bits_b256((bnn_bool_t*)X_ref, (bnn_b256_t*)X, X_HEIGHT * X_WIDTH,
                 CHANS_IN);

  nn_image_params_t x;
  x.height = X_HEIGHT;
  x.width = X_WIDTH;
  x.channels = CHANS_IN;
  nn_image_params_t y;
  y.height = Y_HEIGHT;
  y.width = Y_WIDTH;
  y.channels = CHANS_OUT;
  nn_window_params_t k;
  k.shape.height = K_HEIGHT;
  k.shape.width = K_WIDTH;
  k.start.column = H_OFFSET;
  k.start.row = V_OFFSET;
  k.stride.horizontal = H_STRIDE;
  k.stride.vertical = V_STRIDE;

  int16_t thresholds[CHANS_OUT];
  for (unsigned i = 0; i < CHANS_OUT; i++)
    thresholds[i] = i + ((CHANS_IN * K_HEIGHT * K_WIDTH - CHANS_OUT) / 2);

  nn_bnn_conv2d_bin_out_ref_plan_t plan_ref;
  bnn_conv2d_bin_out_ref_init(&plan_ref, &x, &y, &k);
  bnn_conv2d_bin_out_ref((bnn_bool_t*)&Y_ref, (bnn_bool_t*)&X_ref,
                         (bnn_bool_t*)&K_ref, thresholds, &plan_ref);

  nn_bnn_conv2d_bin_out_plan_t plan;
  bnn_conv2d_bin_out_init(&plan, &x, &y, &k);
  bnn_conv2d_bin_out((bnn_b32_t*)&Y, (bnn_b256_t*)&X, (bnn_b256_t*)&K,
                     thresholds, &plan);

  int l = 0;
  for (unsigned h = 0; h < Y_HEIGHT; h++) {
    for (unsigned w = 0; w < Y_WIDTH; w++) {
      for (unsigned co = 0; co < CHANS_OUT; co++) {
        TEST_ASSERT_EQUAL_INT8(get_bit_b32(Y[h][w], co), Y_ref[h][w][co]);
      }
    }
  }
}

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

void test_bnn_conv2d_bin_out_ref_directed() {
#define H_STRIDE 1
#define V_STRIDE 1
#define H_OFFSET 0
#define V_OFFSET 0

#include "dir.inc"

  bnn_bool_t WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];

  memset(Y, 0, sizeof(Y));

  nn_image_params_t x;
  x.height = X_HEIGHT;
  x.width = X_WIDTH;
  x.channels = CHANS_IN;
  nn_image_params_t y;
  y.height = Y_HEIGHT;
  y.width = Y_WIDTH;
  y.channels = CHANS_OUT;
  nn_window_params_t k;
  k.shape.height = K_HEIGHT;
  k.shape.width = K_WIDTH;
  k.start.column = H_OFFSET;
  k.start.row = V_OFFSET;
  k.stride.horizontal = H_STRIDE;
  k.stride.vertical = V_STRIDE;

  nn_bnn_conv2d_bin_out_ref_plan_t plan;
  bnn_conv2d_bin_out_ref_init(&plan, &x, &y, &k);
  bnn_conv2d_bin_out_ref((bnn_bool_t*)&Y, (bnn_bool_t*)&X, (bnn_bool_t*)&K,
                         thresholds, &plan);

  for (unsigned ch = 0; ch < CHANS_OUT; ch++) {
    for (unsigned h = 0; h < y.height; h++) {
      for (unsigned w = 0; w < y.width; w++) {
        TEST_ASSERT_EQUAL_INT8(Y_expected[h][w][ch], Y[h][w][ch]);
      }
    }
  }
}

void test_bnn_conv2d() {
  UNITY_SET_FILE();

  RUN_TEST(test_bnn_conv2d_bin_out_pseudo_random);
  // RUN_TEST(test_bnn_conv2d_bin_out_ref_directed);
  // RUN_TEST(test_bnn_conv2d_bin_out_directed);
}