
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

void larq_ref_bconv2d(const nn_image_params_t* x, const nn_image_params_t* y,
                      const nn_window_params_t* k,
                      const int32_t* packed_input_data,
                      const int32_t* packed_filter_data,
                      int32_t* packed_output_data, const long* thresholds);

void run_config(bnn_b32_t* Y_p, bnn_b32_t* Y_ref_p, bnn_b256_t* X_ref,
                bnn_b256_t* K_p, bnn_b256_t* K_ref_p, int32_t* thresholds_ref,
                int32_t* thresholds_p, unsigned x_height, unsigned x_width,
                unsigned k_height, unsigned k_width, unsigned chans_in,
                unsigned chans_out, unsigned h_stride, unsigned v_stride) {
  unsigned y_height = x_height - k_height + 1;
  unsigned y_width = x_width - k_width + 1;

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

  unsigned X_bytes = (x_height * x_width * chans_in) / 8;
  unsigned K_bytes = (k_width * k_height * chans_in * chans_out) / 8;
  unsigned Y_bytes = (y_width * y_height * chans_out) / 8;

  memset(Y_p, 0, Y_bytes);
  memset(Y_ref_p, 0, Y_bytes);

  pseudo_rand_bytes((char*)X_ref, X_bytes);
  pseudo_rand_bytes((char*)K_ref_p, K_bytes);

  for (unsigned i = 0; i < chans_out; i++)
    thresholds_ref[i] = i + ((chans_in * k_height * k_width - chans_out) / 2);

  int16_t* thresholds = (int32_t(*))thresholds_p;

  // boggle the threshold(accum init)
  for (unsigned i = 0; i < chans_out; i++) {
    unsigned bank = i / 16;

    int32_t t = ((k_width * k_height * chans_in) / 2) - thresholds_ref[i] - 1;

    thresholds[(bank * 32) + (i % 16)] = (t >> 0);
    thresholds[(bank * 32) + (i % 16) + 16] = (t >> 16);
  }

  unsigned chan_b32_out = (chans_out + 32 - 1) / 32;
  unsigned chan_b256_in =
      (chans_in + XS3_VPU_VREG_WIDTH_BITS - 1) / XS3_VPU_VREG_WIDTH_BITS;

  bnn_b256_t(*K_ref)[k_height][k_width][chan_b256_in] =
      (bnn_b256_t(*)[k_height][k_width][chan_b256_in])K_ref_p;

  bnn_b256_t(*K)[k_height][k_width][chan_b256_in][16] =
      (bnn_b256_t(*)[k_height][k_width][chan_b256_in][16])K_p;

  // boggle the kernel addresses
  for (unsigned oc = 0; oc < chans_out / 16; oc++) {
    for (unsigned h = 0; h < k_height; h++) {
      for (unsigned w = 0; w < k_width; w++) {
        for (unsigned o = 0; o < 16; o++) {
          for (unsigned ic = 0; ic < chan_b256_in; ic++) {
            for (unsigned i = 0; i < 8; i++) {
              K[oc][h][w][ic][15 - o].d[i] = ~K_ref[oc * 16 + o][h][w][ic].d[i];
            }
          }
        }
      }
    }
  }

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

  assert(((int)K & 0x3) == 0);
  assert(((int)K_ref & 0x3) == 0);
  assert(((int)X_ref & 0x3) == 0);
  assert(((int)Y_ref_p & 0x3) == 0);

  larq_ref_bconv2d(&x, &y, &k, (int32_t*)X_ref, (int32_t*)K_ref,
                   (int32_t*)Y_ref_p, thresholds_ref);

#if 1
  nn_bnn_conv2d_bin_out_asm_plan_t plan;
  bnn_conv2d_bin_out_asm_init(&plan, &x, &y, &k);

  // printf("input_channel_loop_counter %u\n", plan.input_channel_loop_counter);
  // printf("k_height_loop_counter %u\n", plan.k_height_loop_counter);
  // printf("k_width_loop_counter %u\n", plan.k_width_loop_counter);
  // printf("output_channel_loop_counter %u\n",
  // plan.output_channel_loop_counter); printf("x_h_stride %u\n",
  // plan.x_h_stride); printf("x_height_loop_counter %u\n",
  // plan.x_height_loop_counter); printf("x_v_stride %u\n", plan.x_v_stride);
  // printf("x_width_loop_counter %u\n", plan.x_width_loop_counter);
  // printf("y_v_stride %u\n", plan.y_v_stride);

  bnn_conv2d_bin_out_asm((bnn_b32_t*)Y_p, (bnn_b256_t*)X_ref, (bnn_b256_t*)K,
                         (int32_t*)thresholds, &plan);
#else
  nn_bnn_conv2d_bin_out_plan_t plan;
  bnn_conv2d_bin_out_init(&plan, &x, &y, &k);
  bnn_conv2d_bin_out((bnn_b32_t*)Y_p, (bnn_b256_t*)X_ref, (bnn_b256_t*)K_ref,
                     thresholds_ref, &plan);
#endif

  bnn_b32_t(*Y)[y_width][chan_b32_out] =
      (bnn_b32_t(*)[y_width][chan_b32_out])Y_p;

  bnn_b32_t(*Y_ref)[y_width][chan_b32_out] =
      (bnn_b32_t(*)[y_width][chan_b32_out])Y_ref_p;

  int all_equal = 1;
  for (unsigned h = 0; h < y_height; h++) {
    for (unsigned w = 0; w < y_width; w++) {
      for (unsigned co = 0; co < chans_out; co++) {
        // printf("%u %u %u %d %d  \t%u\n", h, w, co, get_bit_b32(Y_ref[h][w],
        // co),
        //        get_bit_b32(Y[h][w], co),
        //        get_bit_b32(Y_ref[h][w], co) == get_bit_b32(Y[h][w], co));
        all_equal &= (get_bit_b32(Y_ref[h][w], co) == get_bit_b32(Y[h][w], co));
        // if (get_bit_b32(Y[h][w], co) != get_bit_b32(Y_ref[h][w], co)) {
        //   printf("Error\n");
        //   return;
        // }
      }
    }
  }

  TEST_ASSERT_EQUAL_INT(all_equal, 1);
}

void test_bnn_conv2d_bin_out_pseudo_random() {
#define H_STRIDE 1
#define V_STRIDE 1
#define H_OFFSET 0
#define V_OFFSET 0
#define K_HEIGHT 1
#define K_WIDTH 1
#define CHANS_IN 256
#define CHANS_OUT 32
#define X_HEIGHT 1
#define X_WIDTH 1
#define Y_HEIGHT (((X_HEIGHT - V_OFFSET) / V_STRIDE) - K_HEIGHT + 1)
#define Y_WIDTH (((X_WIDTH - H_OFFSET) / H_STRIDE) - K_WIDTH + 1)

#define CHAN_WORDS_IN \
  ((CHANS_IN + XS3_VPU_VREG_WIDTH_BITS - 1) / XS3_VPU_VREG_WIDTH_BITS)
#define CHAN_WORDS_OUT ((CHANS_OUT + 32 - 1) / 32)
  int64_t f;
  bnn_b256_t WORD_ALIGNED K_ref[CHANS_OUT][K_HEIGHT][K_WIDTH][CHAN_WORDS_IN];
  bnn_b256_t WORD_ALIGNED
      K[CHANS_OUT / 16][K_HEIGHT][K_WIDTH][CHAN_WORDS_IN][16];

  bnn_b256_t WORD_ALIGNED X_ref[X_HEIGHT][X_WIDTH][CHAN_WORDS_IN];
  bnn_b32_t WORD_ALIGNED Y_ref[Y_HEIGHT][Y_WIDTH][CHAN_WORDS_OUT];
  bnn_b32_t WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHAN_WORDS_OUT];

  int32_t WORD_ALIGNED thresholds_ref[CHANS_OUT];
  int32_t WORD_ALIGNED thresholds[CHANS_OUT];

  run_config((bnn_b32_t*)Y, (bnn_b32_t*)Y_ref, (bnn_b256_t*)X_ref,
             (bnn_b256_t*)K, (bnn_b256_t*)K_ref, (int32_t*)thresholds_ref,
             (int32_t*)thresholds, X_HEIGHT, X_WIDTH, K_HEIGHT, K_WIDTH,
             CHANS_IN, CHANS_OUT, H_STRIDE, V_STRIDE);
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

void test_bnn_conv2d_bin_out_pseudo_random2() {
#define MIN_H_STRIDE 1
#define MIN_V_STRIDE 1
#define MAX_H_STRIDE 2
#define MAX_V_STRIDE 2

#define MIN_H_OFFSET 0
#define MIN_V_OFFSET 0
#define MAX_H_OFFSET 0
#define MAX_V_OFFSET 0

#define MIN_K_HEIGHT 3
#define MIN_K_WIDTH 3
#define MAX_K_HEIGHT 6
#define MAX_K_WIDTH 6

#define CHANS_IN 256
#define CHANS_OUT 32

#define MIN_X_HEIGHT MIN_K_HEIGHT
#define MIN_X_WIDTH MIN_K_WIDTH
#define MAX_X_HEIGHT 8
#define MAX_X_WIDTH 8

#define CHAN_WORDS_IN \
  ((CHANS_IN + XS3_VPU_VREG_WIDTH_BITS - 1) / XS3_VPU_VREG_WIDTH_BITS)
#define CHAN_WORDS_OUT ((CHANS_OUT + 32 - 1) / 32)

#define MAX_Y_HEIGHT \
  (((MAX_X_HEIGHT - MIN_V_OFFSET) / MIN_V_STRIDE) - MIN_K_HEIGHT + 1)
#define MAX_Y_WIDTH \
  (((MAX_X_WIDTH - MIN_H_OFFSET) / MIN_H_STRIDE) - MIN_K_WIDTH + 1)

  // bnn_b256_t WORD_ALIGNED
  //     K[CHANS_OUT / 16][MAX_K_HEIGHT][MAX_K_WIDTH][CHAN_WORDS_IN][16];

  bnn_b256_t WORD_ALIGNED
      K_ref[CHANS_OUT][MAX_K_HEIGHT][MAX_K_WIDTH][CHAN_WORDS_IN];
  bnn_b256_t WORD_ALIGNED
      K[CHANS_OUT][MAX_K_HEIGHT][MAX_K_WIDTH][CHAN_WORDS_IN];

  bnn_b256_t WORD_ALIGNED X_ref[MAX_X_HEIGHT][MAX_X_WIDTH][CHAN_WORDS_IN];
  bnn_b32_t WORD_ALIGNED Y_ref[MAX_Y_HEIGHT][MAX_Y_WIDTH][CHAN_WORDS_OUT];
  bnn_b32_t WORD_ALIGNED Y[MAX_Y_HEIGHT][MAX_Y_WIDTH][CHAN_WORDS_OUT];

  unsigned combinations =
      (MAX_H_STRIDE - MIN_H_STRIDE) * (MAX_V_STRIDE - MIN_V_STRIDE) *
      (MAX_K_HEIGHT - MIN_K_HEIGHT) * (MAX_K_WIDTH - MIN_K_WIDTH) *
      (MAX_X_HEIGHT - MIN_X_HEIGHT) * (MAX_X_WIDTH - MIN_X_WIDTH);

  printf("Combinations = %u\n", combinations);
  srand(69);

  assert(((int)K & 0x3) == 0);
  assert(((int)K_ref & 0x3) == 0);
  assert(((int)X_ref & 0x3) == 0);
  assert(((int)Y & 0x3) == 0);
  assert(((int)Y_ref & 0x3) == 0);

  unsigned chans_in = CHANS_IN;
  unsigned chans_out = CHANS_OUT;

  int32_t thresholds_ref[CHANS_OUT];
  int32_t thresholds[CHANS_OUT];

  for (unsigned h_stride = MIN_H_STRIDE; h_stride < MAX_H_STRIDE; ++h_stride) {
    for (unsigned v_stride = MIN_V_STRIDE; v_stride < MAX_V_STRIDE;
         ++v_stride) {
      for (unsigned k_height = MIN_K_HEIGHT; k_height < MAX_K_HEIGHT;
           ++k_height) {
        for (unsigned k_width = MIN_K_WIDTH; k_width < MAX_K_WIDTH; ++k_width) {
          for (unsigned x_height = k_height; x_height < MAX_X_HEIGHT;
               ++x_height) {
            for (unsigned x_width = k_width; x_width < MAX_X_WIDTH; ++x_width) {
              run_config((bnn_b32_t*)Y, (bnn_b32_t*)Y_ref, (bnn_b256_t*)X_ref,
                         (bnn_b256_t*)K, (bnn_b256_t*)K_ref,
                         (int32_t*)thresholds_ref, (int32_t*)thresholds,
                         x_height, x_width, k_height, k_width, chans_in,
                         chans_out, h_stride, v_stride);
            }
          }
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
}

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
  RUN_TEST(test_bnn_conv2d_bin_out_pseudo_random2);
  // RUN_TEST(test_bnn_conv2d_bin_out_ref_directed);
  // RUN_TEST(test_bnn_conv2d_bin_out_directed);
}