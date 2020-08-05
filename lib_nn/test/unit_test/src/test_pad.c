
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

#include "unity.h"

#define X_HEIGHT 1
#define X_WIDTH 1
#define X_CHANS 512

#define PAD_TOP 0
#define PAD_BOTTOM 0
#define PAD_RIGHT 1
#define PAD_LEFT 0

#define Y_HEIGHT (X_HEIGHT + PAD_TOP + PAD_BOTTOM)
#define Y_WIDTH (X_WIDTH + PAD_LEFT + PAD_RIGHT)

#define X_CHANS_WORDS_IN \
  ((X_CHANS + XS3_VPU_VREG_WIDTH_BITS - 1) / XS3_VPU_VREG_WIDTH_BITS)

void test_pad_directed_b256() {
  bnn_b256_t WORD_ALIGNED X[X_HEIGHT][X_WIDTH][X_CHANS_WORDS_IN];
  bnn_b256_t WORD_ALIGNED Y_ref[Y_HEIGHT][Y_WIDTH][X_CHANS_WORDS_IN];
  bnn_b256_t WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][X_CHANS_WORDS_IN];

  pseudo_rand_bytes((char*)X, sizeof(X));
  pseudo_rand_bytes((char*)Y, sizeof(Y));
  pseudo_rand_bytes((char*)Y_ref, sizeof(Y_ref));

  PaddingValues p;
  p.height = PAD_TOP;
  p.height_offset = PAD_BOTTOM - PAD_TOP;
  p.width = PAD_LEFT;
  p.width_offset = PAD_RIGHT - PAD_LEFT;

  nn_image_params_t xp;
  xp.height = X_HEIGHT;
  xp.width = X_WIDTH;
  xp.channels = X_CHANS;

  unsigned bytes_per_pixel = sizeof(bnn_b256_t) * xp.channels / 256;

  pad_ref((void*)Y_ref, (void*)X, &p, &xp, bytes_per_pixel);
  nn_pad_plan_t plan;
  pad_perpare(&plan, &p, &xp, bytes_per_pixel);
  pad_run((void*)Y, (void*)X, &plan);

  unsigned output_height = xp.height + PAD_TOP + PAD_BOTTOM;
  unsigned output_width = xp.width + PAD_LEFT + PAD_RIGHT;

  TEST_ASSERT_EQUAL_INT8_ARRAY(
      Y, Y_ref,
      output_height * output_width * xp.channels / 256 * sizeof(bnn_b256_t));
}

void test_pad_param_space_b256() {
#define MAX_X_HEIGHT 4
#define MAX_X_WIDTH 4
#define MAX_X_CHANS 512

#define MAX_PAD_LEFT 3
#define MAX_PAD_RIGHT 3
#define MAX_PAD_TOP 3
#define MAX_PAD_BOTTOM 3

#define MAX_Y_HEIGHT (MAX_X_HEIGHT + MAX_PAD_TOP + MAX_PAD_BOTTOM)
#define MAX_Y_WIDTH (MAX_X_WIDTH + MAX_PAD_LEFT + MAX_PAD_RIGHT)

#define MAX_X_CHANS_WORDS \
  ((MAX_X_CHANS + XS3_VPU_VREG_WIDTH_BITS - 1) / XS3_VPU_VREG_WIDTH_BITS)

  bnn_b256_t WORD_ALIGNED X[MAX_X_HEIGHT][MAX_X_WIDTH][MAX_X_CHANS_WORDS];
  bnn_b256_t WORD_ALIGNED Y_ref[MAX_Y_HEIGHT][MAX_Y_WIDTH][MAX_X_CHANS_WORDS];
  bnn_b256_t WORD_ALIGNED Y[MAX_Y_HEIGHT][MAX_Y_WIDTH][MAX_X_CHANS_WORDS];

  pseudo_rand_bytes((char*)X, sizeof(X));

  for (unsigned x_height = 1; x_height <= MAX_X_HEIGHT; ++x_height) {
    for (unsigned x_width = 1; x_width <= MAX_X_WIDTH; ++x_width) {
      for (unsigned x_chans = 256; x_chans <= MAX_X_CHANS; x_chans += 256) {
        for (unsigned pad_top = 0; pad_top <= MAX_PAD_TOP; ++pad_top) {
          for (unsigned pad_bottom = 0; pad_bottom <= MAX_PAD_BOTTOM;
               ++pad_bottom) {
            for (unsigned pad_right = 0; pad_right <= MAX_PAD_RIGHT;
                 ++pad_right) {
              for (unsigned pad_left = 0; pad_left <= MAX_PAD_LEFT;
                   ++pad_left) {
                pseudo_rand_bytes((char*)Y, sizeof(Y));
                pseudo_rand_bytes((char*)Y_ref, sizeof(Y_ref));

                PaddingValues p;
                p.height = pad_top;
                p.height_offset = pad_bottom - pad_top;
                p.width = pad_left;
                p.width_offset = pad_right - pad_left;

                unsigned bytes_per_pixel = sizeof(bnn_b256_t) * x_chans / 256;

                nn_image_params_t xp;
                xp.height = x_height;
                xp.width = x_width;
                xp.channels = x_chans;

                pad_ref((void*)Y_ref, (void*)X, &p, &xp, bytes_per_pixel);
                nn_pad_plan_t plan;
                pad_perpare(&plan, &p, &xp, bytes_per_pixel);
                pad_run((void*)Y, (void*)X, &plan);

                unsigned output_height = xp.height + pad_top + pad_bottom;
                unsigned output_width = xp.width + pad_left + pad_right;

                TEST_ASSERT_EQUAL_INT8_ARRAY(Y, Y_ref,
                                             output_height * output_width *
                                                 xp.channels / 256 *
                                                 sizeof(bnn_b256_t));
              }
            }
          }
        }
      }
    }
  }
}

void test_pad_directed_int8() {
#undef X_CHANS
#define X_CHANS 4

  int8_t WORD_ALIGNED X[X_HEIGHT][X_WIDTH][X_CHANS];
  int8_t WORD_ALIGNED Y_ref[Y_HEIGHT][Y_WIDTH][X_CHANS];
  int8_t WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][X_CHANS];

  pseudo_rand_bytes((char*)X, sizeof(X));
  pseudo_rand_bytes((char*)Y, sizeof(Y));
  pseudo_rand_bytes((char*)Y_ref, sizeof(Y_ref));

  PaddingValues p;
  p.height = PAD_TOP;
  p.height_offset = PAD_BOTTOM - PAD_TOP;
  p.width = PAD_LEFT;
  p.width_offset = PAD_RIGHT - PAD_LEFT;

  nn_image_params_t xp;
  xp.height = X_HEIGHT;
  xp.width = X_WIDTH;
  xp.channels = X_CHANS;

  unsigned bytes_per_pixel = sizeof(int8_t) * xp.channels;

  pad_ref((void*)Y_ref, (void*)X, &p, &xp, bytes_per_pixel);
  nn_pad_plan_t plan;
  pad_perpare(&plan, &p, &xp, bytes_per_pixel);
  pad_run((void*)Y, (void*)X, &plan);

  unsigned output_height = xp.height + PAD_TOP + PAD_BOTTOM;
  unsigned output_width = xp.width + PAD_LEFT + PAD_RIGHT;

  TEST_ASSERT_EQUAL_INT8_ARRAY(
      Y, Y_ref, output_height * output_width * xp.channels * sizeof(int8_t));
}

void test_pad_param_space_int8() {
#undef MAX_X_CHANS
#define MAX_X_CHANS 20

  int8_t WORD_ALIGNED X[MAX_X_HEIGHT][MAX_X_WIDTH][MAX_X_CHANS];
  int8_t WORD_ALIGNED Y_ref[MAX_Y_HEIGHT][MAX_Y_WIDTH][MAX_X_CHANS];
  int8_t WORD_ALIGNED Y[MAX_Y_HEIGHT][MAX_Y_WIDTH][MAX_X_CHANS];

  pseudo_rand_bytes((char*)X, sizeof(X));

  for (unsigned x_height = 1; x_height <= MAX_X_HEIGHT; ++x_height) {
    for (unsigned x_width = 1; x_width <= MAX_X_WIDTH; ++x_width) {
      for (unsigned x_chans = 4; x_chans <= MAX_X_CHANS; x_chans += 4) {
        for (unsigned pad_top = 0; pad_top <= MAX_PAD_TOP; ++pad_top) {
          for (unsigned pad_bottom = 0; pad_bottom <= MAX_PAD_BOTTOM;
               ++pad_bottom) {
            for (unsigned pad_right = 0; pad_right <= MAX_PAD_RIGHT;
                 ++pad_right) {
              for (unsigned pad_left = 0; pad_left <= MAX_PAD_LEFT;
                   ++pad_left) {
                pseudo_rand_bytes((char*)Y, sizeof(Y));
                pseudo_rand_bytes((char*)Y_ref, sizeof(Y_ref));

                PaddingValues p;
                p.height = pad_top;
                p.height_offset = pad_bottom - pad_top;
                p.width = pad_left;
                p.width_offset = pad_right - pad_left;

                unsigned bytes_per_pixel = sizeof(int8_t) * x_chans;

                nn_image_params_t xp;
                xp.height = x_height;
                xp.width = x_width;
                xp.channels = x_chans;

                pad_ref((void*)Y_ref, (void*)X, &p, &xp, bytes_per_pixel);
                nn_pad_plan_t plan;
                pad_perpare(&plan, &p, &xp, bytes_per_pixel);
                pad_run((void*)Y, (void*)X, &plan);

                unsigned output_height = xp.height + pad_top + pad_bottom;
                unsigned output_width = xp.width + pad_left + pad_right;

                TEST_ASSERT_EQUAL_INT8_ARRAY(Y, Y_ref,
                                             output_height * output_width *
                                                 xp.channels * sizeof(int8_t));
              }
            }
          }
        }
      }
    }
  }
}

void test_pad() {
  UNITY_SET_FILE();
  RUN_TEST(test_pad_directed_b256);
  RUN_TEST(test_pad_param_space_b256);
  RUN_TEST(test_pad_directed_int8);
  RUN_TEST(test_pad_param_space_int8);
}