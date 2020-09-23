
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <syscall.h>
#include <xccompat.h>

#include "nn_operator.h"

#include "xcore/hwtimer.h"

#ifdef __xcore__
#define WORD_ALIGNED __attribute__((aligned(4)))
#else
#define WORD_ALIGNED
#endif

void bnn_conv2d_bin_out_asm(nn_bnn_conv2d_bin_out_asm_plan_t * plan);

void bnn_conv2d_bin_out_asm_prepare(
    nn_bnn_conv2d_bin_out_asm_plan_t* plan, bnn_b32_t* Y_p,
    const bnn_b256_t* X_p, const bnn_b256_t* K_p, const int32_t* thresholds_p,
    const nn_image_params_t* x, 
    const nn_image_params_t* y,
    const nn_window_params_t* k, 
    const unsigned y_loc_x, const unsigned y_loc_y,
    const unsigned y_sub_width, const unsigned y_sub_height,
    const unsigned x_loc_x, const unsigned x_loc_y, 
    const unsigned k_loc_x, const unsigned k_loc_y, 
    const unsigned k_sub_width, const unsigned k_sub_height);

unsigned run_config(bnn_b32_t* Y_p, bnn_b256_t* X_p, bnn_b256_t* K_p,
                    int32_t* thresholds_p, unsigned x_height, unsigned x_width,
                    unsigned k_height, unsigned k_width, unsigned chans_in,
                    unsigned chans_out, unsigned h_stride, unsigned v_stride) {
  unsigned y_height = CONV2D_OUTPUT_LENGTH(x_height, k_height, 1, v_stride);
  unsigned y_width = CONV2D_OUTPUT_LENGTH(x_width, k_width, 1, h_stride);

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

  nn_bnn_conv2d_bin_out_asm_plan_t plan;
  bnn_conv2d_bin_out_asm_prepare(&plan, (bnn_b32_t*)Y_p, (bnn_b256_t*)X_p,
                                 (bnn_b256_t*)K_p, thresholds_p, &x, &y, &k,
    0, 0,y_width, y_height, 0, 0, 0, 0, k_width, k_height);

  hwtimer_t t = hwtimer_alloc();

  uint32_t before = hwtimer_get_time(t);

  bnn_conv2d_bin_out_asm(&plan);

  uint32_t after = hwtimer_get_time(t);

  hwtimer_free(t);
  unsigned elapsed = after - before;

  return elapsed;
}

unsigned calc_macc_count(unsigned x_height, unsigned x_width, unsigned k_height,
                         unsigned k_width, unsigned chans_in,
                         unsigned chans_out, unsigned h_stride,
                         unsigned v_stride) {
  unsigned y_height = CONV2D_OUTPUT_LENGTH(x_height, k_height, 1, v_stride);
  unsigned y_width = CONV2D_OUTPUT_LENGTH(x_width, k_width, 1, h_stride);

  return y_height * y_width * k_width * k_height * (chans_in / 256) * chans_out;
}

void benchmark_bnn_conv2d_bin_output(int argc, char** argv) {
#define MIN_H_STRIDE 1
#define MIN_V_STRIDE 1
#define MAX_H_STRIDE 4
#define MAX_V_STRIDE 4

#define MIN_K_HEIGHT 1
#define MIN_K_WIDTH 1
#define MAX_K_HEIGHT 3
#define MAX_K_WIDTH 3

#define MIN_CHANS_IN 512
#define MAX_CHANS_IN 512

#define MIN_CHANS_OUT 32
#define MAX_CHANS_OUT 64

#define MIN_X_HEIGHT MIN_K_HEIGHT
#define MIN_X_WIDTH MIN_K_WIDTH
#define MAX_X_HEIGHT 64
#define MAX_X_WIDTH 64

#define MAX_CHAN_WORDS_IN \
  ((MAX_CHANS_IN + XS3_VPU_VREG_WIDTH_BITS - 1) / XS3_VPU_VREG_WIDTH_BITS)
#define MAX_CHAN_WORDS_OUT ((MAX_CHANS_OUT + 32 - 1) / 32)

#define MAX_Y_HEIGHT (((MAX_X_HEIGHT - MIN_K_HEIGHT + 1) / MIN_V_STRIDE))
#define MAX_Y_WIDTH (((MAX_X_WIDTH - MIN_K_WIDTH + 1) / MIN_H_STRIDE))

  bnn_b256_t WORD_ALIGNED
      K_ref[MAX_CHANS_OUT][MAX_K_HEIGHT][MAX_K_WIDTH][MAX_CHAN_WORDS_IN];

  bnn_b256_t WORD_ALIGNED X_ref[MAX_X_HEIGHT][MAX_X_WIDTH][MAX_CHAN_WORDS_IN];
  bnn_b32_t WORD_ALIGNED Y_ref[MAX_Y_HEIGHT][MAX_Y_WIDTH][MAX_CHAN_WORDS_OUT];

  int32_t WORD_ALIGNED thresholds_ref[MAX_CHANS_OUT];

  assert(((int)K_ref & 0x3) == 0);
  assert(((int)X_ref & 0x3) == 0);
  assert(((int)Y_ref & 0x3) == 0);

  assert(((int)thresholds_ref & 0x3) == 0);

  unsigned elapsed_timer_ticks =
      run_config((bnn_b32_t*)Y_ref, (bnn_b256_t*)X_ref, (bnn_b256_t*)K_ref,
                 (int32_t*)thresholds_ref, 6, 6, 3, 3, 256, 64, 1, 1);

  float system_freq = 800000000.;

  float ns_per_cycle = 1e9 / (system_freq / 5);

  float elapsed_ns = (float)elapsed_timer_ticks * 10;

  float cycles_executed = elapsed_ns / ns_per_cycle;

  unsigned macc_count = calc_macc_count(6, 6, 3, 3, 256, 64, 1, 1);

  float efficiency = ((float)macc_count) / cycles_executed;

  printf("system_freq:     %f\n", system_freq);
  printf("ns_per_cycle:    %f\n", ns_per_cycle);
  printf("elapsed_ns:      %f\n", elapsed_ns);
  printf("cycles_executed: %f\n", cycles_executed);
  printf("macc_count:      %u\n", macc_count);
  printf("efficiency:      %f\n", efficiency);

}