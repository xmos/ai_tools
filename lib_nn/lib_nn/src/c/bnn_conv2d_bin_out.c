
#include <stdint.h>
#include <string.h>
#include <assert.h>

#include "nn_operator.h"
#include "../nn_op_helper.h"
#include "nn_op_structs.h"

#include "xs3_vpu.h"

#if defined(__XS3A__)
// #if 1

void bnn_conv2d_bin_out_asm_prepare(
    nn_bnn_conv2d_bin_out_asm_plan_t* plan, bnn_b32_t* Y_p,
    const bnn_b256_t* X_p, const bnn_b256_t* K_p, const int32_t* thresholds_p,
    const nn_image_params_t* x, const nn_image_params_t* y,
    const nn_window_params_t* k, const unsigned y_loc_x, const unsigned y_loc_y,
    const unsigned x_loc_x, const unsigned x_loc_y, const unsigned k_loc_x,
    const unsigned k_loc_y, const unsigned y_full_width,
    const unsigned x_full_width, const unsigned k_full_width) {

  plan->Y = (bnn_b32_t*)Y_p;
  plan->X = (bnn_b256_t*)X_p;
  plan->K = (bnn_b256_t*)K_p;
  plan->threshold_p = (int32_t *)thresholds_p;

  unsigned bytes_per_input_channel = x->channels / 8;
  unsigned bytes_per_output_channel = y->channels / 8;

  // This is 32 to make it easier and be more compatable with larq
  const unsigned out_chans_multiplier = 32;

  assert((x->channels % XS3_VPU_VREG_WIDTH_BITS) == 0);
  assert((y->channels % out_chans_multiplier) == 0);

  unsigned k_width = k->shape.width;

  plan->k_height_loop_counter = k->shape.height - 1;
  plan->k_width_loop_counter = k_width - 1;

  plan->input_channel_loop_counter =
      (x->channels / XS3_VPU_VREG_WIDTH_BITS) - 1;
  plan->output_channel_loop_counter = (y->channels / out_chans_multiplier) - 1;
  plan->x_height_loop_counter = y->height;
  plan->x_width_loop_counter = y->width - 1;

  unsigned h_dilation = 1;
  unsigned v_dilation = 1;  // unused

  // Inner Loop
  // minus one to account for the auto increament in the loop
  plan->inner_x_h_step = bytes_per_input_channel * (h_dilation - 1);

  // TODO multiply x->width by dilation
  plan->inner_x_v_step =
      (bytes_per_input_channel * ((x->width - k_width))) - plan->inner_x_h_step;

  // Outer Loop
  plan->outer_x_h_step = bytes_per_input_channel * (k->stride.horizontal);

  unsigned remainder_to_end_of_line = x->width % k->stride.horizontal;

  // TODO this shouldn't be a loop!
  while (k_width > remainder_to_end_of_line) {
    remainder_to_end_of_line += k->stride.horizontal;
  }

  plan->outer_x_v_step =
      bytes_per_input_channel *
          (remainder_to_end_of_line + (x->width * (k->stride.vertical - 1))) -
      plan->outer_x_h_step;

  // TODO these are for implementing sub-kernels
  plan->k_v_step = 0;
  plan->k_h_step = 0;

  // TODO this will be for when writing to a sub-rectangle of an image
  plan->y_v_step = 0;
}

void bnn_conv2d_bin_out(bnn_b32_t* Y_p,
    const bnn_b256_t* X_p, const bnn_b256_t* K_p, const int32_t* thresholds_p,
    const nn_image_params_t* x, const nn_image_params_t* y,
    const nn_window_params_t* k, const unsigned y_loc_x, const unsigned y_loc_y,
    const unsigned x_loc_x, const unsigned x_loc_y, const unsigned k_loc_x,
    const unsigned k_loc_y, const unsigned y_full_width,
    const unsigned x_full_width, const unsigned k_full_width){

      nn_bnn_conv2d_bin_out_asm_plan_t plan;
  bnn_conv2d_bin_out_asm_prepare(&plan, Y_p,
     X_p,  K_p, thresholds_p,
    x,  y,
     k, y_loc_x, y_loc_y,
    x_loc_x, x_loc_y, k_loc_x,
    k_loc_y, y_full_width,
     x_full_width,  k_full_width);

     bnn_conv2d_bin_out_asm(&plan);
    

}
#endif