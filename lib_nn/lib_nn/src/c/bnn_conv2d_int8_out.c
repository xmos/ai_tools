
#include <stdint.h>
#include <string.h>
#include <assert.h>

#include "nn_operator.h"
#include "../nn_op_helper.h"
// #include "nn_op_structs.h"

// #include "xs3_vpu.h"

#if defined(__XS3A__)

void bnn_conv2d_int8_out_asm(nn_bnn_conv2d_int8_out_asm_plan_t * plan);

void bnn_conv2d_int8_out_asm_prepare(
    nn_bnn_conv2d_int8_out_asm_plan_t* plan, int8_t* Y_p,
    const bnn_b256_t* X_p, const bnn_b256_t* K_p, 
    
    const int16_t* post_activation_multiplier_q, 
    const int16_t* post_activation_bias_q,
    const int accu_shr,
    const int final_shr,

    const nn_image_params_t* x, 
    const nn_image_params_t* y,
    const nn_window_params_t* k, 
    const unsigned y_loc_x, const unsigned y_loc_y,
    const unsigned y_sub_width, const unsigned y_sub_height,
    const unsigned x_loc_x, const unsigned x_loc_y, 
    const unsigned k_loc_x, const unsigned k_loc_y, 
    const unsigned k_sub_width, const unsigned k_sub_height) {

  const unsigned chan_b256_in = (x->channels + XS3_VPU_VREG_WIDTH_BITS - 1) / XS3_VPU_VREG_WIDTH_BITS;
  const unsigned chans_out = y->channels;

  int8_t (*Y)[y->width][chans_out] =
      (int8_t (*)[y->width][chans_out])Y_p;

  bnn_b256_t(*X)[x->width][chan_b256_in] =
      (bnn_b256_t(*)[x->width][chan_b256_in])X_p;

  bnn_b256_t(*K)[k->shape.height][k->shape.width][chan_b256_in] =
      (bnn_b256_t(*)[k->shape.height][k->shape.width][chan_b256_in])K_p;

//relocate the pointers to the start of the region we care about.
  plan->Y = (int8_t*)Y[y_loc_y][y_loc_x];
  plan->X = (bnn_b256_t*)X[x_loc_y][x_loc_x];
  plan->K = (bnn_b256_t*)K[k_loc_y][k_loc_x];

  //This could go into the constant pool but it would make the loading
  //slower within the kernel(2 loops in).
  plan->mask = 0xaaaaaaaa; 
  
  plan->post_activation_mul = (int16_t *)post_activation_multiplier_q;
  plan->post_activation_bias = (int16_t *)post_activation_bias_q;
  plan->final_shr = final_shr;

  if(accu_shr >= 0){
    plan->vlsat = accu_shr;
    plan->ashr = 0;
  } else {
    plan->vlsat = 0;
    plan->ashr = accu_shr;
  }

  unsigned bytes_per_input_channel = x->channels / 8;
  unsigned bytes_per_output_channel = y->channels;

  const unsigned out_chans_multiplier = 16;

  assert((x->channels % XS3_VPU_VREG_WIDTH_BITS) == 0);
  assert((y->channels % out_chans_multiplier) == 0);

  plan->k_height_loop_counter = k_sub_height - 1;
  plan->k_width_loop_counter = k_sub_width - 1;

  unsigned h_dilation = k->dilation.horizontal;
  unsigned v_dilation = k->dilation.vertical;

  unsigned h_stride = k->stride.horizontal;
  unsigned v_stride = k->stride.vertical;

  unsigned x_sub_height = CONV2D_INPUT_LENGTH(y_sub_height, k_sub_height, v_dilation, v_stride );
  unsigned x_sub_width = CONV2D_INPUT_LENGTH(y_sub_width, k_sub_width, h_dilation, h_stride );

  plan->input_channel_loop_counter =
      (x->channels / XS3_VPU_VREG_WIDTH_BITS) - 1;
  plan->output_channel_loop_counter = (y->channels / out_chans_multiplier) - 1;

  unsigned x_height_loops = y_sub_height;
  unsigned x_width_loops = y_sub_width;

  plan->x_height_loop_counter = x_height_loops;
  plan->x_width_loop_counter = x_width_loops - 1;

  // Inner Loop
  // minus one to account for the auto increament in the loop
  plan->inner_x_h_step = bytes_per_input_channel * (h_dilation - 1);

  // TODO multiply x->width by dilation
  plan->inner_x_v_step =
      (bytes_per_input_channel * ((x->width - k_sub_width))) - plan->inner_x_h_step;

  // Outer Loop
  plan->outer_x_h_step = bytes_per_input_channel * k->stride.horizontal;

  plan->outer_x_v_step = (bytes_per_input_channel * x->width *k->stride.vertical) 
     - (plan->outer_x_h_step * x_width_loops);

  // TODO these are for implementing sub-kernels
  assert(k_sub_height == k->shape.height); //until the following two lines are working
  assert(k_sub_width == k->shape.width); //until the following two lines are working
  plan->k_v_step = 0;
  plan->k_h_step = 0;

  plan->y_v_step = sizeof(int8_t)*16 * (y->width - y_sub_width); //TODO check this
  
}

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

    nn_bnn_conv2d_int8_out_asm_plan_t plan;

    bnn_conv2d_int8_out_asm_prepare(&plan, Y_p,
        X_p,  K_p, 
        post_activation_multiplier_q, 
        post_activation_bias_q,
        accu_shr,
        final_shr,
        x, y, k, 
        y_loc_x, y_loc_y, y_sub_width, y_sub_height,
        x_loc_x, x_loc_y, 
        k_loc_x, k_loc_y, k_sub_width, k_sub_height);

    bnn_conv2d_int8_out_asm(&plan);
}
#endif
