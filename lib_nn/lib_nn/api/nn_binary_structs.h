#include "nn_bin_types.h"

#define NN_BCONV2D_KERNEL_OVERRUN_WORDS 8

/**
 * Struct represents the parameters needed by each
 * `bnn_conv2d_bin_out_asm()` job.
 *
 * Values are set by `bnn_conv2d_bin_out_asm_prepare()`.
 *
 * @note This struct is intended to be opaque.
 */
typedef struct {

    //These are in a specific order - do not change

  int outer_x_h_step;
  unsigned output_channel_loop_counter;
  void* threshold_p;
  int inner_x_v_step;
  unsigned k_v_step;
  int inner_x_h_step;
  int k_h_step;
  int outer_x_v_step;

  int y_v_step;
  unsigned k_height_loop_counter;
  unsigned k_width_loop_counter;
  unsigned x_height_loop_counter;
  unsigned x_width_loop_counter;
  unsigned input_channel_loop_counter;
  bnn_b32_t* Y;
  bnn_b256_t* X;

  bnn_b256_t* K;
} nn_bnn_conv2d_bin_out_asm_plan_t;

/**
 * Struct represents the parameters needed by each
 * `bnn_conv2d_bin_out_patch_asm()` job.
 *
 * Values are set by `bnn_conv2d_bin_out_patch_asm_prepare()`.
 *
 * @note This struct is intended to be opaque.
 */
typedef struct {

    //These are in a pacific order - do not change

  unsigned k_height_loop_counter;
  bnn_b32_t * data_scratch;
  unsigned k_width_loop_counter;
  int inner_x_v_step;
  int inner_x_h_step;
  int data_scratch_adjust;
  unsigned output_channel_loop_counter;
  int32_t * threshold_p;

  bnn_b32_t* X;
  int outer_x_h_step;
  int outer_x_v_step;
  int y_v_step;
  unsigned patch_loop_counter;
  unsigned x_width_loop_counter;
  bnn_b32_t* K;
  unsigned x_height_loop_counter;

  unsigned input_channel_loop_counter;
  int k_p_adjust;    //the amount to advance the kernel pointer after applying it
  bnn_b32_t* Y;

} nn_bnn_conv2d_bin_out_SISO_asm_plan_t;


/**
 * Struct represents the parameters needed by each
 * `bnn_conv2d_int8_out_asm()` job.
 *
 * Values are set by `bnn_conv2d_int8_out_asm_prepare()`.
 *
 * @note This struct is intended to be opaque.
 */
typedef struct {
    //These are in a specific order - do not change
  bnn_b256_t* X;
  int outer_x_h_step;
  unsigned output_channel_loop_counter;
  bnn_b256_t* K;
  int inner_x_v_step;
  int k_v_step;
  int inner_x_h_step;
  int k_h_step;

  int outer_x_v_step;
  int y_v_step;
  unsigned k_height_loop_counter;
  unsigned k_width_loop_counter;
  unsigned x_height_loop_counter;
  unsigned x_width_loop_counter;
  int16_t* cur_post_activation_mul;  //These are needed to hold variables that will
  int16_t* cur_post_activation_bias; //be indexed with ldd

  unsigned vlsat;
  unsigned ashr;
  int final_shr;
  unsigned mask;
  int16_t* post_activation_mul;  
  int16_t* post_activation_bias; 
  unsigned input_channel_loop_counter;
  int8_t* Y;

} nn_bnn_conv2d_int8_out_asm_plan_t;


/**
 * Struct represents the parameters needed by each
 * `bnn_conv2d_int8_out_asm()` job.
 *
 * Values are set by `bnn_conv2d_int8_out_asm_prepare()`.
 *
 * @note This struct is intended to be opaque.
 */
typedef struct {

    //These are in a specific order - do not change

  int inner_x_h_step;
  int data_scratch_adjust;
  unsigned k_height_loop_counter;
  bnn_b32_t * data_scratch;
  unsigned k_width_loop_counter;
  int inner_x_v_step;
  int outer_x_v_step;
  int y_v_step;

  unsigned output_channel_loop_counter;
  bnn_b256_t* K;
  int16_t* cur_post_activation_mul;  //These are needed to hold variables that will
  int16_t* cur_post_activation_bias; //be indexed with ldd
  unsigned vlsat;
  unsigned ashr;
  int16_t* post_activation_mul;  
  int16_t* post_activation_bias; 

  unsigned input_channel_loop_counter;
  int8_t* Y;
  bnn_b256_t* X;
  int outer_x_h_step;
  int k_p_adjust;
  int patch_branch;
  unsigned final_channels_mask;
  unsigned final_channels_bytes;
  unsigned patch_loop_counter;

  int final_shr;
  int k_p_rewind;
  unsigned x_width_loop_counter;
  unsigned x_height_loop_counter;

} nn_bnn_conv2d_int8_out_SISO_asm_plan_t;
