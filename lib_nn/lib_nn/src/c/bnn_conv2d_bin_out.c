
#include <stdint.h>
#include <string.h>
#include <assert.h>

#include "nn_operator.h"
#include "../nn_op_helper.h"

#include "xs3_vpu.h"
#include "vpu_sim.h"

static void compute_bin_kernel(xs3_vpu * vpu, nn_bconv2d_bin_DI_impl_plan_t * plan, 
    void ** threshold_current, void* X_p, void ** K_p,  void * partial_res){

    vpu_vector_t zero_mem;
    memset(&zero_mem, 0, sizeof(zero_mem));

    void * X_cur_p = X_p;

    VLDR(vpu, *threshold_current);
    *threshold_current += 32;
    VLDD(vpu, *threshold_current);
    *threshold_current += 32;

    for (int kh = plan->k_height_loop_counter; kh >= 0 ; kh-- )  {
        for (int kw = plan->k_width_loop_counter; kw >= 0 ; kw-- )  {
        for (int ic = plan->input_channel_loop_counter; ic >= 0 ; ic-- ) {
            VLDC(vpu, X_cur_p);
            X_cur_p += 32;

            for(unsigned l=0; l<16; l++){
              VLMACCR1(vpu, *K_p);
              *K_p += 32;
            }
        }
        X_cur_p += plan->inner_x_h_step;
        *K_p += plan->k_h_step;
        }
        X_cur_p += plan->inner_x_v_step;
        *K_p += plan->k_v_step;
    }

    VLSAT(vpu, &zero_mem);
    VDEPTH1(vpu);
    VSTRPV(vpu, partial_res, 0x3);
        
}

WEAK_FUNC
void bconv2d_bin_DI_impl(nn_bconv2d_bin_DI_impl_plan_t * plan){

  xs3_vpu vpu_data;
  memset(&vpu_data, 0, sizeof(vpu_data));
  xs3_vpu * vpu = &vpu_data;

  VSETC(vpu, MODE_S16);

  void * X_p = plan->X;
  void * Y_p = plan->Y;

  unsigned partial_res_0_15=0;
  void* partial_res_0_15_p = & partial_res_0_15;
  unsigned partial_res_16_31=0;
  void* partial_res_16_31_p = & partial_res_16_31;

  for (int xh = plan->x_height_loop_counter; xh > 0 ; xh-- ) {
    for (int xv = plan->x_width_loop_counter; xv >= 0 ; xv-- ) {

      void * threshold_current = plan->threshold_p;
      void * K_p = plan->K;
      for (int oc = plan->output_channel_loop_counter; oc >= 0 ; oc-- ) {

        compute_bin_kernel(vpu, plan, &threshold_current, X_p, &K_p, partial_res_0_15_p);
        compute_bin_kernel(vpu, plan, &threshold_current, X_p, &K_p, partial_res_16_31_p);

        unsigned result = (partial_res_16_31<<16) + partial_res_0_15;

        ((unsigned*)Y_p)[0] = result;
        Y_p += 4;
      }
      X_p += plan->outer_x_h_step;
    }
    X_p += plan->outer_x_v_step;
    Y_p += plan->y_v_step;
  }
}

static void make_patch(xs3_vpu * vpu, nn_bconv2d_bin_impl_plan_t * plan, void * X_p){

    void * X_cur_p = X_p;
    void * D_p = plan->data_scratch;

    for (int kh = plan->k_height_loop_counter; kh >= 0 ; kh-- )  {
        for (int kw = plan->k_width_loop_counter; kw >= 0 ; kw-- )  {
            for (int ic = plan->input_channel_loop_counter; ic >= 0 ; ic-- ) {
                VLDD(vpu, X_cur_p);
                X_cur_p += 32;
                VSTD(vpu, D_p);
                D_p += 32;
            }
            X_cur_p += plan->inner_x_h_step;
            D_p += plan->data_scratch_adjust;
        }
        X_cur_p += plan->inner_x_v_step;
    }
    VCLRDR(vpu);
    VSTD(vpu, D_p);
}

static void compute_patch(xs3_vpu * vpu, nn_bconv2d_bin_impl_plan_t * plan,
    void ** threshold_current, void **K_p, unsigned * partial_res){

    vpu_vector_t zero_mem;
    memset(&zero_mem, 0, sizeof(zero_mem));

    VLDR(vpu, *threshold_current);
    *threshold_current += 32;
    VLDD(vpu, *threshold_current);
    *threshold_current += 32;

    void * D_p = plan->data_scratch;
    
    for (int p=plan->patch_loop_counter; p>0; p--){
        VLDC(vpu, D_p);
        D_p += 32;
        for (unsigned i=0;i<16;i++){
            VLMACCR1(vpu, *K_p);
            *K_p += 32;
        }
    }
    VLDC(vpu, D_p);
    for (unsigned i=0;i<16;i++){
        VLMACCR1(vpu, *K_p);
        *K_p += plan->k_p_adjust;
    }

    VLSAT(vpu, &zero_mem);
    VDEPTH1(vpu);
    VSTRPV(vpu, partial_res, 0x3);
    
}

//Patch to Col version
WEAK_FUNC
void bconv2d_bin_impl(nn_bconv2d_bin_impl_plan_t * plan){

  xs3_vpu vpu_data;
  memset(&vpu_data, 0, sizeof(vpu_data));
  xs3_vpu * vpu = &vpu_data;

  VSETC(vpu, MODE_S16);

  void * X_p = plan->X;
  void * Y_p = plan->Y;

  unsigned partial_res_0_15 = 0;
  void* partial_res_0_15_p = & partial_res_0_15;
  unsigned partial_res_16_31 = 0;
  void* partial_res_16_31_p = & partial_res_16_31;

  for (int xh = plan->x_height_loop_counter; xh > 0 ; xh-- ) {
    for (int xv = plan->x_width_loop_counter; xv >= 0 ; xv-- ) {

      make_patch(vpu, plan, X_p);

      void * K_p = plan->K;
      void * threshold_current = plan->threshold_p;
      for (int oc = plan->output_channel_loop_counter; oc >= 0 ; oc-- ) {

        compute_patch(vpu, plan, &threshold_current, &K_p, partial_res_0_15_p);
        compute_patch(vpu, plan, &threshold_current, &K_p, partial_res_16_31_p);

        unsigned result = (partial_res_16_31<<16) + partial_res_0_15;

        ((unsigned*)Y_p)[0] = result;
        Y_p += 4;
      }
      X_p += plan->outer_x_h_step;
    }
    X_p += plan->outer_x_v_step;
    Y_p += plan->y_v_step;
  }
}

void bconv2d_bin_DI_prepare(
    nn_bconv2d_bin_DI_impl_plan_t* plan, bnn_b32_t* Y_p,
    const bnn_b256_t* X_p, const bnn_b256_t* K_p, const int32_t* thresholds_p,
    const nn_image_params_t* x, 
    const nn_image_params_t* y,
    const nn_window_params_t* k, 
    const unsigned y_loc_x, const unsigned y_loc_y,
    const unsigned y_sub_width, const unsigned y_sub_height,
    const unsigned x_loc_x, const unsigned x_loc_y) {

  const unsigned chan_b256_in = (x->channels + XS3_VPU_VREG_WIDTH_BITS - 1) / XS3_VPU_VREG_WIDTH_BITS;
  const unsigned chan_b32_out = (y->channels + 32 - 1) / 32;

  bnn_b32_t(*Y)[y->width][chan_b32_out] =
      (bnn_b32_t(*)[y->width][chan_b32_out])Y_p;

  bnn_b256_t(*X)[x->width][chan_b256_in] =
      (bnn_b256_t(*)[x->width][chan_b256_in])X_p;

  bnn_b256_t(*K)[k->shape.height][k->shape.width][chan_b256_in] =
      (bnn_b256_t(*)[k->shape.height][k->shape.width][chan_b256_in])K_p;

//relocate the pointers to the start of the region we care about.
  plan->Y = (bnn_b32_t*)Y[y_loc_y][y_loc_x];
  plan->X = (bnn_b256_t*)X[x_loc_y][x_loc_x];
  plan->K = K;

  plan->threshold_p = (int32_t *)thresholds_p;

  unsigned bytes_per_input_channel = x->channels / 8;
  unsigned bytes_per_output_channel = y->channels / 8;

  // This is 32 to make it easier and be more compatable with larq
  const unsigned out_chans_multiplier = 32;

  assert((x->channels % XS3_VPU_VREG_WIDTH_BITS) == 0);
  assert((y->channels % out_chans_multiplier) == 0);

  plan->k_height_loop_counter = k->shape.height - 1;
  plan->k_width_loop_counter = k->shape.width - 1;

  unsigned h_dilation = k->dilation.horizontal;
  unsigned v_dilation = k->dilation.vertical;

  unsigned h_stride = k->stride.horizontal;
  unsigned v_stride = k->stride.vertical;

  unsigned x_sub_height = CONV2D_INPUT_LENGTH(y_sub_height, k->shape.height, v_dilation, v_stride );
  unsigned x_sub_width = CONV2D_INPUT_LENGTH(y_sub_width, k->shape.width, h_dilation, h_stride );

  plan->input_channel_loop_counter =
      (x->channels / XS3_VPU_VREG_WIDTH_BITS) - 1;
  plan->output_channel_loop_counter = (y->channels / out_chans_multiplier) - 1;

  unsigned x_height_loops = y_sub_height;
  unsigned x_width_loops = y_sub_width;

  plan->x_height_loop_counter = x_height_loops;
  plan->x_width_loop_counter = x_width_loops - 1;

 // Inner Loop
  // minus one to account for the auto increment in the loop
  plan->inner_x_h_step = bytes_per_input_channel * (h_dilation - 1);

  // TODO multiply x->width by dilation
  plan->inner_x_v_step =
      (bytes_per_input_channel * ((x->width - k->shape.width))) - plan->inner_x_h_step;

  // Outer Loop
  plan->outer_x_h_step = bytes_per_input_channel * k->stride.horizontal;

  plan->outer_x_v_step = (int)(bytes_per_input_channel * x->width * k->stride.vertical) 
     - (int)(plan->outer_x_h_step * x_width_loops);

  // TODO these are for implementing sub-kernels
  plan->k_v_step = 0;
  plan->k_h_step = 0;

  plan->y_v_step = chan_b32_out*sizeof(bnn_b32_t) * (y->width - y_sub_width);
  
}

/*
 * optimisation: if there are no dilations then for anything greater than a 1x1 pretend that the 
 * kernel is a nx1 i.e. long rows with a single coloumn, that way the pixel copies will be merged
 * and fewer loads and stores will execute with less loop overhead. 
 */
void bconv2d_bin_prepare(
    nn_bconv2d_bin_impl_plan_t* plan, bnn_b32_t* Y_p,
    const bnn_b32_t* X_p, const bnn_b32_t* K_p, const int32_t* thresholds_p,
    bnn_b32_t * data_scratch,
    const nn_image_params_t* x, 
    const nn_image_params_t* y,
    const nn_window_params_t* k, 
    const unsigned y_loc_x, const unsigned y_loc_y,
    const unsigned y_sub_width, const unsigned y_sub_height,
    const unsigned x_loc_x, const unsigned x_loc_y) {

  const unsigned outputs_per_b32 = 32;
  const unsigned chan_b32_in = (x->channels + outputs_per_b32 - 1) / outputs_per_b32;
  const unsigned chan_b32_out = (y->channels + outputs_per_b32 - 1) / outputs_per_b32;

  bnn_b32_t(*Y)[y->width][chan_b32_out] =
      (bnn_b32_t(*)[y->width][chan_b32_out])Y_p;

  bnn_b32_t(*X)[x->width][chan_b32_in] =
      (bnn_b32_t(*)[x->width][chan_b32_in])X_p;

  bnn_b32_t(*K)[k->shape.height][k->shape.width][chan_b32_in] =
      (bnn_b32_t(*)[k->shape.height][k->shape.width][chan_b32_in])K_p;

//relocate the pointers to the start of the region we care about.
  plan->Y = (bnn_b32_t*)Y[y_loc_y][y_loc_x];
  plan->X = (bnn_b32_t*)X[x_loc_y][x_loc_x];
  plan->K = K;
  plan->threshold_p = (int32_t *)thresholds_p;
  plan->data_scratch = data_scratch;

  unsigned bytes_per_input_channel = x->channels / 8;
  unsigned bytes_per_output_channel = y->channels / 8;

  assert((x->channels % 32) == 0);
  assert((y->channels % outputs_per_b32) == 0);

  plan->k_height_loop_counter = k->shape.height - 1;
  plan->k_width_loop_counter = k->shape.width - 1;

  unsigned h_dilation = k->dilation.horizontal;
  unsigned v_dilation = k->dilation.vertical;

  unsigned h_stride = k->stride.horizontal;
  unsigned v_stride = k->stride.vertical;

  unsigned x_sub_height = CONV2D_INPUT_LENGTH(y_sub_height, k->shape.height, v_dilation, v_stride );
  unsigned x_sub_width = CONV2D_INPUT_LENGTH(y_sub_width, k->shape.width, h_dilation, h_stride );

  //We are going to copy (in chunks of XS3_VPU_VREG_WIDTH_BITS) each of the 
  plan->input_channel_loop_counter =
      ((x->channels + XS3_VPU_VREG_WIDTH_BITS - 1) / XS3_VPU_VREG_WIDTH_BITS) - 1;

  plan->data_scratch_adjust = -(int)((XS3_VPU_VREG_WIDTH_BITS - x->channels % XS3_VPU_VREG_WIDTH_BITS) % XS3_VPU_VREG_WIDTH_BITS)/8;

  unsigned total_bits_copied_to_scratch = x->channels * k->shape.height * k->shape.width;

  //the final loop copies 32-256 bits(not 0)
  int remainder_bits = total_bits_copied_to_scratch % XS3_VPU_VREG_WIDTH_BITS;
  if (!remainder_bits) {
    remainder_bits = XS3_VPU_VREG_WIDTH_BITS;
  }
  plan->k_p_adjust = remainder_bits / 8;


  total_bits_copied_to_scratch -= plan->k_p_adjust;  
  plan->patch_loop_counter = total_bits_copied_to_scratch / XS3_VPU_VREG_WIDTH_BITS;

  plan->output_channel_loop_counter = (y->channels / outputs_per_b32) - 1;

  unsigned x_height_loops = y_sub_height;
  unsigned x_width_loops = y_sub_width;

  plan->x_height_loop_counter = x_height_loops;
  plan->x_width_loop_counter = x_width_loops - 1;

 // Inner Loop
  // minus one to account for the auto increment in the loop
  int bytes_per_input_channel_rounded_up = ((bytes_per_input_channel + 32 - 1)/32)*32;
  plan->inner_x_h_step = (int)(bytes_per_input_channel * h_dilation) - bytes_per_input_channel_rounded_up;

  // TODO multiply x->width by dilation
  plan->inner_x_v_step = 
      (bytes_per_input_channel * ((x->width - k->shape.width)));
  // Outer Loop
  plan->outer_x_h_step = bytes_per_input_channel * k->stride.horizontal;

  plan->outer_x_v_step = (int)(bytes_per_input_channel * x->width *k->stride.vertical) 
     - (int)(plan->outer_x_h_step * x_width_loops);

  plan->y_v_step = chan_b32_out * sizeof(bnn_b32_t) * (y->width - y_sub_width);
}

void bconv2d_bin_DI(bnn_b32_t* Y_p,
    const bnn_b256_t* X_p, const bnn_b256_t* K_p, const int32_t* thresholds_p,
    
    const nn_image_params_t* x, //The full image of x
    const nn_image_params_t* y, // the full image of y
    const nn_window_params_t* k, //the full kernel k
    
    const unsigned y_loc_x, const unsigned y_loc_y,
    const unsigned y_sub_width, const unsigned y_sub_height,

    const unsigned x_loc_x, const unsigned x_loc_y
) {

    nn_bconv2d_bin_DI_impl_plan_t plan;

    bconv2d_bin_DI_prepare(&plan, Y_p,
        X_p,  K_p, thresholds_p,
        x,  y, k, 
        y_loc_x, y_loc_y, y_sub_width, y_sub_height,
        x_loc_x, x_loc_y);

    bconv2d_bin_DI_impl(&plan);
}

void bconv2d_bin(bnn_b32_t* Y_p,
    const bnn_b32_t* X_p, const bnn_b32_t* K_p, const int32_t* thresholds_p,
    bnn_b32_t * data_scratch, 
    const nn_image_params_t* x, //The full image of x
    const nn_image_params_t* y, // the full image of y
    const nn_window_params_t* k, //the full kernel k
    
    const unsigned y_loc_x, const unsigned y_loc_y,
    const unsigned y_sub_width, const unsigned y_sub_height,

    const unsigned x_loc_x, const unsigned x_loc_y
) {

    nn_bconv2d_bin_impl_plan_t plan;

    bconv2d_bin_prepare(&plan, Y_p,
        X_p,  K_p, thresholds_p, data_scratch, 
        x,  y, k, 
        y_loc_x, y_loc_y, y_sub_width, y_sub_height,
        x_loc_x, x_loc_y);

    bconv2d_bin_impl(&plan);
}