#include <stdint.h>
#include <string.h>
#include <assert.h>

#include "nn_operator.h"
#include "../nn_op_helper.h"

#include "xs3_vpu.h"
#include "vpu_sim.h"

static int64_t saturate_non_sym(
    const int64_t input,
    const unsigned bits)
{
    const int64_t max_val = (((int64_t)1)<<(bits-1))-1;
    const int64_t min_val = -max_val - 1;
    
    return (input > max_val)?  max_val : (input < min_val)? min_val : input;
}

// This is an implementation of VDEPTH8 where the rounding is asymetric
static void VDEPTH8_FIXED(xs3_vpu* vpu){

    vpu_vector_t vec_tmp;
    memcpy(&vec_tmp, &(vpu->vR), sizeof(vpu_vector_t));
    memset(&(vpu->vR), 0, sizeof(vpu_vector_t));
    
    for(int i = 0; i < VPU_INT16_EPV; i++){
        int32_t elm = ((int32_t)vec_tmp.s16[i]) + (1 << 7);
        vpu->vR.s8[i] = saturate_non_sym(elm >> 8, 8);
    }
}

WEAK_FUNC
void bnn_conv2d_int8_out_asm(nn_bnn_conv2d_int8_out_asm_plan_t * plan){

  xs3_vpu vpu_data;
  xs3_vpu * vpu = &vpu_data;

  vpu_vector_t bias_shift;
  vpu_vector_t final_shr;
  vpu_vector_t sat_mem;
  vpu_vector_t temp_mem;
  VSETC(vpu, MODE_S16);


  for(unsigned i=0;i<VPU_INT16_EPV;i++){
    sat_mem.s16[i] = plan->vlsat;
    bias_shift.s16[i] = plan->bias_multiplier;
    final_shr.s16[i] = plan->final_shr;
  }

  void * X_p = (void *)plan->X;
  void * Y_p = (void *)plan->Y;

  for (int xh = plan->x_height_loop_counter; xh > 0 ; xh-- ) {
    for (int xv = plan->x_width_loop_counter; xv >= 0 ; xv-- ) {

      void * cur_post_activation_mul = plan->post_activation_mul;
      void * cur_post_activation_bias = plan->post_activation_bias;
      void * K_p = (void *)plan->K;
      for (int oc = plan->output_channel_loop_counter; oc >= 0 ; oc-- ) {

        void * X_cur_p = X_p;
        VCLRDR(vpu);

        for (int kh = plan->k_height_loop_counter; kh >= 0 ; kh-- )  {
          for (int kw = plan->k_width_loop_counter; kw >= 0 ; kw-- )  {
            for (int ic = plan->input_channel_loop_counter; ic >= 0 ; ic-- ) {
              VLDC(vpu, X_cur_p);
              X_cur_p += 32;

              for(unsigned l=0; l<16; l++){
                VLMACCR1(vpu, K_p);
                K_p += 32;
              }
            }
            X_cur_p += plan->inner_x_h_step;
            K_p += plan->k_h_step;
          }
          X_cur_p += plan->inner_x_v_step;
          K_p += plan->k_v_step;
        }

        VLSAT(vpu, &sat_mem);
        VSTR(vpu, &temp_mem);
        VLASHR(vpu, &temp_mem, plan->ashr);

        VSTR(vpu, &temp_mem);
        VCLRDR(vpu);

        VLDC(vpu, cur_post_activation_bias);
        VLMACC(vpu, &bias_shift);
        VLDC(vpu, &temp_mem);
        VLMACC(vpu, cur_post_activation_mul);
        VLSAT(vpu, &final_shr);

        VDEPTH8_FIXED(vpu);
        VSTRPV(vpu, Y_p, 0xffff);
        Y_p += 16;

        cur_post_activation_mul += 32;
        cur_post_activation_bias += 32;
      }
      X_p += plan->outer_x_h_step;
    }
    X_p += plan->outer_x_v_step;
    Y_p += plan->y_v_step;
  }
}

static void make_patch(xs3_vpu * vpu, nn_bnn_conv2d_int8_out_SISO_asm_plan_t * plan, void * X_p){

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

void compute_patch(nn_bnn_conv2d_int8_out_SISO_asm_plan_t *plan, 
  void ** K_p, int step, xs3_vpu * vpu, 
  vpu_vector_t *sat_mem, 
  vpu_vector_t * bias_shift, 
  vpu_vector_t * final_shr, 
  void * cur_post_activation_mul, 
  void * cur_post_activation_bias){

  VCLRDR(vpu);
  void * D_p = plan->data_scratch;
  for (unsigned p = plan->patch_loop_counter; p > 0; p--){
    VLDC(vpu, D_p);
    D_p += 32;
    for(unsigned l=0; l<15; l++){
      VLMACCR1(vpu, *K_p);
      *K_p += 32;
    }
    VLMACCR1(vpu, *K_p);
    *K_p += step;
  }

  // printf("pretail\n");
  // vpu_sim_print(vpu);
  VLDC(vpu, D_p);
  
  unsigned tail_loops = 15 + step/32;
  for(unsigned l=0; l<tail_loops; l++){
    VLMACCR1(vpu, *K_p);
    // printf("loop %d\n", l);
    // vpu_sim_print(vpu);
    *K_p += plan->k_p_adjust;
  }

  // printf("tail finished\n");
  // vpu_sim_print(vpu);
  vpu_vector_t temp_mem;

  memset(&temp_mem, 0, sizeof(temp_mem));

  VLSAT(vpu, sat_mem);
  VSTR(vpu, &temp_mem);
  VLASHR(vpu, &temp_mem, plan->ashr);

  // printf("post ashr\n");
  // vpu_sim_print(vpu);

  VSTR(vpu, &temp_mem);
  VCLRDR(vpu);


  VLDC(vpu, cur_post_activation_bias);
  
  // printf("post load cur_post_activation_bias\n");
  // vpu_sim_print(vpu);

  VLMACC(vpu, bias_shift);

  // printf("post macc 1\n");
  // vpu_sim_print(vpu);

  VLDC(vpu, &temp_mem);

  // printf("post load temp_mem\n");
  // vpu_sim_print(vpu);

  VLMACC(vpu, cur_post_activation_mul);

  // printf("post macc 2\n");
  // vpu_sim_print(vpu);

  VLSAT(vpu, final_shr);

  // printf("post final shr\n");
  // vpu_sim_print(vpu);
  VDEPTH8_FIXED(vpu);

// printf("post make 8 bit\n");
//   vpu_sim_print(vpu);
}

WEAK_FUNC
void bnn_conv2d_int8_out_SISO_asm(nn_bnn_conv2d_int8_out_SISO_asm_plan_t *plan){

  xs3_vpu vpu_data;
  memset(&vpu_data, 0, sizeof(vpu_data));
  xs3_vpu * vpu = &vpu_data;

  vpu_vector_t sat_mem;
  vpu_vector_t bias_shift;
  vpu_vector_t final_shr;
  VSETC(vpu, MODE_S16);

  for(unsigned i=0;i<VPU_INT16_EPV;i++){
    sat_mem.s16[i] = plan->vlsat;
    bias_shift.s16[i] = plan->bias_multiplier;
    final_shr.s16[i] = plan->final_shr;
  }

  void * X_p = (void *)plan->X;
  void * Y_p = (void *)plan->Y;

  for (int xh = plan->x_height_loop_counter; xh > 0 ; xh-- ) {
    for (int xv = plan->x_width_loop_counter; xv >= 0 ; xv-- ) {

      make_patch(vpu, plan, X_p);

      void * cur_post_activation_mul = plan->post_activation_mul;
      void * cur_post_activation_bias = plan->post_activation_bias;
      void * K_p = (void *)plan->K;
      for (int oc = plan->output_channel_loop_counter; oc > 0 ; oc-- ) {

        compute_patch(plan, &K_p, 32, vpu, &sat_mem, &bias_shift, &final_shr,
          cur_post_activation_mul, cur_post_activation_bias);

        VSTRPV(vpu, Y_p, 0xffff);
        Y_p += 16;

        cur_post_activation_mul += 32;
        cur_post_activation_bias += 32;
      }
      
      compute_patch(plan, &K_p, plan->k_p_rewind, vpu, &sat_mem, &bias_shift, &final_shr,
        cur_post_activation_mul, cur_post_activation_bias);


    // printf("before write %08x\n", plan->final_channels_mask);
    // vpu_sim_print(vpu);

      VSTRPV(vpu, Y_p, plan->final_channels_mask);

      Y_p += plan->final_channels_bytes;
      X_p += plan->outer_x_h_step;
    }
    X_p += plan->outer_x_v_step;
    Y_p += plan->y_v_step;
  }

}

static void bnn_conv2d_int8_out_SISO_asm_prepare(
    nn_bnn_conv2d_int8_out_SISO_asm_plan_t* plan, int8_t* Y_p,
    const bnn_b32_t* X_p, const bnn_b32_t* K_p, bnn_b32_t * data_scratch,
    
    const int16_t* post_activation_multiplier_q, 
    const int16_t* post_activation_bias_q,
    const int accu_shr,
    const int16_t bias_multipler,
    const int final_shr,

    const nn_image_params_t* x, 
    const nn_image_params_t* y,
    const nn_window_params_t* k, 
    const unsigned y_loc_x, const unsigned y_loc_y,
    const unsigned y_sub_width, const unsigned y_sub_height,
    const unsigned x_loc_x, const unsigned x_loc_y)
{

  const unsigned bits_per_b32 = 32;
  const unsigned chan_b32_in = (x->channels + bits_per_b32 - 1) / bits_per_b32;
  const unsigned chans_out = y->channels;

  int8_t (*Y)[y->width][chans_out] =
      (int8_t (*)[y->width][chans_out])Y_p;

  bnn_b32_t(*X)[x->width][chan_b32_in] =
      (bnn_b32_t(*)[x->width][chan_b32_in])X_p;

//relocate the pointers to the start of the region we care about.
  plan->Y = (int8_t*)Y[y_loc_y][y_loc_x];
  plan->X = (bnn_b32_t*)X[x_loc_y][x_loc_x];
  plan->K = K_p;
  plan->data_scratch = data_scratch;
  
  plan->post_activation_mul = (int16_t *)post_activation_multiplier_q;
  plan->post_activation_bias = (int16_t *)post_activation_bias_q;
  plan->final_shr = final_shr;
  plan->bias_multiplier = bias_multipler;

  if(accu_shr >= 0){
    plan->vlsat = accu_shr;
    plan->ashr = 0;
  } else {
    plan->vlsat = 0;
    plan->ashr = accu_shr;
  }

  unsigned bytes_per_input_channel = x->channels / 8;

  const unsigned out_chans_multiplier = 4;

  assert(x->channels >0);
  assert(y->channels >0);
  assert((x->channels % bits_per_b32) == 0);
  assert((y->channels % out_chans_multiplier) == 0);

  plan->k_height_loop_counter = k->shape.height - 1;
  plan->k_width_loop_counter = k->shape.width - 1;

  assert(k->dilation.horizontal >= 1);
  assert(k->dilation.vertical >= 1);

  unsigned h_dilation = k->dilation.horizontal;
  unsigned h_stride = k->stride.horizontal;
  unsigned v_stride = k->stride.vertical;

  plan->input_channel_loop_counter =
      ((x->channels + XS3_VPU_VREG_WIDTH_BITS - 1) / XS3_VPU_VREG_WIDTH_BITS) - 1;

  unsigned x_height_loops = y_sub_height;
  unsigned x_width_loops = y_sub_width;

  plan->x_height_loop_counter = x_height_loops;
  plan->x_width_loop_counter = x_width_loops - 1;

  unsigned total_bytes_copied_to_scratch = (x->channels * k->shape.height *  k->shape.width)/8;

  unsigned channels_to_process_on_tail_output_loop = (y->channels - 4) % 16 + 4;

  plan->output_channel_loop_counter = (y->channels-channels_to_process_on_tail_output_loop)/16;

  plan->k_p_rewind = -(16L - 2L - ((y->channels-1)%16))*32;

  if (total_bytes_copied_to_scratch%32){
    plan->k_p_adjust  = total_bytes_copied_to_scratch%32;
  } else {
    plan->k_p_adjust = 32;
  }

  plan->patch_loop_counter = (total_bytes_copied_to_scratch - plan->k_p_adjust) / (256/8);

  plan->final_channels_bytes = channels_to_process_on_tail_output_loop;
  plan->final_channels_mask = ((1 << channels_to_process_on_tail_output_loop)-1) ;

  int t = (x->channels/8)%32;
  if(t == 0)
    plan->data_scratch_adjust = 0;
  else
    plan->data_scratch_adjust = t - 32;
  
  plan->inner_x_h_step = (int)bytes_per_input_channel * ((int)h_dilation - 1) - (32L*(plan->input_channel_loop_counter + 1) - bytes_per_input_channel);

  // TODO multiply x->width by dilation
  plan->inner_x_v_step =
      (bytes_per_input_channel * ((x->width -  k->shape.width))) ;

  // Outer Loop
  plan->outer_x_h_step = bytes_per_input_channel * h_stride;

  plan->outer_x_v_step = (int)((int)bytes_per_input_channel * (int)x->width * (int)v_stride) 
     - (int)((int)plan->outer_x_h_step * (int)x_width_loops);

  plan->y_v_step = chans_out * sizeof(int8_t) * (y->width - y_sub_width);


  // printf("%d\n%d\n%u\n", plan->inner_x_h_step, plan->data_scratch_adjust, plan->y_v_step);
  // printf("%u\n%d\n%d\n", plan->k_width_loop_counter, plan->inner_x_v_step, plan->outer_x_v_step);
  // printf("%d\n%u\n%u\n", plan->y_v_step, plan->output_channel_loop_counter, plan->vlsat);
  // printf("%d\n%u\n%d\n", plan->ashr, plan->input_channel_loop_counter, plan->outer_x_h_step);
  // printf("%d\n%d\n%u\n", plan->k_p_adjust, plan->patch_branch, plan->final_channels_mask);
  // printf("%u\n%u\n%d\n", plan->final_channels_bytes, plan->patch_loop_counter, plan->final_shr);
  // printf("%d\n%u\n%u\n", plan->k_p_rewind, plan->x_width_loop_counter, plan->x_height_loop_counter);
  // printf("%u\n\n", plan->bias_multiplier);
  // printf("%d\n", plan->output_channel_loop_counter);
  // printf("%08x\n", plan->final_channels_mask);
}

static void bnn_conv2d_int8_out_asm_prepare(
    nn_bnn_conv2d_int8_out_asm_plan_t* plan, int8_t* Y_p,
    const bnn_b256_t* X_p, const bnn_b256_t* K_p, 
    
    const int16_t* post_activation_multiplier_q, 
    const int16_t* post_activation_bias_q,
    const int accu_shr,
    const int16_t bias_multiplier,
    const int final_shr,

    const nn_image_params_t* x, 
    const nn_image_params_t* y,
    const nn_window_params_t* k, 

    const unsigned y_loc_x, const unsigned y_loc_y,
    const unsigned y_sub_width, const unsigned y_sub_height,
    const unsigned x_loc_x, const unsigned x_loc_y
  ) {

  const unsigned chan_b256_in = (x->channels + XS3_VPU_VREG_WIDTH_BITS - 1) / XS3_VPU_VREG_WIDTH_BITS;
  const unsigned chans_out = y->channels;

  int8_t (*Y)[y->width][chans_out] =
      (int8_t (*)[y->width][chans_out])Y_p;

  bnn_b256_t(*X)[x->width][chan_b256_in] =
      (bnn_b256_t(*)[x->width][chan_b256_in])X_p;

//relocate the pointers to the start of the region we care about.
  plan->Y = (int8_t*)Y[y_loc_y][y_loc_x];
  plan->X = (bnn_b256_t*)X[x_loc_y][x_loc_x];
  plan->K = K_p;

  //This could go into the constant pool but it would make the loading
  //slower within the kernel(2 loops in).
  plan->bias_multiplier = bias_multiplier;
  
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
  // unsigned bytes_per_output_channel = y->channels;

  const unsigned out_chans_multiplier = 16;

  assert((x->channels % XS3_VPU_VREG_WIDTH_BITS) == 0);
  assert((y->channels % out_chans_multiplier) == 0);

  plan->k_height_loop_counter = k->shape.height - 1;
  plan->k_width_loop_counter = k->shape.width - 1;

  unsigned h_dilation = k->dilation.horizontal;

  unsigned h_stride = k->stride.horizontal;
  unsigned v_stride = k->stride.vertical;

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
  plan->outer_x_h_step = bytes_per_input_channel * h_stride;

  plan->outer_x_v_step = (int)(bytes_per_input_channel * x->width * v_stride) 
     - (int)(plan->outer_x_h_step * x_width_loops);

  plan->y_v_step = chans_out * sizeof(int8_t) * (y->width - y_sub_width);

  // TODO these are for implementing sub-kernels
  plan->k_v_step = 0;
  plan->k_h_step = 0;
  
}

void bnn_conv2d_int8_out(int8_t* Y_p,
    const bnn_b256_t* X_p, const bnn_b256_t* K_p, 
    
    const int16_t* post_activation_multiplier_q, 
    const int16_t* post_activation_bias_q,
    const int accu_shr,
    const int16_t bias_multipler,
    const int final_shr,
    
    const nn_image_params_t* x, //The full image of x
    const nn_image_params_t* y, // the full image of y
    const nn_window_params_t* k, //the full kernel k
    
    const unsigned y_loc_x, const unsigned y_loc_y,
    const unsigned y_sub_width, const unsigned y_sub_height,

    const unsigned x_loc_x, const unsigned x_loc_y
){
  nn_bnn_conv2d_int8_out_asm_plan_t plan;

  bnn_conv2d_int8_out_asm_prepare(&plan, Y_p,
      X_p,  K_p,
      post_activation_multiplier_q, 
      post_activation_bias_q,
      accu_shr,
      bias_multipler,
      final_shr,
      x, y, k, 
      y_loc_x, y_loc_y, y_sub_width, y_sub_height,
      x_loc_x, x_loc_y);

  bnn_conv2d_int8_out_asm(&plan);
}

void bnn_conv2d_int8_out_SISO(int8_t* Y_p,
    const bnn_b32_t* X_p, const bnn_b32_t* K_p, 
    
    const int16_t* post_activation_multiplier_q, 
    const int16_t* post_activation_bias_q,
    const int accu_shr,
    const int16_t bias_multipler,
    const int final_shr,

    bnn_b32_t * data_scratch,
    
    const nn_image_params_t* x, //The full image of x
    const nn_image_params_t* y, // the full image of y
    const nn_window_params_t* k, //the full kernel k
    
    const unsigned y_loc_x, const unsigned y_loc_y,
    const unsigned y_sub_width, const unsigned y_sub_height,

    const unsigned x_loc_x, const unsigned x_loc_y
) {
    nn_bnn_conv2d_int8_out_SISO_asm_plan_t plan;

    bnn_conv2d_int8_out_SISO_asm_prepare(&plan, Y_p,
        X_p,  K_p, data_scratch,
        post_activation_multiplier_q, 
        post_activation_bias_q,
        accu_shr,
        bias_multipler,
        final_shr,
        x, y, k, 
        y_loc_x, y_loc_y, y_sub_width, y_sub_height,
        x_loc_x, x_loc_y);

    bnn_conv2d_int8_out_SISO_asm(&plan);
}
