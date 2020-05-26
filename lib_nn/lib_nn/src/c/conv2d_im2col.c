

#include "nn_operator.h"
#include "../nn_op_helper.h"
#include "nn_op_structs.h"

#include "xs3_vpu.h"
#include "vpu_sim.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>


void conv2d_im2col_init(
    nn_conv2d_im2col_plan_t* plan,
    nn_conv2d_im2col_job_t* jobs,
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_conv2d_job_params_t* job_params,
    const nn_conv2d_window_params_t* conv_window,
    const int8_t zero_point,
    const unsigned job_count)
{

    // The restrict to patch elements <= 128 for the time being, but should be able to relax
    assert(x_params->channels * conv_window->shape.width * conv_window->shape.height <= VPU_INT8_EPV*4);

    // Need at least 1 job
    assert(job_count > 0);
    // job_params can only be NULL if there's exactly 1 job.
    assert(job_count == 1 || job_params != NULL);

    const unsigned k_array_width = VPU_INT8_EPV / x_params->channels;

    const unsigned x_row_bytes = x_params->width * x_params->channels;
    const unsigned y_row_bytes = y_params->width * y_params->channels;

    const mem_stride_t window_start_offset = conv_window->start.row * x_row_bytes 
                                           + conv_window->start.column * x_params->channels;

    plan->channels.X = x_params->channels;
    plan->channels.Y = y_params->channels;

    plan->zero_point = zero_point;

    plan->window.shape.height      = conv_window->shape.height;
    plan->window.shape.width       = conv_window->shape.width;
    plan->window.shape.len_col     = ((conv_window->shape.width * 
                                       conv_window->shape.height * 
                                       x_params->channels + 31)>>5)<<5;
    plan->window.shape.kernel_row_elements     = 
                                      ((conv_window->shape.width * 
                                       conv_window->shape.height * 
                                       x_params->channels + 3)>>2)<<2;

    plan->window.stride.vertical   = conv_window->stride.vertical;
    plan->window.stride.horizontal = conv_window->stride.horizontal;

    plan->stride.X.row = x_row_bytes;
    
    const int32_t init_padding_top    = -conv_window->start.row;
    const int32_t init_padding_bottom =  conv_window->start.row + conv_window->shape.height - x_params->height;
    const int32_t init_padding_left   = -conv_window->start.column;
    const int32_t init_padding_right  =  conv_window->start.column + k_array_width - x_params->width;

    nn_conv2d_job_params_t full_job = {{0,0,0}, {y_params->height, y_params->width, y_params->channels} };

    for(int i = 0; i < job_count; i++){
        const nn_conv2d_job_params_t* params = (job_params != NULL)? &job_params[i] : &full_job;
        nn_conv2d_im2col_job_t* job = &jobs[i];
        
        // Start can never be negative
        assert(params->start.rows >= 0 
            && params->start.cols >= 0 
            && params->start.channels >= 0);

        // Start channel has to be 0 mod 16
        assert(params->start.channels % VPU_INT8_ACC_PERIOD == 0);

        // Make sure we're not trying to compute outputs that go beyond
        //  the bounds of the output image.
        assert(params->start.rows + params->size.rows <= y_params->height);
        assert(params->start.cols + params->size.cols <= y_params->width );
        assert(params->start.channels + params->size.channels <= y_params->channels);

        job->output.rows = params->size.rows;
        job->output.cols = params->size.cols;
        job->output.channels = params->size.channels;

        job->init_padding.top    = init_padding_top    - params->start.rows * plan->window.stride.vertical;
        job->init_padding.left   = init_padding_left   - params->start.cols * plan->window.stride.horizontal;
        job->init_padding.bottom = init_padding_bottom + params->start.rows * plan->window.stride.vertical;
        job->init_padding.right  = init_padding_right  + params->start.cols * plan->window.stride.horizontal;

        job->stride.start.BSO  = (params->start.channels / VPU_INT8_ACC_PERIOD);
        job->stride.start.K    = params->start.channels * plan->window.shape.height * VPU_INT8_EPV;
        job->stride.start.Y    = params->start.rows * y_row_bytes 
                               + params->start.cols * y_params->channels
                               + params->start.channels;

        job->stride.start.X    = window_start_offset 
                               + params->start.rows * plan->window.stride.vertical * x_row_bytes
                               + params->start.cols * plan->window.stride.horizontal * x_params->channels;

        job->stride.row.window  = x_row_bytes * plan->window.stride.vertical;
        job->stride.row.Y       = y_row_bytes;

        job->stride.chan_group.Y = VPU_INT8_ACC_PERIOD - y_row_bytes * job->output.rows;
        job->stride.col.Y = y_row_bytes / job->output.cols;// TODO check
    }
}


#if CONFIG_SYMMETRIC_SATURATION_conv2d_im2col
  #define NEG_SAT_VAL   (-127)
#else
  #define NEG_SAT_VAL   (-128)
#endif 


const extern int16_t vec_0x007F[VPU_INT8_ACC_PERIOD];
const extern int8_t vec_0x80[VPU_INT8_EPV];

void conv2d_im2col(
    nn_image_t* Y,
    const nn_image_t* X,
    const nn_image_t* COL,
    const nn_tensor_t* K,
    const nn_bso_block_t* BSO,
    const nn_conv2d_im2col_plan_t* plan,
    const nn_conv2d_im2col_job_t* job)
{ 
    // int8_t zero_point_vec[VPU_INT8_EPV];
    // memset(zero_point_vec, plan->zero_point, sizeof(zero_point_vec));
    
    xs3_vpu vpu;
    vpu_vector_t vec_tmp;

    VSETC(&vpu, MODE_S8);

    X = ADDR(X,job->stride.start.X);
    Y = ADDR(Y, job->stride.start.Y);
    K = ADDR(K, job->stride.start.K);
    BSO = ADDR(BSO, job->stride.start.BSO);

    const unsigned C_out_tail = plan->channels.Y % VPU_INT8_ACC_PERIOD; // TODO I don't think im2col cares about tails..

    const unsigned vlmaccr_align = 4;
    const unsigned patch_elements = plan->channels.X * plan->window.shape.height * plan->window.shape.width;
    
    int pad_t = job->init_padding.top;
    int pad_b = job->init_padding.bottom;
    //Iterate once per row of the output region
    for(unsigned output_rows = job->output.rows; output_rows > 0; output_rows--){
        //Iterate once per col of the output region
        for(unsigned output_cols = job->output.cols; output_cols > 0; output_cols--){

            //This points it at the top-left cell of the patch.
            const nn_image_t* patch_X = X;
            const nn_image_t* patch_K = K;

            #if !CONFIG_SYMMETRIC_SATURATION_conv2d_im2col
                    VLDR(&vpu, vec_0x80);
                    VSTRPV(&vpu, Y, 0xFFFF);
            #endif
            // set up "column" by copying all patch rows sequentially
            // TODO look at Aaron's VPU memcpy (doesn't require word-aligned)
            const int8_t* C = COL;
            memset(C,0,plan->window.shape.len_col);// initialize pad -- zero point is handled at bias initialization TODO is this still true?
            unsigned len = plan->channels.X * plan->window.shape.width;
            // start paste
            int pad_l = job->init_padding.left;
            int pad_r = job->init_padding.right;

            const int pad_lr_delta = plan->window.stride.horizontal * (job->output.cols - 1);
            const int final_pad_l = pad_l - pad_lr_delta;
            const int final_pad_r = pad_r + pad_lr_delta;

            const int cur_pad_t = (pad_t > 0)? pad_t : 0;
            const int cur_pad_b = (pad_b > 0)? pad_b : 0;

            const unsigned requires_padding = (pad_l       > 0) || (pad_r       > 0) 
                                           || (cur_pad_t   > 0) || (cur_pad_b   > 0) 
                                           || (final_pad_l > 0) || (final_pad_r > 0);
            // end paste

            for(unsigned rows_in_patch = plan->window.shape.height; rows_in_patch > 0; rows_in_patch--){
                if( requires_padding ){
                    int8_t padded_vals[128]={0}; // todo drop the 128 limit or enforce it
                    int k = 0;
                    int tmp = 0;
                    for(int h = 0; h< plan->window.shape.height; h++){
                        tmp = (cur_pad_t -h > 0) || (plan->window.shape.height - h <= cur_pad_b); // ???
                        for(int i = 0; i < plan->window.shape.width; i++){
                            tmp &= (pad_l -i > 0) || (plan->window.shape.width - i <= pad_r); // ???
                            for(int j = 0; j < plan->channels.X; j++){
                                padded_vals[k] = (tmp & 0x1) ? patch_X[k] : plan->zero_point;
                                k++;
                            }
                        }  
                    }
                    memcpy(C, padded_vals, len);
                }
                else{
                    memcpy(C,patch_X,len);
                }
                
                patch_X = ADDR(patch_X,job->stride.row.window);
                C = ADDR(C,len);

            }
            // im2col complete what follows is just a BSO-aware K*COL = Y            
            for(unsigned j = 0; j < plan->channels.Y; j++){ // TODO sub-optimal traversal, should do sub-rows at a time
                
                patch_K = ADDR(patch_K,plan->window.shape.kernel_row_elements); // TODO make this the k row increment at initialization time .  
                COL = ADDR(COL,plan->window.shape.kernel_row_elements);   
                
                const nn_tensor_t* K_tmp = patch_K;
                const nn_image_t* sub_C = COL;
                for(int k=0; k< plan->window.shape.kernel_row_elements/VPU_INT8_ACC_PERIOD; k++){
                    VLDD(&vpu, BSO->bias_hi);
                    VLDR(&vpu, BSO->bias_lo);
                    BSO = ADDR(BSO, 1);// check

                    VLDC(&vpu, sub_C);
                    sub_C = ADDR(sub_C,-VPU_INT8_ACC_PERIOD);

                    for(int i = 0; i < VPU_INT8_ACC_PERIOD; i++){
                        VLMACCR(&vpu, K_tmp);
                        K_tmp = ADDR(K_tmp, -VPU_INT8_ACC_PERIOD);
                    }
                }
                //Done accumulating for the current output channel
                
                //reset COL (part of sub-optimal traversal)
                COL = ADDR(COL,-plan->window.shape.kernel_row_elements);

                //Set mode to 16-bit
                VSETC(&vpu, MODE_S16);

                //Saturate to 16-bit values
                VLSAT(&vpu, BSO->shift1);

                //Load scales into vC
                VLDC(&vpu, BSO->scale);
                VSTR(&vpu, vec_tmp.s16);
                VCLRDR(&vpu);
                VLMACC(&vpu, vec_tmp.s16);
                VLDC(&vpu, BSO->offset_scale);
                VLMACC(&vpu, BSO->offset);

                #if CONFIG_SYMMETRIC_SATURATION_conv2d_im2col

                    //Set mode back to 8-bit
                    VSETC(&vpu, MODE_S8);

                    //Saturate to 8-bit values
                    VLSAT(&vpu, BSO->shift2);

                    //Store result in Y
                    const unsigned mask16 = 0xFFFF;
                    VSTRPV(&vpu, Y, mask16);
                
                #else

                    //Saturate to 8-bit values
                    VLSAT(&vpu, BSO->shift2);

                    VSTR(&vpu, vec_tmp.s16);
                    VLADD(&vpu, vec_0x007F);
                    VDEPTH1(&vpu);
                    uint32_t mask = ~vpu.vR.s32[0];

                    VLASHR(&vpu, vec_tmp.s16, -8);
                    VDEPTH8(&vpu);

                    //Store result in Y
                    mask = mask & 0xFFFF;
                    VSTRPV(&vpu, Y, mask);

                    //Set mode back to 8-bit
                    VSETC(&vpu, MODE_S8);

                #endif
            }
            
            //Move X and Y pointers one pixel to the right
            X = ADDR(X, plan->window.stride.horizontal);// TODO see if we need X and window
            Y = ADDR(Y, job->stride.col.Y);
            
        }
        //Move X and Y pointers to the start of the following row
        X = ADDR(X, job->stride.row.window);
        Y = ADDR(Y, job->stride.row.Y);
        pad_t -= plan->window.stride.vertical;
        pad_b += plan->window.stride.vertical;

    }

}