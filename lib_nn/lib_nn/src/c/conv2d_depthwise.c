
#include "nn_operator.h"
#include "../nn_op_helper.h"
#include "nn_op_structs.h"

#include "xs3_vpu.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

#define ADDR(VR, STR)   printf("!\t%s = 0x%08X\t\t(%s)\n", (#VR), (unsigned) (VR), (STR))


#define INDEX_CAST(X)   ((int32_t)(X))



void conv2d_depthwise_init(
    nn_conv2d_depthwise_plan_t* plan,
    nn_conv2d_depthwise_job_t* jobs,
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_conv2d_job_params_t* job_params,
    const int kernel_start_row,
    const int kernel_start_col,
    const unsigned K_h,
    const unsigned K_w,
    const int v_stride,
    const int h_stride,
    const int8_t zero_point,
    const unsigned job_count)
{
    assert(x_params->channels % 4 == 0);
    assert(x_params->channels == y_params->channels);
    assert(job_count > 0);
    assert(job_count == 1 || job_params != NULL);

    const unsigned x_row_bytes = x_params->width * x_params->channels;
    const unsigned y_row_bytes = y_params->width * y_params->channels;
    const unsigned k_row_bytes = K_w * y_params->channels;

    const int32_t window_start_offset = kernel_start_row * x_row_bytes + x_params->channels * kernel_start_col;

    plan->channels.X = x_params->channels;
    plan->channels.Y = y_params->channels;

    plan->zero_point = zero_point;

    plan->stride.X.row = x_row_bytes - x_params->channels * K_w;
    plan->stride.window.col = x_params->channels * h_stride;

    plan->kernel.height  = K_h;
    plan->kernel.width   = K_w;
    plan->kernel.vstride = v_stride;
    
    const int32_t init_padding_top    = -kernel_start_row;
    const int32_t init_padding_bottom =  kernel_start_row + K_h - x_params->height;
    const int32_t init_padding_left   = -kernel_start_col;
    const int32_t init_padding_right  =  kernel_start_col + K_w  - x_params->width;

    nn_conv2d_job_params_t full_job = {{0,0,0}, {y_params->height, y_params->width, y_params->channels} };

    for(int i = 0; i < job_count; i++){
        const nn_conv2d_job_params_t* params = (job_params != NULL)? &job_params[i] : &full_job;
        nn_conv2d_depthwise_job_t* job = &jobs[i];
        
        assert(params->start.rows >= 0 && params->start.cols >= 0 && params->start.channels >= 0);
        assert(params->start.rows + params->size.rows <= y_params->height);
        assert(params->start.cols + params->size.cols <= y_params->width);
        assert(params->start.channels + params->size.channels <= y_params->channels);

        job->output.rows = params->size.rows;
        job->output.cols = params->size.cols;
        job->output.channels = params->size.channels;

        job->init_padding.top    = init_padding_top    - params->start.rows * plan->kernel.vstride;
        job->init_padding.left   = init_padding_left   - params->start.cols * h_stride;
        job->init_padding.bottom = init_padding_bottom + params->start.rows * plan->kernel.vstride;
        job->init_padding.right  = init_padding_right  + params->start.cols * h_stride;

        const int32_t end_padding_top    = init_padding_top    - ((params->start.rows + params->size.rows - 1) * plan->kernel.vstride);
        const int32_t end_padding_left   = init_padding_left   - ((params->start.cols + params->size.cols - 1) * h_stride);
        const int32_t end_padding_bottom = init_padding_bottom + ((params->start.rows + params->size.rows - 1) * plan->kernel.vstride);
        const int32_t end_padding_right  = init_padding_right  + ((params->start.cols + params->size.cols - 1) * h_stride);

        job->init_padding.unpadded =  (job->init_padding.top    <= 0 && job->init_padding.left  <= 0
                                    && job->init_padding.bottom <= 0 && job->init_padding.right <= 0
                                    && end_padding_top          <= 0 && end_padding_left        <= 0
                                    && end_padding_bottom       <= 0 && end_padding_right       <= 0);


        job->stride.start.BSS  = (params->start.channels / VPU_INT8_ACC_PERIOD);
        job->stride.start.K    = params->start.channels;
        job->stride.start.Y    = params->start.rows * y_row_bytes + y_params->channels * params->start.cols + params->start.channels;

        job->stride.start.X    = window_start_offset 
                                + params->start.rows * plan->kernel.vstride * x_row_bytes
                                + params->start.cols * h_stride * x_params->channels
                                + params->start.channels;

        job->stride.row.window  = x_row_bytes * plan->kernel.vstride;
        job->stride.row.Y       = y_row_bytes;

        job->stride.chan_group.X = VPU_INT8_VLMACC_ELMS - x_row_bytes * job->output.rows * plan->kernel.vstride;
        job->stride.chan_group.Y = VPU_INT8_VLMACC_ELMS - y_row_bytes * job->output.rows;
    }
}


void conv2d_depthwise(
    int8_t* Y,
    const int8_t* X,
    const int8_t* K,
    const nn_bss_block_t* BSS,
    const nn_conv2d_depthwise_plan_t* plan,
    const nn_conv2d_depthwise_job_t* job)
{
    // ADDR(X, "initial");
    // ADDR(Y, "initial");
    // ADDR(K, "initial");

    int8_t zero_point_vec[VPU_INT8_VLMACC_ELMS];
    memset(zero_point_vec, plan->zero_point, sizeof(zero_point_vec));

    X = &X[INDEX_CAST(job->stride.start.X)];
    Y = &Y[INDEX_CAST(job->stride.start.Y)];
    K = &K[INDEX_CAST(job->stride.start.K)];
    BSS = &BSS[INDEX_CAST(job->stride.start.BSS)];

    // ADDR(X, "start strided");
    // ADDR(Y, "start strided");
    // ADDR(K, "start strided");

    for(int out_chan = 0; out_chan < job->output.channels; out_chan += VPU_INT8_VLMACC_ELMS){

        const unsigned cur_chans = (job->output.channels - out_chan >= VPU_INT8_VLMACC_ELMS)? VPU_INT8_VLMACC_ELMS : job->output.channels - out_chan; 

        int pad_t = job->init_padding.top;
        int pad_b = job->init_padding.bottom;

        for(int out_row = 0; out_row < job->output.rows; out_row++){
            // ADDR(X, "out row start");
            // ADDR(Y, "out row start");
            // ADDR(K, "out row start");

            const int cur_pad_t = (pad_t > 0)? pad_t : 0;
            const int cur_pad_b = (pad_b > 0)? pad_b : 0;
            
            if(job->init_padding.unpadded){
                nn_compute_hstrip_depthwise(Y, X, K, BSS, plan->kernel.height, plan->kernel.width,
                        plan->channels.X, plan->stride.X.row,
                        plan->stride.window.col, plan->channels.Y, job->output.cols, cur_chans);
            } else {
                nn_compute_hstrip_depthwise_padded(Y, X, K, BSS, plan->kernel.height, plan->kernel.width,
                            cur_pad_t, job->init_padding.left, cur_pad_b, job->init_padding.right,
                            plan->channels.X, plan->stride.X.row, 
                            plan->stride.window.col, plan->channels.Y, job->output.cols, 
                            cur_chans, zero_point_vec);
            }

            pad_t -= plan->kernel.vstride;
            pad_b += plan->kernel.vstride;

            X = &X[INDEX_CAST(job->stride.row.window)];
            Y = &Y[INDEX_CAST(job->stride.row.Y)];
            
        }

        X = &X[INDEX_CAST(job->stride.chan_group.X)];
        Y = &Y[INDEX_CAST(job->stride.chan_group.Y)];

        BSS = &BSS[INDEX_CAST(1)];
        K = &K[INDEX_CAST(VPU_INT8_VLMACC_ELMS)];
    }

}
