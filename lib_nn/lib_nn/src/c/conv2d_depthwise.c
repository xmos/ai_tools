
#include "nn_operator.h"
#include "../nn_op_helper.h"
#include "nn_op_structs.h"

#include "xs3_vpu.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>


typedef struct {
    int top;
    int left;
    int bottom;
    int right;
} incl_bounds_t;

/**
 * Compute the bounds of the convolution window (in an input image's coordinate space) corresponding to a given 
 * pixel location in the output image's coordinate space.
 * 
 * The bottom and right coordinates are inclusive.
 */
static incl_bounds_t inverse_map(
    const nn_window_params_t* conv_window,
    const int out_row,
    const int out_col)
{
    incl_bounds_t res;
    res.top    = conv_window->start.row    + conv_window->stride.vertical   * out_row;
    res.left   = conv_window->start.column + conv_window->stride.horizontal * out_col;
    res.bottom = res.top  + conv_window->shape.height - 1; //inclusive bottom
    res.right  = res.left + conv_window->shape.width  - 1; //inclusive right
    return res;
}

#ifndef CONV2D_INIT_ERROR_DETECTION_ENABLE
  #define CONV2D_INIT_ERROR_DETECTION_ENABLE     (1)
#endif

void conv2d_depthwise_init(
    nn_conv2d_depthwise_plan_t* plan,
    nn_conv2d_depthwise_job_t* jobs,
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_conv2d_job_params_t* job_params,
    const nn_window_params_t* conv_window,
    const int8_t zero_point,
    const unsigned job_count)
{
    if(CONV2D_INIT_ERROR_DETECTION_ENABLE){
        assert(x_params->channels % 4 == 0);
        assert(x_params->channels == y_params->channels);
        assert(job_count > 0);
        assert(job_count == 1 || job_params != NULL);
    }

    const unsigned x_row_bytes = x_params->width * x_params->channels;
    const unsigned y_row_bytes = y_params->width * y_params->channels;
    const unsigned k_row_bytes = conv_window->shape.width * y_params->channels;

    const int32_t window_start_offset = conv_window->start.row * x_row_bytes 
                                        + x_params->channels * conv_window->start.column;

    plan->channels.X = x_params->channels;
    plan->channels.Y = y_params->channels;

    plan->zero_point = zero_point;

    plan->stride.X.row = x_row_bytes - x_params->channels * conv_window->shape.width;
    plan->stride.window.col = x_params->channels * conv_window->stride.horizontal;

    plan->kernel.height  = conv_window->shape.height;
    plan->kernel.width   = conv_window->shape.width;
    plan->kernel.vstride = conv_window->stride.vertical;
    
    const int32_t init_padding_top    = -conv_window->start.row;
    const int32_t init_padding_bottom =  conv_window->start.row + conv_window->shape.height - x_params->height;
    const int32_t init_padding_left   = -conv_window->start.column;
    const int32_t init_padding_right  =  conv_window->start.column + conv_window->shape.width  - x_params->width;

    nn_conv2d_job_params_t full_job = {{0,0,0}, {y_params->height, y_params->width, y_params->channels} };

    for(int i = 0; i < job_count; i++){
        const nn_conv2d_job_params_t* params = (job_params != NULL)? &job_params[i] : &full_job;
        nn_conv2d_depthwise_job_t* job = &jobs[i];
        
        if(CONV2D_INIT_ERROR_DETECTION_ENABLE){
            assert(params->start.rows >= 0 && params->start.cols >= 0 && params->start.channels >= 0);
            assert(params->start.rows + params->size.rows <= y_params->height);
            assert(params->start.cols + params->size.cols <= y_params->width);
            assert(params->start.channels + params->size.channels <= y_params->channels);

            // Make sure the convolution window is never entirely outside the input image.
            //   (If it is, it would have to also be for the first and/or last pixel)
            {
                int first_row_y = params->start.rows;
                int final_row_y = first_row_y + params->size.rows - 1;
                int first_col_y = params->start.cols;
                int final_col_y = first_col_y + params->size.cols - 1;

                incl_bounds_t bounds = inverse_map(conv_window, first_row_y, first_col_y);

                assert(bounds.bottom >= 0);
                assert(bounds.right >= 0);

                bounds = inverse_map(conv_window, final_row_y, final_col_y);

                assert(bounds.top  < ((int)x_params->height));
                assert(bounds.left < ((int)x_params->width));
            }
        }

        job->output.rows = params->size.rows;
        job->output.cols = params->size.cols;
        job->output.channels = params->size.channels;

        job->init_padding.top    = init_padding_top    - params->start.rows * plan->kernel.vstride;
        job->init_padding.left   = init_padding_left   - params->start.cols * conv_window->stride.horizontal;
        job->init_padding.bottom = init_padding_bottom + params->start.rows * plan->kernel.vstride;
        job->init_padding.right  = init_padding_right  + params->start.cols * conv_window->stride.horizontal;

        const int32_t end_padding_top    = init_padding_top    - ((params->start.rows + params->size.rows - 1) * plan->kernel.vstride);
        const int32_t end_padding_left   = init_padding_left   - ((params->start.cols + params->size.cols - 1) * conv_window->stride.horizontal);
        const int32_t end_padding_bottom = init_padding_bottom + ((params->start.rows + params->size.rows - 1) * plan->kernel.vstride);
        const int32_t end_padding_right  = init_padding_right  + ((params->start.cols + params->size.cols - 1) * conv_window->stride.horizontal);

        job->init_padding.unpadded =  (job->init_padding.top    <= 0 && job->init_padding.left  <= 0
                                    && job->init_padding.bottom <= 0 && job->init_padding.right <= 0
                                    && end_padding_top          <= 0 && end_padding_left        <= 0
                                    && end_padding_bottom       <= 0 && end_padding_right       <= 0);


        job->stride.start.BSO  = (params->start.channels / VPU_INT8_ACC_PERIOD);
        job->stride.start.K    = params->start.channels;
        job->stride.start.Y    = params->start.rows * y_row_bytes + y_params->channels * params->start.cols + params->start.channels;

        job->stride.start.X    = window_start_offset 
                                + params->start.rows * plan->kernel.vstride * x_row_bytes
                                + params->start.cols * conv_window->stride.horizontal * x_params->channels
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
    const nn_bso_block_t* BSO,
    const nn_conv2d_depthwise_plan_t* plan,
    const nn_conv2d_depthwise_job_t* job)
{

    int8_t zero_point_vec[VPU_INT8_VLMACC_ELMS];
    memset(zero_point_vec, plan->zero_point, sizeof(zero_point_vec));

    X = ADDR(X, job->stride.start.X);
    Y = ADDR(Y, job->stride.start.Y);
    K = ADDR(K, job->stride.start.K);
    BSO = ADDR(BSO, job->stride.start.BSO);

    for(int out_chan = 0; out_chan < job->output.channels; out_chan += VPU_INT8_VLMACC_ELMS){

        const unsigned cur_chans = (job->output.channels - out_chan >= VPU_INT8_VLMACC_ELMS)? VPU_INT8_VLMACC_ELMS : job->output.channels - out_chan; 

        int pad_t = job->init_padding.top;
        int pad_b = job->init_padding.bottom;

        for(int out_row = 0; out_row < job->output.rows; out_row++){

            const int cur_pad_t = (pad_t > 0)? pad_t : 0;
            const int cur_pad_b = (pad_b > 0)? pad_b : 0;
            
            if(job->init_padding.unpadded){
                nn_conv2d_hstrip_depthwise(Y, X, K, BSO, plan->kernel.height, plan->kernel.width,
                        plan->channels.X, plan->stride.X.row,
                        plan->stride.window.col, plan->channels.Y, job->output.cols, cur_chans);
            } else {
                nn_conv2d_hstrip_depthwise_padded(Y, X, K, BSO, plan->kernel.height, plan->kernel.width,
                            cur_pad_t, job->init_padding.left, cur_pad_b, job->init_padding.right,
                            plan->channels.X, plan->stride.X.row, 
                            plan->stride.window.col, plan->channels.Y, job->output.cols, 
                            cur_chans, zero_point_vec);
            }

            pad_t -= plan->kernel.vstride;
            pad_b += plan->kernel.vstride;

            X = ADDR(X, job->stride.row.window);
            Y = ADDR(Y, job->stride.row.Y);
            
        }

        X = ADDR(X, job->stride.chan_group.X);
        Y = ADDR(Y, job->stride.chan_group.Y);

        BSO = ADDR(BSO, 1);
        K = ADDR(K, VPU_INT8_VLMACC_ELMS);
    }

}
