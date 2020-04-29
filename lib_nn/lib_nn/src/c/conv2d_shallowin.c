

#include "nn_operator.h"
#include "../nn_op_helper.h"
#include "nn_op_structs.h"

#include "xs3_vpu.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>


#define ADDR(V, INDEX)      &V[((int)(INDEX))]


void conv2d_shallowin_init(
    nn_conv2d_shallowin_plan_t* plan,
    nn_conv2d_shallowin_job_t* jobs,
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_conv2d_job_params_t* job_params,
    const nn_conv2d_window_params_t* conv_window,
    const int8_t zero_point,
    const unsigned job_count)
{
    // Input and output channels must each be a multiple of 4
    assert(x_params->channels % 4 == 0);
    assert(y_params->channels % 4 == 0);
    // The product of the input channels and kernel width must be <= VPU_INT8_EPV
    assert(x_params->channels * conv_window->shape.width <= VPU_INT8_EPV);

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
        nn_conv2d_shallowin_job_t* job = &jobs[i];
        
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

        job->stride.start.BSS  = (params->start.channels / VPU_INT8_ACC_PERIOD);
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
    }
}




void conv2d_shallowin(
    nn_image_t* Y,
    const nn_image_t* X,
    const nn_tensor_t* K,
    const nn_bss_block_t* BSS,
    const nn_conv2d_shallowin_plan_t* plan,
    const nn_conv2d_shallowin_job_t* job)
{ 
    int8_t zero_point_vec[VPU_INT8_EPV];
    memset(zero_point_vec, plan->zero_point, sizeof(zero_point_vec));
    
    X = ADDR(X,job->stride.start.X);
    Y = ADDR(Y, job->stride.start.Y);
    K = ADDR(K, job->stride.start.K);
    BSS = ADDR(BSS, job->stride.start.BSS);

    const unsigned C_out_tail = plan->channels.Y % VPU_INT8_ACC_PERIOD;

    for(int out_chan = 0; out_chan < job->output.channels; out_chan += VPU_INT8_ACC_PERIOD){

        const unsigned cur_chans = (job->output.channels - out_chan >= VPU_INT8_VLMACC_ELMS)? 
            VPU_INT8_VLMACC_ELMS : job->output.channels - out_chan; 

        int pad_t = job->init_padding.top;
        int pad_b = job->init_padding.bottom;

        K = ADDR(K, plan->window.shape.height * VPU_INT8_EPV * (cur_chans - 1));

        const nn_image_t* X_cog = X;

        for(int out_row = 0; out_row < job->output.rows; out_row++){

            int pad_l = job->init_padding.left;
            int pad_r = job->init_padding.right;

            const int pad_lr_delta = plan->window.stride.horizontal * (job->output.cols - 1);
            const int final_pad_l = pad_l - pad_lr_delta;
            const int final_pad_r = pad_r + pad_lr_delta;

            const int cur_pad_t = (pad_t > 0)? pad_t : 0;
            const int cur_pad_b = (pad_b > 0)? pad_b : 0;

            //TODO: Currently job->init_padding.right is forced to make the assumption that the actual width of the
            //      kernel is 32/C_in, regardless of the convolution window width with which the operator was initialized.
            //      That means that we'll end up "requiring padding" when we don't actually have to, because the elements
            //      after the user-supplied convolution window width are guaranteed (per the contract with the user) to
            //      be zeroed out.
            //      At the moment I'm stuffing the user-supplied convolution window width into the plan struct. I'm pretty
            //      sure I can just adjust the conditions for "requires_padding" below using that value to handle this.

            const unsigned requires_padding = (pad_l       > 0) || (pad_r       > 0) 
                                           || (cur_pad_t   > 0) || (cur_pad_b   > 0) 
                                           || (final_pad_l > 0) || (final_pad_r > 0);

            if(cur_chans == VPU_INT8_ACC_PERIOD){
                if(requires_padding){
                    nn_conv2d_hstrip_shallowin_padded(Y, X_cog, K, BSS, plan->window.shape.height,
                        plan->window.stride.horizontal, plan->channels.X, cur_pad_t, cur_pad_b, pad_l, 
                        pad_r, plan->stride.X.row, plan->channels.Y, job->output.cols, zero_point_vec);
                } else {
                    nn_conv2d_hstrip_shallowin( Y, X_cog, K, BSS, plan->window.shape.height,
                        plan->window.stride.horizontal, plan->channels.X, plan->stride.X.row, 
                        plan->channels.Y, job->output.cols);
                }
            } else {
                if(requires_padding){
                    nn_conv2d_hstrip_tail_shallowin_padded( Y, X_cog, K, BSS, plan->window.shape.height, 
                        plan->window.stride.horizontal, plan->channels.X, cur_pad_t, cur_pad_b, pad_l, 
                        pad_r, plan->stride.X.row, plan->channels.Y, job->output.cols, zero_point_vec, 
                        C_out_tail);
                } else {
                    nn_conv2d_hstrip_tail_shallowin( Y, X_cog, K, BSS, plan->window.shape.height, 
                        plan->window.stride.horizontal, plan->channels.X, plan->stride.X.row, 
                        plan->channels.Y, job->output.cols, C_out_tail);
                }
            }

            pad_t -= plan->window.stride.vertical;
            pad_b += plan->window.stride.vertical;

            X_cog = ADDR(X_cog, job->stride.row.window);
            Y = ADDR(Y, job->stride.row.Y);
        }

        K = ADDR(K, plan->window.shape.height * VPU_INT8_EPV);
        Y = ADDR(Y, job->stride.chan_group.Y);
        BSS = ADDR(BSS, 1);
    }
}