

#include "nn_operator.h"
#include "../nn_op_helper.h"
// #include "nn_op_structs.h"

#include "xs3_vpu.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>


#ifndef CONV2D_INIT_ERROR_DETECTION_ENABLE
  #define CONV2D_INIT_ERROR_DETECTION_ENABLE     (1)
#endif

typedef struct {
    int top;
    int left;
    int bottom;
    int right;
} incl_bounds_t;



/**
 * Struct represents the job-specific parameters required to execute a `conv2d_deep()` operation. 
 */
typedef struct {

    struct {

        struct {
            mem_stride_t Y;
            mem_stride_t K;
        } chan_group;

        struct {
            mem_stride_t X;
            mem_stride_t window;
            mem_stride_t Y;
        } row;
    } stride;

} nn_conv2d_deep_job_t;



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


static void conv2d_deep_compute_padding(
    int32_t* top,
    int32_t* bottom,
    int32_t* left,
    int32_t* right,
    const nn_image_params_t* x_params,
    const nn_window_params_t* conv_window,
    const nn_conv2d_job_params_t* job_params)
{
    const int32_t init_padding_top    = -conv_window->start.row;
    const int32_t init_padding_bottom =  conv_window->start.row + conv_window->shape.height - x_params->height;
    const int32_t init_padding_left   = -conv_window->start.column;
    const int32_t init_padding_right  =  conv_window->start.column + conv_window->shape.width - x_params->width;
    
    *top    = init_padding_top    - job_params->start.rows * conv_window->stride.vertical;
    *left   = init_padding_left   - job_params->start.cols * conv_window->stride.horizontal;
    *bottom = init_padding_bottom + job_params->start.rows * conv_window->stride.vertical;
    *right  = init_padding_right  + job_params->start.cols * conv_window->stride.horizontal;
}

static void conv2d_deep_adjust_starts(
    int8_t** Y,
    const int8_t** X,
    const int8_t** K,
    const nn_bso_block_t** BSO,
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_window_params_t* conv_window,
    const nn_conv2d_job_params_t* job_params,
    const nn_conv2d_deep_flags_e flags)
{
    const unsigned x_row_bytes = x_params->width * x_params->channels;
    const unsigned y_row_bytes = y_params->width * y_params->channels;

    const int32_t window_start_offset = conv_window->start.row * x_row_bytes 
                                           + conv_window->start.column * x_params->channels;

    const int32_t start_Y    = job_params->start.rows * y_row_bytes 
                                    + job_params->start.cols * y_params->channels
                                    + job_params->start.channels;

    const int32_t start_X    = window_start_offset 
                                    + job_params->start.rows * conv_window->stride.vertical * x_row_bytes
                                    + job_params->start.cols * conv_window->stride.horizontal * x_params->channels;

                            
    *X = ADDR(*X, start_X);
    *Y = ADDR(*Y, start_Y);

    if( !( flags & CONV2D_DEEP_FLAG_SLICED_K ) ){

        const int32_t stride_K_cout = conv_window->shape.height * conv_window->shape.width * x_params->channels;

        const int32_t start_BSO  = (job_params->start.channels / VPU_INT8_ACC_PERIOD);
        const int32_t start_K    = job_params->start.channels * stride_K_cout;

        *K = ADDR(*K, start_K);
        *BSO = ADDR(*BSO, start_BSO);
    }
}



void conv2d_deep_prepare(
    nn_conv2d_deep_job_t* job,
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_window_params_t* conv_window,
    const nn_conv2d_job_params_t* job_params)
{
    if(CONV2D_INIT_ERROR_DETECTION_ENABLE){
        assert(x_params->channels % 4 == 0); //Input channel count must be multiple of 4.
        assert(y_params->channels % 4 == 0); //Output channel count must be multiple of 4.

        // Start can never be negative
        assert(job_params->start.rows >= 0); // Job start row must not be negative.
        assert(job_params->start.cols >= 0); // Job start column must not be negative.
        assert(job_params->start.channels >= 0); //Job start channel must not be negative.

        assert(job_params->start.channels % VPU_INT8_ACC_PERIOD == 0); // Job start channel must be multiple of 16.

        // Make sure we're not trying to compute outputs that go beyond
        //  the bounds of the output image.
        assert(job_params->start.rows + job_params->size.rows <= y_params->height); // Job extends beyond bottom of output.
        assert(job_params->start.cols + job_params->size.cols <= y_params->width ); // Job extends beyond right of output.
        assert(job_params->start.channels + job_params->size.channels <= y_params->channels); //Job extends beyond channels of output.

        // Make sure the convolution window is never entirely outside the input image.
        //   (If it is, it would have to also be for the first and/or last pixel)
        {
            int first_row_y = job_params->start.rows;
            int final_row_y = first_row_y + job_params->size.rows - 1;
            int first_col_y = job_params->start.cols;
            int final_col_y = first_col_y + job_params->size.cols - 1;

            incl_bounds_t bounds = inverse_map(conv_window, first_row_y, first_col_y);

            assert(bounds.bottom >= 0);
            assert(bounds.right >= 0);

            bounds = inverse_map(conv_window, final_row_y, final_col_y);

            assert(bounds.top  < ((int)x_params->height));
            assert(bounds.left < ((int)x_params->width));
        }
    }

    const unsigned x_row_bytes = x_params->width * x_params->channels;
    const unsigned y_row_bytes = y_params->width * y_params->channels;
    const unsigned patch_width_bytes = conv_window->shape.width * x_params->channels;

    job->stride.row.X       = x_row_bytes - patch_width_bytes;
    job->stride.row.window  = x_row_bytes * conv_window->stride.vertical;
    job->stride.row.Y       = y_row_bytes;
    job->stride.chan_group.Y = VPU_INT8_ACC_PERIOD - y_row_bytes * job_params->size.rows;
    job->stride.chan_group.K = conv_window->shape.height * conv_window->shape.width * x_params->channels;

}


void conv2d_deep(
    int8_t* Y,
    const int8_t* X,
    const int8_t* K,
    const nn_bso_block_t* BSO,
    const int8_t zero_point,
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_window_params_t* conv_window)
{
    const nn_conv2d_job_params_t full_job = {{0,0,0}, {y_params->height, y_params->width, y_params->channels} };

    conv2d_deep_ext(Y, X, K, BSO, zero_point, x_params, y_params, conv_window, &full_job, 0);
}


void conv2d_deep_ext(
    nn_image_t* Y,
    const nn_image_t* X,
    const nn_tensor_t* K,
    const nn_bso_block_t* BSO,
    const int8_t zero_point,
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_window_params_t* conv_window,
    const nn_window_op_job_params_t* job_params,
    const nn_conv2d_deep_flags_e flags)
{ 

    // nn_image_t (*Y_matrix)[y_params->width][y_params->channels] = (nn_image_t (*)[y_params->width][y_params->channels]) Y;

    // printf("Testing: %d!\n", Y_mat[0][0][0]);

    conv2d_deep_adjust_starts(&Y, &X, &K, &BSO, x_params, y_params, conv_window, job_params, flags);

    nn_conv2d_deep_job_t job;

    conv2d_deep_prepare(&job, x_params, y_params, conv_window, job_params);

    int8_t zero_point_vec[VPU_INT8_EPV];
    memset(zero_point_vec, zero_point, sizeof(zero_point_vec));
    
    struct {
        int32_t top;
        int32_t left;
        int32_t bottom;
        int32_t right;
    } init_padding;

    conv2d_deep_compute_padding(&init_padding.top, &init_padding.bottom, 
                                &init_padding.left, &init_padding.right, 
                                x_params, conv_window, job_params);

    const unsigned C_out_tail = y_params->channels % VPU_INT8_ACC_PERIOD;

    for(int out_chan = 0; out_chan < job_params->size.channels; out_chan += VPU_INT8_ACC_PERIOD){

        const unsigned cur_chans = (job_params->size.channels - out_chan >= VPU_INT8_VLMACC_ELMS)? VPU_INT8_VLMACC_ELMS : job_params->size.channels - out_chan; 

        int pad_t = init_padding.top;
        int pad_b = init_padding.bottom;

        K = ADDR(K, job.stride.chan_group.K * (cur_chans - 1));

        const nn_image_t* X_cog = X;

        for(int out_row = 0; out_row < job_params->size.rows; out_row++){

            int pad_l = init_padding.left;
            int pad_r = init_padding.right;

            const int pad_lr_delta = conv_window->stride.horizontal * (job_params->size.cols - 1);
            const int final_pad_l = pad_l - pad_lr_delta;
            const int final_pad_r = pad_r + pad_lr_delta;

            const int cur_pad_t = (pad_t > 0)? pad_t : 0;
            const int cur_pad_b = (pad_b > 0)? pad_b : 0;

            const unsigned requires_padding = (pad_l > 0)       || (pad_r > 0) 
                                           || (cur_pad_t > 0)   || (cur_pad_b > 0) 
                                           || (final_pad_l > 0) || (final_pad_r > 0);

            if(cur_chans == VPU_INT8_ACC_PERIOD){
                if(requires_padding){
                    nn_conv2d_hstrip_deep_padded( Y, X_cog, K, BSO, conv_window->shape.height, conv_window->shape.width, 
                        conv_window->stride.horizontal, x_params->channels, cur_pad_t, cur_pad_b, pad_l, pad_r, 
                        job.stride.row.X, -job.stride.chan_group.K, y_params->channels, job_params->size.cols, zero_point_vec);
                } else {
                    nn_conv2d_hstrip_deep( Y, X_cog, K, BSO, conv_window->shape.height, conv_window->shape.width, 
                        conv_window->stride.horizontal, x_params->channels, job.stride.row.X, -job.stride.chan_group.K, 
                        y_params->channels, job_params->size.cols);
                }
            } else {
                if(requires_padding){
                    nn_conv2d_hstrip_tail_deep_padded( Y, X_cog, K, BSO, conv_window->shape.height, conv_window->shape.width, 
                        conv_window->stride.horizontal, x_params->channels, cur_pad_t, cur_pad_b, pad_l, pad_r, 
                        job.stride.row.X, -job.stride.chan_group.K, y_params->channels, job_params->size.cols, 
                        zero_point_vec, C_out_tail);
                } else {
                    nn_conv2d_hstrip_tail_deep( Y, X_cog, K, BSO, conv_window->shape.height, conv_window->shape.width, 
                        conv_window->stride.horizontal, x_params->channels, job.stride.row.X, -job.stride.chan_group.K, 
                        y_params->channels, job_params->size.cols, C_out_tail);
                }
            }

            pad_t -= conv_window->stride.vertical;
            pad_b += conv_window->stride.vertical;

            X_cog = ADDR(X_cog, job.stride.row.window);
            Y = ADDR(Y, job.stride.row.Y);
        }

        K = ADDR(K, job.stride.chan_group.K);
        Y = ADDR(Y, job.stride.chan_group.Y);
        BSO = ADDR(BSO, 1);
    }
}