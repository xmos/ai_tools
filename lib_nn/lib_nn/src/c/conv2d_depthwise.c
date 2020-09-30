
#include "nn_operator.h"
#include "../nn_op_helper.h"
// #include "nn_op_structs.h"

#include "xs3_vpu.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>



#ifndef CONV2D_PREPARE_ERROR_DETECTION_ENABLE
  #define CONV2D_PREPARE_ERROR_DETECTION_ENABLE     (1)
#endif


/**
 * Struct represents the job-specific parameters required to execute a 
 * `conv2d_depthwise()` operation. 
 */
typedef struct {

    struct {
        struct {
            int32_t X;
            int32_t Y;
        } chan_group;

        struct {
            int32_t X;
            int32_t window;
            int32_t Y;
        } row;

        struct {
            int32_t window;
        } col;
    } stride;

} nn_conv2d_depthwise_job_t;



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

static unsigned conv2d_depthwise_compute_padding(
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
    const int32_t init_padding_right  =  conv_window->start.column + conv_window->shape.width  - x_params->width;
    
    *top    = init_padding_top    - job_params->start.rows * conv_window->stride.vertical;
    *left   = init_padding_left   - job_params->start.cols * conv_window->stride.horizontal;
    *bottom = init_padding_bottom + job_params->start.rows * conv_window->stride.vertical;
    *right  = init_padding_right  + job_params->start.cols * conv_window->stride.horizontal;

    const int32_t end_padding_left   = init_padding_left   - ((job_params->start.cols + job_params->size.cols - 1) * conv_window->stride.horizontal);
    const int32_t end_padding_right  = init_padding_right  + ((job_params->start.cols + job_params->size.cols - 1) * conv_window->stride.horizontal);

    return (   (*left  <= 0) && (end_padding_left  <= 0) 
            && (*right <= 0) && (end_padding_right <= 0));
}

static void conv2d_depthwise_adjust_starts(
    int8_t** Y,
    const int8_t** X,
    const int8_t** K,
    const nn_bso_block_t** BSO,
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_window_params_t* conv_window,
    const nn_conv2d_job_params_t* job_params,
    const nn_conv2d_depthwise_flags_e flags)
{
    const unsigned x_row_bytes = x_params->width * x_params->channels;
    const unsigned y_row_bytes = y_params->width * y_params->channels;

    const int32_t window_start_offset = conv_window->start.row * x_row_bytes 
                                        + x_params->channels * conv_window->start.column;
    
    int32_t start_X    = window_start_offset 
                            + job_params->start.rows * conv_window->stride.vertical * x_row_bytes
                            + job_params->start.cols * conv_window->stride.horizontal * x_params->channels
                            + job_params->start.channels;
    int32_t start_Y    = job_params->start.rows * y_row_bytes + y_params->channels * job_params->start.cols + job_params->start.channels;

    *X = ADDR(*X, start_X);
    *Y = ADDR(*Y, start_Y);

    if(!(flags & CONV2D_DEPTHWISE_FLAG_SLICED_K)){
        int32_t start_K    = job_params->start.channels;
        int32_t start_BSO   = (job_params->start.channels / VPU_INT8_ACC_PERIOD);

        *K = ADDR(*K, start_K);
        *BSO = ADDR(*BSO, start_BSO);
    }
}

static void conv2d_depthwise_prepare(
    nn_conv2d_depthwise_job_t* job,
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_window_params_t* conv_window,
    const nn_conv2d_job_params_t* job_params)
{
    if(CONV2D_PREPARE_ERROR_DETECTION_ENABLE){
        assert(x_params->channels % 4 == 0);
        assert(x_params->channels == y_params->channels);
    }

    const unsigned x_row_bytes = x_params->width * x_params->channels;
    const unsigned y_row_bytes = y_params->width * y_params->channels;
    const unsigned k_row_bytes = conv_window->shape.width * y_params->channels;

    job->stride.row.X = x_row_bytes - x_params->channels * conv_window->shape.width;
    job->stride.col.window = x_params->channels * conv_window->stride.horizontal;
    
    if(CONV2D_PREPARE_ERROR_DETECTION_ENABLE){
        assert(job_params->start.rows >= 0 && job_params->start.cols >= 0 && job_params->start.channels >= 0);
        assert(job_params->start.rows + job_params->size.rows <= y_params->height);
        assert(job_params->start.cols + job_params->size.cols <= y_params->width);
        assert(job_params->start.channels + job_params->size.channels <= y_params->channels);

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

    job->stride.row.window  = x_row_bytes * conv_window->stride.vertical;
    job->stride.row.Y       = y_row_bytes;

    job->stride.chan_group.X = VPU_INT8_VLMACC_ELMS - x_row_bytes * job_params->size.rows * conv_window->stride.vertical;
    job->stride.chan_group.Y = VPU_INT8_VLMACC_ELMS - y_row_bytes * job_params->size.rows;
}


void conv2d_depthwise(
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

    conv2d_depthwise_adv(Y, X, K, BSO, zero_point, x_params, y_params, conv_window, &full_job, 0);
}

void conv2d_depthwise_adv(
    int8_t* Y,
    const int8_t* X,
    const int8_t* K,
    const nn_bso_block_t* BSO,
    const int8_t zero_point,
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_window_params_t* conv_window,
    const nn_window_op_job_params_t* job_params,
    const nn_conv2d_depthwise_flags_e flags)
{
    conv2d_depthwise_adjust_starts(&Y, &X, &K, &BSO, x_params, y_params, conv_window, job_params, flags);

    int8_t zero_point_vec[VPU_INT8_VLMACC_ELMS];
    memset(zero_point_vec, zero_point, sizeof(zero_point_vec));

    nn_conv2d_depthwise_job_t job;
    conv2d_depthwise_prepare(&job, x_params, y_params, conv_window, job_params);

    channel_count_t k_channels = x_params->channels;
    if(flags & CONV2D_DEPTHWISE_FLAG_SLICED_K){
        k_channels = job_params->size.channels;
    }

    struct {
        int32_t top;
        int32_t left;
        int32_t bottom;
        int32_t right;
        unsigned unpadded_lr;
    } init_padding;

    init_padding.unpadded_lr = conv2d_depthwise_compute_padding(&init_padding.top, &init_padding.bottom, 
                                                                &init_padding.left, &init_padding.right, 
                                                                x_params, conv_window, job_params);

    for(int out_chan = 0; out_chan < job_params->size.channels; out_chan += VPU_INT8_VLMACC_ELMS){

        const unsigned cur_chans = (job_params->size.channels - out_chan >= VPU_INT8_VLMACC_ELMS)? VPU_INT8_VLMACC_ELMS 
                                   : job_params->size.channels - out_chan; 

        int pad_t = init_padding.top;
        int pad_b = init_padding.bottom;

        for(int out_row = 0; out_row < job_params->size.rows; out_row++){

            const int cur_pad_t = (pad_t > 0)? pad_t : 0;
            const int cur_pad_b = (pad_b > 0)? pad_b : 0;
            
            if(init_padding.unpadded_lr && cur_pad_t == 0 && cur_pad_b == 0){
                nn_conv2d_hstrip_depthwise(Y, X, K, BSO, conv_window->shape.height, conv_window->shape.width,
                        x_params->channels, k_channels, job.stride.row.X,
                        job.stride.col.window, y_params->channels, job_params->size.cols, cur_chans);
            } else {
                nn_conv2d_hstrip_depthwise_padded(Y, X, K, BSO, conv_window->shape.height, conv_window->shape.width,
                            cur_pad_t, init_padding.left, cur_pad_b, init_padding.right,
                            x_params->channels, k_channels, job.stride.row.X, 
                            job.stride.col.window, y_params->channels, job_params->size.cols, 
                            cur_chans, zero_point_vec);
            }

            pad_t -= conv_window->stride.vertical;
            pad_b += conv_window->stride.vertical;

            X = ADDR(X, job.stride.row.window);
            Y = ADDR(Y, job.stride.row.Y);
            
        }

        X = ADDR(X, job.stride.chan_group.X);
        Y = ADDR(Y, job.stride.chan_group.Y);

        BSO = ADDR(BSO, 1);
        K = ADDR(K, VPU_INT8_VLMACC_ELMS);
    }

}
