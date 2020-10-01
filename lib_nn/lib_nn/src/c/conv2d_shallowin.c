

#include "nn_operator.h"
#include "../nn_op_helper.h"
// #include "nn_op_structs.h"

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
 * Struct represents the job-specific parameters required to execute a `conv2d_deep()` operation. 
 */
typedef struct {

    struct {

        struct {
            int32_t Y;
        } chan_group;

        struct {
            int32_t X;
            int32_t window;
            int32_t Y;
        } row;
    } stride;

} nn_conv2d_shallowin_job_t;





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



static void conv2d_shallowin_compute_padding(
    int32_t* top,
    int32_t* bottom,
    int32_t* left,
    int32_t* right,
    const nn_image_params_t* x_params,
    const nn_window_params_t* conv_window,
    const nn_conv2d_job_params_t* job_params)
{
    const unsigned k_array_width = VPU_INT8_EPV / x_params->channels;

    const int32_t init_padding_top    = -conv_window->start.row;
    const int32_t init_padding_bottom =  conv_window->start.row + conv_window->shape.height - x_params->height;
    const int32_t init_padding_left   = -conv_window->start.column;
    const int32_t init_padding_right  =  conv_window->start.column + k_array_width - x_params->width;
    

    *top    = init_padding_top    - job_params->start.rows * conv_window->stride.vertical;
    *left   = init_padding_left   - job_params->start.cols * conv_window->stride.horizontal;
    *bottom = init_padding_bottom + job_params->start.rows * conv_window->stride.vertical;
    *right  = init_padding_right  + job_params->start.cols * conv_window->stride.horizontal;
}

static void conv2d_shallowin_adjust_starts(
    int8_t** Y,
    const int8_t** X,
    const int8_t** K,
    const nn_bso_block_t** BSO,
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_window_params_t* conv_window,
    const nn_conv2d_job_params_t* job_params,
    const nn_conv2d_shallowin_flags_e flags)
{
    const unsigned x_row_bytes = x_params->width * x_params->channels;
    const unsigned y_row_bytes = y_params->width * y_params->channels;

    const mem_stride_t window_start_offset = conv_window->start.row * x_row_bytes 
                                           + conv_window->start.column * x_params->channels;

    int32_t start_BSO  = (job_params->start.channels / VPU_INT8_ACC_PERIOD);
    int32_t start_K    = job_params->start.channels * conv_window->shape.height * VPU_INT8_EPV;
    int32_t start_Y    = job_params->start.rows * y_row_bytes 
                            + job_params->start.cols * y_params->channels
                            + job_params->start.channels;

    int32_t start_X    = window_start_offset 
                            + job_params->start.rows * conv_window->stride.vertical * x_row_bytes
                            + job_params->start.cols * conv_window->stride.horizontal * x_params->channels;

                            
    *X = ADDR(*X, start_X);
    *Y = ADDR(*Y, start_Y);
    *K = ADDR(*K, start_K);
    *BSO = ADDR(*BSO, start_BSO);
}



#ifndef CONV2D_INIT_ERROR_DETECTION_ENABLE
  #define CONV2D_INIT_ERROR_DETECTION_ENABLE     (1)
#endif
void conv2d_shallowin_prepare(
    nn_conv2d_shallowin_job_t* job,
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_conv2d_job_params_t* job_params,
    const nn_window_params_t* conv_window)
{
    if(CONV2D_INIT_ERROR_DETECTION_ENABLE){
        // Input and output channels must each be a multiple of 4
        assert(x_params->channels % 4 == 0);
        assert(y_params->channels % 4 == 0);
        // The product of the input channels and kernel width must be <= VPU_INT8_EPV
        assert(x_params->channels * conv_window->shape.width <= VPU_INT8_EPV);
    }

    const unsigned k_array_width = VPU_INT8_EPV / x_params->channels;

    const unsigned x_row_bytes = x_params->width * x_params->channels;
    const unsigned y_row_bytes = y_params->width * y_params->channels;

    job->stride.row.X = x_row_bytes;
    
    
    if(CONV2D_INIT_ERROR_DETECTION_ENABLE){
        // Start can never be negative
        assert(job_params->start.rows >= 0 
            && job_params->start.cols >= 0 
            && job_params->start.channels >= 0);

        // Start channel has to be 0 mod 16
        assert(job_params->start.channels % VPU_INT8_ACC_PERIOD == 0);

        // Make sure we're not trying to compute outputs that go beyond
        //  the bounds of the output image.
        assert(job_params->start.rows + job_params->size.rows <= y_params->height);
        assert(job_params->start.cols + job_params->size.cols <= y_params->width );
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

    job->stride.chan_group.Y = VPU_INT8_ACC_PERIOD - y_row_bytes * job_params->size.rows;
}


void conv2d_shallowin(
    nn_image_t* Y,
    const nn_image_t* X,
    const nn_tensor_t* K,
    const nn_bso_block_t* BSO,
    const int8_t zero_point,
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_window_params_t* conv_window)
{    
    nn_conv2d_job_params_t full_job = {{0,0,0}, {y_params->height, y_params->width, y_params->channels} };

    conv2d_shallowin_adv(Y, X, K, BSO, zero_point, x_params, y_params, conv_window, &full_job, 0);
}



void conv2d_shallowin_adv(
    nn_image_t* Y,
    const nn_image_t* X,
    const nn_tensor_t* K,
    const nn_bso_block_t* BSO,
    const int8_t zero_point,
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_window_params_t* conv_window,
    const nn_window_op_job_params_t* job_params,
    const nn_conv2d_shallowin_flags_e flags)
{ 
    conv2d_shallowin_adjust_starts(&Y, &X, &K, &BSO, x_params, y_params, conv_window, job_params, flags);

    nn_conv2d_shallowin_job_t job;

    conv2d_shallowin_prepare(&job, x_params, y_params, job_params, conv_window);

    int8_t zero_point_vec[VPU_INT8_EPV];
    memset(zero_point_vec, zero_point, sizeof(zero_point_vec));
    
    struct {
        int32_t top;
        int32_t left;
        int32_t bottom;
        int32_t right;
    } init_padding;

    conv2d_shallowin_compute_padding(&init_padding.top, &init_padding.bottom, 
                                     &init_padding.left, &init_padding.right, 
                                     x_params, conv_window, job_params);


    const unsigned C_out_tail = y_params->channels % VPU_INT8_ACC_PERIOD;

    for(int out_chan = 0; out_chan < job_params->size.channels; out_chan += VPU_INT8_ACC_PERIOD){

        const unsigned cur_chans = (job_params->size.channels - out_chan >= VPU_INT8_VLMACC_ELMS)? 
            VPU_INT8_VLMACC_ELMS : job_params->size.channels - out_chan; 

        int pad_t = init_padding.top;
        int pad_b = init_padding.bottom;

        K = ADDR(K, conv_window->shape.height * VPU_INT8_EPV * (cur_chans - 1));

        const nn_image_t* X_cog = X;

        for(int out_row = 0; out_row < job_params->size.rows; out_row++){

            int pad_l = init_padding.left;
            int pad_r = init_padding.right;

            const int pad_lr_delta = conv_window->stride.horizontal * (job_params->size.cols - 1);
            const int final_pad_l = pad_l - pad_lr_delta;
            const int final_pad_r = pad_r + pad_lr_delta;

            const int cur_pad_t = (pad_t > 0)? pad_t : 0;
            const int cur_pad_b = (pad_b > 0)? pad_b : 0;

            //TODO: Currently init_padding.right is forced to make the assumption that the actual width of the
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
                    nn_conv2d_hstrip_shallowin_padded(Y, X_cog, K, BSO, conv_window->shape.height,
                        conv_window->stride.horizontal, x_params->channels, cur_pad_t, cur_pad_b, pad_l, 
                        pad_r, job.stride.row.X, y_params->channels, job_params->size.cols, zero_point_vec);
                } else {
                    nn_conv2d_hstrip_shallowin( Y, X_cog, K, BSO, conv_window->shape.height,
                        conv_window->stride.horizontal, x_params->channels, job.stride.row.X, 
                        y_params->channels, job_params->size.cols);
                }
            } else {
                if(requires_padding){
                    nn_conv2d_hstrip_tail_shallowin_padded( Y, X_cog, K, BSO, conv_window->shape.height, 
                        conv_window->stride.horizontal, x_params->channels, cur_pad_t, cur_pad_b, pad_l, 
                        pad_r, job.stride.row.X, y_params->channels, job_params->size.cols, zero_point_vec, 
                        C_out_tail);
                } else {
                    nn_conv2d_hstrip_tail_shallowin( Y, X_cog, K, BSO, conv_window->shape.height, 
                        conv_window->stride.horizontal, x_params->channels, job.stride.row.X, 
                        y_params->channels, job_params->size.cols, C_out_tail);
                }
            }

            pad_t -= conv_window->stride.vertical;
            pad_b += conv_window->stride.vertical;

            X_cog = ADDR(X_cog, job.stride.row.window);
            Y = ADDR(Y, job.stride.row.Y);
        }

        K = ADDR(K, conv_window->shape.height * VPU_INT8_EPV);
        Y = ADDR(Y, job.stride.chan_group.Y);
        BSO = ADDR(BSO, 1);
    }
}