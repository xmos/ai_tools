

#include "nn_operator.h"
#include "../nn_op_helper.h"

#include "xs3_vpu.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>



#ifndef MAXPOOL2D_INIT_ERROR_DETECTION_ENABLE
  #define MAXPOOL2D_INIT_ERROR_DETECTION_ENABLE     (1)
#endif




static void maxpool2d_adjust_starts(
    int8_t** Y,
    const int8_t** X,
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_window_params_t* pooling_window,
    const nn_window_op_job_params_t* job_params,
    const nn_maxpool2d_flags_e flags)
{
    const int32_t x_row_bytes = x_params->width * x_params->channels;

    // The start row and col in X of this particular job
    const int32_t job_start_row_x = pooling_window->start.row + pooling_window->stride.vertical * job_params->start.rows;
    const int32_t job_start_col_x = pooling_window->start.column + pooling_window->stride.horizontal * job_params->start.cols;
    const int32_t start_X = job_start_row_x * x_row_bytes + x_params->channels * job_start_col_x + job_params->start.channels;
    
    const int32_t y_row_bytes = y_params->width * y_params->channels;
    const int32_t start_Y = job_params->start.rows * y_row_bytes + job_params->start.cols * y_params->channels + job_params->start.channels;

    *X = ADDR(*X, start_X);
    *Y = ADDR(*Y, start_Y);
}


static void maxpool2d_prepare(
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_window_params_t* pooling_window,
    const nn_window_op_job_params_t* job_params)
{

    if(MAXPOOL2D_INIT_ERROR_DETECTION_ENABLE){
    
        const unsigned final_row = ((pooling_window->start.row    + pooling_window->shape.height - 1) 
                                    + (pooling_window->stride.vertical * (y_params->height-1)));
        const unsigned final_col = ((pooling_window->start.column + pooling_window->shape.width - 1) 
                                    + (pooling_window->stride.horizontal * (y_params->width-1)));

        //Input and output must have same channel count
        assert(x_params->channels == y_params->channels);
        // Every pixel must be word-aligned.
        assert(x_params->channels % 4 == 0);

        // This operator doesn't support padding
        assert( final_row < x_params->height);
        assert( final_col < x_params->width);
        assert( pooling_window->start.row >= 0 );
        assert( pooling_window->start.column >= 0 );


        assert(job_params->start.rows >= 0);
        assert(job_params->start.cols >= 0);
        assert(job_params->start.channels >= 0);

        assert(job_params->start.rows + job_params->size.rows <= y_params->height);
        assert(job_params->start.cols + job_params->size.cols <= y_params->width);
        assert(job_params->start.channels + job_params->size.channels <= y_params->channels);
        
        //NOTE: unlike most operators, this one has an output channel group size of 32
        assert(job_params->start.channels % 4 == 0);
        assert(job_params->size.channels % 4 == 0);

    }
    
}



void maxpool2d(
    nn_image_t* Y,
    const nn_image_t* X, 
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_window_params_t* pooling_window)
{
    nn_window_op_job_params_t full_job = {{0,0,0},{y_params->height, y_params->width, y_params->channels}};

    maxpool2d_ext(Y, X, x_params, y_params, pooling_window, &full_job, 0);
}


WEAK_FUNC
void maxpool2d_ext(
    nn_image_t* Y,
    const nn_image_t* X, 
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_window_params_t* pooling_window,
    const nn_window_op_job_params_t* job_params,
    const nn_maxpool2d_flags_e flags)
{
    maxpool2d_adjust_starts(&Y, &X, x_params, y_params, pooling_window, job_params, flags);

    maxpool2d_prepare(x_params, y_params, pooling_window, job_params);


    mem_stride_t stride_X_row, stride_Y_row, 
                 stride_window_col, stride_window_row, 
                 stride_X_cog, stride_Y_cog;

    do {

        const int32_t x_row_bytes = x_params->width * x_params->channels;
        const int32_t y_row_bytes = y_params->width * y_params->channels;

        const mem_stride_t win_hstride_from_prev_start = pooling_window->stride.horizontal * x_params->channels;
        const mem_stride_t win_vstride_from_prev_start = pooling_window->stride.vertical * x_row_bytes;

        stride_X_row = x_row_bytes - pooling_window->shape.width * x_params->channels;
        stride_Y_row = y_row_bytes - job_params->size.cols * y_params->channels;

        // The X pointer will be pointing at the start of the first row *not* inside the window 
        // when doing an hstride
        stride_window_col = win_hstride_from_prev_start - pooling_window->shape.height * x_row_bytes;

        // The X pointer will be pointing at the start of the first patch *after* the job's output 
        // columns when doing a vstride.
        stride_window_row = win_vstride_from_prev_start - win_hstride_from_prev_start * job_params->size.cols;

        // For a channel group stride, move back to start of job and add 32
        stride_X_cog = VPU_INT8_EPV - job_params->size.rows * win_vstride_from_prev_start;
        stride_Y_cog = VPU_INT8_EPV - job_params->size.rows * y_row_bytes;

    } while(0); // Adding scope just to clarify what's temporary.


    const uint32_t chan_groups = job_params->size.channels >> VPU_INT8_EPV_LOG2;
    const uint32_t chan_tail   = job_params->size.channels  % VPU_INT8_EPV;

    for(int cog = 0; cog <= chan_groups; cog++){

        unsigned cur_chans = (cog < chan_groups)? VPU_INT8_EPV : chan_tail;

        if(cur_chans == 0)
            break;

        for(int out_row = 0; out_row < job_params->size.rows; out_row++){
            for(int out_col = 0; out_col < job_params->size.cols; out_col++){

                int8_t maxes[VPU_INT8_EPV];
                memset(maxes, -128, sizeof(maxes));

                for(int pool_row = 0; pool_row < pooling_window->shape.height; pool_row++){
                    for(int pool_col = 0; pool_col < pooling_window->shape.width; pool_col++){
                        
                        for(int k = 0; k < VPU_INT8_EPV; k++){
                            maxes[k] = (X[k] > maxes[k])? X[k] : maxes[k];
                        }

                        X = ADDR(X, x_params->channels);
                    }

                    X = ADDR(X, stride_X_row);
                }

                for(int k = 0; k < cur_chans; k++){
                    Y[k] = maxes[k];
                }

                Y = ADDR(Y, y_params->channels);
                X = ADDR(X, stride_window_col);
            }
            
            X = ADDR(X, stride_window_row);
            Y = ADDR(Y, stride_Y_row);
        }
        
        X = ADDR(X, stride_X_cog);
        Y = ADDR(Y, stride_Y_cog);
    }
}




