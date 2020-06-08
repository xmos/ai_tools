

#include "nn_operator.h"
#include "../nn_op_helper.h"

#include "xs3_vpu.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>




WEAK_FUNC
void maxpool2d(
    int8_t* Y,
    const int8_t* X, 
    const nn_maxpool2d_plan_t* plan,
    const nn_maxpool2d_job_t* job)
{
    X = ADDR(X, job->stride.X.start);
    Y = ADDR(Y, job->stride.Y.start);

    const uint32_t chan_groups = job->output.channels >> VPU_INT8_EPV_LOG2;
    const uint32_t chan_tail   = job->output.channels  % VPU_INT8_EPV;

    for(int cog = 0; cog <= chan_groups; cog++){

        unsigned cur_chans = (cog < chan_groups)? VPU_INT8_EPV : chan_tail;

        if(cur_chans == 0)
            break;

        for(int out_row = 0; out_row < job->output.rows; out_row++){
            for(int out_col = 0; out_col < job->output.cols; out_col++){

                int8_t maxes[VPU_INT8_EPV];
                memset(maxes, -128, sizeof(maxes));

                for(int pool_row = 0; pool_row < plan->window.rows; pool_row++){
                    for(int pool_col = 0; pool_col < plan->window.cols; pool_col++){
                        
                        for(int k = 0; k < VPU_INT8_EPV; k++){
                            maxes[k] = (X[k] > maxes[k])? X[k] : maxes[k];
                        }

                        X = ADDR(X, plan->channels.X);
                    }

                    X = ADDR(X, job->stride.X.row);
                }

                for(int k = 0; k < cur_chans; k++){
                    Y[k] = maxes[k];
                }

                Y = ADDR(Y, plan->channels.Y);
                X = ADDR(X, job->stride.window.col);
            }
            
            X = ADDR(X, job->stride.window.row);
            Y = ADDR(Y, job->stride.Y.row);
        }
        
        X = ADDR(X, job->stride.X.cog);
        Y = ADDR(Y, job->stride.Y.cog);
    }
}



void maxpool2d_init(
    nn_maxpool2d_plan_t* plan,
    nn_maxpool2d_job_t* jobs,
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_window_params_t* window_config,
    const nn_conv2d_job_params_t* job_params,
    const unsigned job_count)
{
    // If job_count is 1, job_params can be null
    assert(job_count == 1 || job_params != NULL);
    //Input and output must have same channel count
    assert(x_params->channels == y_params->channels);
    // Every pixel must be word-aligned.
    assert(x_params->channels % 4 == 0);

    plan->channels.X = x_params->channels;
    plan->channels.Y = y_params->channels;
    
    const unsigned final_row = ((window_config->start.row    + window_config->shape.height - 1) + (window_config->stride.vertical * (y_params->height-1)));
    const unsigned final_col = ((window_config->start.column + window_config->shape.width - 1) + (window_config->stride.horizontal * (y_params->width-1)));

    // This operator doesn't support padding
    assert( final_row < x_params->height);
    assert( final_col < x_params->width);
    assert( window_config->start.row >= 0 );
    assert( window_config->start.column >= 0 );

    plan->window.rows = window_config->shape.height;
    plan->window.cols = window_config->shape.width;



    const int32_t x_row_bytes = x_params->width * x_params->channels;
    const int32_t y_row_bytes = y_params->width * y_params->channels;

    const int32_t win_start_pix = window_config->start.row * x_params->width + window_config->start.column;

    const mem_stride_t win_hstride_from_prev_start = window_config->stride.horizontal * x_params->channels;
    const mem_stride_t win_vstride_from_prev_start = window_config->stride.vertical * x_row_bytes;

    const nn_conv2d_job_params_t full_job = { { 0, 0, 0 }, { y_params->height, y_params->width, y_params->channels } };

    for(int i = 0; i < job_count; i++){

        const nn_conv2d_job_params_t* params = (job_params != NULL)? &job_params[i] : &full_job;
        nn_maxpool2d_job_t* job = &jobs[i];

        assert(params->start.rows >= 0);
        assert(params->start.cols >= 0);
        assert(params->start.channels >= 0);

        assert(params->start.rows + params->size.rows <= y_params->height);
        assert(params->start.cols + params->size.cols <= y_params->width);
        assert(params->start.channels + params->size.channels <= y_params->channels);
        
        //NOTE: unlike most operators, this one has an output channel group size of 32
        assert(params->start.channels % 4 == 0);
        assert(params->size.channels % 4 == 0);

        job->output.rows = params->size.rows;
        job->output.cols = params->size.cols;
        job->output.channels = params->size.channels;


        // The start row and col in X of this particular job
        const int32_t job_start_row_x = window_config->start.row + window_config->stride.vertical * params->start.rows;
        const int32_t job_start_col_x = window_config->start.column + window_config->stride.horizontal * params->start.cols;

        job->stride.X.start = job_start_row_x * x_row_bytes + x_params->channels * job_start_col_x + params->start.channels;
        job->stride.Y.start = params->start.rows * y_row_bytes + params->start.cols * y_params->channels + params->start.channels;

        job->stride.X.row = x_row_bytes - window_config->shape.width * x_params->channels;
        job->stride.Y.row = y_row_bytes - params->size.cols * y_params->channels;

        // The X pointer will be pointing at the start of the first row *not* inside the window 
        // when doing an hstride
        job->stride.window.col = win_hstride_from_prev_start - window_config->shape.height * x_row_bytes;

        // The X pointer will be pointing at the start of the first patch *after* the job's output 
        // columns when doing a vstride.
        job->stride.window.row = win_vstride_from_prev_start - win_hstride_from_prev_start * params->size.cols;

        // For a channel group stride, move back to start of job and add 32
        job->stride.X.cog = VPU_INT8_EPV - params->size.rows * win_vstride_from_prev_start;
        job->stride.Y.cog = VPU_INT8_EPV - params->size.rows * y_row_bytes;



    }
    
}


