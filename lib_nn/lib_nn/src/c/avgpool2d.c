

#include "nn_operator.h"
#include "../nn_op_helper.h"

#include "xs3_vpu.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>






#if CONFIG_SYMMETRIC_SATURATION_avgpool2d
  #define NEG_SAT_VAL   (-127)
#else
  #define NEG_SAT_VAL   (-128)
#endif 


WEAK_FUNC
void avgpool2d_gen(
    int8_t* Y,
    const int8_t* X, 
    const nn_avgpool2d_plan_t* plan,
    const nn_pool2d_job_t* job)
{
    X = ADDR(X, job->stride.X.start);
    Y = ADDR(Y, job->stride.Y.start);

    const int8_t shift = plan->shift & 0xFFFF;
    const int8_t scale = plan->scale & 0xFF;

    const unsigned channel_groups = job->output.channels >> VPU_INT8_ACC_PERIOD_LOG2;

    for(unsigned chn_grp = 0; chn_grp <= channel_groups; chn_grp++){

        unsigned iter_chans = VPU_INT8_ACC_PERIOD;
        if(chn_grp == channel_groups)
            iter_chans = job->output.channels - (channel_groups << VPU_INT8_ACC_PERIOD_LOG2);

        if(iter_chans == 0)
            break;

        for(unsigned out_row = 0; out_row < job->output.rows; out_row++){
            for(unsigned out_col = 0; out_col < job->output.cols; out_col++){
                    
                int32_t acc32[VPU_INT8_ACC_PERIOD] = {0};

                for(unsigned w_rows = 0; w_rows < plan->window.rows; w_rows++){
                    for(unsigned w_cols = 0; w_cols < plan->window.cols; w_cols++){

                        for(unsigned k = 0; k < iter_chans; k++){
                            acc32[k] += (((int32_t)X[k]) * scale);
                        }

                        X = ADDR(X, plan->channels);
                    }

                    X = ADDR(X, job->stride.X.row);
                }

                for(unsigned k = 0; k < iter_chans; k++){
                    Y[k] = vlsat_single_s8(acc32[k], shift, NEG_SAT_VAL, VPU_INT8_MAX);
                }

                X = ADDR(X, job->stride.window.col);
                Y = ADDR(Y, plan->channels);
            }

            X = ADDR(X, job->stride.window.row);
            Y = ADDR(Y, job->stride.Y.row);
        }

        X = ADDR(X, job->stride.X.cog);
        Y = ADDR(Y, job->stride.Y.cog);
    }
}

WEAK_FUNC
void avgpool2d_2x2(
    int8_t* Y,
    const int8_t* X, 
    const nn_avgpool2d_plan_t* plan,
    const nn_pool2d_job_t* job)
{
    avgpool2d_gen(Y, X, plan, job);
}

#undef NEG_SAT_VAL


#if CONFIG_SYMMETRIC_SATURATION_avgpool2d_global
  #define NEG_SAT_VAL   (-127)
#else
  #define NEG_SAT_VAL   (-128)
#endif 


WEAK_FUNC
void avgpool2d_global(
    nn_image_t* Y,
    const nn_image_t* X, 
    const int32_t bias,
    const int8_t scale,
    const int16_t shift,
    const nn_avgpool2d_global_plan_t* plan,
    const nn_avgpool2d_global_job_t* job)
{
    Y = ADDR(Y, job->start_stride);
    X = ADDR(X, job->start_stride);

    const unsigned pix = plan->X.pixels;
    
    for(unsigned ch = 0; ch < job->out_channels; ch++){

        int32_t acc = bias;

        for(unsigned p = 0; p < pix; p++){
            int32_t x = X[p*plan->X.channels + ch];
            acc += x * scale;
        }

        
        Y[ch] = vlsat_single_s8(acc, shift, NEG_SAT_VAL, VPU_INT8_MAX);
    }
}

#undef NEG_SAT_VAL


static inline int matches_2x2_impl(
    const nn_window_params_t* conv_window)
{   
    return  (conv_window->shape.height == 2)
         && (conv_window->shape.width  == 2)
         && (conv_window->stride.vertical   == 2)
         && (conv_window->stride.horizontal == 2);
}


static void calcScaleShift(
    nn_avgpool2d_plan_t* plan,
    const unsigned window_pixels)
{
    const int c = ceil_log2(window_pixels);

    if(c == -1) __builtin_trap(); // window_pixels == 0

    int8_t scale;
    int16_t shift;

    if(window_pixels == (1<<c))
    {
        //window pixel count is already a power of 2   (2^c)
        scale = 1;
        shift = c;
    } else {
        const unsigned q = 31 - c - 6;

        // 2^31 / pix
        const unsigned g = 0x80000000 / window_pixels;
        const unsigned h = (g + (1 << (q-1))) >> q; //Rounding down-shift

        assert(h > (1<<6));
        assert(h < (1<<7));

        scale = (int8_t)h;
        shift = c+6;
    }

    plan->shift = 0x00010001 * shift;
    plan->scale = 0x01010101 * scale;
    
}


void avgpool2d_init(
    nn_avgpool2d_plan_t* plan,
    nn_pool2d_job_t* jobs,
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_window_params_t* window_config,
    const nn_window_op_job_params_t* job_params,
    const unsigned job_count)
{
    assert(x_params->channels == y_params->channels);
    plan->channels = x_params->channels;

    assert(window_config->start.row >= 0);
    assert(window_config->start.column >= 0);

    assert(job_count == 1 || job_params != NULL);
    
    plan->window.rows = window_config->shape.height;
    plan->window.cols = window_config->shape.width;

    if(matches_2x2_impl(window_config)){
        plan->impl = AVGPOOL2D_2X2;
        plan->scale = 0x01010101;
        plan->shift = 0x00020002;
    } else {
        plan->impl = AVGPOOL2D_DEFAULT;
        calcScaleShift(plan, window_config->shape.height * window_config->shape.width);
    }

    const int32_t x_row_bytes = x_params->width * x_params->channels;
    const int32_t y_row_bytes = y_params->width * y_params->channels;

    const int32_t win_start_pix = window_config->start.row * x_params->width + window_config->start.column;

    const mem_stride_t win_hstride_from_prev_start = window_config->stride.horizontal * x_params->channels;
    const mem_stride_t win_vstride_from_prev_start = window_config->stride.vertical * x_row_bytes;

    const nn_window_op_job_params_t full_job = { { 0, 0, 0, }, { y_params->height, y_params->width, y_params->channels } };

    for(int k = 0; k < job_count; k++){

        const nn_window_op_job_params_t* params = (job_params != NULL)? &job_params[k] : &full_job;
        nn_pool2d_job_t* job = &jobs[k];

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
        job->stride.X.cog = VPU_INT8_ACC_PERIOD - params->size.rows * win_vstride_from_prev_start;
        job->stride.Y.cog = VPU_INT8_ACC_PERIOD - params->size.rows * y_row_bytes;
    }
}







void avgpool2d_global_init(
    nn_avgpool2d_global_plan_t* plan,
    nn_avgpool2d_global_job_t* jobs,
    const nn_image_params_t* x_params,
    const nn_avgpool2d_global_job_params_t* job_params,
    const unsigned job_count)
{    
    plan->X.channels = x_params->channels;

    //Figure out the scale and shift
    plan->X.pixels = x_params->height * x_params->width;

    // //Find c = ceil(log2(pix)), which can be achieve via clz()
    // const int c = ceil_log2(plan->X.pixels);

    // assert(c != -1); //pix == 0

    // if(plan->X.pixels == (1<<c)){
    //     //window pixel count is already a power of 2   (2^c)
    //     plan->scale = 1;
    //     plan->shift = c;
    // } else {
    //     const unsigned q = 31 - c - 6;
    //     // 2^31 / plan->X.pixels
    //     const unsigned g = 0x80000000 / plan->X.pixels;
    //     const unsigned h = (g + (1 << (q-1))) >> q; //Rounding down-shift

    //     assert(h > (1<<6));
    //     assert(h < (1<<7));

    //     plan->scale = (int8_t)h;
    //     plan->shift = c+6;
    // }

    
    const nn_avgpool2d_global_job_params_t full_job = { 0, x_params->channels };

    for(int k = 0; k < job_count; k++){

        const nn_avgpool2d_global_job_params_t* params = (job_params != NULL)? &job_params[k] : &full_job;
        nn_avgpool2d_global_job_t* job = &jobs[k];

        assert(params->start_channel >= 0);
        assert((params->start_channel + params->out_channels) <= x_params->channels);

        job->start_stride = params->start_channel;
        job->out_channels = params->out_channels;

    }
}   

