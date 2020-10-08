

#include "nn_operator.h"
#include "../nn_op_helper.h"

#include "xs3_vpu.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>



#ifndef AVGPOOL2D_INIT_ERROR_DETECTION_ENABLE
  #define AVGPOOL2D_INIT_ERROR_DETECTION_ENABLE     (1)
#endif


#if CONFIG_SYMMETRIC_SATURATION_avgpool2d
  #define NEG_SAT_VAL   (-127)
#else
  #define NEG_SAT_VAL   (-128)
#endif 




typedef struct {


    struct {
        struct {
            mem_stride_t row;
            mem_stride_t cog;
        } X;

        struct {
            mem_stride_t row;
            mem_stride_t col;
        } window;

        struct {
            mem_stride_t row;
            mem_stride_t cog;
        } Y;

    } stride;

    int32_t scale;
    uint32_t shift;

} nn_avgpool2d_job_t;




static inline int matches_2x2_impl(
    const nn_window_params_t* pooling_window)
{   
    return  (pooling_window->shape.height == 2)
         && (pooling_window->shape.width  == 2)
         && (pooling_window->stride.vertical   == 2)
         && (pooling_window->stride.horizontal == 2);
}


static void avgpool2d_calc_scale_shift(
    nn_avgpool2d_job_t* job,
    const nn_window_params_t* pooling_window)
{
    const unsigned window_pixels = pooling_window->shape.height * pooling_window->shape.width;
    const int c = ceil_log2(window_pixels);

    if(c == -1) __builtin_trap(); // window_pixels == 0

    int8_t scale;
    uint16_t shift;

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

    job->scale = 0x01010101 * ((int32_t)scale);
    job->shift = 0x00010001 * ((int32_t)shift);
    
}


static void avgpool2d_adjust_starts(
    int8_t** Y,
    const int8_t** X,
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_window_params_t* pooling_window,
    const nn_window_op_job_params_t* job_params,
    const nn_avgpool2d_flags_e flags)
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

static void avgpool2d_prepare(
    nn_avgpool2d_job_t* job,
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_window_params_t* pooling_window,
    const nn_window_op_job_params_t* job_params)
{
    if ( AVGPOOL2D_INIT_ERROR_DETECTION_ENABLE ){
        assert(x_params->channels == y_params->channels);

        assert(pooling_window->start.row >= 0);
        assert(pooling_window->start.column >= 0);

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

    const int32_t x_row_bytes = x_params->width * x_params->channels;
    const int32_t y_row_bytes = y_params->width * y_params->channels;

    const int32_t win_start_pix = pooling_window->start.row * x_params->width + pooling_window->start.column;

    const mem_stride_t win_hstride_from_prev_start = pooling_window->stride.horizontal * x_params->channels;
    const mem_stride_t win_vstride_from_prev_start = pooling_window->stride.vertical * x_row_bytes;

    job->stride.X.row = x_row_bytes - pooling_window->shape.width * x_params->channels;
    job->stride.Y.row = y_row_bytes - job_params->size.cols * y_params->channels;

    // The X pointer will be pointing at the start of the first row *not* inside the window 
    // when doing an hstride
    job->stride.window.col = win_hstride_from_prev_start - pooling_window->shape.height * x_row_bytes;

    // The X pointer will be pointing at the start of the first patch *after* the job's output 
    // columns when doing a vstride.
    job->stride.window.row = win_vstride_from_prev_start - win_hstride_from_prev_start * job_params->size.cols;

    // For a channel group stride, move back to start of job and add 32
    job->stride.X.cog = VPU_INT8_ACC_PERIOD - job_params->size.rows * win_vstride_from_prev_start;
    job->stride.Y.cog = VPU_INT8_ACC_PERIOD - job_params->size.rows * y_row_bytes;

}



WEAK_FUNC
void avgpool2d_gen(
    int8_t* Y,
    const int8_t* X, 
    const channel_count_t image_chans,
    const nn_window_params_t* pooling_window,
    const nn_window_op_job_params_t* job_params,
    const nn_avgpool2d_flags_e flags,
    const nn_avgpool2d_job_t* job)
{
    const unsigned channel_groups = job_params->size.channels >> VPU_INT8_ACC_PERIOD_LOG2;

    int32_t scale = job->scale & 0xFF;
    uint16_t shift = job->shift & 0xFFFF;

    for(unsigned chn_grp = 0; chn_grp <= channel_groups; chn_grp++){

        unsigned iter_chans = VPU_INT8_ACC_PERIOD;
        if(chn_grp == channel_groups)
            iter_chans = job_params->size.channels - (channel_groups << VPU_INT8_ACC_PERIOD_LOG2);

        if(iter_chans == 0)
            break;

        for(unsigned out_row = 0; out_row < job_params->size.rows; out_row++){
            for(unsigned out_col = 0; out_col < job_params->size.cols; out_col++){
                    
                int32_t acc32[VPU_INT8_ACC_PERIOD] = {0};

                for(unsigned w_rows = 0; w_rows < pooling_window->shape.height; w_rows++){
                    for(unsigned w_cols = 0; w_cols < pooling_window->shape.width; w_cols++){

                        for(unsigned k = 0; k < iter_chans; k++){
                            acc32[k] += (((int32_t)X[k]) * scale);
                        }

                        X = ADDR(X, image_chans);
                    }

                    X = ADDR(X, job->stride.X.row);
                }

                for(unsigned k = 0; k < iter_chans; k++){
                    Y[k] = vlsat_single_s8(acc32[k], shift, NEG_SAT_VAL, VPU_INT8_MAX);
                }

                X = ADDR(X, job->stride.window.col);
                Y = ADDR(Y, image_chans);
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
    const channel_count_t image_chans,
    const nn_window_params_t* pooling_window,
    const nn_window_op_job_params_t* job_params,
    const nn_avgpool2d_flags_e flags,
    const nn_avgpool2d_job_t* job)
{
    avgpool2d_gen(Y, X, image_chans, pooling_window, job_params, flags, job);
}







void avgpool2d(
    int8_t* Y,
    const int8_t* X, 
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_window_params_t* pooling_window)
{
    nn_window_op_job_params_t full_job = {{0,0,0},{y_params->height, y_params->width, x_params->channels}};

    avgpool2d_ext(Y, X, x_params, y_params, pooling_window, &full_job, AVGPOOL2D_FLAG_NONE);
}

void avgpool2d_ext(
    int8_t* Y,
    const int8_t* X, 
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_window_params_t* pooling_window,
    const nn_window_op_job_params_t* job_params,
    const nn_avgpool2d_flags_e flags)
{
    avgpool2d_adjust_starts(&Y, &X, x_params, y_params, pooling_window, job_params, flags);

    nn_avgpool2d_job_t job;
    avgpool2d_prepare(&job, x_params, y_params, pooling_window, job_params);

    if(  matches_2x2_impl(pooling_window) ){
        avgpool2d_2x2(Y, X, x_params->channels, pooling_window, job_params, flags, &job);
    } else {
        avgpool2d_calc_scale_shift(&job, pooling_window);
        avgpool2d_gen(Y, X, x_params->channels, pooling_window, job_params, flags, &job);
    }
}

#undef NEG_SAT_VAL













#if CONFIG_SYMMETRIC_SATURATION_avgpool2d_global
  #define NEG_SAT_VAL   (-127)
#else
  #define NEG_SAT_VAL   (-128)
#endif 

void avgpool2d_global(
    nn_image_t* Y,
    const nn_image_t* X, 
    const int32_t bias,
    const int8_t scale,
    const uint16_t shift,
    const nn_image_params_t* x_params)
{
    avgpool2d_global_ext(Y, X, bias, scale, shift, x_params, 
                        0, x_params->channels, AVGPOOL2D_GLOBAL_FLAG_NONE);
}

WEAK_FUNC
void avgpool2d_global_ext(
    nn_image_t* Y,
    const nn_image_t* X, 
    const int32_t bias,
    const int8_t scale,
    const uint16_t shift,
    const nn_image_params_t* x_params,
    const unsigned chan_start,
    const unsigned chan_count,
    const nn_avgpool2d_global_flags_e flags)
{
    Y = ADDR(Y, chan_start);
    X = ADDR(X, chan_start);

    const unsigned pix = x_params->height * x_params->width;
    
    for(unsigned ch = 0; ch < chan_count; ch++){

        int32_t acc = bias;

        for(unsigned p = 0; p < pix; p++){
            int32_t x = X[p * x_params->channels + ch];
            acc += x * scale;
        }

        
        Y[ch] = vlsat_single_s8(acc, shift, NEG_SAT_VAL, VPU_INT8_MAX);
    }
}

#undef NEG_SAT_VAL



