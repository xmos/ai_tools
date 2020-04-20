

#include "nn_operator.h"
#include "../nn_op_helper.h"

#include "xs3_vpu.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>






WEAK_FUNC
void avgpool2d_gen(
    int8_t* Y,
    const int8_t* X, 
    const nn_avgpool2d_plan_t* plan)
{
    X = &X[plan->window.start_stride.x];
    Y = &Y[plan->window.start_stride.y];

    // int8_t* Y_start = Y;

    const int8_t shift = plan->shift & 0xFFFF;
    const int8_t scale = plan->scale & 0xFF;

    const unsigned channel_groups = plan->window.output.channels >> VPU_INT8_ACC_PERIOD_LOG2;

    for(unsigned chn_grp = 0; chn_grp <= channel_groups; chn_grp++){

        unsigned iter_chans = VPU_INT8_ACC_PERIOD;
        if(chn_grp == channel_groups)
            iter_chans = plan->window.output.channels - (channel_groups << VPU_INT8_ACC_PERIOD_LOG2);

        if(iter_chans == 0)
            break;

        for(unsigned out_row = 0; out_row < plan->window.output.rows; out_row++){
            for(unsigned out_col = 0; out_col < plan->window.output.cols; out_col++){
                    
                int32_t acc32[VPU_INT8_ACC_PERIOD] = {0};

                for(unsigned w_rows = 0; w_rows < plan->window.window.rows; w_rows++){
                    for(unsigned w_cols = 0; w_cols < plan->window.window.cols; w_cols++){

                        for(unsigned k = 0; k < iter_chans; k++){
                            acc32[k] += (((int32_t)X[k]) * scale);
                        }

                        X = &X[plan->window.inner_stride.horizontal.x];
                    }

                    X = &X[plan->window.inner_stride.vertical.x];
                }

                for(unsigned k = 0; k < iter_chans; k++){
                    Y[k] = vlsat_single_s8(acc32[k], shift);
                }

                X = &X[plan->window.outer_stride.horizontal.x];
                Y = &Y[plan->window.outer_stride.horizontal.y];
            }

            X = &X[plan->window.outer_stride.vertical.x];
            Y = &Y[plan->window.outer_stride.vertical.y];
        }

        X = &X[plan->window.chan_grp_stride.x];
        Y = &Y[plan->window.chan_grp_stride.y];
    }
}

WEAK_FUNC
void avgpool2d_2x2(
    int8_t* Y,
    const int8_t* X, 
    const nn_avgpool2d_plan_t* plan)
{
    avgpool2d(Y, X, plan);
}




WEAK_FUNC
void avgpool2d_global(
    nn_image_t* Y,
    const nn_image_t* X, 
    const uint32_t x_height, 
    const uint32_t x_width,
    const channel_count_t x_chans,
    const int32_t  bias,
    const uint32_t shift,
    const uint32_t scale)
{
    const unsigned pix = x_height * x_width;

    const uint32_t sh = shift;
    const uint32_t sc = scale;
    
    for(unsigned ch = 0; ch < x_chans; ch++){

        int32_t acc = bias;

        for(unsigned p = 0; p < pix; p++){
            int32_t x = X[p*x_chans + ch];
            acc += x * sc;
        }

        Y[ch] = vlsat_single_s8(acc, sh);
    }
}



static inline int matches_2x2_impl(
    const nn_window_op_config_t* config)
{   
    return  (config->window.shape.height == 2)
         && (config->window.shape.width  == 2)
         && (config->window.outer_stride.horizontal.rows == 0)
         && (config->window.outer_stride.horizontal.cols == 2)
         && (config->window.outer_stride.horizontal.channels == 0)
         && (config->window.outer_stride.vertical.rows == 2)
         && (config->window.outer_stride.vertical.cols == 0)
         && (config->window.outer_stride.vertical.channels == 0)
         && (config->window.inner_stride.horizontal.rows == 0)
         && (config->window.inner_stride.horizontal.cols == 1)
         && (config->window.inner_stride.horizontal.channels == 0)
         && (config->window.inner_stride.vertical.rows == 1)
         && (config->window.inner_stride.vertical.cols == 0)
         && (config->window.inner_stride.vertical.channels == 0);
}




void avgpool2d_init(
    nn_avgpool2d_plan_t* plan,
    const nn_image_params_t* x,
    const nn_image_params_t* y,
    const nn_window_op_config_t* config)
{
    //Initialize the stride plan
    nn_window_op_init(&plan->window, x, y, config, VPU_INT8_ACC_PERIOD);

    //Figure out the scale and shift

    const unsigned pix = config->window.shape.height * config->window.shape.width;
    //Find c = ceil(log2(pix)), which can be achieve via clz()
    const int c = ceil_log2(pix);

    if(c == -1) __builtin_trap(); //pix == 0

    int8_t scale;
    int16_t shift;
    if(pix == (1<<c)){
        //window pixel count is already a power of 2   (2^c)
        scale = 1;
        shift = c;
        // printf("scale: %d\nshift: %d\ncl2: %d\npix: %u\n", scale, shift, ceil_log2(pix), pix);
        // printf("win_h: %u\nwin_w:%u\n", win->window.height, win->window.width);
    } else {
        const unsigned q = 31 - c - 6;
        // 2^31 / pix
        const unsigned g = 0x80000000 / pix;
        const unsigned h = (g + (1 << (q-1))) >> q; //Rounding down-shift

        assert(h > (1<<6));
        assert(h < (1<<7));

        scale = (int8_t)h;
        shift = c+6;
    }

    plan->shift = 0x00010001 * shift;
    plan->scale = 0x01010101 * scale;
    
    //Decide on an implementation (if assembly is being used at all)
    plan->impl = AVGPOOL2D_DEFAULT;
    
    if(matches_2x2_impl(config)){
        plan->impl = AVGPOOL2D_2X2;
    }
}











void avgpool2d_2x2_init(
    nn_avgpool2d_plan_t* plan,
    const nn_image_params_t* x,
    const nn_image_params_t* y,
    const nn_image_vect_t* x_start,
    const nn_image_vect_t* y_start,
    const unsigned out_rows,
    const unsigned out_cols,
    const unsigned out_chans)
{
    nn_window_op_config_t config;
    nn_window_op_config_simple(&config, x, y, 2, 2, 2, 2);

    config.window.start = *x_start;
    config.output.start = *y_start;

    config.output.shape.height = out_rows;
    config.output.shape.width = out_cols;
    config.output.shape.channels = out_chans;

    //Initialize the stride plan
    nn_window_op_init(&plan->window, x, y, &config, VPU_INT8_ACC_PERIOD);

    plan->shift = 0x00010001 * 2;
    plan->scale = 0x01010101 * 1;

    plan->impl = AVGPOOL2D_2X2;
}












void avgpool2d_global_init(
    uint32_t* shift,
    uint32_t* scale,
    const uint32_t x_height,
    const uint32_t x_width)
{    
    //Figure out the scale and shift
    const unsigned pix = x_height * x_width;
    //Find c = ceil(log2(pix)), which can be achieve via clz()
    const int c = ceil_log2(pix);

    if(c == -1) __builtin_trap(); //pix == 0

    if(pix == (1<<c)){
        //window pixel count is already a power of 2   (2^c)
        *scale = 1;
        *shift = c;
        // printf("scale: %d\nshift: %d\ncl2: %d\npix: %u\n", scale, shift, ceil_log2(pix), pix);
        // printf("win_h: %u\nwin_w:%u\n", win->window.height, win->window.width);
    } else {
        const unsigned q = 31 - c - 6;
        // 2^31 / pix
        const unsigned g = 0x80000000 / pix;
        const unsigned h = (g + (1 << (q-1))) >> q; //Rounding down-shift

        // printf("! pix: %u\n", pix);
        // printf("! c: %d\n", c);
        // printf("! q: %u\n", q);
        // printf("! g: 0x%08X\n", g);
        // printf("! h: 0x%02X\n", h);
        assert(h > (1<<6));
        assert(h < (1<<7));

        *scale = (int8_t)h;
        *shift = c+6;
    }

    // (*scale) *= 0x01010101;
    // (*shift) *= 0x00010001;
}   

