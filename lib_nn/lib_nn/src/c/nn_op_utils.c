

#include "nn_operator.h"
#include "../nn_op_helper.h"

#include "xs3_vpu.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>



#define BSS_INNER_SIZE  (5)
void nn_standard_BSS_layout(
    data16_t* bss_out,
    int32_t* bias,
    int16_t* shift1,
    int16_t* scale,
    int16_t* shift2,
    data16_t* scratch,
    const unsigned C_out)
{
    const unsigned ceil_C_out = (((C_out + (VPU_INT8_ACC_PERIOD - 1)) >> VPU_INT8_ACC_PERIOD_LOG2) << VPU_INT8_ACC_PERIOD_LOG2);

    data16_t* buff = NULL;

    if(((void*)bias) == ((void*)bss_out)){
        //bss_out is being updated in-place. We will need to use a scratch buffer

        if(scratch != NULL){
            //scratch buffer was provided by user
            buff = scratch;
        } else {
            //need to malloc a scratch buffer.
            buff = (data16_t*) malloc(C_out * BSS_INNER_SIZE *  sizeof(data16_t));

            if(buff == NULL){
                printf("Failed to allocate scratch buffer.");
                __builtin_trap();
            }
        }

    } else {
        //bss_out is not being updated in-place, just copy from the inputs to
        //  bss_out.
    }


    if(buff != NULL){
        memcpy(&buff[0], bias, C_out * sizeof(int32_t));
        memcpy(&buff[2*C_out], shift1, C_out*sizeof(data16_t));
        memcpy(&buff[3*C_out], scale, C_out*sizeof(data16_t));
        memcpy(&buff[4*C_out], shift2, C_out*sizeof(data16_t));

        bias = (int32_t*) &buff[0];
        shift1 = (int16_t*) &buff[2*C_out];
        scale = (int16_t*) &buff[3*C_out];
        shift2 = (int16_t*) &buff[4*C_out];
    }

    const unsigned C_out_groups = ceil_C_out >> VPU_INT8_ACC_PERIOD_LOG2;

    for(int cog = 0; cog < C_out_groups; cog++){

        const unsigned cog_offset = VPU_INT8_ACC_PERIOD * BSS_INNER_SIZE * cog;

        for(int coff = 0; coff < VPU_INT8_ACC_PERIOD; coff++){

            const unsigned cout = cog * VPU_INT8_ACC_PERIOD + coff;

            int32_t b      = bias[cout];
            data16_t shr1  = shift1[cout];
            data16_t scl   = scale[cout];
            data16_t shr2  = shift2[cout];

            data16_t b_lo = b & 0xFFFF;
            data16_t b_hi = (b & 0xFFFF0000) >> 16;

            bss_out[cog_offset + 0 * VPU_INT8_ACC_PERIOD + coff] = b_hi;
            bss_out[cog_offset + 1 * VPU_INT8_ACC_PERIOD + coff] = b_lo;
            bss_out[cog_offset + 2 * VPU_INT8_ACC_PERIOD + coff] = shr1;
            bss_out[cog_offset + 3 * VPU_INT8_ACC_PERIOD + coff] = scl;
            bss_out[cog_offset + 4 * VPU_INT8_ACC_PERIOD + coff] = shr2;
            
        }
    }


    if(buff != NULL && scratch == NULL){
        free(buff);
    }
}





static inline int32_t image_vect_addr(
    const nn_image_params_t* params,
    const nn_image_vect_t* vec)
{
    return IMG_ADDRESS_VECT(params, vec->rows, vec->cols, vec->channels);
}




void nn_window_op_config_simple(
    nn_window_op_config_t* config,
    const nn_image_params_t* x,
    const nn_image_params_t* y,
    const unsigned window_height,
    const unsigned window_width,
    const unsigned window_v_stride,
    const unsigned window_h_stride)
{
    memset(config, 0, sizeof(nn_window_op_config_t));

    config->output.shape.channels = y->channels;
    config->window.shape.height = window_height;
    config->window.shape.width  = window_width;

    config->output.stride.vertical.rows = 1;
    config->output.stride.horizontal.cols = 1;

    config->window.inner_stride.horizontal.cols = 1;
    config->window.inner_stride.vertical.rows = 1;

    config->window.outer_stride.vertical.rows = window_v_stride;
    config->window.outer_stride.horizontal.cols = window_h_stride;

    config->output.shape.height = ((x->height - window_height) / window_v_stride) + 1;
    config->output.shape.width = ((x->width - window_width) / window_h_stride) + 1;

}



void nn_window_op_init(
    nn_window_op_plan_t* plan,
    const nn_image_params_t* x,
    const nn_image_params_t* y,
    const nn_window_op_config_t* config,
    const unsigned channels_per_group)
{
    assert(config->output.shape.channels <= x->channels);
    assert(config->output.shape.channels <= y->channels);

    //TODO: Do more error-checking
    plan->output.rows = config->output.shape.height;
    plan->output.cols  = config->output.shape.width;
    plan->output.channels = config->output.shape.channels;

    plan->window.rows = config->window.shape.height;
    plan->window.cols = config->window.shape.width;

    plan->start_stride.x = image_vect_addr(x, &config->window.start);

    plan->inner_stride.horizontal.x = image_vect_addr(x, &config->window.inner_stride.horizontal);
    plan->inner_stride.vertical.x   = image_vect_addr(x, &config->window.inner_stride.vertical)
                                     - config->window.shape.width * image_vect_addr(x, &config->window.inner_stride.horizontal);
    plan->outer_stride.horizontal.x = image_vect_addr(x, &config->window.outer_stride.horizontal)
                                     - config->window.shape.height * image_vect_addr(x, &config->window.inner_stride.vertical);
    plan->outer_stride.vertical.x   = image_vect_addr(x, &config->window.outer_stride.vertical)
                                     - config->output.shape.width * image_vect_addr(x, &config->window.outer_stride.horizontal);
    plan->chan_grp_stride.x         = IMG_ADDRESS_VECT(x, 0, 0, channels_per_group)
                                     - config->output.shape.height * image_vect_addr(x, &config->window.outer_stride.vertical);

    plan->start_stride.y = image_vect_addr(y, &config->output.start);
    plan->outer_stride.horizontal.y = image_vect_addr(y, &config->output.stride.horizontal);
    plan->outer_stride.vertical.y   = image_vect_addr(y, &config->output.stride.vertical)
                                     - config->output.shape.width * image_vect_addr(y, &config->output.stride.horizontal);
    plan->chan_grp_stride.y         = IMG_ADDRESS_VECT(y, 0, 0, channels_per_group)
                                     - config->output.shape.height * image_vect_addr(y, &config->output.stride.vertical);
}


