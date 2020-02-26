

#include "nn_operator.h"
#include "../nn_op_helper.h"

#include "xs3_vpu.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>





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


