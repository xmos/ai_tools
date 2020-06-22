
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <syscall.h>

#include "meas_common.h"

#include "nn_operator.h"
#include "xs3_vpu.h"


static nn_bso_block_t bso;


static void benchmark_conv2d_deep_case(
    nn_image_params_t* x_params,
    nn_image_params_t* y_params,
    nn_window_params_t* window)
{
    unsigned X_bytes = x_params->height * y_params->width * x_params->channels;
    unsigned Y_bytes = y_params->height * y_params->width * y_params->channels;
    unsigned K_bytes = y_params->channels * window->shape.height * window->shape.width * x_params->channels;
    unsigned BSO_bytes = ((y_params->channels + (VPU_INT8_ACC_PERIOD-1)) / VPU_INT8_ACC_PERIOD) *  sizeof(nn_bso_block_t);

    nn_image_t* X = (nn_image_t*) malloc(X_bytes);
    nn_tensor_t* K = (nn_tensor_t*) malloc(K_bytes);
    nn_image_t* Y = (nn_image_t*) malloc(Y_bytes);
    nn_bso_block_t* BSO = (nn_bso_block_t*) malloc(BSO_bytes);

    assert(X);
    assert(K);
    assert(Y);
    assert(BSO);

    memset(&bso, 0, BSO_bytes);

    nn_conv2d_deep_plan_t plan;
    nn_conv2d_deep_job_t job;

    conv2d_deep_init(&plan, &job, x_params, y_params, NULL, window, 0, 1);

    conv2d_deep(Y, X, K, BSO, &plan, &job);

    free(X);
    free(K);
    free(Y);
    free(BSO);
}


void benchmark_conv2d_deep(int argc, char** argv){
    assert(argc >= 12);

    while(argc >= 12){

        unsigned X_height   = atoi((char*)argv[0]);
        unsigned X_width    = atoi((char*)argv[1]);
        unsigned X_chans    = atoi((char*)argv[2]);

        unsigned Y_height   = atoi((char*)argv[3]);
        unsigned Y_width    = atoi((char*)argv[4]);
        unsigned Y_chans    = atoi((char*)argv[5]);
        
        unsigned K_h        = atoi((char*)argv[6]);
        unsigned K_w        = atoi((char*)argv[7]);
        unsigned start_row  = atoi((char*)argv[8]);
        unsigned start_col  = atoi((char*)argv[9]);
        unsigned vstride    = atoi((char*)argv[10]);
        unsigned hstride    = atoi((char*)argv[11]);

        nn_image_params_t x_params = { X_height, X_width, X_chans };
        nn_image_params_t y_params = { Y_height, Y_width, Y_chans };

        nn_window_params_t window = { { K_h, K_w }, { start_row, start_col }, { vstride, hstride } };

        benchmark_conv2d_deep_case(&x_params, &y_params, &window);

        argc -= 12;
        argv = &(argv[12]);
    }
}