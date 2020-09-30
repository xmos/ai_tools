
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <syscall.h>

#include "meas_common.h"

#include "nn_operator.h"

// void benchmark_avgpool2d_case(
//     const nn_image_params_t* x,
//     const nn_image_params_t* y,
//     const nn_window_op_config_t* config)
// {

//     int8_t* X = (int8_t*) malloc(x->height * x->width * x->channels);
//     int8_t* Y = (int8_t*) malloc(y->height * y->width * y->channels);
    
//     assert(X);
//     assert(Y);

//     nn_avgpool2d_plan_t plan;

//     avgpool2d_init(&plan, x, y, config);

//     avgpool2d(Y, X, &plan);

//     free(X);
//     free(Y);
// }


#define REQ_ARGS    (7)
void benchmark_avgpool2d(int argc, char** argv){
/*
    nn_window_op_config_t config;
    nn_image_params_t x, y;
    
    memset(&config, 0, sizeof(config));

    config.output.stride.vertical.rows = 1;
    config.output.stride.horizontal.cols = 1;
    config.window.inner_stride.vertical.rows = 1;
    config.window.inner_stride.horizontal.cols = 1;

    while(argc >= REQ_ARGS){

        int i = 0;

        config.output.shape.height = atoi((char*)argv[i++]);
        config.output.shape.width = atoi((char*)argv[i++]);
        config.output.shape.channels = atoi((char*)argv[i++]);
        config.window.shape.height = atoi((char*)argv[i++]);
        config.window.shape.width = atoi((char*)argv[i++]);
        config.window.outer_stride.vertical.rows = atoi((char*)argv[i++]);
        config.window.outer_stride.horizontal.cols = atoi((char*)argv[i++]);
        
        x.height = config.output.shape.height * config.window.outer_stride.vertical.rows 
                    + config.window.shape.height ;
        x.width = config.output.shape.width * config.window.outer_stride.horizontal.cols 
                    + config.window.shape.width ;
        x.channels = config.output.shape.channels;
        y.height = config.output.shape.height;
        y.width = config.output.shape.width;
        y.channels = config.output.shape.channels;


        benchmark_avgpool2d_case(&x, &y, &config);

        argc -= REQ_ARGS;
        argv = &(argv[REQ_ARGS]);
    }

*/
}