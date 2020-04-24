
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


static void benchmark_nn_conv2d_hstrip_deep_case(unsigned K_h, unsigned K_w, unsigned C_in, unsigned out_cols)
{
    const unsigned X_width = out_cols + K_w - 1;

    unsigned X_bytes = C_in * K_h * X_width;
    unsigned Y_bytes = VPU_INT8_ACC_PERIOD * out_cols;
    unsigned K_bytes = VPU_INT8_ACC_PERIOD * C_in * K_h * K_w;

    int8_t* X = (int8_t*) malloc(X_bytes);
    int8_t* K = (int8_t*) malloc(K_bytes);
    int8_t* Y = (int8_t*) malloc(Y_bytes);

    assert(X);
    assert(K);
    assert(Y);

    memset(&bso, 0, sizeof(bso));

    int8_t* K_start = &K[(VPU_INT8_ACC_PERIOD-1)*K_h*K_w*C_in];

    nn_conv2d_hstrip_deep(Y, X, K_start, &bso, K_h, K_w, 1, C_in, 
                        (X_width-K_w)*C_in, -C_in*K_w*K_h, 
                        VPU_INT8_ACC_PERIOD, out_cols);

    free(X);
    free(K);
    free(Y);
}


void benchmark_nn_conv2d_hstrip_deep(int argc, char** argv){
    assert(argc >= 4);

    while(argc >= 4){
        
        unsigned K_h      = atoi((char*)argv[0]);
        unsigned K_w      = atoi((char*)argv[1]);
        unsigned C_in     = atoi((char*)argv[2]);
        unsigned out_cols = atoi((char*)argv[3]);
        benchmark_nn_conv2d_hstrip_deep_case(K_h, K_w, C_in, out_cols);

        argc -= 4;
        argv = &(argv[4]);
    }
}