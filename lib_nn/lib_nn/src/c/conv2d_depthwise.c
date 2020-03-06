
#include "nn_operator.h"
#include "../nn_op_helper.h"
#include "nn_op_structs.h"

#include "xs3_vpu.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

#define ADDR(VR, STR)   printf("!\t%s = 0x%08X\t\t(%s)\n", (#VR), (unsigned) (VR), (STR))

void vlmacc8(
    int32_t* acc,
    const int8_t* X,
    const int8_t* W)
{
    for(int k = 0; k < VPU_INT8_VLMACC_ELMS; k++){
        // printf("!@ %d\t%d\t%d\n", k, X[k], W[k]);
        acc[k] += ((int32_t)X[k]) * W[k];
    }
}




void conv2d_depthwise_init(
    nn_conv2d_depthwise_plan_t* plan,
    nn_conv2d_depthwise_job_t* jobs,
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_conv2d_job_params_t* job_params,
    const unsigned kernel_start_row,
    const unsigned kernel_start_col,
    const unsigned K_h,
    const unsigned K_w,
    const unsigned v_stride,
    const unsigned h_stride,
    const int8_t zero_point,
    const unsigned job_count)
{
    assert(x_params->channels % 4 == 0);
    assert(x_params->channels == y_params->channels);
    assert(job_count > 0);
    assert(job_count == 1 || job_params != NULL);

    const unsigned x_row_bytes = x_params->width * x_params->channels;
    const unsigned y_row_bytes = y_params->width * y_params->channels;
    const unsigned k_row_bytes = K_w * y_params->channels;

    const int32_t window_start_offset = kernel_start_row * x_row_bytes + x_params->channels * kernel_start_col;

    plan->stride.X.inner.col = x_params->channels;
    plan->stride.X.inner.row = x_row_bytes - x_params->channels * K_w;
    plan->stride.X.outer.col = x_params->channels * h_stride - x_row_bytes * K_h;

    plan->stride.Y.col = y_params->channels;

    plan->stride.K.chan_group = VPU_INT8_VLMACC_ELMS - (K_h * K_w * y_params->channels);

    plan->zero_point = zero_point;

    plan->kernel.height  = K_h;
    plan->kernel.width   = K_w;
    plan->kernel.vstride = v_stride;
    plan->kernel.hstride = h_stride;
    
    const int32_t init_padding_top    = -kernel_start_row;
    const int32_t init_padding_bottom =  kernel_start_row + K_h - x_params->height;
    const int32_t init_padding_left   = -kernel_start_col;
    const int32_t init_padding_right  =  kernel_start_col + K_w  - x_params->width;

    nn_conv2d_job_params_t full_job = {{0,0,0}, y_params->height, y_params->width, y_params->channels };

    for(int i = 0; i < job_count; i++){
        const nn_conv2d_job_params_t* params = (job_params != NULL)? &job_params[i] : &full_job;
        nn_conv2d_depthwise_job_t* job = &jobs[i];
        
        assert(params->y_start.rows >= 0 && params->y_start.cols >= 0 && params->y_start.channels >= 0);
        assert(params->y_start.rows + params->out_rows <= y_params->height);
        assert(params->y_start.cols + params->out_cols <= y_params->width);
        assert(params->y_start.channels + params->out_channels <= y_params->channels);

        job->output.rows = params->out_rows;
        job->output.cols = params->out_cols;
        job->output.channels = params->out_channels;

        job->init_padding.top    = init_padding_top    - params->y_start.rows * plan->kernel.vstride;
        job->init_padding.left   = init_padding_left   - params->y_start.cols * plan->kernel.hstride;
        job->init_padding.bottom = init_padding_bottom + params->y_start.rows * plan->kernel.vstride;
        job->init_padding.right  = init_padding_right  + params->y_start.cols * plan->kernel.hstride;

        const int32_t end_padding_top    = init_padding_top    - ((params->y_start.rows + params->out_rows - 1) * plan->kernel.vstride);
        const int32_t end_padding_left   = init_padding_left   - ((params->y_start.cols + params->out_cols - 1) * plan->kernel.hstride);
        const int32_t end_padding_bottom = init_padding_bottom + ((params->y_start.rows + params->out_rows - 1) * plan->kernel.vstride);
        const int32_t end_padding_right  = init_padding_right  + ((params->y_start.cols + params->out_cols - 1) * plan->kernel.hstride);

        job->init_padding.unpadded = (job->init_padding.top <= 0 && job->init_padding.left <= 0
                                    && job->init_padding.bottom <= 0 && job->init_padding.right <= 0
                                    && end_padding_top <= 0 && end_padding_left <= 0
                                    && end_padding_bottom <= 0 && end_padding_right <= 0);

        // printf("job->init_padding.unpadded = %u\n", job->init_padding.unpadded);
        // printf("job->init_padding.top = %ld\n", job->init_padding.top);
        // printf("job->init_padding.left = %ld\n", job->init_padding.left);
        // printf("job->init_padding.bottom = %ld\n", job->init_padding.bottom);
        // printf("job->init_padding.right = %ld\n\n", job->init_padding.right);
        
        // printf("end_padding_top = %ld\n", end_padding_top);
        // printf("end_padding_left = %ld\n", end_padding_left);
        // printf("end_padding_bottom = %ld\n", end_padding_bottom);
        // printf("end_padding_right = %ld\n", end_padding_right);

        job->stride.BSS.start = (params->y_start.channels / VPU_INT8_ACC_PERIOD) * sizeof(nn_bss_block_t);
        job->stride.K.start   = params->y_start.channels;
        job->stride.Y.start   = params->y_start.rows * y_row_bytes + y_params->channels * params->y_start.cols + params->y_start.channels;

        job->stride.X.start = window_start_offset 
                            + params->y_start.rows * plan->kernel.vstride * x_row_bytes
                            + params->y_start.cols * plan->kernel.hstride * x_params->channels
                            + params->y_start.channels;

        job->stride.X.outer.row = x_row_bytes * plan->kernel.vstride - plan->kernel.hstride * x_params->channels * job->output.cols;
        job->stride.X.chan_group = VPU_INT8_VLMACC_ELMS - x_row_bytes * plan->kernel.vstride * job->output.rows;

        job->stride.Y.row = y_row_bytes - job->output.cols * y_params->channels;
        job->stride.Y.chan_group = VPU_INT8_VLMACC_ELMS - y_row_bytes * job->output.rows;
    }
}

void conv2d_depthwise_c(
    int8_t* Y,
    const int8_t* X,
    const int8_t* K,
    const nn_bss_block_t* BSS,
    const nn_conv2d_depthwise_plan_t* plan,
    const nn_conv2d_depthwise_job_t* job)
{
    // ADDR(X, "initial");
    // ADDR(Y, "initial");
    // ADDR(K, "initial");
    const int8_t* K_initial = K;

    int8_t zero_point_vec[VPU_INT8_VLMACC_ELMS];
    memset(zero_point_vec, plan->zero_point, sizeof(zero_point_vec));

    X = &X[job->stride.X.start];
    Y = &Y[job->stride.Y.start];
    K = &K[job->stride.K.start];
    BSS = (nn_bss_block_t*) &(((int8_t*)BSS)[job->stride.BSS.start]);

    // ADDR(X, "start strided");
    // ADDR(Y, "start strided");
    // ADDR(K, "start strided");

    for(int out_chan = 0; out_chan < job->output.channels; out_chan += VPU_INT8_VLMACC_ELMS){

        const unsigned cur_chans = (job->output.channels - out_chan >= VPU_INT8_VLMACC_ELMS)? VPU_INT8_VLMACC_ELMS : job->output.channels - out_chan; 

        int pad_t = job->init_padding.top;
        int pad_b = job->init_padding.bottom;

        for(int out_row = 0; out_row < job->output.rows; out_row++){
            // ADDR(X, "out row start");
            // ADDR(Y, "out row start");
            // ADDR(K, "out row start");

            int pad_l = job->init_padding.left;
            int pad_r = job->init_padding.right;

            const int cur_pad_t = (pad_t > 0)? pad_t : 0;
            const int cur_pad_b = (pad_b > 0)? pad_b : 0;

            for(int out_col = 0; out_col < job->output.cols; out_col++){

                // ADDR(X, "out col start");
                // ADDR(Y, "out col start");
                // ADDR(K, "out col start");

                const int cur_pad_l = (pad_l > 0)? pad_l : 0;
                const int cur_pad_r = (pad_r > 0)? pad_r : 0;

                // printf("pads:  (%d, %d, %d, %d)\n", pad_t, pad_l, pad_l, pad_r);
                // printf("cur_pads:  (%d, %d, %d, %d)\n", cur_pad_t, cur_pad_l, cur_pad_l, cur_pad_r);
                // printf("@\t&X = 0x%08X\t\t&Y = 0x%08X\t\t&K = 0x%08X\n", (unsigned) X, (unsigned) Y, (unsigned) K);
                
                const int8_t* cur_K = K;
                const nn_bss_block_t* cur_BSS = BSS;

                int32_t accs[VPU_INT8_VLMACC_ELMS];

                for(int k = 0; k < VPU_INT8_VLMACC_ELMS; k++)
                    accs[k] = ((int32_t)cur_BSS->bias_hi[k]) << VPU_INT8_ACC_VR_BITS;
                
                for(int k = 0; k < VPU_INT8_VLMACC_ELMS; k++)
                    accs[k] |= cur_BSS->bias_lo[k];
                
                //THIS LOOP IS IN PADDING (above image)
                for(int i = cur_pad_t; i > 0; i--){
                    // printf("PAD_T??\t%d\t%d\n", cur_pad_t, i);
                    for(int j = plan->kernel.width; j > 0; j--){
                        vlmacc8(accs, zero_point_vec, cur_K);
                        X = &X[plan->stride.X.inner.col];
                        cur_K = &cur_K[plan->stride.Y.col];
                    }
                    X = &X[plan->stride.X.inner.row];
                }

                // These rows are inside image (vertically)
                for(int i = plan->kernel.height - (cur_pad_t + cur_pad_b); i > 0; i--){

                    //THIS LOOP IS IN PADDING (left of image)
                    for(int j = cur_pad_l; j > 0; j--){
                        // printf("PAD_L??\t%d\t%d\n", cur_pad_l, j);
                        vlmacc8(accs, zero_point_vec, cur_K);
                        X = &X[plan->stride.X.inner.col];
                        cur_K = &cur_K[plan->stride.Y.col];
                    }

                    for(int j = plan->kernel.width - (cur_pad_l + cur_pad_r); j > 0; j--){
                        vlmacc8(accs, X, cur_K);
                        X = &X[plan->stride.X.inner.col];
                        cur_K = &cur_K[plan->stride.Y.col];
                    }

                    //THIS LOOP IS IN PADDING (right of image)
                    for(int j = cur_pad_r; j > 0; j--){
                        // printf("PAD_R??\t%d\t%d\n", cur_pad_r, j);
                        vlmacc8(accs, zero_point_vec, cur_K);
                        X = &X[plan->stride.X.inner.col];
                        cur_K = &cur_K[plan->stride.Y.col];
                    }

                    X = &X[plan->stride.X.inner.row];
                }
                
                //THIS LOOP IS IN PADDING (below image)
                for(int i = cur_pad_b; i > 0; i--){
                    // printf("PAD_B??\t%d\t%d\n", cur_pad_b, i);
                    for(int j = plan->kernel.width; j > 0; j--){
                        // ADDR(cur_K - K_initial, "pad_b");
                        // ADDR(cur_K, "pad_b");
                        vlmacc8(accs, zero_point_vec, cur_K);
                        X = &X[plan->stride.X.inner.col];
                        cur_K = &cur_K[plan->stride.Y.col];
                    }
                    X = &X[plan->stride.X.inner.row];
                }

                // printf("&Y[0] = 0x%08X\taccs[0] = %ld\n", (unsigned) &Y[0], accs[0]);

                for(int k = 0; k < cur_chans; k++){
                    int16_t shift1  = cur_BSS->shift1[k];
                    int16_t scale   = cur_BSS->scale[k];
                    int16_t shift2  = cur_BSS->shift2[k];
                    accs[k] = vlsat_single_s16(accs[k], shift1);
                    accs[k] = accs[k] * scale;
                    accs[k] = vlsat_single_s8(accs[k], shift2);
                    Y[k] = (int8_t) accs[k];
                }
                // printf("@\t&X = 0x%08X\t\t&Y = 0x%08X\t\t&K = 0x%08X\n", (unsigned) X, (unsigned) Y, (unsigned) K);
                
                pad_l -= (int) plan->kernel.hstride;
                pad_r += (int) plan->kernel.hstride;
                
                X = &X[plan->stride.X.outer.col];
                Y = &Y[plan->stride.Y.col];
            }

            pad_t -= plan->kernel.vstride;
            pad_b += plan->kernel.vstride;

            X = &X[job->stride.X.outer.row];
            Y = &Y[job->stride.Y.row];
        }

        X = &X[job->stride.X.chan_group];
        Y = &Y[job->stride.Y.chan_group];

        BSS = &BSS[1];
        K = &K[VPU_INT8_VLMACC_ELMS];
    }

}

void nn_compute_hstrip_depthwise_padded_asm(
    int8_t* Y,
    const int8_t* X, 
    const int8_t* K,
    const nn_bss_block_t* BSS,
    const unsigned K_h,
    const unsigned K_w,
    const unsigned pad_t,
    const unsigned pad_l_initial,
    const unsigned pad_b,
    const unsigned pad_r_initial,
    const int32_t xk_col_stride,
    const int32_t x_row_stride,
    const int32_t window_hstride,
    const int32_t y_col_stride,
    const unsigned out_cols,
    const unsigned chans_to_write,
    const int8_t* zero_point_vec);

void nn_compute_hstrip_depthwise_asm(
    int8_t* Y,
    const int8_t* X, 
    const int8_t* K,
    const nn_bss_block_t* BSS,
    const unsigned K_h,
    const unsigned K_w,
    const int32_t xk_col_stride,
    const int32_t x_row_stride,
    const int32_t window_hstride,
    const int32_t y_col_stride,
    const unsigned out_cols,
    const unsigned chans_to_write);

void conv2d_depthwise_asm(
    int8_t* Y,
    const int8_t* X,
    const int8_t* K,
    const nn_bss_block_t* BSS,
    const nn_conv2d_depthwise_plan_t* plan,
    const nn_conv2d_depthwise_job_t* job)
{
    // ADDR(X, "initial");
    // ADDR(Y, "initial");
    // ADDR(K, "initial");

    int8_t zero_point_vec[VPU_INT8_VLMACC_ELMS];
    memset(zero_point_vec, plan->zero_point, sizeof(zero_point_vec));

    X = &X[job->stride.X.start];
    Y = &Y[job->stride.Y.start];
    K = &K[job->stride.K.start];
    BSS = (nn_bss_block_t*) &(((int8_t*)BSS)[job->stride.BSS.start]);

    // ADDR(X, "start strided");
    // ADDR(Y, "start strided");
    // ADDR(K, "start strided");

    for(int out_chan = 0; out_chan < job->output.channels; out_chan += VPU_INT8_VLMACC_ELMS){

        const unsigned cur_chans = (job->output.channels - out_chan >= VPU_INT8_VLMACC_ELMS)? VPU_INT8_VLMACC_ELMS : job->output.channels - out_chan; 

        int pad_t = job->init_padding.top;
        int pad_b = job->init_padding.bottom;

        for(int out_row = 0; out_row < job->output.rows; out_row++){
            // ADDR(X, "out row start");
            // ADDR(Y, "out row start");
            // ADDR(K, "out row start");

            int pad_l = job->init_padding.left;
            int pad_r = job->init_padding.right;

            const int cur_pad_t = (pad_t > 0)? pad_t : 0;
            const int cur_pad_b = (pad_b > 0)? pad_b : 0;
            
            if(job->init_padding.unpadded){
                nn_compute_hstrip_depthwise_asm(Y, X, K, BSS, plan->kernel.height, plan->kernel.width,
                        plan->stride.X.inner.col, plan->stride.X.inner.row,
                        plan->kernel.hstride * plan->stride.X.inner.col, plan->stride.Y.col, job->output.cols, cur_chans);
            } else {
                nn_compute_hstrip_depthwise_padded_asm(Y, X, K, BSS, plan->kernel.height, plan->kernel.width,
                            cur_pad_t, job->init_padding.left, cur_pad_b, job->init_padding.right,
                            plan->stride.X.inner.col, plan->stride.X.inner.row, 
                            plan->kernel.hstride * plan->stride.X.inner.col, plan->stride.Y.col, job->output.cols, 
                            cur_chans, zero_point_vec);
            }

            pad_t -= plan->kernel.vstride;
            pad_b += plan->kernel.vstride;
            // printf("!! %ld\n", (int32_t) plan->stride.X.inner.col * plan->kernel.hstride * job->output.cols + job->stride.X.outer.row);
            X = &X[plan->stride.X.inner.col * plan->kernel.hstride * job->output.cols + job->stride.X.outer.row];
            Y = &Y[plan->stride.Y.col * job->output.cols + job->stride.Y.row];
            
        }

        X = &X[job->stride.X.chan_group];
        Y = &Y[job->stride.Y.chan_group];

        BSS = &BSS[1];
        K = &K[VPU_INT8_VLMACC_ELMS];
    }

}