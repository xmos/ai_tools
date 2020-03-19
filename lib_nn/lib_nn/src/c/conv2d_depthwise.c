
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
    const int kernel_start_row,
    const int kernel_start_col,
    const unsigned K_h,
    const unsigned K_w,
    const int v_stride,
    const int h_stride,
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

    plan->channels.X = x_params->channels;
    plan->channels.Y = y_params->channels;

    plan->zero_point = zero_point;

    plan->stride.X.row = x_row_bytes - x_params->channels * K_w;
    plan->stride.window.col = x_params->channels * h_stride;

    plan->kernel.height  = K_h;
    plan->kernel.width   = K_w;
    plan->kernel.vstride = v_stride;
    
    const int32_t init_padding_top    = -kernel_start_row;
    const int32_t init_padding_bottom =  kernel_start_row + K_h - x_params->height;
    const int32_t init_padding_left   = -kernel_start_col;
    const int32_t init_padding_right  =  kernel_start_col + K_w  - x_params->width;

    nn_conv2d_job_params_t full_job = {{0,0,0}, {y_params->height, y_params->width, y_params->channels} };

    for(int i = 0; i < job_count; i++){
        const nn_conv2d_job_params_t* params = (job_params != NULL)? &job_params[i] : &full_job;
        nn_conv2d_depthwise_job_t* job = &jobs[i];
        
        assert(params->start.rows >= 0 && params->start.cols >= 0 && params->start.channels >= 0);
        assert(params->start.rows + params->size.rows <= y_params->height);
        assert(params->start.cols + params->size.cols <= y_params->width);
        assert(params->start.channels + params->size.channels <= y_params->channels);

        job->output.rows = params->size.rows;
        job->output.cols = params->size.cols;
        job->output.channels = params->size.channels;

        job->init_padding.top    = init_padding_top    - params->start.rows * plan->kernel.vstride;
        job->init_padding.left   = init_padding_left   - params->start.cols * h_stride;
        job->init_padding.bottom = init_padding_bottom + params->start.rows * plan->kernel.vstride;
        job->init_padding.right  = init_padding_right  + params->start.cols * h_stride;

        const int32_t end_padding_top    = init_padding_top    - ((params->start.rows + params->size.rows - 1) * plan->kernel.vstride);
        const int32_t end_padding_left   = init_padding_left   - ((params->start.cols + params->size.cols - 1) * h_stride);
        const int32_t end_padding_bottom = init_padding_bottom + ((params->start.rows + params->size.rows - 1) * plan->kernel.vstride);
        const int32_t end_padding_right  = init_padding_right  + ((params->start.cols + params->size.cols - 1) * h_stride);

        job->init_padding.unpadded =  (job->init_padding.top    <= 0 && job->init_padding.left  <= 0
                                    && job->init_padding.bottom <= 0 && job->init_padding.right <= 0
                                    && end_padding_top          <= 0 && end_padding_left        <= 0
                                    && end_padding_bottom       <= 0 && end_padding_right       <= 0);


        job->stride.start.BSS  = (params->start.channels / VPU_INT8_ACC_PERIOD);
        job->stride.start.K    = params->start.channels;
        job->stride.start.Y    = params->start.rows * y_row_bytes + y_params->channels * params->start.cols + params->start.channels;

        job->stride.start.X    = window_start_offset 
                                + params->start.rows * plan->kernel.vstride * x_row_bytes
                                + params->start.cols * h_stride * x_params->channels
                                + params->start.channels;

        job->stride.row.window  = x_row_bytes * plan->kernel.vstride;
        job->stride.row.Y       = y_row_bytes;

        job->stride.chan_group.X = VPU_INT8_VLMACC_ELMS - x_row_bytes * job->output.rows * plan->kernel.vstride;
        job->stride.chan_group.Y = VPU_INT8_VLMACC_ELMS - y_row_bytes * job->output.rows;
    }
}


static void nn_compute_hstrip_depthwise_padded_c(
    int8_t* Y,
    const int8_t* X_in, 
    const int8_t* K_in,
    const nn_bss_block_t* BSS,
    const unsigned pad_t,
    const unsigned pad_b,
    const unsigned chans_to_write,
    const int8_t* zero_point_vec,
    const nn_conv2d_depthwise_plan_t* plan,
    const nn_conv2d_depthwise_job_t* job)
{

    int pad_l = job->init_padding.left * plan->channels.X;
    int pad_r = job->init_padding.right * plan->channels.X;

    int center_cols = plan->channels.X * plan->kernel.width;
    if(pad_l >= 0)  center_cols -= pad_l;
    if(pad_r >= 0)  center_cols -= pad_r;

    for(int out_col = 0; out_col < job->output.cols; out_col++){

        const int8_t* X = X_in;
        const int8_t* K = K_in;
        // ADDR(X, "out col start");
        // ADDR(Y, "out col start");
        // ADDR(K, "out col start");

        const int cur_pad_l = (pad_l > 0)? pad_l : 0;
        const int cur_pad_r = (pad_r > 0)? pad_r : 0;

        // printf("pads:  (%d, %d, %d, %d)\n", pad_t, pad_l, pad_l, pad_r);
        // printf("cur_pads:  (%d, %d, %d, %d)\n", pad_t, cur_pad_l, cur_pad_l, cur_pad_r);

        int32_t accs[VPU_INT8_VLMACC_ELMS];

        for(int k = 0; k < VPU_INT8_VLMACC_ELMS; k++)
            accs[k] = ((int32_t)BSS->bias_hi[k]) << VPU_INT8_ACC_VR_BITS;
        
        for(int k = 0; k < VPU_INT8_VLMACC_ELMS; k++)
            accs[k] |= BSS->bias_lo[k];
        
        //THIS LOOP IS IN PADDING (above image)
        for(int i = pad_t; i > 0; i--){
            // printf("PAD_T??\t%d\t%d\n", pad_t, i);
            for(int j = plan->kernel.width; j > 0; j--){
                vlmacc8(accs, zero_point_vec, K);
                X = &X[plan->channels.X];
                K = &K[plan->channels.X];
            }
            X = &X[plan->stride.X.row];
        }

        // These rows are inside image (vertically)
        for(int i = plan->kernel.height - (pad_t + pad_b); i > 0; i--){

            //THIS LOOP IS IN PADDING (left of image)
            for(int j = cur_pad_l; j > 0; j -= plan->channels.X){
                // printf("PAD_L??\t%d\t%d\n", cur_pad_l, j);
                vlmacc8(accs, zero_point_vec, K);
                X = &X[plan->channels.X];
                K = &K[plan->channels.X];
            }

            for(int j = center_cols; j > 0; j-= plan->channels.X){
                vlmacc8(accs, X, K);
                X = &X[plan->channels.X];
                K = &K[plan->channels.X];
            }

            //THIS LOOP IS IN PADDING (right of image)
            for(int j = cur_pad_r; j > 0; j -= plan->channels.X){
                // printf("PAD_R??\t%d\t%d\n", cur_pad_r, j);
                vlmacc8(accs, zero_point_vec, K);
                X = &X[plan->channels.X];
                K = &K[plan->channels.X];
            }

            X = &X[plan->stride.X.row];
        }
        
        //THIS LOOP IS IN PADDING (below image)
        for(int i = pad_b; i > 0; i--){
            // printf("PAD_B??\t%d\t%d\n", pad_b, i);
            for(int j = plan->kernel.width; j > 0; j--){
                vlmacc8(accs, zero_point_vec, K);
                X = &X[plan->channels.X];
                K = &K[plan->channels.X];
            }
            X = &X[plan->stride.X.row];
        }

        for(int k = 0; k < chans_to_write; k++){
            int16_t shift1  = BSS->shift1[k];
            int16_t scale   = BSS->scale[k];
            int16_t shift2  = BSS->shift2[k];
            accs[k] = vlsat_single_s16(accs[k], shift1);
            accs[k] = accs[k] * scale;
            accs[k] = vlsat_single_s8(accs[k], shift2);
            Y[k] = (int8_t) accs[k];
        }

        if(pad_l > 0){
            int tmp = (pad_l <= plan->stride.window.col)? pad_l : plan->stride.window.col;
            center_cols += tmp;
        }

        pad_l -= (int) plan->stride.window.col;
        pad_r += (int) plan->stride.window.col;

        if(pad_r > 0){
            int tmp = (pad_r <= plan->stride.window.col)? pad_r : plan->stride.window.col;
            center_cols -= tmp;
        }
        
        X_in = &X_in[plan->stride.window.col];
        Y = &Y[plan->channels.Y];

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



    int8_t zero_point_vec[VPU_INT8_VLMACC_ELMS];
    memset(zero_point_vec, plan->zero_point, sizeof(zero_point_vec));

    X = &X[job->stride.start.X];
    Y = &Y[job->stride.start.Y];
    K = &K[job->stride.start.K];
    BSS = &BSS[job->stride.start.BSS];

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

            nn_compute_hstrip_depthwise_padded_c( Y, X, K, BSS, cur_pad_t, cur_pad_b, cur_chans, zero_point_vec, plan, job);

            pad_t -= plan->kernel.vstride;
            pad_b += plan->kernel.vstride;

            X = &X[job->stride.row.window];
            Y = &Y[job->stride.row.Y];
        }

        X = &X[job->stride.chan_group.X];
        Y = &Y[job->stride.chan_group.Y];

        BSS = &BSS[1];
        K = &K[VPU_INT8_VLMACC_ELMS];
    }

}


#if defined(__XS3A__)

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

    X = &X[job->stride.start.X];
    Y = &Y[job->stride.start.Y];
    K = &K[job->stride.start.K];
    BSS = &BSS[job->stride.start.BSS];

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

            const int cur_pad_t = (pad_t > 0)? pad_t : 0;
            const int cur_pad_b = (pad_b > 0)? pad_b : 0;
            
            if(job->init_padding.unpadded){
                nn_compute_hstrip_depthwise_asm(Y, X, K, BSS, plan->kernel.height, plan->kernel.width,
                        plan->channels.X, plan->stride.X.row,
                        plan->stride.window.col, plan->channels.Y, job->output.cols, cur_chans);
            } else {
                nn_compute_hstrip_depthwise_padded_asm(Y, X, K, BSS, plan->kernel.height, plan->kernel.width,
                            cur_pad_t, job->init_padding.left, cur_pad_b, job->init_padding.right,
                            plan->channels.X, plan->stride.X.row, 
                            plan->stride.window.col, plan->channels.Y, job->output.cols, 
                            cur_chans, zero_point_vec);
            }

            pad_t -= plan->kernel.vstride;
            pad_b += plan->kernel.vstride;

            X = &X[job->stride.row.window];
            Y = &Y[job->stride.row.Y];
            
        }

        X = &X[job->stride.chan_group.X];
        Y = &Y[job->stride.chan_group.Y];

        BSS = &BSS[1];
        K = &K[VPU_INT8_VLMACC_ELMS];
    }

}

#endif //defined(__XS3A__)