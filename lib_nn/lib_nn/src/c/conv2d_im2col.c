

#include "nn_operator.h"
#include "../nn_op_helper.h"
#include "nn_op_structs.h"

#include "xs3_vpu.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>


void conv2d_im2col_init(
    nn_conv2d_im2col_plan_t* plan,
    nn_conv2d_im2col_job_t* jobs,
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_conv2d_job_params_t* job_params,
    const nn_conv2d_window_params_t* conv_window,
    const int8_t zero_point,
    const unsigned job_count)
{

    // The restrict to patch elements <= 128 for the time being, but should be able to relax
    assert(x_params->channels * conv_window->shape.width * conv_window->shape.height <= VPU_INT8_EPV*4);

    // Need at least 1 job
    assert(job_count > 0);
    // job_params can only be NULL if there's exactly 1 job.
    assert(job_count == 1 || job_params != NULL);

    const unsigned k_array_width = VPU_INT8_EPV / x_params->channels;

    const unsigned x_row_bytes = x_params->width * x_params->channels;
    const unsigned y_row_bytes = y_params->width * y_params->channels;

    const mem_stride_t window_start_offset = conv_window->start.row * x_row_bytes 
                                           + conv_window->start.column * x_params->channels;

    plan->channels.X = x_params->channels;
    plan->channels.Y = y_params->channels;

    plan->zero_point = zero_point;

    plan->window.shape.height      = conv_window->shape.height;
    plan->window.shape.width       = conv_window->shape.width;
    plan->window.shape.len_col     = ((conv_window->shape.width * 
                                       conv_window->shape.height * 
                                       x_params->channels + 31)>>5)<<5;
    plan->window.shape.kernel_row_elements     = 
                                      ((conv_window->shape.width * 
                                       conv_window->shape.height * 
                                       x_params->channels + 3)>>2)<<2;

    plan->window.stride.vertical   = conv_window->stride.vertical;
    plan->window.stride.horizontal = conv_window->stride.horizontal;

    plan->stride.X.row = x_row_bytes;
    
    const int32_t init_padding_top    = -conv_window->start.row;
    const int32_t init_padding_bottom =  conv_window->start.row + conv_window->shape.height - x_params->height;
    const int32_t init_padding_left   = -conv_window->start.column;
    const int32_t init_padding_right  =  conv_window->start.column + k_array_width - x_params->width;

    nn_conv2d_job_params_t full_job = {{0,0,0}, {y_params->height, y_params->width, y_params->channels} };

    for(int i = 0; i < job_count; i++){
        const nn_conv2d_job_params_t* params = (job_params != NULL)? &job_params[i] : &full_job;
        nn_conv2d_im2col_job_t* job = &jobs[i];
        
        // Start can never be negative
        assert(params->start.rows >= 0 
            && params->start.cols >= 0 
            && params->start.channels >= 0);

        // Start channel has to be 0 mod 16
        assert(params->start.channels % VPU_INT8_ACC_PERIOD == 0);

        // Make sure we're not trying to compute outputs that go beyond
        //  the bounds of the output image.
        assert(params->start.rows + params->size.rows <= y_params->height);
        assert(params->start.cols + params->size.cols <= y_params->width );
        assert(params->start.channels + params->size.channels <= y_params->channels);

        job->output.rows = params->size.rows;
        job->output.cols = params->size.cols;
        job->output.channels = params->size.channels;

        job->init_padding.top    = init_padding_top    - params->start.rows * plan->window.stride.vertical;
        job->init_padding.left   = init_padding_left   - params->start.cols * plan->window.stride.horizontal;
        job->init_padding.bottom = init_padding_bottom + params->start.rows * plan->window.stride.vertical;
        job->init_padding.right  = init_padding_right  + params->start.cols * plan->window.stride.horizontal;

        job->stride.start.BSO  = (params->start.channels / VPU_INT8_ACC_PERIOD);
        job->stride.start.K    = params->start.channels * plan->window.shape.height * VPU_INT8_EPV;
        job->stride.start.Y    = params->start.rows * y_row_bytes 
                               + params->start.cols * y_params->channels
                               + params->start.channels;

        job->stride.start.X    = window_start_offset 
                               + params->start.rows * plan->window.stride.vertical * x_row_bytes
                               + params->start.cols * plan->window.stride.horizontal * x_params->channels;

        job->stride.row.window  = x_row_bytes * plan->window.stride.vertical;
        job->stride.row.Y       = y_row_bytes;

        job->stride.chan_group.Y = VPU_INT8_ACC_PERIOD - y_row_bytes * job->output.rows;
        job->stride.col.Y = y_row_bytes / job->output.cols;// TODO check
    }
}




void conv2d_im2col(
    nn_image_t* Y,
    const nn_image_t* X,
    const nn_image_t* COL,
    const nn_tensor_t* K,
    const nn_bso_block_t* BSO,
    const nn_conv2d_im2col_plan_t* plan,
    const nn_conv2d_im2col_job_t* job)
{ 
    printf("1\n");
    int8_t zero_point_vec[VPU_INT8_EPV];
    memset(zero_point_vec, plan->zero_point, sizeof(zero_point_vec));
    
    X = ADDR(X,job->stride.start.X);
    Y = ADDR(Y, job->stride.start.Y);
    K = ADDR(K, job->stride.start.K);
    BSO = ADDR(BSO, job->stride.start.BSO);

    const unsigned C_out_tail = plan->channels.Y % VPU_INT8_ACC_PERIOD; // TODO I don't think im2col cares about tails..

    const unsigned vlmaccr_align = 4;
    const unsigned patch_elements = plan->channels.X * plan->window.shape.height * plan->window.shape.width;
    
    int pad_t = job->init_padding.top;
    int pad_b = job->init_padding.bottom;
    printf("2\n");
    //Iterate once per row of the output region
    for(unsigned output_rows = job->output.rows; output_rows > 0; output_rows--){
        //Iterate once per col of the output region
        for(unsigned output_cols = job->output.cols; output_cols > 0; output_cols--){

            //V is the X-pointer for the patch. This points it at the top-left cell of the patch.
            const int8_t* V = X;
            const int8_t* L = K;
            // set up "column" by copying all patch rows sequentially
            // TODO look at Aaron's VPU memcpy (doesn't require word-aligned)
            const int8_t* C = COL;
            memset(C,0,plan->window.shape.len_col);// initialize pad -- zero point is handled at bias initialization TODO is this still true?
            unsigned len = plan->channels.X * plan->window.shape.width;
            printf("len = %d\n", plan->window.shape.len_col);
            // start paste
            int pad_l = job->init_padding.left;
            int pad_r = job->init_padding.right;

            const int pad_lr_delta = plan->window.stride.horizontal * (job->output.cols - 1);
            const int final_pad_l = pad_l - pad_lr_delta;
            const int final_pad_r = pad_r + pad_lr_delta;

            const int cur_pad_t = (pad_t > 0)? pad_t : 0;
            const int cur_pad_b = (pad_b > 0)? pad_b : 0;

            const unsigned requires_padding = (pad_l       > 0) || (pad_r       > 0) 
                                           || (cur_pad_t   > 0) || (cur_pad_b   > 0) 
                                           || (final_pad_l > 0) || (final_pad_r > 0);
            // end paste

            for(unsigned rows_in_patch = plan->window.shape.height; rows_in_patch > 0; rows_in_patch--){
                printf("3\n");
                if( requires_padding ){
                    memcpy(C,V,len);
                    printf("4a\n");
                    // int8_t padded_vals[VPU_INT8_EPV]={0};
                    // int k = 0;
                    // int tmp = 0;
                    // for(int i = 0; i < plan->window.shape.width; i++){
                    //     for(int j = 0; j < plan->channels.X; j++){
                    //         padded_vals[k] = (tmp & 0x1) ? V[k] : 0;
                    //         k++;
                    //     }
                    //     tmp >>=1;
                    // }  
                    // memcpy(C, padded_vals, len);
                }
                else{
                    memcpy(C,V,len);
                    printf("4b\n");
                }
                
                V = ADDR(V,job->stride.row.window);
                C = ADDR(C,len);

           }
           printf("5\n");

            /*print debug info*/
            // if(output_rows == block->output.rows && output_cols == block->output.cols){
                // printf("\n\n %d %d \n", output_rows, output_cols);

                // unsigned l= job->output.channels;
                // if( l < plan->window.shape.kernel_row_elements) l = plan->window.shape.kernel_row_elements;
                
                // for(unsigned j=0; j<l; j++){
                //     if(j < plan->window.shape.kernel_row_elements){ 
                //         printf("\n | %d | ", COL[j]);}
                //     else{
                //         printf("\n       | ");}
                    
                //     if( j < job->output.channels){
                //         printf("|");
                //         for(unsigned i = 0; i< plan->window.shape.kernel_row_elements; i++){
                //             printf("%d ",L[i]);// TODO define L[] to reduce pointer math
                //         }
                //         L = &L[plan->window.shape.kernel_row_elements];
                //         printf("|");
                //     }
                // }
                // L = K;
            // }
            /*     */
                                
            unsigned jj=0;
            for(unsigned j = 0; j < plan->channels.Y; j++){ //TODO make this use the new VLMACCR macros etc.!
                //Because this mirrors the assembly, we need to keep accumulators for each
                // of the channels in an output channel
                int32_t patch_acc[VPU_INT8_ACC_PERIOD];
                jj = j%VPU_INT8_ACC_PERIOD;
                if(jj==0){
                    for(int k = 0; k < VPU_INT8_ACC_PERIOD; k++){
                        int32_t bias = (BSO->bias_hi[k]<<16) + BSO->bias_lo[k];
                        patch_acc[k] = bias;
                    }
                    if(j!=0){
                        BSO = ADDR(BSO, 1);
                    }
                }
                // printf("\nj=%d patch_acc[%d] = %d \t",j,jj,patch_acc[jj]);
                for(unsigned i = 0; i< plan->window.shape.kernel_row_elements; i++){
                    patch_acc[jj]  += (int32_t)(COL[i]*L[i]);
                }
                // printf("patch_acc[%d] = %d\t", jj, patch_acc[jj]);

                L = &L[plan->window.shape.kernel_row_elements];
                 //L = &L[plan->window.shape.kernel_row_elements]; // TODO make this the k row increment at initialization time
                // The shape of the `scales` tensor is (C_out // 16, 2, 16). 
                // The first index is the channel group, second indicates shift (0) or scale (1), 
                // and the third is the channel offset within the channel group.
                int16_t res16;
                int8_t res8;

                res16 = vlsat_single_s16(patch_acc[jj], BSO->shift1[jj]);
                // printf("scales[0][jj]: %d\tscales[1][jj]: %d\t", cales[jj],cales[ jj+VPU_INT8_ACC_PERIOD]);
                res16 = vlmul_single_s16(res16, BSO->scale[jj]);
                res8 = vdepth8_single_s16(res16);
                // printf("res16= 0x%04X\tres8 = %d",res16, res8);
                Y[j] =res8;      
                // move BSO to the next channel-output-group

            }
            printf("6\n");
            //Move X and Y pointers one pixel to the right
            X = ADDR(X, plan->window.stride.horizontal);// TODO see if we need X and window
            Y = ADDR(Y, job->stride.col.Y);
            
        }
        printf("7\n");
        //Move X and Y pointers to the start of the following row
        X = ADDR(X, job->stride.row.window);
        Y = ADDR(Y, job->stride.row.Y);
        pad_t -= plan->window.stride.vertical;
        pad_b += plan->window.stride.vertical;

    }

}