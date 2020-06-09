

#include "nn_operator.h"
#include "../nn_op_helper.h"
#include "nn_op_structs.h"

#include "xs3_vpu.h"
// #include "../asm/asm_constants.h"
#include "vpu_sim.h"

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

    // const unsigned k_array_width = VPU_INT8_EPV / x_params->channels;

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
                                       x_params->channels + VPU_INT8_EPV-1)>>VPU_INT8_EPV_LOG2)<<VPU_INT8_EPV_LOG2;
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
    const int32_t init_padding_right  =  conv_window->start.column + plan->window.shape.width - x_params->width;

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


        job->stride.row.K       = plan->window.shape.kernel_row_elements;
        job->stride.chan_group.K = plan->window.shape.kernel_row_elements * VPU_INT8_ACC_PERIOD;
        job->stride.start.K    = params->start.channels * plan->window.shape.kernel_row_elements;
        job->stride.start.Y    = params->start.rows * y_row_bytes 
                               + params->start.cols * y_params->channels
                               + params->start.channels;

        job->stride.start.X    = window_start_offset 
                               + params->start.rows * plan->window.stride.vertical * x_row_bytes
                               + params->start.cols * plan->window.stride.horizontal * x_params->channels;

        job->stride.row.window  = (plan->window.stride.vertical-1) * x_row_bytes //newline
                                  - plan->channels.X * (plan->window.stride.horizontal-1); // carriage return
        
        const unsigned job_row_bytes = job->output.cols * y_params->channels;
        job->stride.row.Y       = y_row_bytes - job_row_bytes;

        job->stride.chan_group.Y = VPU_INT8_ACC_PERIOD;
        job->stride.col.Y = y_params->channels; // TODO should this account for multiple bytes per channel?
    }
}


#if CONFIG_SYMMETRIC_SATURATION_conv2d_im2col
  #define NEG_SAT_VAL   (-127)
#else
  #define NEG_SAT_VAL   (-128)
#endif 


void conv2d_im2col(
    nn_image_t* Y,
    const nn_image_t* X,
    const nn_image_t* COL,
    const nn_tensor_t* K,
    const nn_bso_block_t* BSO,
    const nn_conv2d_im2col_plan_t* plan,
    const nn_conv2d_im2col_job_t* job)
{ 

    xs3_vpu vpu;
    vpu_vector_t vec_tmp;

    const int16_t vec_0x007F[16] = {0x007f};
    const int8_t vec_0x80[30] = {0x80};

    VSETC(&vpu, MODE_S8);

    X = ADDR(X,job->stride.start.X);
    Y = ADDR(Y, job->stride.start.Y);
    K = ADDR(K, job->stride.start.K);
    BSO = ADDR(BSO, job->stride.start.BSO);

    int pad_t = job->init_padding.top;
    int pad_b = job->init_padding.bottom;
    
    //Iterate once per row of the output region
    for(unsigned output_rows = job->output.rows; output_rows > 0; output_rows--){

        int pad_l = job->init_padding.left;
        int pad_r = job->init_padding.right;

        const int pad_lr_delta = plan->window.stride.horizontal * (job->output.cols - 1);
        const int final_pad_l = pad_l - pad_lr_delta;
        const int final_pad_r = pad_r + pad_lr_delta;

        const int cur_pad_t = (pad_t > 0)? pad_t : 0;
        const int cur_pad_b = (pad_b > 0)? pad_b : 0;

        const unsigned requires_padding = ( pad_l       > 0) || (pad_r       > 0) 
                                        || (cur_pad_t   > 0) || (cur_pad_b   > 0) 
                                        || (final_pad_l > 0) || (final_pad_r > 0);
        //Iterate once per col of the output region
        for(unsigned output_cols = job->output.cols; output_cols > 0; output_cols--){
            //This points it at the top-left cell of the patch.
            const nn_image_t* patch_X = X;
            const nn_image_t* C = COL;
            memset(C,0,plan->window.shape.len_col);// initialize pad
            unsigned len = plan->channels.X * plan->window.shape.width;
            if( requires_padding ){
                    int8_t padded_vals[128]={0}; // todo drop the 128 limit or enforce it
                    len *= plan->window.shape.height;
                    int k = 0;
                    int pad = 0;
                    int pad_tb = 0; // ??? do we need to distinguish?
                    for(int h = 0; h< plan->window.shape.height; h++){
                        int p = 0;
                        pad_tb = (cur_pad_t -h > 0) || (plan->window.shape.height - pad_b) <= h; 
                        for(int i = 0; i < plan->window.shape.width; i++){
                            pad =  ((pad_l-i > 0) || (plan->window.shape.width - pad_r) <= i) || pad_tb;
                            for(int j = 0; j < plan->channels.X; j++){
                                if(pad){
                                    padded_vals[k] = plan->zero_point;
                                }
                                else{
                                    padded_vals[k] = patch_X[p]; // can be flattened and use more memcpy
                                    
                                }
                                p++;
                                k++;
                            }
                        }  
                        patch_X = ADDR(patch_X, plan->stride.X.row);

                    }
                    memcpy(C, padded_vals, len);// TODO  vpu memcpy
                }
            else{
                    for(unsigned rows_in_patch = plan->window.shape.height; rows_in_patch > 0; rows_in_patch--){
                        memcpy(C,patch_X,len);// TODO  vpu memcpy
                        patch_X = ADDR(patch_X, plan->stride.X.row );
                        C = ADDR(C,len);
                    }
                }
            // im2col complete what follows is just a BSO-aware K*COL + BSO = Y  
            #if !CONFIG_SYMMETRIC_SATURATION_conv2d_im2col
                // VLDR(&vpu, vec_0x80);
                // VSTRPV(&vpu, Y, y_mask);// @tail chicken and egg problem with vec registers
                memset(Y,0x80,job->output.channels);
            #endif

            int mat_col_chunks = plan->window.shape.kernel_row_elements % VPU_INT8_EPV == 0 ?
                                 plan->window.shape.kernel_row_elements / VPU_INT8_EPV      :
                                 plan->window.shape.kernel_row_elements / VPU_INT8_EPV +1;
            int mat_row_chunks = job->output.channels % VPU_INT8_ACC_PERIOD == 0 ?
                                 job->output.channels / VPU_INT8_ACC_PERIOD      :
                                 job->output.channels / VPU_INT8_ACC_PERIOD +1;  
            

            nn_image_t* Y_cog = ADDR(Y, 0);
            for(int m_row_chunk = 0; m_row_chunk< mat_row_chunks; m_row_chunk++){
                const nn_tensor_t* patch_K = ADDR(K,m_row_chunk*job->stride.chan_group.K);
                const nn_image_t* sub_vector = COL;
                //get BSO for this group of output channels
                nn_bso_block_t * bso = &BSO[m_row_chunk];
                uint16_t y_mask = 0x0000;
                //load vR and vD with appropriate bso vectors
                VLDD(&vpu, bso->bias_hi);
                VLDR(&vpu, bso->bias_lo);

                for(int m_col_chunk = 0; m_col_chunk < mat_col_chunks; m_col_chunk++){
    
                        y_mask = 0xFFFF;
                        
                        // load the 32 element sub-vector into vC
                        VLDC(&vpu, sub_vector);

                        // VLMACCR each output channel for up to 16 channels
                        int sub_mat_row_end = m_row_chunk < (mat_row_chunks-1) ? VPU_INT8_ACC_PERIOD : job->output.channels % VPU_INT8_ACC_PERIOD;
                        if(sub_mat_row_end == 0) sub_mat_row_end = VPU_INT8_ACC_PERIOD; // todo find cleaner way to unwrap modulus
                        const nn_tensor_t* sub_matrix = ADDR( patch_K, job->stride.row.K*(sub_mat_row_end));
                        
                        //TODO still need to find the best way to add a <=32 byte pad to K
                        for(int m_row = 0; m_row < VPU_INT8_ACC_PERIOD-sub_mat_row_end; m_row++) {VLMACCR(&vpu, COL); y_mask>>=1; } // DUMMY on safe memory to rotate @tail
                        for(int m_row = 0; m_row < sub_mat_row_end; m_row++){                                            
                            sub_matrix = ADDR(sub_matrix, -job->stride.row.K);
                            VLMACCR(&vpu, sub_matrix);
                        }
                                        
                        // go to next section of vector
                        sub_vector = ADDR(sub_vector,VPU_INT8_EPV);
                        patch_K = ADDR(patch_K, VPU_INT8_EPV );
                }

                // Done with A*x + b need to groom and extract results now

                //Set mode to 16-bit
                VSETC(&vpu, MODE_S16);
                //Saturate to 16-bit values
                VLSAT(&vpu, bso->shift1);

                //Load scales into vC
                VLDC(&vpu, bso->scale);
                VSTR(&vpu, vec_tmp.s16);
                VCLRDR(&vpu);
            
                VLMACC(&vpu, vec_tmp.s16);
                VLDC(&vpu, bso->offset_scale);
                VLMACC(&vpu, bso->offset);
                

                #if CONFIG_SYMMETRIC_SATURATION_conv2d_im2col

                    //Set mode back to 8-bit
                    VSETC(&vpu, MODE_S8);

                    //Saturate to 8-bit values
                    VLSAT(&vpu, bso->shift2);

                    //Store result in Y
                    const unsigned mask16 = y_mask;
                    VSTRPV(&vpu, Y_cog, mask16);
                
                #else

                    //Saturate to 8-bit values
                    VLSAT(&vpu, bso->shift2);
                    VSTR(&vpu, vec_tmp.s16);
                    VLADD(&vpu, vec_0x007F);
                    VDEPTH1(&vpu);

                    // uint32_t mask = ~vpu.vR.s32[0];

                    VLASHR(&vpu, vec_tmp.s16, -8);
                    VDEPTH8(&vpu);
                    //Store result in Y
                    // mask = mask & 0xFFFF;
                    // printf( "0x%0.4X\n",y_mask);
                    // for(int i=0; i<32; i++)printf("vpu.vR.s8[%d] = %d\n", i, vpu.vR.s8[i]);
                    
                    VSTRPV(&vpu, Y_cog, y_mask);
                    
                    //Set mode back to 8-bit
                    VSETC(&vpu, MODE_S8);

                #endif
                // printf("row chunk %d  Y  %d   Y_cog-Y  %d\n", m_row_chunk, Y, Y_cog-Y);
                if( mat_row_chunks > 1 ) Y_cog = ADDR(Y_cog, job->stride.chan_group.Y);
                    
            }
            //Move X and Y pointers one pixel to the right
            X = ADDR(X, plan->channels.X * (plan->window.stride.horizontal));
            Y = ADDR(Y, job->stride.col.Y);

            pad_l -= plan->window.stride.horizontal;
            pad_r += plan->window.stride.horizontal;
            
        }
        X = ADDR(X,job->stride.row.window);
        Y = ADDR(Y, job->stride.row.Y);
        pad_t -= plan->window.stride.vertical;
        pad_b += plan->window.stride.vertical;

    }
}
