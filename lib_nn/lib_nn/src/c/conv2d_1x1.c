
#include "nn_operator.h"
#include "../nn_op_helper.h"
// #include "nn_op_structs.h"

#include "xs3_vpu.h"
#include "vpu_sim.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>


#if CONFIG_SYMMETRIC_SATURATION_conv2d_1x1
  #define NEG_SAT_VAL   (-127)
#else
  #define NEG_SAT_VAL   (-128)
#endif 





WEAK_FUNC
void conv2d_1x1(
    nn_image_t* Y,
    const nn_image_t* X,
    const nn_tensor_t* K,
    const nn_bso_block_t* BSO,
    const nn_conv2d_1x1_plan_t* plan,
    const nn_conv2d_1x1_job_t* job)
{
    const uint32_t C_out_groups = job->output.channels >> VPU_INT8_ACC_PERIOD_LOG2;
    const uint32_t C_out_tail   = job->output.channels %  VPU_INT8_ACC_PERIOD;
    const uint32_t C_in_groups  = plan->channels.X     >> VPU_INT8_EPV_LOG2;
    const uint32_t C_in_tail    = plan->channels.X     %  VPU_INT8_EPV;

    // Stride required to move to next channel input group for current output group in 
    // the kernel tensor
    const mem_stride_t cig_stride_body = (VPU_INT8_ACC_PERIOD-1) * plan->channels.X + VPU_INT8_EPV;
    const mem_stride_t cig_stride_tail = (C_out_tail - 1) * plan->channels.X + VPU_INT8_EPV;

    //Point K at the start of the first channel out group to be
    //processed by this job.
    K = ADDR(K, job->start.K);

    // Same for these
    const nn_bso_block_t* BSO_start = ADDR(BSO, job->start.BSO/sizeof(nn_bso_block_t));
    const nn_image_t* X_start = ADDR(X, job->start.X);
    nn_image_t* Y_next_cog = ADDR(Y, job->start.Y);

    
    for(int cog = 0; cog <= C_out_groups; cog++){

        //Reset Y, BSO and X fort eh current output group
        Y = Y_next_cog;
        BSO = BSO_start;
        X = X_start;

        //Y for next output group will start 16 channels after this one
        Y_next_cog = ADDR(Y_next_cog, VPU_INT8_ACC_PERIOD);

        //This stride moves K from the first row of the current output group to the final row of the
        // current output group, plus one C_in group (32 channels). It moves K to where it needs to be 
        // after processing all output channels (of the current group) on a given input channel group.
        const int32_t cig_stride = (cog < C_out_groups)? cig_stride_body : cig_stride_tail;

        // K currently points to the beginning of the current output group (e.g. 0 mod 16).
        // Because the channels within the channel group are processed in reverse order, we
        // need to move it to the final channel of the current group
        K = ADDR(K, cig_stride - VPU_INT8_EPV);

        // Number of channels in the current ouput group (16 unless it's the tail)
        const unsigned cog_chans = (cog < C_out_groups)? VPU_INT8_ACC_PERIOD : C_out_tail;

        //Exit early if we're on the tail and there is no tail.
        if(!cog_chans) break;

        //Process all of this job's pixels for the current output group
        for(int pix = 0; pix < job->output.pixels; pix++){

            // Accumulators for the output channels
            int64_t acc64[VPU_INT8_ACC_PERIOD];

            //Biases
            for(int k = 0; k < VPU_INT8_ACC_PERIOD; k++)
                acc64[k] = (BSO->bias_hi[k] << VPU_INT8_ACC_VR_BITS)
                         | (BSO->bias_lo[k] << 0);

            //Process all input channel groups for this pixel
            for(unsigned cig = 0; cig <= C_in_groups; cig++){

                // The final input channel group may have fewer than 32 channels
                const unsigned cig_chans  = (cig < C_in_groups)? VPU_INT8_EPV : plan->channels.X % VPU_INT8_EPV;

                if(!cig_chans) break;

                //Do one VLMACCR for each output channel in the output group,
                // in reverse order (because that's what the assembly must do)
                for(int k = cog_chans-1; k >= 0; k--){

                    //This is all a single VLMACCR
                    for(unsigned cin = 0; cin < cig_chans; cin++){
                        acc64[k] += ((int32_t)X[cin]) * K[cin];
                        acc64[k] = sat_s32(acc64[k]);
                    }

                    //After each VLMACCR (except the last), move K up one row (out channel).
                    K = ADDR(K, (k != 0) * -plan->channels.X);

                }
                
                //Move X to point at next input group
                X = ADDR(X, cig_chans);

                //Move K to final row of current output group, then over one input group.
                K = ADDR(K, cig_stride);
            }

            //After processing the final input channel group, if there was an input tail
            // K will be pointing (32-C_in_tail) bytes into the next output channel group. 
            // Subtracting that will make K point right at the beginning of the next output 
            // channel group, which is where it would be pointing if there was no input tail.
            if(C_in_tail){
                K = ADDR(K, C_in_tail - 32);
            }

            //Turn 32-bit accumulators into 8-bit outputs
            for(unsigned k = 0; k < cog_chans; k++){
                int16_t shift1  = BSO->shift1[k];
                int16_t scale   = BSO->scale[k];
                int16_t shift2  = BSO->shift2[k];
                int16_t offset_scale = BSO->offset_scale[k];
                int16_t offset       = BSO->offset[k];
                
                int32_t res = vlsat_single_s16((int32_t)acc64[k], shift1);
                res = res * scale;
                res = res + ((int32_t)offset_scale) * offset;
                
                res = vlsat_single_s8(res, shift2, NEG_SAT_VAL, VPU_INT8_MAX);

                Y[k] = (int8_t) res;
            }

            //After going through all the input channel groups (and tail), K should be
            //pointing at the start of the next channel group. Subtracting C_in moves it
            //back to pointing at the start of the final channel for the current channel
            //output group, which is where it needs to be for the next pixel.
            K = ADDR(K,-plan->channels.X);

            //Increment Y to point at the next output pixel.
            Y = ADDR(Y, plan->channels.Y);
        }

        //After each output pixel we move K to point back at the start of the final output
        //channel for that channel output group. The channel output group loop expects it
        //to be pointing at the start of the channel group to be processed. Incrementing K
        //by C_in moves it down one row in the kernel matrix.
        K = ADDR(K, plan->channels.X);

        //Move BSO_start to point at the next C_out group.
        BSO_start = ADDR(BSO, 1);
    }
}
#undef ADDR









void conv2d_1x1_init(
    nn_conv2d_1x1_plan_t* plan,
    nn_conv2d_1x1_job_t* jobs,
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_conv2d_1x1_job_params_t* job_params,
    const unsigned job_count)
{
    // Input and output images must have the same spatial dimensions
    assert(x_params->height == y_params->height);
    assert(x_params->width == y_params->width);

    // Input and output images must have multiple of 4 channels
    assert( (x_params->channels % 4) == 0);
    assert( (y_params->channels % 4) == 0);

    // Either a (non-NULL) job params pointer must be provided, or 
    // job_count must be 1.
    assert( job_count == 1 || job_params != NULL);

    
    // Total number of pixels in output image
    const uint32_t y_pixels = y_params->height * y_params->width;

    // Number of channels in the output tail
    const uint32_t C_out_tail = y_params->channels % VPU_INT8_ACC_PERIOD;

    // Input and output image channels
    plan->channels.X = x_params->channels;
    plan->channels.Y = y_params->channels;

    // Params for processing the entire output image
    const nn_conv2d_1x1_job_params_t full_job = { {0, 0, 0}, { y_pixels, y_params->channels } };

    for(int i = 0; i < job_count; i++){

        // Job params to pull info from. If NULL was provided to the job_params argument,
        // we'll use the full job.
        const nn_conv2d_1x1_job_params_t* params = (job_params != NULL)? &job_params[i] : &full_job;
        nn_conv2d_1x1_job_t* job = &jobs[i];

        // Start can never be negative
        assert(params->start.rows >= 0 
            && params->start.cols >= 0 
            && params->start.channels >= 0);

        // Start channel has to be 0 mod 16
        assert(params->start.channels % VPU_INT8_ACC_PERIOD == 0);

        const uint32_t start_pix = params->start.rows * y_params->width + params->start.cols;

        // Make sure we're not trying to compute outputs that go beyond
        //  the bounds of the output image.
        assert(start_pix + params->size.pixels <= y_pixels);
        assert(params->start.channels + params->size.channels <= y_params->channels);

        //Make sure channels to be processed is a multiple of 4
        assert(params->size.channels % 4 == 0);

        //Set job params
        job->output.pixels = params->size.pixels;
        job->output.channels = params->size.channels;

        const uint32_t start_cog = (params->start.channels / VPU_INT8_ACC_PERIOD);

        job->start.X = start_pix * plan->channels.X;
        job->start.Y = start_pix * plan->channels.Y + params->start.channels;
        job->start.K = start_cog * VPU_INT8_ACC_PERIOD * plan->channels.X;
        job->start.BSO = start_cog  * sizeof(nn_bso_block_t);

    }

}
