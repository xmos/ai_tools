

#include "nn_operator.h"
#include "nn_op_helper.h"
#include "nn_op_structs.h"

#include "xs3_vpu.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>


///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////


/**
    Takes bias tensor B in standard form (just a vector of 32-bit ints) and rearranges it (in place)
    for use with conv2d_deepin_deepout().

    Returns B, cast to a data16_t pointer.
*/
data16_t* conv2d_boggle_B(
    int32_t* B,
    const unsigned C_out)
{
    const unsigned cog = C_out >> VPU_INT8_ACC_PERIOD_LOG2;


    for(int i = 0; i < cog; i++){

        int32_t* B_grp = &B[i * VPU_INT8_ACC_PERIOD];

        data16_t* B_hi = (data16_t*) B_grp;
        data16_t* B_lo = &B_hi[VPU_INT8_ACC_PERIOD];
        
        int32_t buff[VPU_INT8_ACC_PERIOD];

        //Just copy first to make it simple
        memcpy(buff, B_grp, sizeof(buff));

        for(int k = 0; k < VPU_INT8_ACC_PERIOD; k++){
            int32_t val = buff[k];

            data16_t hi = (val & 0xFFFF0000) >> 16;
            data16_t lo = (val & 0x0000FFFF) >>  0;

            B_hi[k] = hi;
            B_lo[k] = lo;
        }

    }

    return (data16_t*) B;
}


/**
    Transform the kernel tensor K from the standard layout to the boggled layout
    required for conv2d_deepin_deepout().

    Standard layout has shape  (C_out, K_h, K_w, C_in)

    Modified layout has shape  (C_out // 16, K_h, K_w, C_in // 32, 16, 32)
        where the axes are: 
            - out group
            - kernel row
            - kernel col
            - in group
            - out chan (mod 16) (reversed)
            - in chan (mod 32)
*/
void conv2d_dido_boggle_K(
    int8_t* K,
    const unsigned K_h,
    const unsigned K_w,
    const unsigned C_in,
    const unsigned C_out)
{

    const unsigned C_out_groups = C_out >> VPU_INT8_ACC_PERIOD_LOG2;
    const unsigned C_in_groups = C_in >> VPU_INT8_EPV_LOG2;

    typedef struct {
        int8_t ch[VPU_INT8_EPV];
    } block_t;

    //No point in malloc()ing more than one output channel group's worth
    const size_t cout_group_bytes = K_h * K_w * C_in * VPU_INT8_ACC_PERIOD * sizeof(int8_t);


    block_t* Kb = (block_t*) K;
    block_t* K_tmp = malloc(cout_group_bytes);

    if(!K_tmp){
        //Trap?
        printf("malloc() failed.!\n");
    }


    for(int cog = 0; cog < C_out_groups; cog++){

        unsigned orig_offset = cog * (K_h * K_w * VPU_INT8_ACC_PERIOD * C_in_groups);

        block_t* K_cog_src = (block_t*) &K[orig_offset];
        block_t* K_cog_dst = (block_t*) &Kb[cog * K_h * K_w * VPU_INT8_ACC_PERIOD * C_in_groups];

        //The output channel groups don't move, so we can just iterate over those, shuffling their innards.
        
        memcpy(K_tmp, K_cog_src, cout_group_bytes);

        for(int kr = 0; kr < K_h; kr++){
            for(int kc = 0; kc < K_w; kc++){
                for(int cig = 0; cig < C_in_groups; cig++){
                    for(int co = 0; co < VPU_INT8_ACC_PERIOD; co++){

                        block_t* K_src = &K_tmp[ co*K_h*K_w*C_in_groups  +  kr*K_w*C_in_groups  +  kc*C_in_groups  +  cig ];
                        block_t* K_dst = &K_cog_dst[ (kr*K_w*C_in_groups + kc*C_in_groups + cig) * VPU_INT8_ACC_PERIOD  +  co ];
                        *K_dst = *K_src;
                    }
                }
            }
        }
    }

    if(K_tmp){
        free(K_tmp);
    }
}


void conv2d_sido_boggle_K(
    int8_t* K,
    const unsigned K_h,
    const unsigned K_w,
    const unsigned C_in,
    const unsigned C_out)
{
    //Hate to have to malloc for this, but... it is what it is.

    assert(C_in % 4 == 0);
    assert(C_in * K_w == VPU_INT8_EPV);

    const unsigned C_out_groups = C_out >> VPU_INT8_ACC_PERIOD_LOG2;

    const unsigned C_out_group_size_elms = K_h * VPU_INT8_ACC_PERIOD * VPU_INT8_EPV;

    int8_t* K_buff = malloc( C_out_group_size_elms * sizeof(int8_t) );

    assert(K_buff);

    for(int cog = 0; cog < C_out_groups; cog++){

        memcpy(K_buff, &K[cog * C_out_group_size_elms], C_out_group_size_elms * sizeof(int8_t));

        for(int cout = 0; cout < VPU_INT8_ACC_PERIOD; cout++){
            
            const unsigned rcout = (VPU_INT8_ACC_PERIOD - 1) - cout;

            for(int kr = 0; kr < K_h; kr++){
                for(int kc = 0; kc < K_w; kc++){

                    unsigned in_b = (cog * VPU_INT8_ACC_PERIOD + cout) * (K_h * K_w * C_in)
                                + kr * (K_w * C_in)
                                + kc * (C_in);
                    unsigned out_b = cog * (K_h * VPU_INT8_ACC_PERIOD * VPU_INT8_EPV) 
                                + kr * (VPU_INT8_ACC_PERIOD * VPU_INT8_EPV) 
                                + rcout * (VPU_INT8_EPV) 
                                + kc * (C_in);

                    for(int cin = 0; cin < C_in; cin++){

                        unsigned in_index = in_b + cin;
                        unsigned out_index = out_b + cin;

                        K[out_index] = K_buff[in_index];
                    }
                }
            }
        }
    }

    if(K_buff) free(K_buff);

}

static inline unsigned conv2d_block_output_bounds(
    uint32_t* Y_t,
    uint32_t* Y_l,
    uint32_t* Y_b,
    uint32_t* Y_r,
    const unsigned kr,
    const unsigned kc,
    const uint32_t Y_height,
    const uint32_t Y_width,
    const uint32_t K_h,
    const uint32_t K_w,
    const uint32_t region_t,
    const uint32_t region_l,
    const uint32_t region_rows,
    const uint32_t region_cols)
{
    const int P_h = K_h >> 1;
    const int P_w = K_w >> 1;

    unsigned region_b = region_t + region_rows;
    unsigned region_r = region_l + region_cols;
    
    *Y_t  = (kr <= P_h)? kr : Y_height - K_h + kr;
    *Y_l  = (kc <= P_w)? kc : Y_width  - K_w + kc;
    *Y_b  = *Y_t + (  (kr == P_h)? Y_height - 2 * P_h : 1  );
    *Y_r  = *Y_l + (  (kc == P_w)? Y_width  - 2 * P_w : 1  );

    if(*Y_t < region_t)
        *Y_t = region_t;
    else if(*Y_t > region_b)
        *Y_t = region_b;

    if(*Y_b < region_t)
        *Y_b = region_t;
    else if(*Y_b > region_b)
        *Y_b = region_b; 

    if(*Y_l < region_l)
        *Y_l = region_l;
    else if(*Y_l > region_r)
        *Y_l = region_r;
    
    if(*Y_r < region_l)
        *Y_r = region_l;
    else if(*Y_r > region_r)
        *Y_r = region_r;

    return (*Y_b > *Y_t && *Y_r > *Y_l);
}

void conv2d_deepin_deepout_init(
    nn_conv2d_dido_params_t* params,
    const uint32_t X_height,
    const uint32_t X_width,
    const uint32_t K_h,
    const uint32_t K_w,
    const uint32_t C_in,
    const uint32_t C_out,
    const padding_mode_t pad_mode,
    const int8_t zero_point,
    const uint32_t region_top,
    const uint32_t region_left,
    const uint32_t region_rows,
    const uint32_t region_cols)
{
    const int P_h = K_h >> 1;
    const int P_w = K_w >> 1;

    uint32_t Y_height, Y_width;

    if(pad_mode == PADDING_VALID){
        Y_height = X_height - 2*P_h;
        Y_width  = X_width  - 2*P_w;
    } else {
        Y_height = X_height;
        Y_width  = X_width;
    }

    //Check parameters
    unsigned check_errors = 1;
    if(check_errors){

        unsigned error = 0;
        if(region_rows > Y_height)
            error = 1;
        if(region_cols > Y_width)
            error = 1;
        if(region_top >= Y_height)
            error = 1;
        if(region_top + region_rows > Y_height)
            error = 1;
        if(region_left >= Y_width)
            error = 1;
        if(region_left + region_cols > Y_width)
            error = 1;

        if(error){
            printf("Region parameters invalid.\n");
            __builtin_trap();
        }
    }
    
    params->chans_in = C_in;
    params->chans_out = C_out;
    params->C_in_groups = C_in >> VPU_INT8_EPV_LOG2;
    params->C_out_groups = C_out >> VPU_INT8_ACC_PERIOD_LOG2;
    params->zero_point = zero_point;

    if(pad_mode == PADDING_VALID){

        nn_conv2d_dido_block_params_t* blocks = malloc(sizeof(nn_conv2d_dido_block_params_t));

        if(!blocks){
            printf("Failed to allocation blocks.\n");
            __builtin_trap();
        }

        params->block_count = 1;
        params->blocks = blocks;


        nn_conv2d_dido_block_params_t* block = &blocks[0];

        uint32_t Y_start_top = region_top;
        uint32_t Y_start_left = region_left;

        uint32_t X_start_top = Y_start_top;
        uint32_t X_start_left = Y_start_left;

        
        block->init.start_offset.X = C_in * (X_start_top * X_width + X_start_left);
        block->init.start_offset.Y = C_out * (Y_start_top * Y_width + Y_start_left);
        block->init.start_offset.K = 0;
        block->init.padding_cells = 0;

        block->patch.maccs_per_row = K_w * params->C_in_groups;
        block->patch.rows = K_h;
        block->patch.row_incr.X = C_in * (X_width - K_w);

        //This needs to be enough to pull K to the beginning of the current row, then down a row
        block->patch.row_incr.K = 0;

        //This needs to be enough to push K to the end of the current channel group, plus the start offset
        block->cout_group_incr.K = 0;
        block->output.rows = region_rows;
        block->output.cols = region_cols;
        block->output.row_incr.X = C_in * (X_width - block->output.cols);
        block->output.row_incr.Y = C_out * (Y_width - block->output.cols);
    } else {

        unsigned block_count = 0;

        //First, determine which blocks have any work to be done
        for(int kr = 0; kr < K_h; kr++){
            for(int kc = 0; kc < K_w; kc++){
                uint32_t aa, bb, cc, dd;
                if(conv2d_block_output_bounds(&aa, &bb, &cc, &dd,
                                              kr, kc, Y_height, Y_width, K_h, K_w,
                                              region_top, region_left, region_rows, region_cols))
                    block_count++;
            }
        }

        //Now, allocate the memory
        nn_conv2d_dido_block_params_t* blocks = malloc(block_count * sizeof(nn_conv2d_dido_block_params_t));

        if(!blocks){
            printf("Failed to allocation blocks.\n");
            __builtin_trap();
        }

        params->block_count = block_count;
        params->blocks = blocks;

        //Finally, actually initialize the block parameters

        unsigned block_dex = 0;
        for(int kr = 0; kr < K_h; kr++){
            for(int kc = 0; kc < K_w; kc++){

                nn_conv2d_dido_block_params_t* block = &blocks[block_dex];

                uint32_t Y_top, Y_left, Y_bottom, Y_right;

                if(!conv2d_block_output_bounds(&Y_top, &Y_left, &Y_bottom, &Y_right,
                                               kr, kc, Y_height, Y_width, K_h, K_w,
                                               region_top, region_left, region_rows, region_cols))
                    continue;
                
                block_dex++;

                unsigned pad_t = (kr  > P_h)?  0 : P_h - kr;
                unsigned pad_b = (kr <= P_h)?  0 : P_h - (K_h-1-kr);
                unsigned pad_l = (kc  > P_w)?  0 : P_w - kc; 
                unsigned pad_r = (kc <= P_w)?  0 : P_w - (K_w-1-kc);
                
                unsigned out_rows = Y_bottom - Y_top;
                unsigned out_cols = Y_right  - Y_left;

                // printf("\t (t, l)\t(rows x cols) = (%u, %u)\t(%ux%u)\n", Y_top, Y_left, out_rows, out_cols);

                unsigned X_top    = (Y_top  <= P_h)? 0 : Y_top  - P_h;
                unsigned X_left   = (Y_left <= P_w)? 0 : Y_left - P_w;
                
                unsigned patch_rows = K_h - pad_t - pad_b;
                unsigned patch_cols = K_w - pad_l - pad_r;


                block->init.start_offset.X = C_in * (X_left + X_top * X_width);
                block->init.start_offset.Y = C_out * (Y_left + Y_top * Y_width);
                block->init.start_offset.K = VPU_INT8_ACC_PERIOD * VPU_INT8_EPV * (pad_l + pad_t * K_w); 
                block->init.padding_cells = (pad_t + pad_b) * K_w + (pad_l + pad_r) * patch_rows;

                block->patch.maccs_per_row = patch_cols * params->C_in_groups;
                block->patch.rows = patch_rows;
                block->patch.row_incr.X = C_in * (X_width - patch_cols);

                //This needs to be enough to pull K to the beginning of the current row, then down a row
                block->patch.row_incr.K = (K_w - patch_cols) * (C_in * VPU_INT8_ACC_PERIOD);

                //This needs to be enough to push K to the end of the current channel group, plus the start offset
                block->cout_group_incr.K = (pad_r + K_w * pad_b) * (C_in * VPU_INT8_ACC_PERIOD) + block->init.start_offset.K;
                block->output.rows = out_rows;
                block->output.cols = out_cols;
                block->output.row_incr.X = C_in * (X_width - block->output.cols);
                block->output.row_incr.Y = C_out * (Y_width - block->output.cols);
            }
        }
    }

}

void conv2d_deepin_deepout_block_c(
    int8_t* Y,
    const nn_conv2d_dido_params_t* params,
    const nn_conv2d_dido_block_params_t* block,
    const int8_t* X,
    const int8_t* K,
    const data16_t* B,
    const int16_t* shifts,
    const int16_t* scales)
{

    X = &X[block->init.start_offset.X];
    Y = &Y[block->init.start_offset.Y];
    K = &K[block->init.start_offset.K];

    int32_t bias_adjustment = params->zero_point * block->init.padding_cells;

    
    //Iterate once per row of the output region
    for(unsigned output_rows = block->output.rows; output_rows > 0; output_rows--){

        //Iterate once per col of the output region
        for(unsigned output_cols = block->output.cols; output_cols > 0; output_cols--){

            const int8_t* L = K;
            const data16_t* D = B;
            const int16_t* hifts = shifts;
            const int16_t* cales = scales;
            
            for(unsigned cout_groups = params->C_out_groups; cout_groups > 0; cout_groups--){

                //V is the X-pointer for the patch. This points it at the top-left cell of the patch.
                const int8_t* V = X;
                
                //Because this mirrors the assembly, we need to keep accumulators for each
                // of the channels in an output channel
                int32_t patch_acc[VPU_INT8_ACC_PERIOD] = {0};

                for(int k = 0; k < VPU_INT8_ACC_PERIOD; k++){
                    int32_t bias = D[k];
                    bias = (bias << 16) | D[k + VPU_INT8_ACC_PERIOD];
                    patch_acc[k] = bias + bias_adjustment;
                }
                
                D = &D[2*VPU_INT8_ACC_PERIOD];

                for(unsigned rows_in_patch = block->patch.rows; rows_in_patch > 0; rows_in_patch--){

                    for(unsigned maccs = block->patch.maccs_per_row; maccs > 0; maccs--){

                        //This loop represents one "VLMACCR group"
                        for(int k = VPU_INT8_ACC_PERIOD - 1; k >= 0; k--){
                            
                            //This loop is a single VLMACCR
                            for(int i = 0; i < VPU_INT8_EPV; i++){
                                patch_acc[k] += ((int32_t)L[i]) * V[i];
                            }

                            L = &L[VPU_INT8_EPV];
                        }
                        V = &V[VPU_INT8_EPV];
                    }
                    
                    V = &V[block->patch.row_incr.X];
                    L = &L[block->patch.row_incr.K];
                }

                //Do the shifting and scaling
                for(int k = 0; k < VPU_INT8_ACC_PERIOD; k++){

                    int16_t res16 = vlsat_single_s16(patch_acc[k], hifts[k]);

                    res16 = vlmul_single_s16(res16, cales[k]);

                    int8_t res8 = vdepth8_single_s16(res16);

                    Y[k] = res8;
                }
                hifts = &hifts[VPU_INT8_ACC_PERIOD];
                cales = &cales[VPU_INT8_ACC_PERIOD];


                Y = &Y[VPU_INT8_ACC_PERIOD];
                L = &L[block->cout_group_incr.K];
            }

            X = &X[params->chans_in];

        }

        X = &X[block->output.row_incr.X];
        Y = &Y[block->output.row_incr.Y];

    }
}


void conv2d_shallowin_deepout_init(
    nn_conv2d_sido_params_t* params,
    const uint32_t X_height,
    const uint32_t X_width,
    const uint32_t K_h,
    const uint32_t K_w,
    const uint32_t C_in,
    const uint32_t C_out,
    const padding_mode_t pad_mode,
    const int8_t zero_point,
    const uint32_t region_top,
    const uint32_t region_left,
    const uint32_t region_rows,
    const uint32_t region_cols)
{
    const int P_h = K_h >> 1;
    const int P_w = K_w >> 1;

    uint32_t Y_height, Y_width;

    if(pad_mode == PADDING_VALID){
        Y_height = X_height - 2*P_h;
        Y_width  = X_width  - 2*P_w;
    } else {
        Y_height = X_height;
        Y_width  = X_width;
    }

    //Check parameters
    unsigned check_errors = 1;
    if(check_errors){

        unsigned error = 0;
        if(region_rows > Y_height)
            error = 1;
        if(region_cols > Y_width)
            error = 1;
        if(region_top >= Y_height)
            error = 1;
        if(region_top + region_rows > Y_height)
            error = 1;
        if(region_left >= Y_width)
            error = 1;
        if(region_left + region_cols > Y_width)
            error = 1;

        if(error){
            printf("Region parameters invalid.\n");
            __builtin_trap();
        }
    }
    
    params->chans_in = C_in;
    params->chans_out = C_out;
    params->C_in_groups = C_in >> VPU_INT8_EPV_LOG2;
    params->C_out_groups = C_out >> VPU_INT8_ACC_PERIOD_LOG2;
    params->zero_point = zero_point * C_in;

    if(pad_mode == PADDING_VALID){

        nn_conv2d_sido_block_params_t* blocks = malloc(sizeof(nn_conv2d_sido_block_params_t));

        if(!blocks){
            printf("Failed to allocation blocks.\n");
            __builtin_trap();
        }

        params->block_count = 1;
        params->blocks = blocks;

        nn_conv2d_sido_block_params_t* block = &blocks[0];

        uint32_t Y_start_top = region_top;
        uint32_t Y_start_left = region_left;

        uint32_t X_start_top = Y_start_top;
        uint32_t X_start_left = Y_start_left;

        
        block->init.start_offset.X = C_in * (X_start_top * X_width + X_start_left);
        block->init.start_offset.Y = C_out * (Y_start_top * Y_width + Y_start_left);
        block->init.start_offset.K = 0;
        block->init.padding_cells = 0;

        // block->patch.maccs_per_row = K_w * params->C_in_groups;
        block->patch.pad_mask = 0xFFFFFFFF;
        block->patch.rows = K_h;
        block->patch.row_incr.X = C_in * X_width;

        //This needs to be enough to pull K to the beginning of the current row, then down a row
        block->patch.row_incr.K = 0;

        //This needs to be enough to push K to the end of the current channel group, plus the start offset
        block->cout_group_incr.K = 0;
        block->output.rows = region_rows;
        block->output.cols = region_cols;
        block->output.row_incr.X = C_in * (X_width - block->output.cols);
        block->output.row_incr.Y = C_out * (Y_width - block->output.cols);
    } else {

        unsigned block_count = 0;

        //First, determine which blocks have any work to be done
        for(int kr = 0; kr < K_h; kr++){
            for(int kc = 0; kc < K_w; kc++){
                uint32_t aa, bb, cc, dd;
                if(conv2d_block_output_bounds(&aa, &bb, &cc, &dd,
                                              kr, kc, Y_height, Y_width, K_h, K_w,
                                              region_top, region_left, region_rows, region_cols))
                    block_count++;
            }
        }

        //Now, allocate the memory
        nn_conv2d_sido_block_params_t* blocks = malloc(block_count * sizeof(nn_conv2d_sido_block_params_t));

        if(!blocks){
            printf("Failed to allocation blocks.\n");
            __builtin_trap();
        }

        params->block_count = block_count;
        params->blocks = blocks;

        //Finally, actually initialize the block parameters

        unsigned block_dex = 0;
        for(int kr = 0; kr < K_h; kr++){
            for(int kc = 0; kc < K_w; kc++){

                nn_conv2d_sido_block_params_t* block = &blocks[block_dex];

                uint32_t Y_top, Y_left, Y_bottom, Y_right;

                if(!conv2d_block_output_bounds(&Y_top, &Y_left, &Y_bottom, &Y_right,
                                               kr, kc, Y_height, Y_width, K_h, K_w,
                                               region_top, region_left, region_rows, region_cols))
                    continue;
                
                block_dex++;

                unsigned pad_t = (kr  > P_h)?  0 : P_h - kr;
                unsigned pad_b = (kr <= P_h)?  0 : P_h - (K_h-1-kr);
                unsigned pad_l = (kc  > P_w)?  0 : P_w - kc; 
                unsigned pad_r = (kc <= P_w)?  0 : P_w - (K_w-1-kc);
                
                unsigned out_rows = Y_bottom - Y_top;
                unsigned out_cols = Y_right  - Y_left;

                // printf("\t (t, l)\t(rows x cols) = (%u, %u)\t(%ux%u)\n", Y_top, Y_left, out_rows, out_cols);

                unsigned X_top    = (Y_top  <= P_h)? 0 : Y_top  - P_h;
                unsigned X_left   = (Y_left <= P_w)? 0 : Y_left - P_w;
                
                unsigned patch_rows = K_h - pad_t - pad_b;
                // unsigned patch_cols = K_w - pad_l - pad_r;


                block->init.start_offset.X = C_in * (X_left + X_top * X_width);
                block->init.start_offset.Y = C_out * (Y_left + Y_top * Y_width);
                //Always start Kernel from left edge of a row. The padmask will take care of it.
                block->init.start_offset.K = VPU_INT8_ACC_PERIOD * VPU_INT8_EPV * pad_t;
                block->init.padding_cells = (pad_t + pad_b) * K_w + (pad_l + pad_r) * patch_rows;

                uint32_t padding_mask = (pad_r == 0)? 0xFFFFFFFF : ((((uint32_t)1)<<(32-(C_in*pad_r)))-1);
                padding_mask = padding_mask ^ ((1<<(C_in*pad_l))-1);
                block->patch.pad_mask = padding_mask;
                block->patch.rows = patch_rows;
                //One row is a single VLMACCR group
                block->patch.row_incr.X = C_in * X_width;

                //This needs to be enough to pull K to the beginning of the current row, then down a row
                block->patch.row_incr.K = 0;

                //This needs to be enough to push K to the end of the current channel group, plus the start offset
                block->cout_group_incr.K = (pad_r + 8 * pad_b) * (C_in * VPU_INT8_ACC_PERIOD) + block->init.start_offset.K;
                block->output.rows = out_rows;
                block->output.cols = out_cols;
                block->output.row_incr.X = C_in * (X_width - block->output.cols);
                block->output.row_incr.Y = C_out * (Y_width - block->output.cols);
            }
        }
    }

}

void conv2d_shallowin_deepout_block_c(
    int8_t* Y,
    const nn_conv2d_sido_params_t* params,
    const nn_conv2d_sido_block_params_t* block,
    const int8_t* X,
    const int8_t* K,
    const data16_t* B,
    const int16_t* shifts,
    const int16_t* scales)
{

    X = &X[block->init.start_offset.X];
    Y = &Y[block->init.start_offset.Y];
    K = &K[block->init.start_offset.K];

    int32_t bias_adjustment = params->zero_point * block->init.padding_cells;

    
    //Iterate once per row of the output region
    for(unsigned output_rows = block->output.rows; output_rows > 0; output_rows--){

        //Iterate once per col of the output region
        for(unsigned output_cols = block->output.cols; output_cols > 0; output_cols--){

            const int8_t* L = K;
            const data16_t* D = B;
            const int16_t* hifts = shifts;
            const int16_t* cales = scales;
            
            for(unsigned cout_groups = params->C_out_groups; cout_groups > 0; cout_groups--){

                //V is the X-pointer for the patch. This points it at the top-left cell of the patch.
                const int8_t* V = X;
                
                //Because this mirrors the assembly, we need to keep accumulators for each
                // of the channels in an output channel
                int32_t patch_acc[VPU_INT8_ACC_PERIOD] = {0};

                for(int k = 0; k < VPU_INT8_ACC_PERIOD; k++){
                    int32_t bias = D[k];
                    bias = (bias << 16) | D[k + VPU_INT8_ACC_PERIOD];
                    patch_acc[k] = bias + bias_adjustment;
                }
                
                D = &D[2*VPU_INT8_ACC_PERIOD];

                for(unsigned rows_in_patch = block->patch.rows; rows_in_patch > 0; rows_in_patch--){

                    if( block->patch.pad_mask == 0xFFFFFFFF ){

                        //This loop represents one "VLMACCR group"
                        for(int k = VPU_INT8_ACC_PERIOD - 1; k >= 0; k--){
                            
                            //This loop is a single VLMACCR
                            for(int i = 0; i < VPU_INT8_EPV; i++){
                                // printf("%d ", L[i]);
                                patch_acc[k] += ((int32_t)L[i]) * V[i];
                            }

                            //Move to the next output channel's coefficients for the same patch row
                            L = &L[VPU_INT8_EPV];
                        }
                        
                        //Move to the start of the next patch row
                        V = &V[block->patch.row_incr.X];
                        L = &L[block->patch.row_incr.K];

                    } else {
                        int8_t padded_vals[VPU_INT8_EPV];

                        for(int i = 0; i < VPU_INT8_EPV; i++){
                            unsigned tmp = ((block->patch.pad_mask >> i) & 0x1);
                            padded_vals[i] = tmp? V[i] : 0;
                        }   

                        //This loop represents one "VLMACCR group"
                        for(int k = VPU_INT8_ACC_PERIOD - 1; k >= 0; k--){
                            
                            //This loop is a single VLMACCR
                            for(int i = 0; i < VPU_INT8_EPV; i++){
                                patch_acc[k] += ((int32_t)L[i]) * padded_vals[i];
                            }

                            //Move to the next output channel's coefficients for the same patch row
                            L = &L[VPU_INT8_EPV];
                        }
                        
                        //Move to the start of the next patch row
                        V = &V[block->patch.row_incr.X];
                        L = &L[block->patch.row_incr.K];

                    }
                }

                //Do the shifting and scaling
                for(int k = 0; k < VPU_INT8_ACC_PERIOD; k++){

                    // printf("Bleh %ld: 0x%04X\n", k, patch_acc[k]);

                    int16_t res16 = vlsat_single_s16(patch_acc[k], hifts[k]);

                    res16 = vlmul_single_s16(res16, cales[k]);

                    int8_t res8 = vdepth8_single_s16(res16);

                    Y[k] = res8;
                }
                hifts = &hifts[VPU_INT8_ACC_PERIOD];
                cales = &cales[VPU_INT8_ACC_PERIOD];

                //Move Y pointer to the next set of output channels (or the next output pixel)
                Y = &Y[VPU_INT8_ACC_PERIOD];

                //Move K pointer to the start of the next output channel group
                L = &L[block->cout_group_incr.K];
            }

            //Move X pointer one pixel to the right
            X = &X[params->chans_in];

        }

        //Move X and Y pointers to the start of the following row
        X = &X[block->output.row_incr.X];
        Y = &Y[block->output.row_incr.Y];

    }
}


