

#include "nn_operator.h"
#include "../nn_op_helper.h"
#include "nn_op_structs.h"

#include "xs3_vpu.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>



#ifndef TRUE
#define TRUE (1)
#endif
#ifndef FALSE
#define FALSE (0)
#endif








////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
/*
    
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
static inline unsigned img_pxl_offset(
    const unsigned img_width,
    const unsigned channels,
    const unsigned pixel_row,
    const unsigned pixel_col)
{
    const unsigned bytes_per_row = channels * img_width;
    return bytes_per_row * pixel_row + channels * pixel_col;
}








// ////////////////////////////////////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////////////////////////////////////
// /*
//     Get the offset of a coefficient in the BOGGLED shallowin-deepout kernel tensor.
// */
// ///////////////////////////////////////////////////////////////////////////////////////////////////
// ///////////////////////////////////////////////////////////////////////////////////////////////////
// static inline unsigned sido_coef_offset(
//     const nn_conv2d_init_params_t* init,
//     const unsigned cout,
//     const unsigned kernel_row,
//     const unsigned kernel_col,
//     const unsigned cin)
// {
//     //boggled sido kernel tensor has shape:   (C_out/16,  K_h,  16,  32/C_in,  C_in)

//     const unsigned cog = cout >> VPU_INT8_ACC_PERIOD_LOG2;
//     const unsigned cof = (VPU_INT8_ACC_PERIOD - 1) - (cout % VPU_INT8_ACC_PERIOD);

//     return  cog * (init->K_h * VPU_INT8_ACC_PERIOD * VPU_INT8_EPV)
//           + kernel_row * (VPU_INT8_ACC_PERIOD * VPU_INT8_EPV)
//           + cof * (VPU_INT8_EPV)
//           + kernel_col * (C_in)
//           + cin;
// }















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

            data16_t hi = (val & 0xFFFF0000) >> VPU_INT8_ACC_VR_BITS;
            data16_t lo = (val & 0x0000FFFF) >>  0;

            B_hi[k] = hi;
            B_lo[k] = lo;
        }

    }

    return (data16_t*) B;
}

/**
 * Re-layout the shift-scale tensor to the format expected by the convolution kernels.
 * 
 * The input tensor should contain all of the shifts followed by all of the scales, in
 * channel order. 
 *
 * A scratch buffer parameter may optionally be supplied (same size as `shiftscales`).
 * If `scratch` is `NULL`, a buffer will be `malloc`ed (and `free`ed).
 *
 * \param shiftscales   The shift/scale tensor. Updated in-place
 * \param C_out         The number of output channels
 * \param scratch       Optional scratch buffer.
 */
void conv2d_boggle_shift_scale(
    int16_t* shiftscales,
    const unsigned C_out,
    int16_t* scratch)
{
    assert((C_out & ((1 << VPU_INT8_ACC_PERIOD_LOG2)-1)) == 0);

    const unsigned C_out_groups = C_out >> VPU_INT8_ACC_PERIOD_LOG2;

    unsigned malloced = 0;
    if(scratch == NULL){
        scratch = malloc(2 * C_out * sizeof(int16_t));

        if(scratch == NULL){
            printf("Failed to allocate scratch buffer.\n");
            __builtin_trap();
        }

        malloced = 1;
    }
    
    memcpy(scratch, shiftscales, 2*C_out*sizeof(int16_t));

    for(int cog = 0; cog < C_out_groups; cog++){
        const unsigned out_shifts = cog * 2*VPU_INT8_ACC_PERIOD;
        const unsigned out_scales = out_shifts + VPU_INT8_ACC_PERIOD;
        const unsigned in_shifts = cog * VPU_INT8_ACC_PERIOD;
        const unsigned in_scales = in_shifts + C_out;

        memcpy(&shiftscales[out_shifts],&scratch[in_shifts], VPU_INT8_ACC_PERIOD * sizeof(int16_t));
        memcpy(&shiftscales[out_scales],&scratch[in_scales], VPU_INT8_ACC_PERIOD * sizeof(int16_t)); 
    }

    if(malloced)
        free(scratch);
}









////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
/*
    
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
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
                for(int kc = 0; kc < VPU_INT8_EPV/C_in; kc++){

                    unsigned in_b = cout * K_h * VPU_INT8_EPV
                                  + kr * VPU_INT8_EPV
                                  + kc * C_in;
                    
                    unsigned out_b = cog * K_h * VPU_INT8_ACC_PERIOD * VPU_INT8_EPV
                                   + kr * VPU_INT8_ACC_PERIOD * VPU_INT8_EPV
                                   + rcout * VPU_INT8_EPV
                                   + kc * C_in;

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








////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
/*
    
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
static int32_t conv2d_sido_coef_sum(
    const int8_t* K,
    const nn_conv2d_init_params_t* init,
    const unsigned chan_out,
    const unsigned row,
    const unsigned col)
{
    // The shape of K is:   int8_t K[C_out/16][K_h][16][32/C_in][C_in]  
    //      ->  (Channel out group, kernel row, out channel offset (reversed), kernel col (padded out), channel in)

    const unsigned cog = chan_out / VPU_INT8_ACC_PERIOD;
    const unsigned cout = (VPU_INT8_ACC_PERIOD - 1) - (chan_out % VPU_INT8_ACC_PERIOD);

    const unsigned t3a = cog  * (init->K_h * VPU_INT8_ACC_PERIOD * VPU_INT8_EPV)
                       + row  * ( VPU_INT8_ACC_PERIOD * VPU_INT8_EPV)
                       + cout * (VPU_INT8_EPV)
                       + col  * (init->C_in);

    const int8_t* K_tmp = &K[t3a];
    
    int32_t acc = 0;

    for(int cin = 0; cin < init->C_in; cin++){
        acc += K_tmp[cin];
    }

    return acc;
}








////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
/*
    
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
static data16_t* conv2d_adjusted_biases(
    const nn_conv2d_init_params_t* init,
    const unsigned pad_t,
    const unsigned pad_l,
    const unsigned pad_b,
    const unsigned pad_r,
    const int8_t* K,
    const data16_t* B,
    const unsigned is_deepin)
{
    const unsigned C_out_groups = init->C_out >> VPU_INT8_ACC_PERIOD_LOG2;

    //allocate space for the block's bias tensor
    data16_t* biases = malloc(2 * sizeof(data16_t) * init->C_out);

    if(!biases){
        printf("Failed to allocate biases for block.\n");
        __builtin_trap();
    }

    for(unsigned cog = 0; cog < C_out_groups; cog++){

        const data16_t* B_cog_hi = &B[cog * 2 * VPU_INT8_ACC_PERIOD];
        const data16_t* B_cog_lo = &B[((cog * 2) + 1) * VPU_INT8_ACC_PERIOD];

        data16_t* B_out_hi = &biases[cog * 2 * VPU_INT8_ACC_PERIOD];
        data16_t* B_out_lo = &biases[((cog * 2) + 1) * VPU_INT8_ACC_PERIOD];

        for(unsigned ofst = 0; ofst < VPU_INT8_ACC_PERIOD; ofst++){
            unsigned cout = VPU_INT8_ACC_PERIOD * cog + ofst;

            //initialize bias to the actual bias
            int32_t bias = (B_cog_hi[ofst] << VPU_INT8_ACC_VR_BITS) | B_cog_lo[ofst];

            //iterate over kernel cells and determine whether each is in padding.
            for(unsigned gr = 0; gr < init->K_h; gr++){
                for(unsigned gc = 0; gc < init->K_w; gc++){
                    //Is this kernel cell in the padding?
                    if(!   ((gr < pad_t)
                        || (gc < pad_l)
                        || (gr >= (init->K_h - pad_b) )
                        || (gc >= (init->K_w - pad_r) )))
                        continue;

                    //It is in the padding
                    bias += init->zero_point * conv2d_sido_coef_sum( K, init, cout, gr, gc );
                }
            }

            B_out_hi[ofst] = bias >> VPU_INT8_ACC_VR_BITS;
            B_out_lo[ofst] = bias & 0xFFFF;
        }
    }

    return biases;
}








////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
/*
    
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
static inline unsigned conv2d_block_output_bounds(
    uint32_t* Y_t,
    uint32_t* Y_l,
    uint32_t* Y_b,
    uint32_t* Y_r,
    const unsigned kr,
    const unsigned kc,
    const uint32_t Y_height,
    const uint32_t Y_width,
    const nn_conv2d_init_params_t* init,
    const nn_conv2d_region_params_t* region)
{
    const int P_h = init->K_h >> 1;
    const int P_w = init->K_w >> 1;

    unsigned reg_bot = region->top + region->rows;
    unsigned reg_rig = region->left + region->cols;
    
    *Y_t  = (kr <= P_h)? kr : Y_height - init->K_h + kr;
    *Y_l  = (kc <= P_w)? kc : Y_width  - init->K_w + kc;
    *Y_b  = *Y_t + (  (kr == P_h)? Y_height - 2 * P_h : 1  );
    *Y_r  = *Y_l + (  (kc == P_w)? Y_width  - 2 * P_w : 1  );

    if(*Y_t < region->top)
        *Y_t = region->top;
    else if(*Y_t > reg_bot)
        *Y_t = reg_bot;

    if(*Y_b < region->top)
        *Y_b = region->top;
    else if(*Y_b > reg_bot)
        *Y_b = reg_bot; 

    if(*Y_l < region->left)
        *Y_l = region->left;
    else if(*Y_l > reg_rig)
        *Y_l = reg_rig;
    
    if(*Y_r < region->left)
        *Y_r = region->left;
    else if(*Y_r > reg_rig)
        *Y_r = reg_rig;

    return (*Y_b > *Y_t && *Y_r > *Y_l);
}








////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
/*
    
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
static void conv2d_check_params_for_errors(
    const uint32_t Y_height,
    const uint32_t Y_width,
    const nn_conv2d_init_params_t* init,
    const nn_conv2d_region_params_t* region)
{
    unsigned error = 0;

    if(init->C_out % VPU_INT8_ACC_PERIOD != 0){
        printf("C_out must be a multiple of VPU_INT8_ACC_PERIOD (%u).\n", VPU_INT8_ACC_PERIOD);
        __builtin_trap();
    }

    if(region->rows > Y_height)
        error = 1;
    
    if(region->cols > Y_width)
        error = 1;
    
    if(region->top >= Y_height)
        error = 1;
    
    if(region->top + region->rows > Y_height)
        error = 1;
    
    if(region->left >= Y_width)
        error = 1;
    
    if(region->left + region->cols > Y_width)
        error = 1;

    if(error){
        printf("Region parameters invalid.\n");
        __builtin_trap();
    }
}















////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
/*
    
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

static void conv2d_shallowin_deepout_init_valid(
    nn_conv2d_sido_params_t* params,
    const nn_conv2d_init_params_t* init,
    const nn_conv2d_region_params_t* region,
    const int8_t* K,
    const data16_t* B)
{
    const int P_h = init->K_h >> 1;
    const int P_w = init->K_w >> 1;

    const uint32_t Y_height = init->X_height - 2 * P_h;
    const uint32_t Y_width = init->X_width - 2 * P_w;

    const nn_conv2d_region_params_t def_region = {0, 0, Y_height, Y_width};

    if(region == NULL)
        region = &def_region;

    conv2d_check_params_for_errors(Y_height, Y_width, init, region);

    
    params->chans_in = init->C_in;
    params->chans_out = init->C_out;
    params->C_in_groups = init->C_in >> VPU_INT8_EPV_LOG2;
    params->C_out_groups = init->C_out >> VPU_INT8_ACC_PERIOD_LOG2;
    params->zero_point = init->zero_point;

    nn_conv2d_sido_block_params_t* blocks = malloc(sizeof(nn_conv2d_sido_block_params_t));

    if(!blocks){
        printf("Failed to allocation blocks.\n");
        __builtin_trap();
    }

    params->block_count = 1;
    params->blocks = blocks;

    nn_conv2d_sido_block_params_t* block = &blocks[0];

    uint32_t Y_start_top = region->top;
    uint32_t Y_start_left = region->left;

    uint32_t X_start_top = Y_start_top;
    uint32_t X_start_left = Y_start_left;

    
    block->init.start_offset.X = img_pxl_offset(init->X_width, init->C_in, X_start_top, X_start_left);
    block->init.start_offset.Y = img_pxl_offset(Y_width, init->C_out, Y_start_top, Y_start_left);
    block->init.start_offset.K = 0;

    block->init.biases = conv2d_adjusted_biases(init, 0, 0, 0, 0, K, B, FALSE);

    // block->patch.maccs_per_row = K_w * params->C_in_groups;
    block->patch.pad_mask = 0xFFFFFFFF;
    block->patch.rows = init->K_h;
    block->patch.row_incr.X = init->C_in * init->X_width;

    //This needs to be enough to pull K to the beginning of the current row, then down a row
    block->patch.row_incr.K = 0;

    //This needs to be enough to push K to the end of the current channel group, plus the start offset
    block->cout_group_incr.K = 0;
    block->output.rows = region->rows;
    block->output.cols = region->cols;
    block->output.row_incr.X = init->C_in * (init->X_width - block->output.cols);
    block->output.row_incr.Y = init->C_out * (Y_width - block->output.cols);
}









////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
/*
    
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

static void conv2d_shallowin_deepout_init_same(
    nn_conv2d_sido_params_t* params,
    const nn_conv2d_init_params_t* init,
    const nn_conv2d_region_params_t* region,
    const int8_t* K,
    const data16_t* B)
{
    
    const int P_h = init->K_h >> 1;
    const int P_w = init->K_w >> 1;

    const uint32_t Y_height = init->X_height;
    const uint32_t Y_width = init->X_width;

    const nn_conv2d_region_params_t def_region = {0, 0, Y_height, Y_width};

    if(region == NULL)
        region = &def_region;

    conv2d_check_params_for_errors(Y_height, Y_width, init, region);
    
    params->chans_in = init->C_in;
    params->chans_out = init->C_out;
    params->C_in_groups = init->C_in >> VPU_INT8_EPV_LOG2;
    params->C_out_groups = init->C_out >> VPU_INT8_ACC_PERIOD_LOG2;
    params->zero_point = init->zero_point;


    unsigned block_count = 0;

    //First, determine which blocks have any work to be done
    for(int kr = 0; kr < init->K_h; kr++){
        for(int kc = 0; kc < init->K_w; kc++){
            uint32_t aa, bb, cc, dd;
            if(conv2d_block_output_bounds(&aa, &bb, &cc, &dd,
                                            kr, kc, Y_height, Y_width, init, region))
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
    for(int kr = 0; kr < init->K_h; kr++){
        for(int kc = 0; kc < init->K_w; kc++){

            nn_conv2d_sido_block_params_t* block = &blocks[block_dex];

            uint32_t Y_top, Y_left, Y_bottom, Y_right;

            if(!conv2d_block_output_bounds(&Y_top, &Y_left, &Y_bottom, &Y_right,
                                            kr, kc, Y_height, Y_width, init, region))
                continue;
            
            block_dex++;

            unsigned pad_t = (kr  > P_h)?  0 : P_h - kr;
            unsigned pad_b = (kr <= P_h)?  0 : P_h - (init->K_h-1-kr);
            unsigned pad_l = (kc  > P_w)?  0 : P_w - kc; 
            unsigned pad_r = (kc <= P_w)?  0 : P_w - (init->K_w-1-kc);
            
            unsigned out_rows = Y_bottom - Y_top;
            unsigned out_cols = Y_right  - Y_left;

            // printf("\t (t, l)\t(rows x cols) = (%u, %u)\t(%ux%u)\n", Y_top, Y_left, out_rows, out_cols);

            unsigned X_top    = (Y_top  <= P_h)? 0 : Y_top  - P_h;
            // unsigned X_left   = (Y_left <= P_w)? 0 : Y_left - P_w;
            unsigned X_left   = Y_left - P_w;
            
            unsigned patch_rows = init->K_h - pad_t - pad_b;
            // unsigned patch_cols = K_w - pad_l - pad_r;


            block->init.start_offset.X = img_pxl_offset(init->X_width, init->C_in, X_top, X_left);
            block->init.start_offset.Y = img_pxl_offset(Y_width, init->C_out, Y_top, Y_left);
            //Always start Kernel from left edge of a row. The padmask will take care of it.
            block->init.start_offset.K = VPU_INT8_ACC_PERIOD * VPU_INT8_EPV * pad_t;
            // block->init.padding_cells = (pad_t + pad_b) * init->K_w + (pad_l + pad_r) * patch_rows;

            uint32_t padding_mask = 0;
            // uint32_t padding_mask = (pad_r == 0)? 0xFFFFFFFF : ((((uint32_t)1)<<(32-(init->C_in*pad_r)))-1);
            // padding_mask = padding_mask ^ ((1<<(init->C_in*pad_l))-1);

            // if(pad_r == 0 && pad_l == 0)
            //     padding_mask = 0xFFFFFFFF;
            // else {

            //     padding_mask = 0;

            for(int i = VPU_INT8_EPV/init->C_in - 1; i >= 0; i--){

                //Determine whether this kernel cell is in padding
                unsigned in_img = 1;

                if(i >= init->K_w)
                    //Beyond K_w, the coefficients are supposed to be zero,
                    //  so we'll say it's not in padding, which allows us
                    //  to speed up processing in the interior region
                    in_img = 1;
                else if(i < pad_l) {
                    in_img = 0;
                } else if(init->K_w-i-1 < pad_r) {
                    in_img = 0;
                }

                for(int k = 0; k < init->C_in; k++){
                    padding_mask = padding_mask << 1;
                    padding_mask |= (in_img!=0);
                }
            }
            // }

            block->patch.pad_mask = padding_mask;
            block->patch.rows = patch_rows;
            //One row is a single VLMACCR group
            block->patch.row_incr.X = init->C_in * init->X_width;

            //This needs to be enough to pull K to the beginning of the current row, then down a row
            block->patch.row_incr.K = 0;

            //This needs to be enough to push K to the end of the current channel group, plus the start offset
            block->cout_group_incr.K = (8 * pad_b) * (init->C_in * VPU_INT8_ACC_PERIOD) + block->init.start_offset.K;
            block->output.rows = out_rows;
            block->output.cols = out_cols;
            block->output.row_incr.X = init->C_in * (init->X_width - block->output.cols);
            block->output.row_incr.Y = init->C_out * (Y_width - block->output.cols);

            block->init.biases = conv2d_adjusted_biases(init, pad_t, pad_l, pad_b, pad_r, K, B, FALSE);


            // printf("block_dex = %u\n", block_dex-1);
            // printf("pad_t = %u\n", pad_t);
            // printf("pad_l = %u\n", pad_l);
            // printf("pad_b = %u\n", pad_b);
            // printf("pad_r = %u\n", pad_r);
            // printf("out_rows = %u\n", out_rows);
            // printf("out_cols = %u\n", out_cols);
            // printf("X_top = %d\n", X_top);
            // printf("X_left = %d\n", X_left);
            // printf("block->init.start_offset.X = %ld\n", block->init.start_offset.X);
            // printf("block->init.start_offset.Y = %ld\n", block->init.start_offset.Y);
            // printf("block->init.start_offset.K = %ld\n", block->init.start_offset.K);
            // printf("block->patch.pad_mask = 0x%08X\n", (unsigned) block->patch.pad_mask);
            // printf("block->patch.rows = %u\n", block->patch.rows);
            // printf("block->patch.row_incr.X = %ld\n", block->patch.row_incr.X);
            // printf("block->patch.row_incr.K = %ld\n", block->patch.row_incr.K);
            // printf("block->cout_group_incr.K = %ld\n", block->cout_group_incr.K);
            // printf("block->output.rows = %u\n", block->output.rows);
            // printf("block->output.cols = %u\n", block->output.cols);
            // printf("block->output.row_incr.X = %ld\n", block->output.row_incr.X);
            // printf("block->output.row_incr.Y = %ld\n", block->output.row_incr.Y);
            // printf("\n\n\n");
            
        }
    }

}








////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
/*
    
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
void conv2d_shallowin_deepout_init(
    nn_conv2d_sido_params_t* params,
    const nn_conv2d_init_params_t* init_params,
    const nn_conv2d_region_params_t* region_params,
    const int8_t* K,
    const data16_t* B)
{
    //Check parameters
    if(init_params->C_in % sizeof(int) != 0){
        printf("C_in must be a multiple of 4 bytes.\n");
        __builtin_trap();
    }

    if(init_params->C_in * init_params->K_w > VPU_INT8_EPV){
        printf("C_in * K_w cannot exceed VPU_INT8_EPV (%u).\n", VPU_INT8_EPV);
        __builtin_trap();
    }

    if(init_params->pad_mode == PADDING_VALID){
        conv2d_shallowin_deepout_init_valid(params, init_params, region_params, K, B);
    } else {
        conv2d_shallowin_deepout_init_same(params, init_params, region_params, K, B);
    }
}








////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
/*
    
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
void conv2d_shallowin_deepout_deinit(
    nn_conv2d_sido_params_t* params)
{
    if(params->blocks){

        for(unsigned i = 0; i < params->block_count; i++){
            if(params->blocks[i].init.biases){
                free(params->blocks[i].init.biases);
            }
        }

        free(params->blocks);
    }
}








////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
/*
    
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
void conv2d_shallowin_deepout_block_c(
    int8_t* Y,
    const nn_conv2d_sido_params_t* params,
    const nn_conv2d_sido_block_params_t* block,
    const int8_t* X,
    const int8_t* K,
    const int16_t* scales)
{
    X = &X[block->init.start_offset.X];
    Y = &Y[block->init.start_offset.Y];
    K = &K[block->init.start_offset.K];

    const data16_t* B = block->init.biases;
    
    //Iterate once per row of the output region
    for(unsigned output_rows = block->output.rows; output_rows > 0; output_rows--){

        //Iterate once per col of the output region
        for(unsigned output_cols = block->output.cols; output_cols > 0; output_cols--){

            const int8_t* L = K;
            const data16_t* D = B;
            const int16_t* cales = scales;
            
            for(unsigned cout_groups = params->C_out_groups; cout_groups > 0; cout_groups--){
                // const unsigned coutgroup = params->C_out_groups - cout_groups;

                //V is the X-pointer for the patch. This points it at the top-left cell of the patch.
                const int8_t* V = X;
                
                //Because this mirrors the assembly, we need to keep accumulators for each
                // of the channels in an output channel
                int32_t patch_acc[VPU_INT8_ACC_PERIOD] = {0};

                for(int k = 0; k < VPU_INT8_ACC_PERIOD; k++){
                    int32_t bias = D[k];
                    bias = (bias << VPU_INT8_ACC_VR_BITS) | D[k + VPU_INT8_ACC_PERIOD];
                    patch_acc[k] = bias;
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

                    int16_t res16 = vlsat_single_s16(patch_acc[k], cales[k]);

                    res16 = vlmul_single_s16(res16, cales[k+VPU_INT8_ACC_PERIOD]);

                    int8_t res8 = vdepth8_single_s16(res16);

                    Y[k] = res8;
                }
                cales = &cales[2*VPU_INT8_ACC_PERIOD];

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






