

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
    for use with conv2d_deepin_deepout_relu().

    Returns B, cast to a data16_t pointer.
*/
data16_t* conv2d_dido_boggle_B(
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
    required for conv2d_deepin_deepout_relu().

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

    //This can be done without malloc(), but it's way more complicated.


    const unsigned C_in_groups = C_in >> VPU_INT8_EPV_LOG2;
    const unsigned C_out_groups = C_out >> VPU_INT8_ACC_PERIOD_LOG2;

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



    // ///////TODO: Would like to do it without malloc(), but it isn't immediately obvious how


    // //First, go ahead and apply the reversal, which happens within each C_out_group
    // if(1) {

    //     //The size of a single output channel slice in blocks
    //     const unsigned c_out_slice_blks = K_h * K_w * C_in >> VPU_INT8_EPV_LOG2;
        
    //     //The size of an output channel group slice in blocks
    //     const unsigned c_out_group_slice_blks = c_out_slice_blks * VPU_INT8_ACC_PERIOD;

    //     for(int cog = 0; cog < C_out_groups; cog++){

    //         block_t* K_cog = &K_blocks[cog * c_out_group_slice_blks];

    //         //The reversal is just a series of swaps, from the outside in
    //         for(int co_lo = 0; co_lo < (VPU_INT8_ACC_PERIOD / 2); co_lo++){


    //             //The two indices (chan % 16)  to be swapped are co_lo and co_hi
    //             int co_hi = (VPU_INT8_ACC_PERIOD - 1) - co_lo;

    //             block_t* K_lo = &K_cog[co_lo * c_out_slice_blks];
    //             block_t* K_hi = &K_cog[co_hi * c_out_slice_blks];

    //             unsigned swaps_remaining = c_out_slice_blks;
    //             while(swaps_remaining--){
    //                 block_t swap_buff = *K_hi;
    //                 *K_hi = *K_lo;
    //                 *K_lo = swap_buff;
    //                 K_lo = &K_lo[1];
    //                 K_hi = &K_hi[1];
    //             }
    //         }
    //     }
    // }

    // //Now, K has the shape of  (C_out_groups, 16, K_h, K_w, C_in_groups, 32), and we need to
    // //  transpose the axes so that its shape is  (C_out_groups, K_h, K_w, C_in_groups, 16, 32)

    // for(int cog = 0; cog < C_out_groups; cog++){

    //     block_t* K_cog_dst = &Kb[cog * K_h * K_w * C_in_groups * VPU_INT8_ACC_PERIOD];
    //     block_t* K_cog_src = K_cog_dst;

    //     for(int kr = 0; kr < K_h; kr++){
    //         for(int kc = 0; kc < K_w; kc++){
    //             for(int cig = 0; cig < C_in_groups; cig++){
    //                 for(int co = 0; co < VPU_INT8_ACC_PERIOD){

    //                     unsigned src_elm = co * (K_h*K_w*C_in_groups) + kr * (K_w*C_in_groups) + kc * (C_in_groups) + cig;
    //                     block_t* src = &K_cog_src[ src_elm ];

    //                     while(src < K_cog_dst){
    //                         //The target element has already been swapped to the source location
    //                         //  if we can find where it was moved to, we can just follow where it went
    //                         //  until we hit a memory location after K_cog_dst. The problem is that this
    //                         //  involves a lot of division and potentially many iterations of the while loop.
    //                         //   Is there an efficient way to do this?
    //                     }

    //                     K_cog_dst = &K_cog_dst[1];
    //                 }
    //             }
    //         }
    //     }

    // }


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

void conv2d_deepin_deepout_relu_init(
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

void conv2d_deepin_deepout_relu_block(
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


void conv2d_deepin_deepout_relu_c2(
    const int8_t* K, 
    const data16_t* B,
    const int8_t* X, 
    int8_t* Y,
    const int16_t* shifts, 
    const int16_t* scales,
    const nn_conv2d_dido_params_t* params)
{
    for(int block = 0; block < params->block_count; block++){
        conv2d_deepin_deepout_relu_block(
            Y, 
            params,
            &params->blocks[block],
            X, K, B, shifts, scales);

    }
}




void conv2d_deepin_deepout_relu_c(
    const int8_t* K, 
    const data16_t* B,
    const int8_t* X, 
    int8_t* Y,
    const int32_t height, 
    const int32_t width,
    const int32_t K_h, 
    const int32_t K_w,
    const int32_t C_out, 
    const int32_t C_in,
    const int16_t* shifts, 
    const int16_t* scales)
{
    const int P_h = K_h / 2;
    const int P_w = K_w / 2;
    const int Q_h = P_h - height + 1;
    const int Q_w = P_w - width  + 1;

    for(int chout = 0; chout < C_out; chout++){

        const int cog = chout / VPU_INT8_ACC_PERIOD;
        const int cog_offset = chout % VPU_INT8_ACC_PERIOD;
        
        const int16_t shr = shifts[chout];
        const int16_t scale = scales[chout];

        for(int row = 0; row < height; row++){

            const int pad_top = ((P_h-row) > 0)?  (P_h-row) : 0;
            const int pad_bot = ((Q_h+row) > 0)?  (Q_h+row) : 0;

            for(int col = 0; col < width; col++){

                const int pad_lef = ((P_w-col) > 0)?  (P_w-col) : 0;
                const int pad_rig = ((Q_w+col) > 0)?  (Q_w+col) : 0;

                const int B_offset   = 2 * C_out * (K_w * (P_h + pad_bot - pad_top) + (P_w + pad_rig - pad_lef));
                // printf("(%d, %d) -> (%d, %d, %d, %d)\n", row, col, pad_top, pad_bot, pad_lef, pad_rig);
                // printf("(%d, %d) -> %d\n", row, col, B_offset/(2*C_out));

                const data16_t* B_lo = &B[B_offset + 0];
                const data16_t* B_hi = &B[B_offset + C_out];
                const int32_t bias   = (((int32_t)B_hi[chout])<<16) | B_lo[chout];
                // printf("(%d, %d) -> %ld\n", row, col, bias);

                int32_t acc32 = bias;

                for(int kr = -P_h; kr <= P_h; kr++){
                    for(int kc = -P_w; kc <= P_w; kc++){

                        //check if we're in padding:
                        if(row+kr < 0 || row+kr >= height)
                            continue;
                        if(col+kc < 0 || col+kc >= width)
                            continue;

                        const int kkrr = kr + P_h;
                        const int kkcc = kc + P_w;

                        // int64_t acc64 = acc32;

                        //iterate through input channels
                        for(int chin = 0; chin < C_in; chin++){

                            const int cig = chin / VPU_INT8_EPV;
                            const int cig_offset = chin % VPU_INT8_EPV;

                            int k_offset = C_in * VPU_INT8_ACC_PERIOD * K_h * K_w * cog
                                         + C_in * VPU_INT8_ACC_PERIOD * K_w * kkrr
                                         + C_in * VPU_INT8_ACC_PERIOD * kkcc
                                         + VPU_INT8_EPV * VPU_INT8_ACC_PERIOD * cig
                                         + VPU_INT8_EPV * (15 - cog_offset)
                                         + cig_offset;

                            int x_offset = C_in * width * (row+kr)
                                         + C_in * (col+kc)
                                         + chin;

                            const int16_t kernel_val = K[k_offset];
                            const int16_t input_val = X[x_offset];

                            acc32 += kernel_val * input_val;

                        }
                        
                        // acc32 = sat_s32(acc64);
                    }
                }

                // printf("acc32 = %ld\n", acc32);

                int16_t res16 = vlsat_single_s16(acc32, shr);
                
                // printf("res16 = %d\t(%d)\n", res16, shr);
                res16 = vlmul_single_s16(res16, scale);
                
                // printf("res16 = %d\t(%d)\n", res16, scale);
                const int8_t res8 = vdepth8_single_s16(res16);
                
                // printf("res8 = %d\n", res8);
                Y[(row*width+col)*C_out + chout] = res8;
            }
        }
    }
}





#define C_in                            (4)
#define K_W_DIM                         (8)
void conv2d_shallowin_deepout_relu_c(
    const int8_t* K, 
    const data16_t* B,
    const int8_t* X, 
    int8_t* Y,
    const int32_t height, 
    const int32_t width,
    const int32_t K_h, 
    const int32_t K_w,
    const int32_t C_out,
    const int16_t* shifts, 
    const int16_t* scales)
{
    const int P_h = K_h / 2;
    const int P_w = K_w / 2;
    const int Q_h = P_h - height + 1;
    const int Q_w = P_w - width  + 1;

    const int K_row_bytes = K_W_DIM * C_in * VPU_INT8_ACC_PERIOD;
    const int K_cout_group_bytes = K_row_bytes * K_h;

    const int X_pxl_bytes = C_in;
    const int X_row_bytes = width * X_pxl_bytes;

    for(int ch_out_grp = 0; ch_out_grp < (C_out/(VPU_INT8_ACC_PERIOD)); ch_out_grp++){

        const int k_group_offset = ch_out_grp * K_cout_group_bytes;

        for(int ch_out = 0; ch_out < VPU_INT8_ACC_PERIOD; ch_out++){

            const unsigned actual_chout = ch_out_grp * VPU_INT8_ACC_PERIOD + ch_out;

            const int16_t shr = shifts[actual_chout];
            const int16_t scale = scales[actual_chout];

            for(int row = 0; row < height; row++){

                const int pad_top = ((P_h-row) > 0)?  (P_h-row) : 0;
                const int pad_bot = ((Q_h+row) > 0)?  (Q_h+row) : 0;

                for(int col = 0; col < width; col++){

                    const int pad_lef = ((P_w-col) > 0)?  (P_w-col) : 0;
                    const int pad_rig = ((Q_w+col) > 0)?  (Q_w+col) : 0;

                    const int B_offset   = 2 * C_out * (K_w * (P_h + pad_bot - pad_top) + (P_w + pad_rig - pad_lef));
                    // printf("(%d, %d) -> (%d, %d, %d, %d)\n", row, col, pad_top, pad_bot, pad_lef, pad_rig);
                    // printf("(%d, %d) -> %d\n", row, col, B_offset/(2*C_out));

                    const data16_t* B_lo = &B[B_offset + 0];
                    const data16_t* B_hi = &B[B_offset + C_out];
                    const int32_t bias   = (((int32_t)B_hi[actual_chout])<<16) | B_lo[actual_chout];
                    // printf("(%d, %d) -> %ld\n", row, col, bias);

                    int32_t acc32 = bias;

                    for(int kr = -P_h; kr <= P_h; kr++){
                        
                        if(row+kr < 0 || row+kr >= height)
                            continue;

                        int64_t acc64 = acc32;

                        const int k_row_offset = k_group_offset + (kr + P_h) * K_row_bytes;
                        const int x_row_offset = (row + kr) * X_row_bytes;

                        for(int kc = -P_w; kc <= P_w; kc++){

                            if(col+kc < 0 || col+kc >= width)
                                continue;

                            const int k_cout_offset = k_row_offset + ((VPU_INT8_ACC_PERIOD-1)-ch_out) * K_W_DIM * C_in;
                            const int k_col_offset = k_cout_offset + (kc + P_w) * C_in;
                            const int x_col_offset = x_row_offset + (col + kc) * X_pxl_bytes;
                            
                            for(unsigned ch_in = 0; ch_in < C_in; ch_in++){

                                const int k_elm_offset = k_col_offset + ch_in;
                                const int x_elm_offset = x_col_offset + ch_in;

                                const int8_t K_elm = K[k_elm_offset];
                                const int8_t X_elm = X[x_elm_offset];

                                // printf("K[%d, %d, %d, %d] -> K[%d]: %d\n", actual_chout, (kr+P_h), (kc+P_w), ch_in, k_elm_offset, K_elm);
                                
                                acc64 += ((int32_t)K_elm) * X_elm;
                            }
                        }
                        
                        acc32 = sat_s32(acc64);
                    }
            
                    int16_t res16 = vlsat_single_s16(acc32, shr);
                    res16 = vlmul_single_s16(res16, scale);
                    int8_t res8 = vdepth8_single_s16(res16);
                    Y[(row*width+col)*C_out + actual_chout] = res8;

                    // return;
                }
            }
        }
    }
}
#undef K_W_DIM
#undef C_in

