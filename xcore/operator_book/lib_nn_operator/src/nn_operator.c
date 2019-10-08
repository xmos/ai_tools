

#include "nn_operator.h"
#include "xs3_vpu.h"

#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

static inline int8_t sat_s8(const int32_t acc32)
{
    if(acc32 >= VPU_INT8_MAX)
        return VPU_INT8_MAX;
    if(acc32 <= VPU_INT8_MIN)
        return VPU_INT8_MIN;
    
    return (int8_t) acc32;
}

static inline int16_t sat_s16(const int32_t acc32)
{
    if(acc32 >= VPU_INT16_MAX)
        return VPU_INT16_MAX;
    if(acc32 <= VPU_INT16_MIN)
        return VPU_INT16_MIN;
    
    return (int16_t) acc32;
}

static inline int32_t sat_s32(const int64_t acc64)
{
    if(acc64 >= VPU_INT32_MAX)
        return VPU_INT32_MAX;
    if(acc64 <= VPU_INT32_MIN)
        return VPU_INT32_MIN;
    
    return (int32_t) acc64;
}

// static inline void mulsat_s32(int32_t* acc32, const int8_t a, const int8_t b)
// {
//     int64_t acc64 = *acc32 + a*b;
//     *acc32 = sat_s32(acc64);
// }

static inline int8_t vlsat_single_s8(int32_t acc, int16_t shr)
{
    if(shr > 0) acc += 1<<(shr-1);
    return sat_s8(acc >> shr);
}

static inline int16_t vlsat_single_s16(int32_t acc, int16_t shr)
{
    if(shr > 0) acc += 1<<(shr-1);
    return sat_s16(acc >> shr);
}

static inline int16_t vlmul_single_s16(int16_t vR, int16_t mem){
    int32_t p = ((int32_t)vR) * mem;
    p = vlsat_single_s16(p, 14);
    return (int16_t)p;
}

static inline int8_t vdepth8_single_s16(int16_t vR){
    return vlsat_single_s8(vR, 8);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

void nn_mat_vec_mul_s8_c(
    const int8_t* W,
    const int8_t* x,
    const unsigned N_bands,
    const unsigned N_chunks,
    const int16_t* shr,
    int8_t* y)
{
    typedef struct {
        int8_t w[VPU_INT8_ACC_PERIOD][VPU_INT8_EPV];
    } chunk_t;

    const chunk_t* W_chunks = (chunk_t*) W;

    memset(y, 0, N_chunks * VPU_INT8_ACC_PERIOD * sizeof(int8_t));

    for(unsigned row = 0; row < ( N_bands * VPU_INT8_ACC_PERIOD); row++){
        
        const unsigned band_index = row / VPU_INT8_ACC_PERIOD;
        const chunk_t* band_start = &W_chunks[band_index * N_chunks];

        const unsigned chunk_row = (VPU_INT8_ACC_PERIOD - 1) - (row % VPU_INT8_ACC_PERIOD);

        int32_t accumulator = 0;

        for(unsigned ch = 0; ch < N_chunks; ch++){
            const chunk_t* chunk = &band_start[ch];

            for(unsigned col = 0; col < VPU_INT8_EPV; col++){
                const int8_t w = chunk->w[chunk_row][col];
                const int8_t xx = x[ch*32+col];
                int64_t acc64 = ((int64_t)accumulator) + w*xx;
                
                // printf("@@ %lld\t\t%d\t%d\t%d\n", accumulator, w, xx, w*xx);
                if(acc64 > VPU_INT32_MAX)
                    acc64 = VPU_INT32_MAX;
                if(acc64 < VPU_INT32_MIN)
                    acc64 = VPU_INT32_MIN;

                accumulator = (int32_t) acc64;
            }
        }

        y[row] = vlsat_single_s8(accumulator, shr[row]);
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

    const data16_t* B_hi = &B[C_out];
    const data16_t* B_lo = &B[0];


    for(int chout = 0; chout < C_out; chout++){

        const int cog = chout / VPU_INT8_ACC_PERIOD;
        const int cog_offset = chout % 16;
        
        const int32_t bias = (((int32_t)B_hi[chout])<<16) | B_lo[chout];
        const int16_t shr = shifts[chout];
        const int16_t scale = scales[chout];

        for(int row = 0; row < height; row++){
            for(int col = 0; col < width; col++){
                
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

                        int64_t acc64 = acc32;

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

                            acc64 += kernel_val * input_val;

                        }
                        
                        acc32 = sat_s32(acc64);
                    }
                }

                int16_t res16 = vlsat_single_s16(acc32, shr);

                if(res16 < 0) res16 = 0;

                res16 = res16 - ((1<<14)-1);

                res16 = vlmul_single_s16(res16, scale);

                Y[(row*width+col)*C_out + chout] = vdepth8_single_s16(res16);
            }
        }
    }
}



#define KERNEL_GROUP(C_OUT_G_DEX)       K[(C_OUT_G_DEX) * K_h * K_w * C_in * (VPU_INT8_ACC_PERIOD)]
#define KERNEL(C_OUT_DEX)               (kernel_group[(15-(C_OUT_DEX)) * K_h * K_w * C_in ])
#define KERNEL_VAL(ROW, COL, CH_IN)     (kernel[(ROW)*K_w*C_in + (COL)*C_in + CH_IN])
#define X_VAL(ROW, COL, CH_IN)          (X[(ROW)*width*C_in + (COL)*C_in + CH_IN])
void conv2d_deepin_deepout_relu_c_old(
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

    assert( vlmul_single_s16(8318, 16384) == 8318  );

    const int P_h = K_h / 2;
    const int P_w = K_w / 2;

    const data16_t* B_hi = &B[C_out];
    const data16_t* B_lo = &B[0];

    for(int ch_out_grp = 0; ch_out_grp < (C_out/(VPU_INT8_ACC_PERIOD)); ch_out_grp++){

        const int8_t* kernel_group = &KERNEL_GROUP(ch_out_grp);

        for(int ch_out = 0; ch_out < VPU_INT8_ACC_PERIOD; ch_out++){

            const unsigned actual_chout = ch_out_grp * VPU_INT8_ACC_PERIOD + ch_out;

            const int8_t* kernel = &KERNEL(ch_out);
            const int32_t bias = (B_hi[actual_chout] << 16) | B_lo[actual_chout];
            const int16_t shr = shifts[actual_chout];
            const int16_t scale = scales[actual_chout];

            for(int row = 0; row < height; row++){
                for(int col = 0; col < width; col++){
                    int32_t acc32 = bias;

                    for(int kr = -P_h; kr <= P_h; kr++){
                        for(int kc = -P_w; kc <= P_w; kc++){

                            if(row+kr < 0 || row+kr >= height)
                                continue;
                            if(col+kc < 0 || col+kc >= width)
                                continue;
                            
                            for(unsigned ch_in_grp = 0; ch_in_grp < (C_in/VPU_INT8_EPV); ch_in_grp++){
                                int64_t acc64 = acc32;
                                for(unsigned ch_in = 0; ch_in < VPU_INT8_EPV; ch_in++){
                                    acc64 += KERNEL_VAL(kr+P_h, kc+P_w, (ch_in_grp*VPU_INT8_EPV)+ch_in) * X_VAL(row+kr,col+kc,(ch_in_grp*VPU_INT8_EPV)+ch_in);
                                }

                                acc32 = sat_s32(acc64);
                            }
                        }
                    }

                    if(K_h == 3 && K_w == 3 && row == 0 && col == 0 && actual_chout == 0){
                        printf("acc32 = %ld\n", acc32);
                        return;
                    }

                    // if( actual_chout == 2 && row == 1 && col == 0)
                    //     printf("2!!  %ld\n", acc32);

                    int16_t res16 = vlsat_single_s16(acc32, shr);
                    
                    // if( actual_chout == 2 && row == 1 && col == 0)
                    //     printf("3!!  %d\n", res16);

                    if(res16 < 0) res16 = 0;
                    
                    // if( actual_chout == 2 && row == 1 && col == 0)
                    //     printf("4!!  %d\n", res16);

                    res16 = res16 - ((1<<14)-1);
                    
                    // if( actual_chout == 2 && row == 1 && col == 0)
                    //     printf("5!!  %d\n", res16);

                    res16 = vlmul_single_s16(res16, scale);
                    
                    // if( actual_chout == 2 && row == 1 && col == 0)
                    //     printf("6!!  %d\n", res16);
                    
                    // if( actual_chout == 2 && row == 1 && col == 0)
                    //     printf("7!!  %d\n", vdepth8_single_s16(res16));

                    Y[(row*width+col)*C_out + actual_chout] = vdepth8_single_s16(res16);
                }
            }
        }
    }

}
#undef X_VAL
#undef KERNEL_VAL
#undef KERNEL
#undef KERNEL_GROUP



#define C_in                            (4)
#define K_W_DIM                         (8)
#define KERNEL_GROUP(C_OUT_G_DEX)       K[(C_OUT_G_DEX) * K_h * K_W_DIM * C_in * (VPU_INT8_ACC_PERIOD)]
#define KERNEL(C_OUT_DEX)               (kernel_group[(15-(C_OUT_DEX)) * K_h * K_W_DIM * C_in ])
#define KERNEL_VAL(ROW, COL, CH_IN)     (kernel[(ROW)*K_W_DIM*C_in + (COL)*C_in + CH_IN])
#define X_VAL(ROW, COL, CH_IN)          (X[(ROW)*width*C_in + (COL)*C_in + CH_IN])
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

    const data16_t* B_hi = &B[C_out];
    const data16_t* B_lo = &B[0];

    for(int ch_out_grp = 0; ch_out_grp < (C_out/(VPU_INT8_ACC_PERIOD)); ch_out_grp++){

        const int8_t* kernel_group = &KERNEL_GROUP(ch_out_grp);

        for(int ch_out = 0; ch_out < VPU_INT8_ACC_PERIOD; ch_out++){

            const unsigned actual_chout = ch_out_grp * VPU_INT8_ACC_PERIOD + ch_out;

            const int8_t* kernel = &KERNEL(ch_out);
            const int32_t bias = (B_hi[actual_chout] << 16) | B_lo[actual_chout];
            const int16_t shr = shifts[actual_chout];
            const int16_t scale = scales[actual_chout];

            for(int row = 0; row < height; row++){
                for(int col = 0; col < width; col++){
                    int32_t acc32 = bias;

                    for(int kr = -P_h; kr <= P_h; kr++){

                        //shallowin-deepout processes an entire row of a patch as a single instruction

                        int64_t acc64 = acc32;

                        for(int kc = -P_w; kc <= P_w; kc++){

                            if(row+kr < 0 || row+kr >= height)
                                continue;
                            if(col+kc < 0 || col+kc >= width)
                                continue;
                            
                            for(unsigned ch_in = 0; ch_in < C_in; ch_in++){
                                acc64 += KERNEL_VAL(kr+P_h, kc+P_w, ch_in) * X_VAL(row+kr,col+kc,ch_in);
                            }
                        }
                        
                        acc32 = sat_s32(acc64);
                    }
            
                    // printf("2!!  %ld\n", acc32);

                    int16_t res16 = vlsat_single_s16(acc32, shr);

                    // printf("3!!  %d\n", res16);

                    if(res16 < 0) res16 = 0;
                    // printf("4!!  %d\n", res16);
                    res16 = res16 - ((1<<14)-1);

                    // printf("5!!  %d\n", res16);

                    res16 = vlmul_single_s16(res16, scale);

                    // printf("6!!  %d\n", res16);

                    int8_t res8 = vdepth8_single_s16(res16);
                    
                    // printf("7!!  %d\n\n", res8);

                    Y[(row*width+col)*C_out + actual_chout] = res8;
                    // return;
                }
            }
        }
    }
}
#undef X_VAL
#undef KERNEL_VAL
#undef KERNEL
#undef K_W_DIM
#undef C_in



#define KERNEL_SIZE (2)
#define KERNEL_STRIDE (KERNEL_SIZE)
#define IN_INDEX(ROW, COL, CHAN, WIDTH, CHANS_IN)   ((ROW)*((WIDTH)*(CHANS_IN)) + (COL)*(CHANS_IN) + (CHAN))
#define OUT_INDEX(ROW, COL, CHAN, WIDTH, CHANS_IN)   ((ROW)*(((WIDTH)/2)*(CHANS_IN)) + (COL)*(CHANS_IN) + (CHAN))

void maxpool2d_deep_c(
    const int8_t* X, 
    int8_t* Y,
    const int32_t height, 
    const int32_t width,
    const int32_t C_in)
{
    //int8_t X[height][width][C_in]
    //int8_t Y[height/2][width/2][C_in]

    assert((height & 1) == 0);
    assert((width  & 1) == 0);

    for(unsigned row = 0; row < height; row += KERNEL_STRIDE){
        for(unsigned col = 0; col < width; col += KERNEL_STRIDE){
            for(unsigned ch = 0; ch < C_in; ch++){

                unsigned out_dex = OUT_INDEX(row/2, col/2, ch, width, C_in);

                int8_t* out_val = &Y[out_dex];
                *out_val = INT8_MIN;

                for(unsigned krow = 0; krow < KERNEL_SIZE; krow++){
                    for(unsigned kcol = 0; kcol < KERNEL_SIZE; kcol++){
                        unsigned in_dex = IN_INDEX(row+krow, col+kcol, ch, width, C_in);

                        int8_t in_val = X[in_dex];

                        if(in_val > *out_val)
                            *out_val = in_val;
                    }
                }
            }
        }
    }
}

#undef OUT_INDEX
#undef IN_INDEX
#undef KERNEL_STRIDE
#undef KERNEL_SIZE


void fc_deepin_shallowout_lin_c(
    const int8_t* W, 
    const int32_t* B,
    const int8_t* X, 
    int16_t* Y,
    const int32_t C_out, 
    const int32_t C_in,
    const uint16_t* shifts, 
    const int16_t* scales)
{
    assert(C_in % VPU_INT8_EPV == 0);
    assert(C_out <= 16);

    const int row_vlmaccrs = C_in / VPU_INT8_EPV;

    //Compute outputs one at a time
    for(unsigned k = 0; k < C_out; k++){

        //Compute pre-activation value
        int32_t acc32 = B[k];
        // printf("@@\t%ld\n", acc32);

        const int8_t* W_row = &W[k * C_in];

        for(unsigned v = 0; v < row_vlmaccrs; v++){

            int32_t vacc = 0;

            const int8_t* W_v = &W_row[v*VPU_INT8_EPV];
            const int8_t* X_v = &X[v*VPU_INT8_EPV];

            for(unsigned i = 0; i < VPU_INT8_EPV; i++)
                vacc += W_v[i] * X_v[i];

            int64_t acc64 = acc32 + vacc;

            if(acc64 > VPU_INT32_MAX)
                acc64 = VPU_INT32_MAX;
            if(acc64 < VPU_INT32_MIN)
                acc64 = VPU_INT32_MIN;

            acc32 = acc64;
        }

        //Compute shifts
        int16_t res = vlsat_single_s16(acc32, shifts[k]);

        //Compute scales
        res = vlmul_single_s16(res, scales[k]);

        Y[k] = res;
    }
}
