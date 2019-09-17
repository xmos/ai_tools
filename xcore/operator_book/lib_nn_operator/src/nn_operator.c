

#include "nn_operator.h"
#include "xs3_vpu.h"

#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>


static inline int8_t vlsat_single(int32_t acc, int16_t shr){
    if(shr > 0){
        int32_t tmp = (acc >> shr) << shr;
        int32_t diff = acc - tmp;
        uint32_t threshold = 1<<(shr-1);
        
        acc >>= shr;

        if (diff == threshold){
            // Round to nearest even.
            if(acc & 0x01)
                acc++;
        } else if(diff > threshold){
            //Round up (more positive, less negative)
            acc += 1;
        } else {
            //Round down (do nothing)
        }
    } else {
        acc >>= shr;
    }
    if(acc > VPU_INT8_MAX)
        acc = VPU_INT8_MAX;
    if(acc < VPU_INT8_MIN)
        acc = VPU_INT8_MIN;

    return (int8_t) acc;
}




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

        y[row] = vlsat_single(accumulator, shr[row]);
    }

}





void conv2d_deepin_deepout_relu_c(
    const int8_t* K, 
    const uint16_t* B,
    const int8_t* X, 
    int8_t* Y,
    const int32_t height, 
    const int32_t width,
    const int32_t K_h, 
    const int32_t K_w,
    const int32_t C_out, 
    const int32_t C_in,
    const uint16_t* shifts, 
    const int16_t* scales)
{
    
}


void conv2d_shallowin_deepout_relu_c(
    const int8_t* K, 
    const uint16_t* B,
    const int8_t* X, 
    int8_t* Y,
    const int32_t height, 
    const int32_t width,
    const int32_t K_h, 
    const int32_t K_w,
    const int32_t C_out,
    const uint16_t* shifts, 
    const int16_t* scales)
{

}



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
    const uint16_t* B,
    const int8_t* X, 
    int16_t* Y,
    const int32_t C_out, 
    const int32_t C_in,
    const uint16_t* shifts, 
    const int16_t* scales)
{
}
