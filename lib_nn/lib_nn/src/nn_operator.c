

#include "nn_operator.h"
#include "nn_op_helper.h"

#include "xs3_vpu.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>





///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

void fc_boggle_BSS(
    data16_t* bss_out,
    int32_t* bias,
    int16_t* shift,
    int16_t* scale,
    data16_t* scratch,
    const unsigned C_out)
{
    const unsigned ceil_C_out = (((C_out + (VPU_INT8_ACC_PERIOD - 1)) >> VPU_INT8_ACC_PERIOD_LOG2) << VPU_INT8_ACC_PERIOD_LOG2);

    data16_t* buff = NULL;

    if(((void*)bias) == ((void*)bss_out)){
        //bss_out is being updated in-place. We will need to use a scratch buffer

        if(scratch != NULL){
            //scratch buffer was provided by user
            buff = scratch;
        } else {
            //need to malloc a scratch buffer.
            buff = (data16_t*) malloc(C_out * 4 *  sizeof(data16_t));

            if(buff == NULL){
                printf("Failed to allocate scratch buffer.");
                __builtin_trap();
            }
        }

    } else {
        //bss_out is not being updated in-place, just copy from the inputs to
        //  bss_out.
    }


    if(buff != NULL){
        memcpy(&buff[0], bias, C_out * sizeof(int32_t));
        memcpy(&buff[2*C_out], shift, C_out*sizeof(data16_t));
        memcpy(&buff[3*C_out], scale, C_out*sizeof(data16_t));

        bias = (int32_t*) &buff[0];
        shift = (int16_t*) &buff[2*C_out];
        scale = (int16_t*) &buff[3*C_out];
    }

    const unsigned C_out_groups = ceil_C_out >> VPU_INT8_ACC_PERIOD_LOG2;

    for(int cog = 0; cog < C_out_groups; cog++){

        const unsigned cog_offset = VPU_INT8_ACC_PERIOD * 4 * cog;

        for(int coff = 0; coff < VPU_INT8_ACC_PERIOD; coff++){

            const unsigned cout = cog * VPU_INT8_ACC_PERIOD + coff;

            int32_t b      = bias[cout];
            data16_t shr   = shift[cout];
            data16_t scl   = scale[cout];

            data16_t b_lo = b & 0xFFFF;
            data16_t b_hi = (b & 0xFFFF0000) >> 16;

            bss_out[cog_offset + 0 * VPU_INT8_ACC_PERIOD + coff] = b_hi;
            bss_out[cog_offset + 1 * VPU_INT8_ACC_PERIOD + coff] = b_lo;
            bss_out[cog_offset + 2 * VPU_INT8_ACC_PERIOD + coff] = shr;
            bss_out[cog_offset + 3 * VPU_INT8_ACC_PERIOD + coff] = scl;
            
        }
    }


    if(buff != NULL && scratch == NULL){
        free(buff);
    }
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




///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

void fc_deepin_shallowout_16_c(
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

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

void fully_connected_16_c(
    int16_t* Y,
    const int8_t* W, 
    const int8_t* X, 
    const data16_t* BSS,
    const unsigned C_in, 
    const unsigned C_out)
{
    const unsigned ACCS = VPU_INT8_ACC_PERIOD;

    for(unsigned cout = 0; cout < C_out; cout++){

        const unsigned cog = cout >> VPU_INT8_ACC_PERIOD_LOG2;
        const unsigned coff = cout & (ACCS - 1);

        const data16_t* BSS_cog = &BSS[4*ACCS * cog];
        const int8_t* W_row = &W[cout * C_in];
        const int32_t bias_hi = BSS_cog[coff + 0*ACCS];
        const int32_t bias_lo = BSS_cog[coff + 1*ACCS];
        const int16_t shift   = BSS_cog[coff + 2*ACCS];
        const int16_t scale   = BSS_cog[coff + 3*ACCS];

        int64_t acc64 = (bias_hi << 16) | bias_lo;

        //For VERY deep inputs, it is possible that this doesn't match the assembly.
        for(unsigned cin = 0; cin < C_in; cin++){
            int32_t x = X[cin];
            int32_t w = W_row[cin];
            int32_t p = x * w;
            acc64 += p;
        }

        // printf("acc64 = %ld\n", acc64);

        acc64 =   (acc64 >= VPU_INT32_MAX)? VPU_INT32_MAX
                : (acc64 <= VPU_INT32_MIN)? VPU_INT32_MIN
                : acc64;

        int16_t res16 = vlsat_single_s16((int32_t)acc64, shift);
        res16 = vlmul_single_s16(res16, scale);

        // int8_t res8 = vdepth8_single_s16(res16);

        Y[cout] = res16;
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////


void fc_deepin_shallowout_8_c(
    const int8_t* W, 
    const int32_t* B,
    const int8_t* X, 
    int8_t* Y,
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

        int8_t res8 = vdepth8_single_s16(res);

        Y[k] = res8;
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

void argmax_16_c(
    const int16_t* A,
    int32_t* C,
    const int32_t N)
{
    if( N <= 0) return;

    *C = 0;

    for(int32_t i = 1; i < N; i++){
        if( A[i] > A[*C] ){
            *C = i;
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

void requantize_16_to_8_c(
    int8_t* y,
    const int16_t* x,
    const unsigned n)
{
    for(int i = 0; i < n; i++){
        y[i] = vdepth8_single_s16(x[i]);
    }
}