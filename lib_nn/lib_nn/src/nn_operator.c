

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








///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

#define KERNEL_SIZE (2)
#define KERNEL_STRIDE (KERNEL_SIZE)
#define IN_INDEX(ROW, COL, CHAN, WIDTH, CHANS_IN)   ((ROW)*((WIDTH)*(CHANS_IN)) + (COL)*(CHANS_IN) + (CHAN))
#define OUT_INDEX(ROW, COL, CHAN, WIDTH, CHANS_IN)   ((ROW)*(((WIDTH)/2)*(CHANS_IN)) + (COL)*(CHANS_IN) + (CHAN))

void avgpool2d_2x2_c(
    int8_t* Y,
    const int8_t* X, 
    const uint32_t x_height, 
    const uint32_t x_width,
    const uint32_t x_chans)
{
    //int8_t X[x_height][x_width][C_in]
    //int8_t Y[height/2][x_width/2][C_in]

    assert((x_height & 1) == 0);
    assert((x_width  & 1) == 0);

    for(unsigned row = 0; row < x_height; row += KERNEL_STRIDE){
        for(unsigned col = 0; col < x_width; col += KERNEL_STRIDE){
            for(unsigned ch = 0; ch < x_chans; ch++){

                unsigned out_dex = OUT_INDEX(row/2, col/2, ch, x_width, x_chans);

                int32_t acc = 0;

                int8_t* out_val = &Y[out_dex];

                for(unsigned krow = 0; krow < KERNEL_SIZE; krow++){
                    for(unsigned kcol = 0; kcol < KERNEL_SIZE; kcol++){
                        unsigned in_dex = IN_INDEX(row+krow, col+kcol, ch, x_width, x_chans);

                        int8_t in_val = X[in_dex];

                        acc += in_val;

                        if(in_val > *out_val)
                            *out_val = in_val;
                    }
                }

                acc = (acc + 0x02) >> 2; //Should round appropriately

                if(acc == INT8_MIN)
                    acc = VPU_INT8_MIN;

                *out_val = acc;
            }
        }
    }
}

#undef OUT_INDEX
#undef IN_INDEX
#undef KERNEL_STRIDE
#undef KERNEL_SIZE








///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

void avgpool2d_global_c(
    int8_t* Y,
    const int8_t* X, 
    const uint32_t x_height, 
    const uint32_t x_width,
    const uint32_t x_chans,
    const uint32_t shift,
    const uint32_t scale)
{
    const unsigned pix = x_height * x_width;

    const uint32_t sh = shift & 0xFFFF;
    const uint32_t sc = scale & 0xFF;
    
    for(unsigned ch = 0; ch < x_chans; ch++){

        int32_t acc = 0;

        for(unsigned p = 0; p < pix; p++){
            int32_t x = X[p*x_chans + ch];
            acc += x * sc;
        }

        Y[ch] = vlsat_single_s8(acc, sh);
    }
}

void avgpool2d_global_init(
    uint32_t* shift,
    uint32_t* scale,
    const uint32_t x_height,
    const uint32_t x_width)
{    
    //Figure out the scale and shift
    const unsigned pix = x_height * x_width;
    //Find c = ceil(log2(pix)), which can be achieve via clz()
    const int c = ceil_log2(pix);

    if(c == -1) __builtin_trap(); //pix == 0

    if(pix == (1<<c)){
        //window pixel count is already a power of 2   (2^c)
        *scale = 1;
        *shift = c;
        // printf("scale: %d\nshift: %d\ncl2: %d\npix: %u\n", scale, shift, ceil_log2(pix), pix);
        // printf("win_h: %u\nwin_w:%u\n", win->window.height, win->window.width);
    } else {
        const unsigned q = 31 - c - 6;
        // 2^31 / pix
        const unsigned g = 0x80000000 / pix;
        const unsigned h = (g + (1 << (q-1))) >> q; //Rounding down-shift

        // printf("! pix: %u\n", pix);
        // printf("! c: %d\n", c);
        // printf("! q: %u\n", q);
        // printf("! g: 0x%08X\n", g);
        // printf("! h: 0x%02X\n", h);
        assert(h > (1<<6));
        assert(h < (1<<7));

        *scale = (int8_t)h;
        *shift = c+6;
    }

    (*scale) *= 0x01010101;
    (*shift) *= 0x00010001;
}   




///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

void avgpool2d_c(
    int8_t* Y,
    const int8_t* X, 
    const nn_avgpool_params_t* params)
{
    X = &X[params->start_incr_x];
    Y = &Y[params->start_incr_y];

    // int8_t* Y_start = Y;

    const int8_t shift = params->shift & 0xFF;
    const int8_t scale = params->scale & 0xFF;

    for(unsigned out_row = 0; out_row < params->out_rows; out_row++){
        // printf("out_row = %u\n", out_row);
        for(unsigned out_col = 0; out_col < params->out_cols; out_col++){
            // printf("\tout_col = %u\n", out_col);
            for(int chans = params->out_chans; chans > 0; chans -= 16){
                // printf("\t\tchan = %d\n", (int) params->Y_chans - chans);
                
                unsigned iter_chans = (chans >= VPU_INT8_ACC_PERIOD)? VPU_INT8_ACC_PERIOD : chans;
                int32_t acc32[VPU_INT8_ACC_PERIOD] = {0};

                for(unsigned w_rows = 0; w_rows < params->W_h; w_rows++){
                    for(unsigned w_cols = 0; w_cols < params->W_w; w_cols++){

                        for(unsigned k = 0; k < iter_chans; k++){
                            acc32[k] += (((int32_t)X[k]) * scale);
                        }

                        X = &X[params->win_col_incr_x];
                    }

                    X = &X[params->win_row_incr_x];
                }

                for(unsigned k = 0; k < iter_chans; k++){
                    Y[k] = vlsat_single_s8(acc32[k], shift);
                }

                X = &X[params->chan_incr_x];

                Y = &Y[iter_chans];
            }

            X = &X[params->hstride_incr_x];
        }

        X = &X[params->vstride_incr_x];
        Y = &Y[params->vstride_incr_y];
    }

    // int wrote_bytes = (int32_t) (&Y[0] - &Y_start[0]);
    // printf("Wrote %d bytes to Y\n", wrote_bytes);
}


void avgpool2d_init(
    nn_avgpool_params_t* pool,
    const nn_image_params_t* x,
    const nn_image_params_t* y,
    const nn_window_map_t* win)
{

    pool->out_rows = win->window.vcount;
    pool->out_cols = win->window.hcount;;
    pool->out_chans = y->channels;
    pool->W_h = win->window.height;
    pool->W_w = win->window.width;
    
    pool->start_incr_x = IMG_ADDRESS_VECT(x, win->start.X.rows, win->start.X.cols, win->start.X.channels);
    assert(pool->start_incr_x >= 0);
    assert(pool->start_incr_x < (x->height*x->width*x->channels));

    pool->start_incr_y = IMG_ADDRESS_VECT(y, win->start.Y.rows, win->start.Y.cols, win->start.Y.channels);
    assert(pool->start_incr_y >= 0);
    assert(pool->start_incr_y < (y->height*y->width*y->channels));

    const unsigned chan_groups = OUT_CHANNEL_GROUPS(pool->out_chans);

    //TODO: given the unrolled loop in the assembly, it might make more sense to decide
    //      whether the window should be evaluated on the transpose of the region.
    //      i.e. if window->window.height = 16 (unrolled loop size) and window->window->width = 4,
    //           then the win_col_incr_x should probably be set to traverse within a column of the
    //          window inside the innermost loop. This is possible because the offsets are arbitrary
#define IAV IMG_ADDRESS_VECT    
    pool->win_col_incr_x = IAV(x, 0, 1, 0);
    pool->win_row_incr_x = IAV(x, 1, 0, 0)                   - pool->W_w * IAV(x, 0, 1, 0);
    pool->chan_incr_x    = IAV(x, 0, 0, VPU_INT8_ACC_PERIOD) - pool->W_h * IAV(x, 1, 0, 0);
    pool->hstride_incr_x = IAV(x, 0, win->window.hstride, 0) - chan_groups * IAV(x, 0, 0, VPU_INT8_ACC_PERIOD); 
    pool->vstride_incr_x = IAV(x, win->window.vstride, 0, 0) - win->window.hcount * IAV(x, 0, win->window.hstride, 0);

    pool->vstride_incr_y = IAV(y, 1, 0, 0) - win->window.hcount * IAV(y, 0, 1, 0);
#undef IAV

    //Figure out the scale and shift

    const unsigned pix = win->window.height * win->window.width;
    //Find c = ceil(log2(pix)), which can be achieve via clz()
    const int c = ceil_log2(pix);

    if(c == -1) __builtin_trap(); //pix == 0

    int8_t scale;
    int16_t shift;
    if(pix == (1<<c)){
        //window pixel count is already a power of 2   (2^c)
        scale = 1;
        shift = c;
        // printf("scale: %d\nshift: %d\ncl2: %d\npix: %u\n", scale, shift, ceil_log2(pix), pix);
        // printf("win_h: %u\nwin_w:%u\n", win->window.height, win->window.width);
    } else {
        const unsigned q = 31 - c - 6;
        // 2^31 / pix
        const unsigned g = 0x80000000 / pix;
        const unsigned h = (g + (1 << (q-1))) >> q; //Rounding down-shift

        // printf("! pix: %u\n", pix);
        // printf("! c: %d\n", c);
        // printf("! q: %u\n", q);
        // printf("! g: 0x%08X\n", g);
        // printf("! h: 0x%02X\n", h);
        assert(h > (1<<6));
        assert(h < (1<<7));

        scale = (int8_t)h;
        shift = c+6;
    }

    pool->shift = 0x00010001 * shift;
    pool->scale = 0x01010101 * scale;

    pool->special_case = 0;

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