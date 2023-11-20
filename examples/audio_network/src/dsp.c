#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "xmath/xmath.h"
#include "src.h"
#include "src_ff3_fir_coefs.h"
#include "dsp.h"
#include "log.h"
#include "raised_cosine.h"

static int exponent;
static headroom_t hr;

void dsp_time_to_freq(int64_t fft_output[], int data_to_be_processed[]) {
    static int input_frame[WINDOW_SIZE];    // State for overlap-add; first sample is oldest
    static int32_t state_ds[3][32];         // State for the down sampler
    
    // First advance the window, then fill the last quarter in with downsampled signals

    memmove(input_frame, input_frame + WINDOW_ADVANCE, sizeof(int) * WINDOW_OVERLAP);
    for(int i = 0; i < WINDOW_ADVANCE; i++) {  // 205 us
        src_ff3_96t_ds((int32_t *)data_to_be_processed + UPSAMPLE_RATE*i, (int32_t *)input_frame + WINDOW_OVERLAP + i,
                       src_ff3_fir_coefs, state_ds);
    }
    
    // Use a raised cosing to window it, create two bits of headroom.
    for(int i = 0; i < WINDOW_SIZE; i++) {  // 42 us
        ((int *)fft_output)[i] = (raised_cosine_512[i] * (int64_t)input_frame[i]) >> 32;
    }

    // 41 us
    fft_index_bit_reversal((complex_s32_t *)fft_output, WINDOW_SIZE/2);
    hr = 31;
    exponent = 0;
    fft_dit_forward((complex_s32_t *)fft_output, WINDOW_SIZE/2, &hr, &exponent);
    fft_mono_adjust((complex_s32_t *)fft_output, WINDOW_SIZE, 0);
    int z = vect_s32_shl((int32_t *)fft_output, (int32_t *)fft_output, WINDOW_SIZE, exponent-8);
    printf("Hr out %d exp out %d final hr %d\n", hr, exponent, z);
}

void dsp_freq_to_time(int data_processed[], int64_t fft_input[]) {
    static int output_frame[WINDOW_SIZE];   // State for overlap-add; first sample is oldest
    static int32_t state_us[32];            // State for the up sampler
    int exponent2 = 0;
    int hr2 = 31;
    
    // First inverse the FFT
    
    fft_mono_adjust((complex_s32_t *)fft_input, WINDOW_SIZE, 1);
    fft_index_bit_reversal((complex_s32_t *)fft_input, WINDOW_SIZE/2);
    fft_dit_inverse((complex_s32_t *)fft_input, WINDOW_SIZE/2, &hr2, &exponent2);
    printf("Exponent2 %d hr2 %d\n", exponent2, hr2);
    exponent2 = exponent2 + 9; // compensate for INV FFT subtracting 9.
    vect_s32_shl((int32_t *)fft_input, (int32_t *)fft_input, WINDOW_SIZE, -exponent2 + 8-9);

    // Now add the overlap and fill in the new bit
    // TODO: that memcpy is not needed and can be done in here.
    
    for(int i = 0; i < WINDOW_OVERLAP; i++) {
        output_frame[i] += ((int *)fft_input)[i];
    }
    for(int i = WINDOW_OVERLAP; i < WINDOW_SIZE; i++) {
        output_frame[i] = ((int *)fft_input)[i];
    }
    // Now copy the first advance out and up sample it back to 48 kHz
    for(int i = 0; i < WINDOW_ADVANCE; i++) {
        src_ff3_96t_us((int32_t *)output_frame + i, (int32_t *)data_processed + UPSAMPLE_RATE*i,
                       src_ff3_fir_coefs, state_us);
    }
    // Finally memcpy the overlap - TODO: this is not needed and can be done above.
    memmove(output_frame, output_frame + WINDOW_ADVANCE, sizeof(int) * WINDOW_OVERLAP);
}

#include "mel.h"
#include "mel_parameters.h"

void dsp_calculate_mels(int mels[], int64_t fft_input[]) {
    int magnitudes[WINDOW_SIZE/2+1];

    magnitudes[0            ] = log2_16(((int *)fft_input)[0]);
    magnitudes[WINDOW_SIZE/2] = log2_16(((int *)fft_input)[1]);
    for(int i = 2; i < WINDOW_SIZE; i+=2) {           // 82 us
        int64_t mag = (((int *)fft_input)[i] * (int64_t) ((int *)fft_input)[i] +
                       ((int *)fft_input)[i] * (int64_t) ((int *)fft_input)[i]);
        magnitudes[i/2] = log2_16_64(mag);
    }

    mel_compress(mels, magnitudes,
                 mel_coefficients,
                 mel_bins_in_overlap,
                 WINDOW_SIZE/2+1, MEL_BINS);         // 47 us
}

void dsp_apply_masks(int64_t fft_output[], int masks_mel[]) {
    int masks_freq[WINDOW_SIZE/2+1];

    mel_expand(masks_freq, masks_mel,                 // 68 us
               mel_coefficients,
               mel_bins_in_overlap,
               WINDOW_SIZE/2+1, MEL_BINS);
    for(int i = 0; i < WINDOW_SIZE; i++) {            // 47 us
        ((int *)fft_output)[i] = (((int *)fft_output)[i] * (int64_t) masks_freq) >> 30;
    }
}

#ifdef LOCAL_MAIN

int sin_wave[16] = {
    0, 500000, 707100, 866025, 1000000,
    866025, 707100, 500000, 0, 
    -500000, -707100, -866025, -1000000,
    -866025, -707100, -500000,
};
static int64_t fft_data[WINDOW_SIZE/2];
static int masks[MEL_BINS];
static int mels[MEL_BINS];

int main(void) {
    int cnt = 0;
    int data[UPSAMPLED_WINDOW_ADVANCE];
    float multiplier = 100;// 200;
    for(int j = 0; j < 8000; j+=UPSAMPLED_WINDOW_ADVANCE) {
        for(int i = j; i < j + UPSAMPLED_WINDOW_ADVANCE; i++) {
            data[i-j] = sin_wave[cnt]*multiplier;
            cnt++;
            if (cnt == 16) cnt = 0;
        }
        for(int i = j; i < j + 16; i++) {
            printf("%7d ", data[i-j]);
        }
        printf("   ");
        int t0, t1, t2;
        asm volatile("gettime %0" : "=r" (t0));
        dsp_time_to_freq(fft_data, data);
        dsp_calculate_mels(mels, fft_data);
        asm volatile("gettime %0" : "=r" (t1));
        for(int i = 0; i < MEL_BINS; i++) {
            masks[i] = MEL_ONE_VALUE;
        }
        dsp_apply_masks(fft_data, masks);
        dsp_freq_to_time(data, fft_data);
        asm volatile("gettime %0" : "=r" (t2));
//        printf("\n%d %d   %d us\n", t1-t0, t2-t1, (t2-t0)/100);
        for(int i = 0; i < 32; i++) {
            printf("%7d ", data[i]/2);
        }
        printf("\n");
    }
    
}

#endif

#ifdef LOCAL_MAIN3

int data[257];
int data2[257];

static int mels[MEL_BINS+1];

int main(void) {
    
    for(int i = 0; i < 257; i++) {
        if (i & 1) {
            data[i] = i*3 + 1000;
        } else {
            data[i] = -i;
        }
    }
    mel_compress(mels, data,
                 mel_coefficients,
                 mel_bins_in_overlap,
                 WINDOW_SIZE/2+1, MEL_BINS);
    for(int i = 0; i < 64; i++) {
        printf("%9d\n", mels[i]);
    }
    mel_expand(data2, mels,
               mel_coefficients,
               mel_bins_in_overlap,
               WINDOW_SIZE/2+1, MEL_BINS);
    for(int i = 0; i < 257; i++) {
        printf("%9d %9d\n", data[i], data2[i]);
    }
}

#endif
