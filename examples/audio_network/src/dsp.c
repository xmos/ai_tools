#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <print.h>
#include "xmath/xmath.h"
#include "src.h"
#include "src_ff3_fir_coefs.h"
#include "dsp.h"
#include "log.h"
#include "raised_cosine.h"

int dsp_time_to_freq(int64_t fft_output[], int data_to_be_processed[], fft_state_t *state) {
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
    state->hr = 31;
    state->exponent = 0;
    fft_dit_forward((complex_s32_t *)fft_output, WINDOW_SIZE/2, &state->hr, &state->exponent);
    fft_mono_adjust((complex_s32_t *)fft_output, WINDOW_SIZE, 0);

    return state->exponent;          // Gain applied.
}

void dsp_freq_to_time(int data_processed[], int64_t fft_input[], fft_state_t *state) {
    static int output_frame[WINDOW_SIZE];   // State for overlap-add; first sample is oldest
    static int32_t state_us[32];            // State for the up sampler
    
    // First inverse the FFT
    
    fft_mono_adjust((complex_s32_t *)fft_input, WINDOW_SIZE, 1);
    fft_index_bit_reversal((complex_s32_t *)fft_input, WINDOW_SIZE/2);
    fft_dit_inverse((complex_s32_t *)fft_input, WINDOW_SIZE/2, &state->hr, &state->exponent);
    vect_s32_shl((int32_t *)fft_input, (int32_t *)fft_input, WINDOW_SIZE, state->exponent + 9 + 2 + 1 -10);

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

void dsp_calculate_mels(int mels[], int64_t fft_input[], int gain, int mel_bins,
                        int *mel_coefficients, int *mel_bins_in_overlap) {
    int magnitudes[WINDOW_SIZE/2+1];

    magnitudes[0            ] = log2_16(((int *)fft_input)[0]) + (gain << LOG2_16_Q_VALUE);
    magnitudes[WINDOW_SIZE/2] = log2_16(((int *)fft_input)[1]) + (gain << LOG2_16_Q_VALUE);
    for(int i = 2; i < WINDOW_SIZE; i+=2) {           // 82 us
        int64_t mag = (((int *)fft_input)[i] * (int64_t) ((int *)fft_input)[i] +
                       ((int *)fft_input)[i] * (int64_t) ((int *)fft_input)[i]);
        magnitudes[i/2] = log2_16_64(mag) + (gain << (1+LOG2_16_Q_VALUE));
    }

    mel_compress(mels, magnitudes,
                 mel_coefficients,
                 mel_bins_in_overlap,
                 WINDOW_SIZE/2+1, mel_bins);         // 47 us
}

void dsp_apply_masks(int64_t fft_output[], int masks_mel[], int enabled, int mel_bins,
                        int *mel_coefficients, int *mel_bins_in_overlap) {
    int masks_freq[WINDOW_SIZE/2+1];

    mel_expand(masks_freq, masks_mel,                 // 68 us
               mel_coefficients,
               mel_bins_in_overlap,
               WINDOW_SIZE/2+1, mel_bins);
    if (enabled) {
        int64_t m = masks_freq[0];
        ((int *)fft_output)[0] = (((int *)fft_output)[0] * m) >> 30;
        m = masks_freq[WINDOW_SIZE/2];
        ((int *)fft_output)[1] = (((int *)fft_output)[1] * m) >> 30;
        for(int i = 2; i < WINDOW_SIZE; i+=2) {            // 47 us
            m = masks_freq[i/2];
            ((int *)fft_output)[i]   = (((int *)fft_output)[i]   * m) >> 30;
            ((int *)fft_output)[i+1] = (((int *)fft_output)[i+1] * m) >> 30;
        }
    }
}

#ifdef LOCAL_MAIN
#include "mel_parameters.h"

int sin_wave[16] = {
    0, 382683, 707107, 923879, 1000000,
    923879, 707107, 382683, 0, 
    -382683, -707107, -923879, -1000000,
    -923879, -707107, -382683,
};

int main(void) {
    fft_state_t fft_state;
    int64_t fft_data[WINDOW_SIZE/2];
    int masks[MEL_BINS];
    int mels[MEL_BINS];
    int cnt = 0;
    int data[UPSAMPLED_WINDOW_ADVANCE];
    int multiplier = 200;// 200;
    for(int j = 0; j < 8000; j+=UPSAMPLED_WINDOW_ADVANCE) {
        for(int i = j; i < j + UPSAMPLED_WINDOW_ADVANCE; i++) {
            data[i-j] = sin_wave[cnt]*multiplier;
            cnt++;
            if (cnt == 16) cnt = 0;
        }
        for(int i = j; i < j + 16; i++) {
            printf("%10d ", data[i-j]);
        }
        printf("   ");
        int t0, t1, t2;
        asm volatile("gettime %0" : "=r" (t0));
        int gain = dsp_time_to_freq(fft_data, data, &fft_state);
        dsp_calculate_mels(mels, fft_data, gain, MEL_BINS,                mel_coefficients,
               mel_bins_in_overlap
);
        asm volatile("gettime %0" : "=r" (t1));
        for(int i = 0; i < MEL_BINS; i++) {
            masks[i] = MEL_ONE_VALUE;
        }
        dsp_apply_masks(fft_data, masks, 1, MEL_BINS,                mel_coefficients,
               mel_bins_in_overlap
);
        dsp_freq_to_time(data, fft_data, &fft_state);
        asm volatile("gettime %0" : "=r" (t2));
        for(int i = 0; i < 16; i++) {
            printf("%10d ", data[i]/2);
        }
        printf("\n");
    }
    
}

#endif

#ifdef LOCAL_MAIN3
#include "mel_parameters.h"

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
