#ifndef __dsp_h__
#define __dsp_h__

#include "xmath/xmath.h"

#include <stdint.h>

#define UPSAMPLE_RATE            3
#define WINDOW_SIZE              512
#define WINDOW_ADVANCE           (WINDOW_SIZE/4)
#define WINDOW_OVERLAP           (WINDOW_SIZE - WINDOW_ADVANCE)
#define UPSAMPLED_WINDOW_ADVANCE (WINDOW_ADVANCE * UPSAMPLE_RATE)

typedef struct {
    int exponent;
    headroom_t hr;
} fft_state_t;

int dsp_time_to_freq(int64_t fft_output[], int data_to_be_processed[], fft_state_t *state);
void dsp_freq_to_time(int data_processed[], int64_t fft_input[], fft_state_t *state);

void dsp_calculate_mels(int mels[], int64_t fft_input[], int gain, int mel_bins,
                        int *mel_coefficients, int *mel_bins_in_overlap);
void dsp_apply_masks(int64_t fft_output[], int masks_mel[], int enabled, int mel_bins,
                        int *mel_coefficients, int *mel_bins_in_overlap);

#endif
