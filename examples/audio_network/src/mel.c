#include "mel.h"

void mel_compress(int *mel, int *input_bin,
                  int *fractions,
                  int *number_of_bins_in_overlap,
                  int mel_spectral_bins, int mel_mel_bins) {
    int current_mel = -1;
    int run = 0;
    mel[0] = 0;                       // Set first accumulator to zero
    for(int i = 0; i < mel_spectral_bins; i++) {
        if (run <= 0) {               // See if we are at the end of the run
            current_mel++;            // If so, go to next MEL
            run = number_of_bins_in_overlap[current_mel];
            mel[current_mel+1] = 0;   // Initialise next accumulator
        }
        int contribution =  (input_bin[i] * (int64_t) fractions[i]) >> MEL_Q_VALUE;
        mel[current_mel  ] += contribution;
        mel[current_mel+1] += input_bin[i] - contribution;
        run--;                        // Completed one element of the run
    }
}

void mel_expand(int *output_bin, int *mel,
                int *fractions,
                int *number_of_bins_in_overlap,
                int mel_spectral_bins, int mel_mel_bins) {
    int current_mel = -1;
    int run = 0;
    for(int i = 0; i < mel_spectral_bins; i++) {
        if (run <= 0) {               // See if we are at the end of the run
            current_mel++;            // If so, go to next MEL
            run = number_of_bins_in_overlap[current_mel];
        }
        output_bin[i]  =  ((mel[current_mel  ] *                  (int64_t) fractions[i] ) +
                           (mel[current_mel+1] * (MEL_ONE_VALUE - (int64_t) fractions[i]))) >> MEL_Q_VALUE;
        run--;                        // Completed one element of the run
    }
}
