#ifndef __mel_h__
#define __mel_h__

#include <stdint.h>

#define MEL_Q_VALUE       30
#define MEL_ONE_VALUE     (1 << MEL_Q_VALUE)
/** Library that enables a sparse matrix multiplication where the input matrix
 * fullfills the following criteria:
 *   - All columns add up to 1
 *   - There is at most two non zero values in each column, if there are two then they
 *     in consecutive rows
 *   - For two consecutive columns, the number of zero rows above the non-zero
 *     is either the same or has increased by one. Ie, it is loosely a diagonal matrix
 *
 * The matrix is stored in a compressed format where we only store the first
 * non-zero value of each column; this forms a single vector of elements. As
 * the column adds up to one and there is at most two non-zero values, the
 * other value can be recalculated. The length of this vector is identical to the
 * number of columns in the matrix.
 * 
 * In addition to the vector of non-zero values we need one more vector which counts
 * the number of columns for which this row was the first non-zero value. This is a
 * vector with a length identical to the number of rows in the matrix.
 */

/**
 * Function that compresses a spectogram into a smaller array of MEL values
 * 
 * @param  mel              Array of me_mel_bins
 *                          The MEL values are written into this array
 *
 * @param  input_bin        Array of me_spectral_bins
 *                          The input values are read from this array
 *
 * @param  fractions        Array of me_spectral_bins.
 *                          This stores the first non-zero
 *                          value in each column of the eml filter bank.
 *
 * @param  bins_in_overlap  Array of me_mel_bins
 *                          Defines the number of bins until the current MEL
 *                          is completed. For each of the input bins, a
 *                          fraction is added to the current MEL and
 *                          (1-fraction) is added to the next MEL. When the
 *                          current MEL is completed we move on to the next
 *                          MEL.
 *
 * @param mel_spectral_bins Number of bins in the spectrogram
 *
 * @param mel_mel_bins      Number of MEL values; must be smaller than
 *                          mel_spectral_bins.
 */
void mel_compress(int *mel, int *input_bin,
                  int *fractions,
                  int *bins_in_overlap,
                  int mel_spectral_bins, int mel_mel_bins);

/**
 * Function that expands an array of MEL values into a spectogram
 * 
 * @param  output_bin       Array of me_spectral_bins
 * @param  mel              Array of me_mel_bins
 * @param  fractions        Array of me_spectral_bins
 * @param  bins_in_overlap  Array of me_mel_bins. See mel_compress().
 * @param mel_spectral_bins Number of bins in the spectrogram
 * @param mel_mel_bins      Number of MEL values; must be smaller than
 *                          mel_spectral_bins.
 */
void mel_expand(int *output_bin, int *mel,
                int *fractions,
                int *bins_in_overlap,
                int mel_spectral_bins, int mel_mel_bins);

#endif
