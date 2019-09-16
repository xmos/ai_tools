

#ifndef NN_OPERATOR_INLINE_H_
#define NN_OPERATOR_INLINE_H_

#include "nn_operator_c.h"
#include "nn_operator_asm.h"

#include <stdint.h>

#ifdef __XC__
extern "C" {
#endif



/**
 * Performs a matrix-vector multiplication (each signed 8-bit) for a 32-bit result.
 * 
 *      y = W*x
 * 
 * Idiosyncrasies:
 *      - Expects an unusual memory layout for the matrix W.  
 *          The matrix is first broken up into non-overlapping 'bands', where each band is 16 consecutive rows.
 *          Each band is broken up into non-overlapping 'chunks', where each chunk is 32 consecutive columns. 
 *          So each chunk is a 16x32 submatrix of the original matrix, and the matrix is a tiling of these chunks.
 *          The layout is strictly ordered such that:
 *              - Earlier (i.e. lower row indices) bands appear before later bands.
 *              - Within a band, earlier (i.e. lower column indices) chunks appear before later chunks.
 *              - Within a chunk, *later* (by index) rows appear *earlier*
 *              - Within a chunk-row, individual coefficients are stored in increasing index order.
 *              - No padding is used to separate any of these elements.
 *      - Internally a 32-bit accumulator is used, and a per-element right-shift (the shr parameter) is applied before 
 *        saturating the result to 8 bits.
 * 
 * Limitations:
 *      Can only be used for matrices with a multiple of 16 rows and
 *      a multiple of 32 columns.
 * 
 * \param   W           Coefficient matrix, using the memory layout specified above.
 * \param   x           Input vector, ordered normally
 * \param   N_bands     Number of bands in the matrix W. i.e. number of rows in W divided by 16
 * \param   N_chunks    Number of chunks in a band. i.e. number of columns in W divided by 32
 * \param   shr         Vector specifying (for each element) the number of bits to right-shift the 32-bit accumulator before saturating to 8 bits.
 * \param   y           Output vector, ordered normally
 */
static inline void nn_mat_vec_mul_s8(
    const int8_t* W,
    const int8_t* x,
    const unsigned N_bands,
    const unsigned N_chunks,
    const int16_t* shr,
    int8_t* y)
{
#if defined(__XS3A__) && (USE_ASM_nn_mat_vec_mul_s8)

    nn_mat_vec_mul_s8_asm(W, x, N_bands, N_chunks, shr, y);

#else

    nn_mat_vec_mul_s8_c(W, x, N_bands, N_chunks, shr, y);

#endif
}




#ifdef __XC__
}   //extern "C"
#endif

#endif //NN_OPERATOR_INLINE_H_