

#ifndef NN_OPERATOR_H_
#define NN_OPERATOR_H_

#include "nn_operator_asm.h"
#include "nn_operator_c.h"
#include "nn_operator_inline.h"

#include <stdint.h>

#include "xs3_vpu.h"

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
    int8_t* y);





/**  2D convolution for "deep" input and output tensors.
 *
 *  Stride is 1 in both dimensions. Number of input and output channels must be
 *  divisible by 32 and 16 respectively. Activation is ReLU. Zero padding
 *  should be used on all edges, with size K_h//2 above and below, and K_w//2
 *  on the left and right.
 *
 *  \param  K       Kernel weight tensor of shape (C_out, K_h, K_w, C_in) using
 *                  a non-standard layout such that:
 *                  K[i, j, k, l]  =  K[
 *                    K_h * K_w * C_in * ((i//16 + 1)*16 - i%16 - 1)
 *                    +  K_w * C_in * j  +  C_in * k  +  l
 *                  ]
 *  \param  B       Bias tensor of shape (C_out, 2) using a standard layout
 *                  such that B[i, c]  =  2 * i  +  c. The value B[0, c]
 *                  encodes the lower 16 bits, while B[1, c] encodes the higher
 *                  16 bits of the 32-bit bias value for output channel c.
 *  \param  X       Input tensor of shape (height, width, C_in) using standard
 *                  layout with the last index changing fastest:
 *                  X[h, w, c]  =  X[width * C_in * h  +  C_in * w  +  c]
 *  \param  Y       Output tensor of shape (height, width, C_out)
 *                  using standard layout with the last index changing fastest:
 *                  Y[h, w, c]  =  Y[width * C_out * h  +  C_out * w  +  c]
 *  \param  height  Input tensor/image height.
 *  \param  width   Input tensor/image width.
 *  \param  K_h     Kernel height, must be odd.
 *  \param  K_w     Kernel width, must be odd.
 *  \param  C_out   Number of output channels, must be divisible by 16.
 *  \param  C_in    Number of input channels, must be divisible by 32.
 *  \param  shifts  Shift tensor of shape (C_out) using standard layout.
 *                  Defines the shift used in the 32 to 16 bit intermediate
 *                  conversion via VLSAT.
 *  \param  scales  Scale tensor of shape (C_out) using standard layout.
 *                  Defines the scale applied after the 32 to 16 bit
 *                  intermediate conversion. Optional. Can be assumed to be
 *                  between 0x4000 and 0x7FFF.
 */
static inline void conv2d_deepin_deepout_relu(
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
    const int16_t* scales);

/**  2D convolution for "shallow" input and "deep" output tensors.
 *
 *  Stride is 1 in both dimensions. Number of output channels must be divisible
 *  by 16. Number of input channels is 4. Activation is ReLU. Zero padding
 *  should be used on all edges, with size K_h//2 above and below, and K_w//2
 *  on the left and right.
 *
 *  \param  K       Kernel weight tensor of shape (C_out, K_h, 8, 4) using
 *                  a non-standard layout such that:
 *                  K[i, j, k, l]  =  K[
 *                    K_h * 8 * 4 * ((i//16 + 1)*16 - i%16 - 1)
 *                    +  8 * 4 * j  +  4 * k  +  l
 *                  ]
 *                  The weights are zero padded in the 3rd dimension, i.e.
 *                  K[i, j, k, l] is zero for K_w <= k < 8. There may or may
 *                  not be zero padding in the 4th dimension.
 *  \param  B       Bias tensor of shape (C_out, 2) using a standard layout
 *                  such that B[i, c]  =  2 * i  +  c. The value B[0, c]
 *                  encodes the lower 16 bits, while B[1, c] encodes the higher
 *                  16 bits of the 32-bit bias value for output channel c.
 *  \param  X       Input tensor of shape (height, width, C_in) using standard
 *                  layout with the last index changing fastest:
 *                  X[h, w, c]  =  X[width * 4 * h  +  4 * w  +  c]
 *  \param  Y       Output tensor of shape (height, width, C_out)
 *                  using standard layout with the last index changing fastest:
 *                  Y[h, w, c]  =  Y[width * C_out * h  +  C_out * w  +  c]
 *  \param  height  Input tensor/image height.
 *  \param  width   Input tensor/image width.
 *  \param  K_h     Kernel height, must be odd.
 *  \param  K_w     Kernel width, must be odd and less than 8.
 *  \param  C_out   Number of output channels, must be divisible by 16.
 *  \param  shifts  Shift tensor of shape (C_out) using standard layout.
 *                  Defines the shift used in the 32 to 16 bit intermediate
 *                  conversion via VLSAT.
 *  \param  scales  Scale tensor of shape (C_out) using standard layout.
 *                  Defines the scale applied after the 32 to 16 bit
 *                  intermediate conversion. Optional. Can be assumed to be
 *                  between 0x4000 and 0x7FFF.
 */
static inline void conv2d_shallowin_deepout_relu(
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
    const int16_t* scales);


/**  2D maxpool for "deep" input and output tensors.
 *
 *  Pool size is 2x2, stride is 2 in both dimensions. Number of input channels
 *  must be divisible by 32.
 *
 *  \param  X       Input tensor of shape (height, width, C_in) using standard
 *                  layout with the last index changing fastest:
 *                  X[h, w, c]  =  X[width * C_in * h  +  C_in * w  +  c]
 *  \param  Y       Output tensor of shape (height//2, width//2, C_in) using
 *                  standard layout with the last index changing fastest:
 *                  Y[h, w, c]  =  Y[(width//2) * C_in * h  +  C_in * w  +  c]
 *  \param  height  Input tensor/image height, must be even.
 *  \param  width   Input tensor/image width, must be even.
 *  \param  C_in    Number of input channels, must be divisible by 32.
 */
static inline void maxpool2d_deep(
    const int8_t* X, 
    int8_t* Y,
    const int32_t height, 
    const int32_t width,
    const int32_t C_in);



/**  Fully connected layer for "deep" input and "shallow" output tensors.
 *
 *  Number of inputs must be divisible by 32. Number of outputs must be less
 *  than 16. No activation is applied (i.e. linear).
 *
 *  \param  W       Weight tensor of shape (C_out, C_in) using a non-standard
 *                  layout such that:
 *                  W[i, j]  =  K[C_in * (C_out - 1 - i)  +  j]
 *                  There may or may not be zero padding in the 2nd dimension.
 *  \param  B       Bias tensor of shape (C_out) using a standard layout.
 *  \param  X       Input tensor of shape (C_in) using standard layout.
 *  \param  Y       Output tensor of shape (C_out) using standard layout.
 *  \param  C_out   Number of output channels, must be less than 16.
 *  \param  C_in    Number of input channels, must be divisible by 32.
 *  \param  shifts  Shift tensor of shape (C_out) using standard layout.
 *                  Defines the shift used in the 32 to 16 bit conversion via
 *                  VLSAT. For c >= C_out, the value shifts[y] is undefined.
 *  \param  scales  Scale tensor of shape (C_out) using standard layout.
 *                  Defines the scale applied after the 32 to 16 bit
 *                  conversion. Optional. Can be assumed to be between 0x4000
 *                  and 0x7FFF. For c >= C_out, the value scales[y] is
 *                  undefined.
 */
static inline void fc_deepin_shallowout_lin(
    const int8_t* W, 
    const int32_t* B,
    const int8_t* X, 
    int16_t* Y,
    const int32_t C_out, 
    const int32_t C_in,
    const uint16_t* shifts, 
    const int16_t* scales);




#ifdef __XC__
} // extern "C"
#endif

#endif //NN_OPERATOR_H_