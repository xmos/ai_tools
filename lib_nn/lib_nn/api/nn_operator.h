

#ifndef NN_OPERATOR_H_
#define NN_OPERATOR_H_

#include "nn_types.h"
#include "nn_op_structs.h"
#include "nn_operator_asm.h"
#include "nn_operator_c.h"
#include "nn_operator_inline.h"

#include <stdint.h>

#include "xs3_vpu.h"

#ifdef __XC__
extern "C" {
#endif



/**
Saturation, Accumulation, Shifts and Scales

For the convolution and fully-connected layers, it is important to understand the 
behavior of the XS3 VPU.

Each output comprises a series of 8-bit multiplies which accumulate into a 32-bit 
accumulator -- effectively the 32-bit dot-product of two 8-bit vectors. Prior to
this dot product, the accumulator is seeded with a 32-bit bias value.

While accumulating, if at any point the sum would go beyond the range `(2^-31 + 1, 2^31-1)`,
the accumulator will saturate to `-2^31 + 1` if the sum is negative and `2^31-1` if 
it's positive. The accumulator does not roll-over.

After 32-bit accumulation is completed the following occurs:
- The output has a user-specified arithmetic right bit-shift applied (rounded).
- The result is saturated to 16-bit bounds (2^15+1, 2^15-1) if necessary.
- The result is quantized to 16 bits by discarding the upper half word.

After shifting a scale factor is applied to each output, which consists of:
- Multiplying the 16-bit result by a 16-bit signed integer
- Applying an arithmetic right shift of 14 bits to the 30-bit result (rounded).
- Saturating the result to 16-bit bounds.
(This can be thought of as a rounding, saturating fixed-point multiplication in which
the 16-bit accumulator value is treated as a signed Q15.0 fixed-point value and the
scale factor is treated as a signed Q1.14 fixed-point value).

For `fc_deepin_shallowout_16()` the results are then stored in the output.

For the 2D convolution functions, the final step before storing the result is to
quantize the 16-bit results down to signed 8-bit results by right-shifting 8 bits 
and rounding.

*/

/**
ReLU

The convolution and fully-connected layers do not include a built-in ReLU activation
function, and the ReLU function is not available as a separate function. ReLU (and certain
other piecewise-linear activation functions can be achieved implicitly using the 
convolution or fully-connected operators by making use of the saturation boundaries. 


*/


/**
Standard Tensor Layout

The standard layout of an N-dimensional tensor with shape `(s1, s2, ..., sN)` is the
same as the typical layout of a C array with the same dimensions. In this layout, the
indices of each dimension are iterated over, with indices iterating in ascending order,
and with the later dimensions iterating fastest.

For example, a tensor `A` with shape (2,2,2), in standard tensor layout would be ordered
as:

`A[0,0,0], A[0,0,1], A[0,1,0], A[0,1,1], A[1,0,0], A[1,0,1], A[1,1,0], A[1,1,1]`

*/





/**
Bias Tensor Layout (Form 1)

A bias vector is a `C_out`-element sequence of 32-bit values, where each element seeds a
32-bit accumulator with an initial value. The standard-form bias vector can be 
represented as:

\code
    int32_t biases[C_out];
\endcode

A form 1 bias tensor layout rearranges the data to minimize the pointer arithmetic required
during computation. Form 1 layout can be represented as:

\code
    data16_t B[C_out/16][2][16];
\endcode

In this layout, the biases are taken 16 at a time (due to the 16 accumulators used in the
8- and 16-bit modes of the XS3 VPU), and the upper and lower half-words are separated (also
due to an ideosyncrasy of the XS3 VPU).

More specifically, the data in this tensor is arranged in memory such that:
- The first set of 16 data16_t values are the upper 16 bits of output channels 0-15.
- The second set of 16 data16_t values are the lower 16 bits of output channels 0-15.
- The third set of 16 data16_t values are the upper 16 bits of output channels 16-31.
- The fourth set of 16 data16_t values are the lower 16 bits of output channels 16-31.
- And so on.

The following represents how this rearrangement may be achieved:

\code
    for(int i = 0; i < C_out; i++){
        B[i/16][0][i%16] = biases[i] >> 16;
        B[i/16][1][i%16] = biases[i] & (0xFFFF);
    }
/endcode

Thus, in this layout, the elements `B[i][0][j]` and `B[i][1][j]` represent the upper and
lower (respectively) 16 bits of the 32-bit bias for output channel `(16*i + j)`.
*/














/** Execute an 8-bit 2D convolution on a region of an image.
 *
 * This function performs a 2D convolution of an input image with the specified 
 * kernel. This function requires that the convolution's input have a multiple 
 * of 32 input channels, and that the output have a multiple of 16 output channels.
 *
 * All pointer arguments must refer to word-aligned addresses.
 * 
 * The `nn_conv2d_dido_params_t*` input parameter must have been initialized with 
 * a call to `conv2d_deepin_deepout_init()`.
 *
 * The dimensions and types of the `Y`, `X`, `K`, `shifts` and `scales` must be consistent
 * with those provided to `conv2d_deepin_deepout_init()`.
 * 
 * The height and width of `Y` depends on the `padding_mode` parameter supplied to `conv2d_deepin_deepout_init()`.
 * For `PADDING_MODE_SAME`, `Y`'s shape is  (X_height, X_width, C_out),
 * For `PADDING_MODE_VALID`, `Y`'s shape is (X_height - K_h//2, X_width - K_w//2, C_out).
 * In either case, `Y`'s data layout will be the standard tensor layout, with indices 
 * corresponding to the rows, columns and channels of the image.
 *
 * The shape of `X` is (X_height, X_width, C_in), with a standard tensor layout with indices
 * corresponding to the rows, columns and channels of the image.
 * 
 * The shape of `K` is (C_out // 16, K_h, K_w, C_in // 32, 16, 32). The kernel tensor is layed
 * out so as to minimize overhead from pointer arithmetic. The following pseudocode illustrates
 * this layout:
    \code
      int8_t kernel_tensor[C_out][K_h][K_w][C_in] = {...}; //standard kernel tensor
      int8_t K[C_out/16][K_h][K_w][C_in/32][16][32];  //re-arranged kernel tensor

      for(int krow = 0; krow < K_h; krow++)
        for(int kcol = 0; kcol < K_w; kcol++)
          for(int cout = 0; cout < C_out; cout++)
            for(int cin = 0; cin < C_in; cin++)
              K[cout/16][krow][kcol][cin/32][15-(cout % 16)][cin % 32] = kernel_tensor[cout][krow][kcol][cin];
    \endcode
 *  Note that the fifth dimension of `K` is reversed.
 *
 * The shape of the `scales` tensor is (C_out // 16, 2, 16). The first index is the channel group, second indicates shift (0) or scale (1), and the third is the channel offset within the channel group.
 * 
 * where `X_height`, `X_width`, `C_in`, `C_out`, `K_h` and `K_w` are from the parameters supplied to
 * `conv2d_deepin_deepout_init()`.
 * 
 * 
 * \param Y         Pointer to beginning of output data tensor. Updated by function.
 * \param params    Pointer to `nn_conv2d_dido_params_t` initialized with `conv2d_deepin_deepout_init()`
 * \param X         Pointer to beginning of input data tensor
 * \param K         Pointer to beginning of kernel tensor
 * \param scales    Pointer to beginning of scales tensor
 */
static inline void conv2d_deepin_deepout(
    int8_t* Y,
    const nn_conv2d_dido_params_t* params,
    const int8_t* X,
    const int8_t* K,
    const int16_t* scales);












/** Execute an 8-bit 2D convolution on a region of an image.
 *
 * This function performs a 2D convolution of an input image with the specified 
 * kernel. This function requires that the convolution's 
 * output have a multiple of 16 output channels. Further, this function requires that
 * the product of the kernel width and input channel count (`K_w` * C_in`) be no
 * larger than 32, and `C_in` must be a multiple of 4.
 *
 * All pointer arguments must refer to word-aligned addresses.
 * 
 * The `nn_conv2d_sido_params_t*` input parameter
 * must have been initialized with a call to `conv2d_shallowin_deepout_init()`.
 *
 * The dimensions and types of the `Y`, `X`, `K`, `B`, `shifts` and `scales` must be consistent
 * with those provided to `conv2d_shallowin_deepout_init()`.
 * 
 * The height and width of `Y` depends on the `padding_mode` parameter supplied to `conv2d_shallowin_deepout_init()`.
 * For `PADDING_MODE_SAME`, `Y`'s shape is  (X_height, X_width, C_out),
 * For `PADDING_MODE_VALID`, `Y`'s shape is (X_height - K_h//2, X_width - K_w//2, C_out).
 * In either case, `Y`'s data layout will be the standard layout, with indices 
 * corresponding to the rows, columns and channels of the image.
 *
 * The shape of `X` is (X_height, X_width, C_in), with a standard image data layout. Each pixel
 * of `X` must start of a word-aligned boundary. Hence, if the input image has only 3 channels 
 * (e.g. RGB), the input image must be padded with a 4th channel to ensure word-alignment.
 * 
 * The shape of `K` is (C_out // 16, K_h, 16, 32 / C_in, C_in). The kernel tensor is layed
 * out so as to minimize overhead from pointer arithmetic. The following pseudocode illustrates
 * this layout:
    \code
      int8_t kernel_tensor[C_out][K_h][K_w][C_in] = {...}; //standard kernel tensor
      int8_t K[C_out/16][K_h][16][32 / C_in][C_in] = {{{{{0}}}}};  //re-arranged kernel tensor

      for(int krow = 0; krow < K_h; krow++)
        for(int kcol = 0; kcol < K_w; kcol++)
          for(int cout = 0; cout < C_out; cout++)
            for(int cin = 0; cin < C_in; cin++)
              K[cout/16][krow][15-(cout % 16)][kcol][cin] = kernel_tensor[cout][krow][kcol][cin];
    \endcode
 *  Note that the third dimension of `K` is reversed. Note also that the forth dimension, 
 *  corresponding to kernel width has a size of `32/C_in`, rather than `K_w`. This dimension is
 *  zero-padded (if necessary) so that the product of the final two dimensions is 32.
 *
 * The shape of `B` is (C_out // 16, 2, 16), and is layed out in Bias Tensor Layout (Form 1) 
 * as described above.
 *
 * The shape of the `scales` tensor is (C_out // 16, 2, 16). The first index is the channel group, second indicates shift (0) or scale (1), and the third is the channel offset within the channel group.
 * 
 * where `X_height`, `X_width`, `C_in`, `C_out`, `K_h` and `K_w` are from the parameters supplied to
 * `conv2d_shallowin_deepout_init()`.
 * 
 * \param Y         Pointer to beginning of output data tensor. Updated by function.
 * \param params    Pointer to `nn_conv2d_sido_params_t` initialized with `conv2d_shallowin_deepout_init()`
 * \param block     Pointer to `nn_conv2d_sido_block_params_t` initialized with `conv2d_shallowin_deepout_init()`
 * \param X         Pointer to beginning of input data tensor
 * \param K         Pointer to beginning of kernel tensor
 * \param B         Pointer to beginning of bias tensor
 * \param scales    Pointer to beginning of scales tensor
 */
static inline void conv2d_shallowin_deepout(
    int8_t* Y,
    const nn_conv2d_sido_params_t* params,
    const int8_t* X,
    const int8_t* K,
    const int16_t* scales);

    










/**  2D maxpool for "deep" input and output tensors.
 *
 *  Pool size is 2 x 2, stride is 2 in both dimensions. Number of input channels
 *  must be divisible by 32.
 *
 *  For each 2 x 2 block of input pixels (tiled across the input image), this function
 *  reduces that block to a single pixel, where each channel is handled independently.
 *  The value selected for the output channel is the maximum among the 4 values for that
 *  channel in the input block.
 *
 *  The shape of `X` is (height, width, C_in), with a standard image data layout.
 *  
 *  The shape of `Y` is (height/2, width/2, C_in), with a standard image data layout.
 *
 *  \code
 *      Y[i][j][c] = MAX(X[2*i][2*j][c], X[2*i+1][2*j][c], X[2*i][2*j+1][c], X[2*i+1][2*j+1][c])
 *  \endcode
 *
 *  \param  X       Input data tensor.
 *  \param  Y       Output data tensor. Updated by function.
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


/**  2D average pool for "deep" input and output tensors.
 *
 *  Pool size is 2 x 2, stride is 2 in both dimensions. Number of input channels
 *  must be divisible by 32.
 *
 *  For each 2 x 2 block of input pixels (tiled across the input image), this function
 *  reduces that block to a single pixel, where each channel is handled independently.
 *  The value selected for the output channel is the average of the 4 values for that
 *  channel in the input block (with rounding).
 *
 *  The shape of `X` is (height, width, C_in), with a standard image data layout.
 *  
 *  The shape of `Y` is (height/2, width/2, C_in), with a standard image data layout.
 *
 *  \code
 *      Y[i][j][c] = (X[2*i][2*j][c] + X[2*i+1][2*j][c] + X[2*i][2*j+1][c] + X[2*i+1][2*j+1][c] + 2) / 4
 *  \endcode
 *
 *  \param  X       Input data tensor.
 *  \param  Y       Output data tensor. Updated by function.
 *  \param  height  Input tensor/image height, must be even.
 *  \param  width   Input tensor/image width, must be even.
 *  \param  C_in    Number of input channels, must be divisible by 32.
 */
static inline void avgpool2d_deep(
    const int8_t* X, 
    int8_t* Y,
    const int32_t height, 
    const int32_t width,
    const int32_t C_in);



/**  Fully connected layer for "deep" input and "shallow" output tensors.
 *
 *  Number of inputs must be divisible by 32. No activation is applied (i.e. linear).
 *
 * Weight tensor `W` is a 2D matrix with shape (C_out, C_in) in standard layout.
 *
 * Bias tensor `B` has shape (C_out), and is in standard layout.
 * 
 * Input tensor `X` has shape (C_in), and is in standard layout.
 * 
 * Output tensor `Y` has shape (C_out), and will be in standard layout.
 *
 * The `shifts` tensor has shape (C_out) and is in standard layout. The 32-bit accumulant
 * is arithmetically right-shifted by this number of bits, with rounding to the nearest integer.
 * Saturation to 16-bit bounds is applied immediately after the shift-and-round.
 * 
 * The `scales` tensor has shape (C_out) and is in standard layout. The scales can be
 * interpreted as signed Q1.14 fixed-point values.
 * 
 * Each output `Y[i]` is computed as
 *
 *  Y[i] = ( ( B[i] + sum(W[i,:] * X[:]) ) >> shifts[i] ) * scales[i]
 *
 *
 *  \param  W       Weight tensor
 *  \param  B       Bias tensor
 *  \param  X       Input tensor
 *  \param  Y       Output tensor
 *  \param  C_out   Number of output channels
 *  \param  C_in    Number of input channels, must be divisible by 32.
 *  \param  shifts  Shift tensor
 *  \param  scales  Scale tensor
 */
static inline void fc_deepin_shallowout_16(
    const int8_t* W, 
    const int32_t* B,
    const int8_t* X, 
    int16_t* Y,
    const int32_t C_out, 
    const int32_t C_in,
    const uint16_t* shifts, 
    const int16_t* scales);



// /**  Fully connected layer for "deep" input and "shallow" output tensors.
//  *
//  *  Number of inputs must be divisible by 32. No activation is applied (i.e. linear).
//  *
//  *  \param  W       Weight tensor of shape (C_out, C_in) using standard layout
//  *                  such that:
//  *                      W[i, j]  =  K[C_in * i  +  j]
//  *  \param  B       Bias tensor of shape (C_out) using a standard layout.
//  *  \param  X       Input tensor of shape (C_in) using standard layout.
//  *  \param  Y       Output tensor of shape (C_out) using standard layout.
//  *  \param  C_out   Number of output channels.
//  *  \param  C_in    Number of input channels, must be divisible by 32.
//  *  \param  shifts  Shift tensor of shape (C_out) using standard layout.
//  *                  Defines the shift used in the 32 to 8 bit conversion via
//  *                  VLSAT. For c >= C_out, the value shifts[y] is undefined.
//  *  \param  scales  Scale tensor of shape (C_out) using standard layout.
//  *                  Defines the scale applied after the 32 to 8 bit
//  *                  conversion. Optional. Can be assumed to be between 0x4000
//  *                  and 0x7FFF. For c >= C_out, the value scales[y] is
//  *                  undefined.
//  */
// static inline void fc_deepin_shallowout_8(
//     const int8_t* W, 
//     const int32_t* B,
//     const int8_t* X, 
//     int8_t* Y,
//     const int32_t C_out, 
//     const int32_t C_in,
//     const uint16_t* shifts, 
//     const int16_t* scales);


/**  Determines the index of the largest element of a vector.
 *
 * The output `C` will be set to the index of the largest element of `A`.
 *
 *  \param  A       Tensor of shape (N) using a standard layout.
 *  \param  C       Output tensor of shape (1).
 *  \param  N       Number of elements in the input tensor A.
 */
static inline void argmax_16(
    const int16_t* A,
    int32_t* C,
    const int32_t N);


/** Reduce the bit depth of a 16-bit vector to 8 bits
 * 
 * Each output ``y[i]`` is computed as:
 *
 *  ``y[i] = (int8_t) round( x[i] / 256.0 )``
 *
 *  If ``y`` and ``x`` point to the same address, the operator
 *  will work in-place on the input vector.
 *
 *  Both ``y`` and ``x`` must be word-aligned.
 *
 * \param y     Output tensor
 * \param x     Input tensor
 * \param n     Length of input and output tensors (in elements)
 */
static inline void requantize_16_to_8(
    int8_t* y,
    const int16_t* x,
    const unsigned n);




/** Prepare to execute a 2D deepin-deepout convolution.
 *
 * This function initializes a `nn_conv2d_dido_params_t` struct with
 * the values necessary to perform the specified convolution.
 * 
 * Once initialized, the contents of the `params` struct will not
 * change, so it need only be initialized once for many (identical)
 * convolutions.
 *
 * The convolution itself may require several partial convolutions corresponding
 * to different (non-overlapping) regions of the output image. Each of these 
 * partial convolutions is described by a `nn_conv2d_dido_block_params_t` struct.
 * As the number of these blocks is not known a priori, their memory is
 * allocated from the heap. The `nn_conv2d_dido_params_t.blocks` field of `params` 
 * will point to the (contiguous) array of `nn_conv2d_dido_block_params_t` blocks.
 *
 * The `nn_conv2d_dido_params_t` struct is intended to be opaque, however, because
 * memory is allocated from the heap, if the same params struct is to be 
 * initialized again, or if it is to go out of scope, it should be properly
 * de-initialized using `conv2d_deepin_deepout_deinit()`.
 */
void conv2d_deepin_deepout_init(
    nn_conv2d_dido_params_t* params,
    const nn_conv2d_init_params_t* init_params,
    const nn_conv2d_region_params_t* region_params,
    const int8_t* K,
    const data16_t* B);

/**
 * De-initialize a `nn_conv2d_dido_params_t` struct which
 * has been previously initialized.
 *
 * Because `conv2d_deepin_deepout_init()` uses `malloc()`, these
 * structs should be de-initialized if they are going to be 
 * initialized again or before they are allowed to go out of scope.
 * 
 * This function will free the memory allocated by 
 * `conv2d_deepin_deepout_init()`.
 */
void conv2d_deepin_deepout_deinit(
    nn_conv2d_dido_params_t* params);



/** Prepare to execute a 2D deepin-deepout convolution.
 *
 * This function initializes a `nn_conv2d_dido_params_t` struct with
 * the values necessary to perform the specified convolution.
 * 
 * Once initialized, the contents of the `params` struct will not
 * change, so it need only be initialized once for many (identical)
 * convolutions.
 *
 * The convolution itself may require several partial convolutions corresponding
 * to different (non-overlapping) regions of the output image. Each of these 
 * partial convolutions is described by a `nn_conv2d_dido_block_params_t` struct.
 * As the number of these blocks is not known a priori, their memory is
 * allocated from the heap. The `nn_conv2d_dido_params_t.blocks` field of `params` 
 * will point to the (contiguous) array of `nn_conv2d_dido_block_params_t` blocks.
 *
 * The `nn_conv2d_dido_params_t` struct is intended to be opaque, however, because
 * memory is allocated from the heap, if the same params struct is to be 
 * initialized again, or if it is to go out of scope, it should be properly
 * de-initialized using `conv2d_deepin_deepout_deinit()`.
 */
void conv2d_shallowin_deepout_init(
    nn_conv2d_sido_params_t* params,
    const nn_conv2d_init_params_t* init_params,
    const nn_conv2d_region_params_t* region_params,
    const int8_t* K,
    const data16_t* B);


/**
 * De-initialize a `nn_conv2d_sido_params_t` struct which
 * has been previously initialized.
 *
 * Because `conv2d_shallowin_deepout_init()` uses `malloc()`, these
 * structs should be de-initialized if they are going to be 
 * initialized again or before they are allowed to go out of scope.
 * 
 * This function will free the memory allocated by 
 * `conv2d_shallowin_deepout_init()`.
 */
void conv2d_shallowin_deepout_deinit(
    nn_conv2d_sido_params_t* params);


/**
 *  Rearranges the data in `B` from the standard tensor layout
 * into Bias tensor layout form 1, as required by the 
 * `conv2d_deepin_deepout()` and `conv2d_shallowin_deepout()`
 * functions.
 * 
 * \param B         Bias tensor in standard tensor layout
 * \param C_out     Length of the bias tensor
 * \returns         `B` recast as a `data16_t` pointer.
 */
data16_t* conv2d_boggle_B(
    int32_t* B,
    const unsigned C_out);

/**
 * Rearranges the data in kernel tensor `K`, provided in standard tensor
 * layout ( with shape (C_out, K_h, K_w, C_in) corresponding to the
 * output channel, kernel row, kernel column and input channel
 * respectively) into the layout required by `conv2d_deepin_deepout()`.
 * 
 * \param K         Kernel tensor
 * \param K_h       Kernel height
 * \param K_w       Kernel width
 * \param C_in      Input channel count
 * \param C_out     Output Channel count
 */
void conv2d_dido_boggle_K(
    int8_t* K,
    const unsigned K_h,
    const unsigned K_w,
    const unsigned C_in,
    const unsigned C_out);
    

/**
 * Re-layout the shift-scale tensor to the format expected by the convolution kernels.
 * 
 * The input tensor should contain all of the shifts followed by all of the scales, in
 * channel order. 
 *
 * A scratch buffer parameter may optionally be supplied (same size as `shiftscales`).
 * If `scratch` is `NULL`, a buffer will be `malloc`ed (and `free`ed).
 *
 * \param shiftscales   The shift/scale tensor. Updated in-place
 * \param C_out         The number of output channels
 * \param scratch       Optional scratch buffer.
 */
void conv2d_boggle_shift_scale(
    int16_t* shiftscales,
    const unsigned C_out,
    int16_t* scratch);


/**
 * Rearranges the data in kernel tensor `K`, provided in ..nearly... standard tensor
 * layout ( with shape (C_out, K_h, 32/C_in, C_in) corresponding to the
 * output channel, kernel row, kernel column and input channel
 * respectively) into the layout required by `conv2d_deepin_deepout()`.
 * 
 * \param K         Kernel tensor
 * \param K_h       Kernel height
 * \param K_w       Kernel width
 * \param C_in      Input channel count
 * \param C_out     Output Channel count
 */
void conv2d_sido_boggle_K(
    int8_t* K,
    const unsigned K_h,
    const unsigned K_w,
    const unsigned C_in,
    const unsigned C_out);

    

#if defined(__XS3A__)

/**
 * Copy size bytes from src to dst.
 *   
 * dst and src both must be word-aligned.
 *  
 * \param dst
 * \param src
 * \param size
*/
void vpu_memcpy(
    void* dst,
    void* src,
    unsigned size);

#endif

#ifdef __XC__
} // extern "C"
#endif

#endif //NN_OPERATOR_H_
