

#ifndef NN_OPERATOR_H_
#define NN_OPERATOR_H_

#include "nn_types.h"
#include "nn_op_structs.h"
#include "nn_op_utils.h"
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

    TODO: Review this. I don't believe this is still true.
*/


/**
Standard Tensor Layout

The standard layout of an N-dimensional tensor with shape `(s1, s2, ..., sN)` is the
same as the typical layout of a C array with the same dimensions. In this layout, the
as memory address increases, indices of each dimension are iterated over, with indices 
iterating in ascending order, and with the later dimensions iterating fastest.

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





/**             Notes on Inner Products and Saturation

 Many functions in this API compute inner products between vectors with many elements. These
 inner products are computed as long sequences of multiply-accumulates on the VPU. Unlike on
 the scalar unit, on the VPU multiplications, additions and subtractions *are not associative*. 

 The lack of associativity is due to the saturation logic used on the VPU. Where the scalar
 unit will roll-over in the case of integer overflows, the XS3 VPU will clamp results
 to the bounds appropriate to the element bit-depth, which, for N-bit (signed) integers,
 is the symmetric range  [-(2^(N-1))+1, (2^(N-1))-1]. Macros are provided for convenience.

    Saturation Bounds:

        Bit depth   Min             Min Macro           Max             Max Macro
        =========   ===             =========           ===             =========
        8-bit:      -127            VPU_INT8_MIN        127             VPU_INT8_MAX
        16-bit:     -65535          VPU_INT16_MIN       65535           VPU_INT16_MAX
        32-bit:     -2147483647     VPU_INT32_MIN       2147483647      VPU_INT32_MAX


When computing inner products, saturation occurs based on the *accumulator* bit depth, rather than
the multiplicand (vector element) bit depth.

        Element         Accumulator
        =======         ===========
        8-bit           32-bit
        16-bit          32-bit
        32-bit          40-bit

Most inner products computed in this API use 8-bit input vectors. The product of two 8-bit signed
integers can be no larger in magnitude than 2^14  (from -(2^7)*-(2^7) ). The largest 32-bit accumulator
value is approximately 2^31, which is (2^14 * 2^17). Thus, an 8-bit inner product cannot 
saturate its 32-bit accumulator unless the vectors are about 128,000 elements long.

However, when inner products are computed, the accumulator is (usually) seeded with an arbitrary
user-supplied 32-bit bias. This bias makes it possible for saturation to occur on inner products
with operand vectors of any length.

Further, saturation can occur at any point during the accumulation, and subsequent steps may
move the accumulator away from the saturation point, and so it may not always be obvious whether
saturation has occurred somewhere inside the inner product, skewing the final result.

Finally, *the functions in this API neither specify nor guarantee the order in which elements
are accumulated when computing inner products*.

Therefore, where saturation in unacceptable, it is incumbent upon the *user* of this library to 
ensure that saturation is not possible given the inputs (matrix/kernel coefficients and input
vectors) and other parameters (e.g. input channel count).
 */



/**         Notes on Output Shifts and Scale

Many functions in this API include shifts and a scale on each output prior to writing the result
to memory. For the sake of brevity, the details of these operations are contained here, rather than 
repeating them in each function's description.

In general, the situation looks like this:

        y[i] <- ((acc32[i] >> shr1[i]) * scale[i]) >> shr2[i]             (16-bit outputs)

            or
        
        y[i] <- (((acc32[i] >> shr1[i]) * scale[i]) >> shr2[i]) >> 8        (8-bit outputs)

      where
        i is the index of the output
        y[i] is the either 8- or 16-bit output
        acc32[i] is the 32-bit accumulator value associated with output i
            (acc32[i] is an intermediate value, which may be the result of convolution or an inner product, etc)
        shr1[i] is the first shift value associated with output i
        scale[i] is the scale value associated with output i
        shr2[i] is teh second shift value associated with output i

Shift 1:
    The shift operation performs several actions atomically:
        - First a "virtual" arithmetic right shift of `shr` bits occurs on the 32-bit accumulator `acc32`.
        - Second, the result of the shift is rounded (as though the shift is a rounding real division by `2^-shr`)
        with ties rounding towards positive infinity.
        - Third, saturation logic is applied to the result of the rounding, clamping the result to [-65535, 65535]. Note
        that this saturation is symmetric.
        - Finally, the bit-depth of the result is reduced to 16 bits.
    While `shr` is a signed 16-bit value, *negative values will be treated as zero*.
    As a final ideosyncrasy, the shifting of negative accumulator values will *never result in zeros*. Where
    the result of shifting a negative value would normally underflow to 0, it will instead result as -1.

Scale:
    The scaling is a signed 16-bit integer multiplication for a 32-bit (actual max value is ((2^15)-1)^2). 

Shift 2:
    This behaves exactly like shift 1, but is applied after the scale.

Final shift:
    In functions that output 8-bit results, a final shift of 8 bits is applied after shift 2 to
    reduce the bit-depth from 16 bits to 8 bits. In this case, rounding occurs, as with the other
    two operations, but no saturation is possible.
    In functions that output 16-bit results, no final shift occurs here.

 */

/**         Bias-Shifts-Scale Tensor Layout

In most cases, where a function requires a Bias-Shifts-Scale (BSS) tensor as input, the required layout will
be as specified here [0].

The BSS tensor contains information pertaining to the `C_out` output channels of an operation. Such a tensor
is 3 dimensional and has shape (`ceil(C_out/16.0)`, `5`, `16`) with elements of type `data16_t`.

The first axis corresponds to output channel groups. The `ceil( )` is applied because for the purposes of
a BSS tensor, the output channel tail must be handled as a full output channel group. [1]

The third axis corresponds to the output channel offset within the output channel group. So, all information
corresponding to output channel `k` will be found in the elements `BSS[(k//16), :, (k%16)]`.

The second axis corresponds to the parameters type, where each index has a specific meaning:

    0 - Bias high half-word: The most significant 16-bits of the 32-bit bias for the output channel
    1 - Bias low half-word: The least significant 16-bits of the 32-bit bias for the output channel
    2 - Shift1: The first right-shift value; applied to the 32-bit accumulator for a 16-bit result.
    3 - Scale: The scale value; applied after shift1; multiplied by the 16-bit result for a 32-bit product
    4 - Shift2: The second right-shift value; applied after scale; down-shifts the 32-bit product for an 8-bit result.

[0]     For general information about how shifts and scales are used, see "Notes on Output Shifts and Scale", 
        for information on how biases are used, see "Notes on Inner Products and Saturation"
[1]     See "Notes on Channel Output and Input Groups".


 */

/**         Notes on Channel Output and Input Groups

Most functions in this API are capable of handling many input and output channels. The XS3 VPU is capable
of consuming many input channels, or producing many output channels in a single operation. These lead to the
concept of input channel groups and output channel groups being baked deeply into this API.

Most functions in this API operate on 8-bit inputs and produce 8-bit outputs [0]. The XS3 vector register size is 256
bits, so in 8-bit mode, the XS3 VPU is capable of loading (and operating on) 32 8-bit values simultaneously [1]. When
the operation is a vector multiply-accumulate, it is desirable to have accumulators which are significantly larger 
than the largest possible product of two 8-bit signed integers (`2^14`). To this end, XS3 uses 32-bit accumulators 
for 8-bit mode [2], spread across the `vD` and `vR` vector registers [3]. That leaves room for 16 accuulators in 8-bit 
mode.

The `xs3_vpu.h` header file contains macros reflecting the number of elements and number of accumulators in each
operating mode. In particular, `VPU_INT8_EPV` is 32, and `VPU_INT8_ACC_PERIOD` is 16 [4].

When the documentation for a function in this API has `C_in` input channels and `C_out` output channels, the function
may require that the parameter tensors consumed by that function be formatted in a layout which reflects the 
channel groups. In most cases, the number of input channel groups will be `ceil(C_in / 32.0)` -- the number of `VLMACCR`
instructions required to multiply-accumulate all input channels -- and the number of output channel groups will be 
`ceil(C_out / 16.0)` -- the number of outputs that can be simultaneously calculated. 

Where tensors are layed out in this way, output channel group 'k' refers to output channels `16*k` to `16*k+15` (inclusive).
Likewise, input channel group `k` refers to input channels `32*k` to `32*k+31` (inclusive). Some tensors will be arranged 
such that one of the dimensions correponds, for example, to the ouput channel group, and another dimension will refer to
the output channel offset. That output (or input) offset is just the offset index *within* the channel group of that channel
(i.e. for output channel `c`, `c % 16`).

Where the number of output (input) channels is not a multiple of 16 (32) --and where the function allows this -- there will be 
an ouput (input) channel tail. The tail are the last channels which do not form complete group. Some tensors, in particular 
the bias-shifts-scale tensors, will require that tails be padded. Whether padding must be zeros, or if it is safe to use 
arbitrary values is specified by the function.

[0]     Some functions work on elements that are not 8 bits, and most functions operate on intermediate values of
        16 or 32 bits.
[1]     Most instructionss in 8-bit mode that load elements from memory(VLMACCR, VLMUL, VLDR, VLADD, etc) load 32 8-bit
        values. The main exception to this is the VLMACC instruction, which loads only 16 8-bit values.
[2]     16-bit mode also uses 32-bit accumulators (and so `VPU_INT8_ACC_PERIOD == VPU_INT16_ACC_PERIOD`), 32-bit mode
        uses 40-bit accumulators.
[3]     In 8- and 16-bit modes, `vD` holds the 16 most significant bits of the accumulator and `vR` holds the 16 least
        significant bits.
[4]     "EPV" is elements per vector. "ACC_PERIOD" is the accumulator period. The accumulators have a "period" because of
        the `VLMACCR` instruction, which adds a 32-element (in 8-bit mode) inner product between vector regiser `vC` and a
        sequence of values in memory to a single accumulator, and then does a cyclic rotation of the accumulators. Then, in 
        8-bit mode 16 (`VPU_INT8_ACC_PERIOD`) `VLMACCR` instructions will perform a full rotation of the accumulators.

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


/**
 * Perform a 2D convolution using a 1x1 kernel over the input image.
 * 
 * The operation performed is:
 * \code
 *      Y[i,j,k]  <--  (((bias[k] + sum(X[i,j,:] * K[k,:])) >> shift1[k]) * scale[k]) >> shift2[k]
 * \endcode
 * 
 * The output tensor `Y` has shape (`y->height`, `y->width`, `y->channels`), where `y` is the same
 * `nn_image_params_t*` passed to `conv2d_1x1_init()`.
 * 
 * The input tensor `X` has shape (`x->height`, `x->width`, `x->channels`), where `x` is the same
 * `nn_image_params_t*` passed to `conv2d_1x1_init()`.
 * 
 * The kernel tensor `K` has shape (`C_out`, `C_in`), where `C_out` has the value of
 * `y->channels` that was passed to `conv2d_1x1_init()`, and `C_in` has the value of 
 * `x->channels` that was passed to `conv2d_1x1_init()`. The layout of `K` is the standard
 * layout, in which the element at `K[i][j]` is the weight which input channel `j` contributes
 * to output channel `i`. Both `C_out` and `C_in` must be multiples of 4.
 * 
 * The bias-shifts-scale tensor `BSS` is layed out as specified in "Bias-Shifts-Scale Tensor Layout".
 * The accumulators for each output channel are seeded with the 32-bit biases encoded in `BSS`, and
 * the shifts and scale are used as specified in "Notes on Output Shifts and Scales".
 * 
 * The input `plan` is an execution plan which contains information about how to manage the input
 * and output data for the provided images. The execution plan can be obtained via `conv2d_1x1_init()`.
 * 
 * NOTE: While the `conv2d_deepin_deepout()` function is capable of handling convoutions with 1x1 kernels,
 *       the kernel tensor layout used by this function is NOT compatible with that required by 
 *       `conv2d_deepin_deepout()`.
 */
static inline void conv2d_1x1(
    int8_t* Y,
    const int8_t* X,
    const int8_t* K,
    const data16_t* BSS,
    const nn_conv2d_1x1_plan_t* plan);

    

    
/**  2D maxpool for an image
 *
 * This function requires an execution plan (`plan`) in order to do its work. An execution plan
 * can be generated using the `maxpool2d_init()` function. 
 * 
 * Max pooling 2D uses a moving window which slides across the input image, and for each position
 * in the input window an output is computed. The specific parameters of the window (including its
 * dimensions and stride lengths), as well as the output paramters (such as the region of the 
 * output tensor to be written) are specified in a `nn_window_op_config_t` struct which is provided
 * to `maxpool2d_init()`.
 * 
 * For a given input window location, this function finds the maximum value inside the input window
 * for each channel (where channels are completely independent of one another), and writes those
 * maxima to the output image.
 *
 * The shape of the `X` input tensor must be `(x->height, x->width, x->channels)`, where `x` is the
 * same `nn_image_params_t` struct passed to `maxpool2d_init()` when `plan` was computed.
 *  
 * The shape of the `Y` output tensor must be `(y->height, y->width, y->channels)`, where `y` is the
 * same `nn_image_params_t` struct passed to `maxpool2d_init()` when `plan` was computed.
 * 
 * \param Y     The output image tensor.
 * \param X     The input image tensor.
 * \param plan  The execution plan.
 *
 */
static inline void maxpool2d(
    int8_t* Y,
    const int8_t* X, 
    const nn_window_op_plan_t* plan);


/** 2D average pooling for an image.
 * 
 * This function applies the average pool operation on the input image `X` to produce the output
 * image `Y`.
 * 
 * A 2D average pool operation applies a spatial window at various locations in an image, and for
 * each of those locations produces an output pixel whose value for each channel `c` is the average
 * of the values for channel `c` within the window in the input image. In other words, the pixels
 * within the window are added up and divided by the number of pixels in the window, where each
 * channel is handled independently.
 * 
 * The parameters of the operation was given in `params`, which should have been initialized using
 * the `avgpool2d_init()` function.
 * 
 * This function supports arbitrary window sizes, strides, and image spatial dimensions. The only
 * restriction is that the number of input (and output) channels must be a multiple of 4, such that
 * every pixel begins at a word-aligned address.
 * 
 * The `Y` parameter is the output image, as was described in the parameters to `avgpool2d_init()`
 * when `params` was initialized.
 * 
 * The `X` parameter is the output image, as was described in the parameters to `avgpool2d_init()`
 * when `params` was initialized.
 * 
 * \param Y         Output image
 * \param X         Input image
 * \param params    Parameters for the operation
 */
static inline void avgpool2d(
    int8_t* Y,
    const int8_t* X, 
    const nn_avgpool2d_plan_t* plan);

/** 2D global average pooling for an 8-bit image
 * 
 * This function is an implementation of `avgpool2d()` optimized for the common case where the window size
 * is equal to the input image size.
 * 
 * For each channel, this function sums all values for that channel from the input image and divides
 * by the number of pixels in the input image, producing the average value for that channel.
 * 
 * This function takes two parameters, `shift` and `scale` which are calculated by the `avpool2d_global_init()`
 * function. That function should be called (just once) at initialization time to obtain the values
 * to be supplied for these parameters.
 * 
 * `x_chans` must be a multiple of 4.
 * 
 * \param Y         Output image
 * \param X         Input image
 * \param x_height  Height of input image (in pixels)
 * \param x_width   Width of input image (in pixels)
 * \param x_chans   Number of channels in the input image
 * \param bias      Initial 32-bit accumulator value. Shared by all channels.
 * \param shift     Shift parameter. Shared by all channels.
 * \param scale     Scale parameter. Shared by all channels.
 */
static inline void avgpool2d_global(
    int8_t* Y,
    const int8_t* X, 
    const uint32_t x_height, 
    const uint32_t x_width,
    const uint32_t x_chans,
    const int32_t  bias,
    const uint32_t shift,
    const uint32_t scale);



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




/** Generalized fully-connected layer with 16-bit outputs.
 * 
 * The logical operation performed is the following:
 * 
 *      Y[i] = ((dot(W[i][], X[] + bias[i]) >> shift[i]) * scale[i]) >> shift2[i];
 * 
 *   where
 *      W[i][] represents the `i`th row of the weight matrix
 *      dot(A,B) is the 32-bit inner product of 8-bit vectors A and B
 *      bias, shift and scale are encoded in `BSS`
 * 
 * `C_in` is the number of elements in `X` as well as the number of columns in `W`. 
 * `C_in` must be a multiple of `4`.
 * 
 * There are no restrictions on the value of `C_out`.
 * 
 * `Y` is the 16-bit output vector and has a shape of `(C_out)`. `Y` must begin at a word-
 * aligned address.
 * 
 * `W` is the 8-bit weight matrix with a shape of `(C_out, C_in)`. `W` must begin at a word-
 * aligned address.
 * 
 * `X` is the 8-bit input vector with a shape of `(C_in)`. `X` must begin at a word-aligned address.
 * 
 * `BSS` is the bias-shift-scale tensor with a shape of `(ceil(C_out/16), 5, 16)`. This tensor
 * encodes the bias, shift and scale for each output channel into a single linear block of memory,
 * allowing a more efficient implementation of this operator. The function `fc_boggle_BSS()` is
 * provided to simplify the layout of this tensor. Use of `fc_boggle_BSS` is not required, but
 * refer to its documentation if you wish to layout this tensor manually (to reduce initialization
 * time). 
 * 
 * NOTE: This function computes inner products of arbitrary length and is thus susceptible to accumulator
 * saturation. See the Notes on Inner Products and Saturation section above for more information.
 * 
 * NOTE: See the section Notes on Output Shift and Scale for more details about how the final shift
 * and scale are applied to get the final result.
 * 
 * NOTE: This implementation will be most efficient when `C_in` is a multiple of `32`, and `C_out` is
 * a multiple of `16`.
 * 
 * NOTE: The requirement that `C_in` be a multiple of 4 is imposed by the word-alignment constraint of the VPU.
 * To use this function with input vectors whose length is not a multiple of `4`, pad the matrix `W` on 
 * right with zeros until its width is a multiple of `4`, and use new width as `C_in`. So long as `W` is 
 * padded with zeros, `X` does not need to be padded out. Alternatively, if setting many elements of 
 * `W` to zero is an insufferable cost, `X` can padded out with zeros so that its size is a multiple 
 * of 4 elements, in which case it does not matter what `W` is padded with, *though it must still be 
 * padded*.
 * 
 */
static inline void fully_connected_16(
    int16_t* Y,
    const int8_t* W, 
    const int8_t* X, 
    const data16_t* BSS,
    const nn_fully_connected_plan_t* plan);



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



/** 8-bit Look-up Table
 * 
 * `lut` is used as a look-up table mapping 8-bit inputs to 8-bit outputs.
 * 
 * The following operation is applied:
 *  
 * \code
 *      Y[i] <- lut[X[i]];
 * 
 *         for  0 <= i < length
 * \endcode 
 * 
 * NOTE: This function can safely operate in-place on a buffer. To do the look-up in-place
 *        just pass the same address for both `X` and `Y`.
 * 
 * \param Y         
 * \param X         
 * \param lut       
 * \param length    
 */
static inline void lookup8(
    uint8_t* Y,
    const uint8_t* X,
    const uint8_t* lut,
    const unsigned length);




    

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
