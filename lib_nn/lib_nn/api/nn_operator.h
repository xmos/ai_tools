

#ifndef NN_OPERATOR_H_
#define NN_OPERATOR_H_

#include "nn_types.h"
#include "nn_op_structs.h"
#include "nn_conv2d_structs.h"
#include "nn_op_utils.h"
#include "nn_op_init.h"
#include "nn_operator_asm.h"
#include "nn_operator_c.h"
#include "nn_operator_inline.h"
#include "nn_op_advanced.h"

#include <stdint.h>

#include "xs3_vpu.h"

#ifdef __XC__
extern "C" {
#endif


/**
 * @brief Perform a deep 2D convolution of an input image.
 * 
 * The convolution performed by this function is "deep" in the sense that many input channels and many
 * output channels are supported, though both the input channel count @math{X_c} and output channel count 
 * @math{Y_c} must be a multiple of `4`. 
 * 
 * This function also supports implied padding of the input image in which a specified value is used as 
 * the padding value.
 * 
 * Before performing a 2D convolution using this function, a call must be made to `conv2d_deep_init()` to
 * initialize a `nn_conv2d_deep_plan_t` struct and one or more `nn_conv2d_deep_job_t` structs. Each job,
 * together with the plan shared by all jobs, contains the information required to compute a subset of the
 * pixels in the output image. More specifically, each job corresponds to a rectangular subtensor of @tensor{Y},
 * which is to be computed by that job.
 * 
 * Splitting the convolution into multiple jobs may serve several ends, including parallelization, returning
 * control to caller to service other application resources, or even to improve the performance of the 
 * operation itself.
 * 
 * `Y` points to the output image tensor @tensor{Y} with shape @tensor_shape{Y_h, Y_w, Y_c}, which correspond to
 * the output image rows, columns and channels respectively. The dimensions of @tensor{Y} must be as specified 
 * when `plan` was initialized. The address supplied for `Y` should be the start address of the output image
 * tensor, *not* the start address of the sub-tensor being computed by the current job.
 * 
 * `X` points to the input image tensor @tensor{X} with shape @tensor_shape{X_h, X_w, X_c}, which correspond to
 * the input image rows, columns and channels respectively. The dimensions of @tensor{X} must be as specified
 * when `plan` was initialized. The address supplied for `X` should be the start address of input image tensor,
 * *not* the address at which the convolution window starts for the job being processed.
 * 
 * The memory layout of @tensor{Y} and @tensor{X} are the standard memory layout for image tensors (see @ref 
 * standard_layout).
 * 
 * `K` points to the kernel tensor @tensor{K} with shape @tensor_shape{Y_c, K_h, K_w, X_c}, which correspond to
 * the output image channels, convolution window rows and columns, and the input image channels respectively.
 * The dimensions of @tensor{K} must be as specified when `plan` was initialized. The address supplied for `K`
 * should be the start address of the kernel tensor.
 * 
 * The memory layout of @tensor{K} is the standard memory layout for 4D tensors (see @ref standard_layout).
 * 
 * `BSS` points to an array of bias-shifts-scale parameters required for this convolution. Each 
 * `nn_bss_block_t` in the array contains the bias-shifts-scale parameters for a single output channel group,
 * (@ttref{VPU_INT8_ACC_PERIOD} output channels). If @math{Y_c} is not a multiple of @ttref{VPU_INT8_ACC_PERIOD}, 
 * then the output channel tail ( the last @math{(Y_c mod 16)} output channels) also gets `nn_bss_block_t`, where
 * the entries corresponding to channels beyond @math{Y_c} are ignored. The address supplied for `BSS` should be
 * the start address of the the array, *not* the address of the `nn_bss_block_t` corresponding of the first output
 * channel of the job being processed.
 * 
 * `plan` points to the `nn_conv2d_deep_plan_t` which was previously initialized with a call to `conv2d_deep_init()`.
 * 
 * `job` points to the job to be performed in this call, which was previously initialized along-side `plan`. 
 * 
 * Note that a single call to this function processes only a *single job*. If multiple jobs were initialized,
 * performing the complete convolution requires multiple calls to this function. In such a case, the `Y`, `X`,
 * `K`, `BSS`, and `plan` pointers will be identical in each call, and the `job` pointer will be different with
 * each call.
 * 
 * @requires_word_alignment{Y,X,K,BSS}
 * 
 * @param[out] Y        The output image @tensor{Y}
 * @param[in]  X        The input image @tensor{X}
 * @param[in]  K        The kernel tensor @tensor{K}
 * @param[in]  BSS      The bias-shifts-scale parameters
 * @param[in]  plan     The convolution plan
 * @param[in]  job      The convolution job
 */
void conv2d_deep(
    nn_image_t* Y,
    const nn_image_t* X,
    const nn_tensor_t* K,
    const nn_bss_block_t* BSS,
    const nn_conv2d_deep_plan_t* plan,
    const nn_conv2d_deep_job_t* job);








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
void conv2d_1x1(
    int8_t* Y,
    const int8_t* X,
    const int8_t* K,
    const data16_t* BSS,
    const nn_conv2d_1x1_plan_t* plan);




/**
 * Perform a depthwise 2D convolution of an image.
 * 
 * A depthwise 2D convolution is one in which the number of output channels is equal to the
 * number of input channels, and where the `k`th output channel receives contributions from 
 * only the `k`th input channel (and the bias, shifts and scale associated with channel `k`). 
 * 
 * This function requires a plan (`nn_conv2d_depthwise_plan_t`) and a job (`nn_conv2d_depthwise_job_t`)
 * which specify the work to be performed and how to perform it. These structs are initialized
 * with a call to `conv2d_depthwise_init()`.
 * 
 * The computation of the output image can be done all at once with a single call, or may be 
 * divided among multiple calls. In either case, the particular work to be done by a given call
 * is specified via a job. Each job specifies a rectangular region of the output image to be computed
 * in a call to `conv2d_depthwise()`.
 * 
 * Dividing the work among several jobs can be used to parallelize the
 * work to be done across cores, or, for example, to periodically return control to the caller 
 * so that other resources can be serviced in a timely manner.
 * 
 * `Y` is a pointer to the output image to be computed (either fully or partially, depending on `job`).
 * The output image must have the same dimensions as were supplied via the `y_params` argument to
 *  `conv2d_depthwise_init()` when `plan` was initialized.
 * 
 * `X` is a pointer to the input image. The input image must have the same dimensions as were supplied
 * via the `x_params` argument to `conv2d_depthwise_init()` when `plan` was initialized.
 * 
 * `K` is a pointer to the kernel tensor which is to be convolved with the input image to produce the
 * output image. The shape of the kernel tensor must be `(K_h, K_w, C_in)`, where `K_h` and `K_w` are the
 * height and width respectively of the kernel tensor, and `C_in` is the number of channels in the input 
 * image (via `x_params->channels`) as supplied to `conv2d_depthwise_init()` when `plan` was initialized. 
 * 
 * `BSS` is a pointer to the bias-shifts-scale tensor. `BSS` is layed out as specified in "Bias-Shifts-
 * Scale Tensor Layout". The accumulators for each output channel are seeded with the 32-bit biases 
 * encoded in `BSS`, and the shifts and scale are used as specified in "Notes on Output Shifts 
 * and Scales".
 * 
 * `plan` is a pointer to the depthwise convolution plan initialized by a previous call to
 * `conv2d_depthwise_init()`.
 * 
 * `job` is a pointer to a single depthwise convolution job initialized with `plan` in a previous call
 * to `conv2d_depthwise_init()`. If more than one job was initialized with `plan`, then to perform
 * all jobs, the user must call this function multiple times, each time passing the same plan, 
 * and a different one of the initialized jobs.
 * 
 * Constraints:
 *  - `Y`, `X`, `K` and `BSS` must all point to word-aligned addresses.
 *  - The input and output images must each have a multiple of 4 channels.
 * 
 * \param Y     The output image.
 * \param X     The input image.
 * \param K     The kernel tensor.
 * \param BSS   The bias-shifts-scale tensor.
 * \param plan  The execution plan initialized by `conv2d_depthwise_init()`.
 * \param job   The (single) job to be performed.
 */
void conv2d_depthwise(
    int8_t* Y,
    const int8_t* X,
    const int8_t* K,
    const nn_bss_block_t* BSS,
    const nn_conv2d_depthwise_plan_t* plan,
    const nn_conv2d_depthwise_job_t* job);

    

    
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
void maxpool2d(
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
void avgpool2d_global(
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
void fully_connected_16(
    int16_t* Y,
    const int8_t* W, 
    const int8_t* X, 
    const data16_t* BSS,
    const nn_fully_connected_plan_t* plan);




/**  Determines the index of the largest element of a vector.
 *
 * The output `C` will be set to the index of the largest element of `A`.
 *
 *  \param  A       Tensor of shape (N) using a standard layout.
 *  \param  C       Output tensor of shape (1).
 *  \param  N       Number of elements in the input tensor A.
 */
void argmax_16(
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
void requantize_16_to_8(
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
void lookup8(
    uint8_t* Y,
    const uint8_t* X,
    const uint8_t* lut,
    const unsigned length);



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


#ifdef __XC__
} // extern "C"
#endif

#endif //NN_OPERATOR_H_
