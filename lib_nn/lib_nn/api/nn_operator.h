

#ifndef NN_OPERATOR_H_
#define NN_OPERATOR_H_

#include "nn_config.h"
#include "nn_types.h"
#include "nn_op_structs.h"
#include "nn_conv2d_structs.h"
#include "nn_op_utils.h"
#include "nn_op_init.h"
#include "nn_op_advanced.h"

#include <stdint.h>

#include "xs3_vpu.h"

#ifdef __XC__
extern "C" {
#endif

/**
 * @brief Execute @oper{conv2d_deep} job.
 *
 * See @oper_ref{conv2d_deep} for more details about the @oper{conv2d_deep}
 * operator.
 *
 * An instance of the @oper{conv2d_deep} operator requires an initialized plan
 * and one or more jobs. See conv2d_deep_init() for more details.
 *
 * `Y` points to the output image @tensor{Y} with shape @tensor_shape{Y_h, Y_w,
 * Y_c}. The address supplied for `Y` should be the start address of the output
 * image (for any job being processed).
 *
 * `X` points to the input image @tensor{X} with shape @tensor_shape{X_h, X_w,
 * X_c}. The address supplied for `X` should be the start address of the input
 * image (for any job being processed).
 *
 * The memory layout of @tensor{Y} and @tensor{X} are the standard memory layout
 * for image tensors (see @ref standard_layout).
 *
 * `K` points to the kernel tensor @tensor{K} with shape @tensor_shape{Y_c, K_h,
 * K_w, X_c}, which correspond to the output image channels, convolution window
 * rows and columns, and the input image channels respectively. The address
 * supplied for `K` should be the start address of the kernel tensor (for any
 * job being processed).
 *
 * The memory layout of @tensor{K} is the standard memory layout for 4D tensors
 * (see @ref standard_layout).
 *
 * `BSO` points to an array of bias-scale-offset parameters required for this
 * convolution. See @ref bso_layout for details on the encoding of this array.
 * The address supplied for `BSO` should be the start address of the the array
 * (for any job being processed).
 *
 * `plan` points to the (initialized) plan associated with this instance of the
 * @oper{conv2d_deep} operator.
 *
 * `job` points to the (initialized) job to be performed with this call.
 *
 * @requires_word_alignment{Y,X,K,BSO}
 *
 * @param Y    [out]    The output image @tensor{Y}
 * @param X    [in]     The input image @tensor{X}
 * @param K    [in]     The kernel tensor @tensor{K}
 * @param BSO  [in]     The bias-scale-offset array
 * @param plan [in]     The @oper{conv2d_deep} plan to be processed
 * @param job  [in]     The @oper{conv2d_deep} job to be processed
 */
void conv2d_deep(nn_image_t* Y, const nn_image_t* X, const nn_tensor_t* K,
                 const nn_bso_block_t* BSO, const nn_conv2d_deep_plan_t* plan,
                 const nn_conv2d_deep_job_t* job);

/**
 * @brief Execute @oper{conv2d_shallowin} job.
 *
 * See @oper_ref{conv2d_shallowin} for more details about the
 * @oper{conv2d_shallowin} operator.
 *
 * An instance of the @oper{conv2d_shallowin} operator requires an initialized
 * plan and one or more jobs. See conv2d_shallowin_init() for more details.
 *
 * `Y` points to the output image @tensor{Y} with shape @tensor_shape{Y_h, Y_w,
 * Y_c}. The address supplied for `Y` should be the start address of the output
 * image (for any job being processed).
 *
 * `X` points to the input image @tensor{X} with shape @tensor_shape{X_h, X_w,
 * X_c}. The address supplied for `X` should be the start address of the input
 * image (for any job being processed).
 *
 * The memory layout of @tensor{Y} and @tensor{X} are the standard memory layout
 * for image tensors (see @ref standard_layout).
 *
 * `K` points to the kernel tensor @tensor{K} encoded with shape
 * @tensor_shape{Y_c, K_h, \hat{K_w}, X_c}, where
 * @math{Y_c}, @math{K_h} and @math{X_c} correspond to the output image
 * channels, convolution window rows and the input image channel count
 * respectively. @math{\hat{K_w}} is the augmented convolution window width,
 * which must be exactly
 * @math{32/X_c}, regardless of the convolution window width @math{K_w}. The
 * address supplied for `K` should be the start address of the kernel tensor
 * (for any job being processed).
 *
 * The memory layout of @tensor{K} (with the modified 3rd dimension) is the
 * standard memory layout for 4D tensors (see
 * @ref standard_layout). Further, the coefficients for all elements
 * @math{K\left[i,j,k,l\right]} where @math{k\geq K_w} must have the value 0.
 *
 * `BSO` points to an array of bias-scale-offset parameters required for this
 * convolution. See @ref bso_layout for details on the encoding of this array.
 * The address supplied for `BSO` should be the start address of the the array
 * (for any job being processed).
 *
 * `plan` points to the (initialized) plan associated with this instance of the
 * @oper{conv2d_shallowin} operator.
 *
 * `job` points to the (initialized) job to be performed with this call.
 *
 * @requires_word_alignment{Y,X,K,BSO}
 *
 * @param Y    [out]    The output image @tensor{Y}
 * @param X    [in]     The input image @tensor{X}
 * @param K    [in]     The kernel tensor @tensor{K}
 * @param BSO  [in]     The bias-scale-offset array
 * @param plan [in]     The @oper{conv2d_shallowin} plan to be processed
 * @param job  [in]     The @oper{conv2d_shallowin} job to be processed
 */
void conv2d_shallowin(nn_image_t* Y, const nn_image_t* X, const nn_tensor_t* K,
                      const nn_bso_block_t* BSO,
                      const nn_conv2d_shallowin_plan_t* plan,
                      const nn_conv2d_shallowin_job_t* job);

/**
 * @brief Perform a 2D convolution of a shallow input image.
 *
 * Perform a 2D convolution of kernel tensor @tensor{K} with input image
 * @tensor{X} to produce output image @tensor{Y}.
 *
 * This function is optimized for input images that have 3 channels, but will
 * work with any number of input channels. This will use more memory than the
 * TFLite reference implementation, but run much faster:
 *
 * Additional memory: 1 patch worth (K_w * K_h * C_in) bytes + 32 bytes + some
 * code Performance gain: Depends on patch size vs # output channels, and
 * padding, but approximately: 16x faster convolutions - PATCH_SIZE copy
 * operations
 *
 * @note multiples of 16 output channels will run fastest input channels not
 * imporant, but a PATCH_SIZE that is a multiple of 32 will be the fastest, most
 * memory effcient
 *
 * `Y` points to the output image tensor @tensor{Y} with shape
 * @tensor_shape{Y_h, Y_w, Y_c}, which correspond to the output image rows,
 * columns and channels respectively. The dimensions of @tensor{Y} must be as
 * specified when `plan` was initialized. The address supplied for `Y` should be
 * the start address of the output image tensor, *not* the start address of the
 * sub-tensor being computed by the current job.
 *
 * `X` points to the input image tensor @tensor{X} with shape @tensor_shape{X_h,
 * X_w, X_c}, which correspond to the input image rows, columns and channels
 * respectively. The dimensions of @tensor{X} must be as specified when `plan`
 * was initialized. The address supplied for `X` should be the start address of
 * input image tensor, *not* the address at which the convolution window starts
 * for the job being processed.
 *
 * The memory layout of @tensor{Y} and @tensor{X} are the standard memory layout
 * for image tensors (see @ref standard_layout).
 *
 * `COL` points to a caller supplied buffer that will be used for the internal
 * im2col() transformation. This buffer needs to be word-aligned and a multiple
 * of 32-bytes in length, no shorter than: (K_w * K_h * C_in) bytes.
 *
 * `K` points to the kernel tensor @tensor{K} with shape @tensor_shape{Y_c, K_h
 * * K_w * X_c}, where @math{Y_c},
 * @math{K_h} @math{K_w} and @math{X_c} correspond to the output image channels,
 * convolution window rows, convolution window columns, and the input image
 * channel count respectively.
 *
 * The memory layout of @tensor{K} is a row-major 2D matrix
 *
 * `BSO` points to an array of bias-shifts-scale parameters required for this
 * convolution. Each `nn_bso_block_t` in the array contains the
 * bias-shifts-scale parameters for a single output channel group,
 * (@ttref{VPU_INT8_ACC_PERIOD} output channels). If @math{Y_c} is not a
 * multiple of @ttref{VPU_INT8_ACC_PERIOD}, then the output channel tail ( the
 * last @math{(Y_c mod 16)} output channels) also gets `nn_bso_block_t`, where
 * the entries corresponding to channels beyond @math{Y_c} are ignored. The
 * address supplied for `BSO` should be the start address of the the array,
 * *not* the address of the `nn_bso_block_t` corresponding of the first output
 * channel of the job being processed.
 *
 * `plan` points to the `nn_conv2d_im2col_plan_t` which was previously
 * initialized with a call to `conv2d_im2col_init()`.
 *
 * `job` points to the job to be performed in this call, which was previously
 * initialized along-side `plan`.
 *
 * @requires_word_alignment{Y,X,COL,K,BSO}
 *
 * @param[out] Y        The output image @tensor{Y}
 * @param[in]  X        The input image @tensor{X}
 * @param[in]  COL      Scratch space for im2col (multiple of 32 words >= |K|)
 * @param[in]  K        The kernel tensor @tensor{K}
 * @param[in]  BSO      The bias-shifts-scale parameters
 * @param[in]  plan     The convolution plan
 * @param[in]  job      The convolution job
 */
void conv2d_im2col(nn_image_t* Y, const nn_image_t* X, const nn_image_t* COL,
                   const nn_tensor_t* K, const nn_bso_block_t* BSO,
                   const nn_conv2d_im2col_plan_t* plan,
                   const nn_conv2d_im2col_job_t* job);

/**
 * @brief Execute @oper{conv2d_1x1} job.
 *
 * See @oper_ref{conv2d_1x1} for more details about the @oper{conv2d_1x1}
 * operator.
 *
 * An instance of the @oper{conv2d_1x1} operator requires a plan and one or more
 * jobs, which are represented by the `nn_conv2d_1x1_plan_t` and
 * `nn_conv2d_1x1_job_t` structs. Before performing a 2D convolution using this
 * function, a call must be made to conv2d_1x1_init() to initialize the plan and
 * any jobs.
 *
 * An instance of the @oper{conv2d_1x1} operator requires an initialized plan
 * and one or more jobs. See conv2d_1x1_init() for more details.
 *
 * `Y` points to the output image @tensor{Y} with shape @tensor_shape{Y_h, Y_w,
 * Y_c}. The address supplied for `Y` should be the start address of the output
 * image (for any job being processed).
 *
 * `X` points to the input image @tensor{X} with shape @tensor_shape{X_h, X_w,
 * X_c}. The address supplied for `X` should be the start address of the input
 * image (for any job being processed).
 *
 * The memory layout of @tensor{Y} and @tensor{X} are the standard memory layout
 * for image tensors (see @ref standard_layout).
 *
 * `K` points to the kernel tensor @tensor{K} with shape @tensor_shape{Y_c,
 * X_c}, which correspond to the output image channels and input image channels
 * respectively. The address supplied for `K` should be the start address of the
 * kernel tensor (for any job being processed).
 *
 * The memory layout of @tensor{K} is the standard memory layout for 2D tensors
 * (see @ref standard_layout).
 *
 * `BSO` points to an array of bias-scale-offset parameters required for this
 * convolution. See @ref bso_layout for details on the encoding of this array.
 * The address supplied for `BSO` should be the start address of the the array
 * (for any job being processed).
 *
 * `plan` points to the (initialized) plan associated with this instance of the
 * @oper{conv2d_deep} operator.
 *
 * `job` points to the (initialized) job to be performed with this call.
 *
 * @requires_word_alignment{Y,X,K,BSO}
 *
 * @param Y    [out]    The output image @tensor{Y}
 * @param X    [in]     The input image @tensor{X}
 * @param K    [in]     The kernel tensor @tensor{K}
 * @param BSO  [in]     The bias-scale-offset array
 * @param plan [in]     The @oper{conv2d_1x1} plan to be processed
 * @param job  [in]     The @oper{conv2d_1x1} job to be processed
 */
void conv2d_1x1(nn_image_t* Y, const nn_image_t* X, const nn_tensor_t* K,
                const nn_bso_block_t* BSO, const nn_conv2d_1x1_plan_t* plan,
                const nn_conv2d_1x1_job_t* job);

// Binary operators

#define CONV2D_OUTPUT_LENGTH(input_length, filter_size, dilation, stride)     \
  (((input_length - (filter_size + (filter_size - 1) * (dilation - 1)) + 1) + \
    stride - 1) /                                                             \
   stride)

void bnn_reorder_threshold_tensor(const int32_t* thresh_boggled,
                                  const int32_t* thresholds_ref,
                                  const unsigned chans_out,
                                  const unsigned receptive_field);

void bnn_reorder_kernel_tensor(const bnn_b256_t* K_p, const bnn_b256_t* K_ref_p,
                               const unsigned k_height, const unsigned k_width,
                               const unsigned chans_in,
                               const unsigned chans_out);

// This is the actual kernel
void bnn_conv2d_bin_out_asm(const nn_bnn_conv2d_bin_out_asm_plan_t* plan);

void bnn_conv2d_bin_out_asm_prepare(
    nn_bnn_conv2d_bin_out_asm_plan_t* plan, const bnn_b32_t* Y_p,
    const bnn_b256_t* X_p, const bnn_b256_t* K_p, const int32_t* thresholds_p,
    const nn_image_params_t* x, const nn_image_params_t* y,
    const nn_window_params_t* k, const unsigned y_loc_x, const unsigned y_loc_y,
    const unsigned x_loc_x, const unsigned x_loc_y, const unsigned k_loc_x,
    const unsigned k_loc_y, const unsigned y_full_width,
    const unsigned x_full_width, const unsigned k_full_width);

void bnn_conv2d_bin_out_threshold_prepare(
    const nn_bnn_conv2d_bin_out_asm_plan_t* plan, int32_t* kernel_weight,
    int32_t* unmodified_thresholds);

void bnn_conv2d_bin_out(bnn_b32_t* Y_p, const bnn_b256_t* X_p,
                        const bnn_b256_t* K_p,
                        int32_t* thresholds,  //[out_channel];
                        const nn_bnn_conv2d_bin_out_plan_t* plan);

void bnn_conv2d_bin_out_init(nn_bnn_conv2d_bin_out_plan_t* plan,
                             const nn_image_params_t* x,
                             const nn_image_params_t* y,
                             const nn_window_params_t* k);

/**
 * @brief Execute @oper{conv2d_depthwise} job.
 *
 * See @oper_ref{conv2d_depthwise} for more details about the
 * @oper{conv2d_depthwise} operator.
 *
 * An instance of the @oper{conv2d_depthwise} operator requires an initialized
 * plan and one or more jobs. See conv2d_depthwise_init() for more details.
 *
 * `Y` points to the output image @tensor{Y} with shape @tensor_shape{Y_h, Y_w,
 * X_c}. The address supplied for `Y` should be the start address of the output
 * image (for any job being processed).
 *
 * `X` points to the input image @tensor{X} with shape @tensor_shape{X_h, X_w,
 * X_c}. The address supplied for `X` should be the start address of the input
 * image (for any job being processed).
 *
 * The memory layout of @tensor{Y} and @tensor{X} are the standard memory layout
 * for image tensors (see @ref standard_layout).
 *
 * `K` points to the kernel tensor @tensor{K} with shape @tensor_shape{K_h, K_w,
 * X_c}, which correspond to the convolution window rows and columns and the
 * input image channels respectively. The address supplied for `K` should be the
 * start address of the kernel tensor (for any job being processed).
 *
 * The memory layout of @tensor{K} is the standard memory layout for 3D tensors
 * (see @ref standard_layout).
 *
 * `BSO` points to an array of bias-scale-offset parameters required for this
 * convolution. See @ref bso_layout for details on the encoding of this array.
 * The address supplied for `BSO` should be the start address of the the array
 * (for any job being processed).
 *
 * `plan` points to the (initialized) plan associated with this instance of the
 * @oper{conv2d_depthwise} operator.
 *
 * `job` points to the (initialized) job to be performed with this call.
 *
 * @requires_word_alignment{Y,X,K,BSO}
 *
 * @param Y    [out]    The output image @tensor{Y}
 * @param X    [in]     The input image @tensor{X}
 * @param K    [in]     The kernel tensor @tensor{K}
 * @param BSO  [in]     The bias-scale-offset array
 * @param plan [in]     The @oper{conv2d_depthwise} plan to be processed
 * @param job  [in]     The @oper{conv2d_depthwise} job to be processed
 */
void conv2d_depthwise(nn_image_t* Y, const nn_image_t* X, const nn_tensor_t* K,
                      const nn_bso_block_t* BSO,
                      const nn_conv2d_depthwise_plan_t* plan,
                      const nn_conv2d_depthwise_job_t* job);

/**
 * @brief Execute @oper{maxpool2d} job.
 *
 * See @oper_ref{maxpool2d} for more details about the @oper{maxpool2d}
 * operator.
 *
 * An instance of the @oper{maxpool2d} operator requires an initialized plan and
 * one or more jobs. See maxpool2d_init() for more details.
 *
 * `Y` points to the output image @tensor{Y} with shape @tensor_shape{Y_h, Y_w,
 * X_c}. The address supplied for `Y` should be the start address of the output
 * image (for any job being processed).
 *
 * `X` points to the input image @tensor{X} with shape @tensor_shape{X_h, X_w,
 * X_c}. The address supplied for `X` should be the start address of the input
 * image (for any job being processed).
 *
 * The memory layout of @tensor{Y} and @tensor{X} are the standard memory layout
 * for image tensors (see @ref standard_layout).
 *
 * `plan` points to the (initialized) plan associated with this instance of the
 * @oper{maxpool2d} operator.
 *
 * `job` points to the (initialized) job to be performed with this call.
 *
 * @requires_word_alignment{Y,X}
 *
 * @param Y    [out]    The output image @tensor{Y}
 * @param X    [in]     The input image @tensor{X}
 * @param plan [in]     The @oper{maxpool2d} plan to be processed
 * @param job  [in]     The @oper{maxpool2d} job to be processed
 */
void maxpool2d(nn_image_t* Y, const nn_image_t* X,
               const nn_maxpool2d_plan_t* plan, const nn_pool2d_job_t* job);

/**
 * @brief Execute @oper{avgpool2d} job.
 *
 * See @oper_ref{avgpool2d} for more details about the @oper{avgpool2d}
 * operator.
 *
 * An instance of the @oper{avgpool2d} operator requires an initialized plan and
 * one or more jobs. See avgpool2d_init() for more details.
 *
 * `Y` points to the output image @tensor{Y} with shape @tensor_shape{Y_h, Y_w,
 * X_c}. The address supplied for `Y` should be the start address of the output
 * image (for any job being processed).
 *
 * `X` points to the input image @tensor{X} with shape @tensor_shape{X_h, X_w,
 * X_c}. The address supplied for `X` should be the start address of the input
 * image (for any job being processed).
 *
 * The memory layout of @tensor{Y} and @tensor{X} are the standard memory layout
 * for image tensors (see @ref standard_layout).
 *
 * `plan` points to the (initialized) plan associated with this instance of the
 * @oper{avgpool2d} operator.
 *
 * `job` points to the (initialized) job to be performed with this call.
 *
 * @requires_word_alignment{Y,X}
 *
 * @param Y    [out]    The output image @tensor{Y}
 * @param X    [in]     The input image @tensor{X}
 * @param plan [in]     The @oper{avgpool2d} plan to be processed
 * @param job  [in]     The @oper{avgpool2d} job to be processed
 */
static inline void avgpool2d(nn_image_t* Y, const nn_image_t* X,
                             const nn_avgpool2d_plan_t* plan,
                             const nn_pool2d_job_t* job) {
  switch (plan->impl) {
    case AVGPOOL2D_2X2:
      avgpool2d_2x2(Y, X, plan, job);
      break;
    default:
      avgpool2d_gen(Y, X, plan, job);
      break;
  }
}

/**
 * @brief Execute @oper{avgpool2d_global} job.
 *
 * See @oper_ref{avgpool2d_global} for more details about the
 * @oper{avgpool2d_global} operator.
 *
 * An instance of the @oper{avgpool2d_global} operator requires an initialized
 * plan and one or more jobs. See avgpool2d_global_init() for more details.
 *
 * `Y` points to the output vector @tensor{y} with shape @tensor_shape{X_c}. The
 * address supplied for `Y` should be the start address of the output vector
 * (for any job being processed).
 *
 * `X` points to the input image @tensor{X} with shape @tensor_shape{X_h, X_w,
 * X_c}. The address supplied for `X` should be the start address of the input
 * image (for any job being processed).
 *
 * The memory layout of @tensor{Y} and @tensor{X} are the standard memory layout
 * for image tensors (see @ref standard_layout).
 *
 * `bias` is the bias @math{b} with which the accumulators are initialized.
 *
 * `plan` points to the (initialized) plan associated with this instance of the
 * @oper{avgpool2d_global} operator.
 *
 * `job` points to the (initialized) job to be performed with this call.
 *
 * @requires_word_alignment{Y,X}
 *
 * @param Y    [out]    The output vector @tensor{y}
 * @param X    [in]     The input image @tensor{X}
 * @param bias [in]     Initial 32-bit accumulator value @math{B}. Shared by all
 * channels.
 * @param plan [in]     The @oper{avgpool2d_global} plan to be processed
 * @param job  [in]     The @oper{avgpool2d_global} job to be processed
 */
void avgpool2d_global(int8_t* Y, const int8_t* X, const int32_t bias,
                      const nn_avgpool2d_global_plan_t* plan,
                      const nn_avgpool2d_global_job_t* job);

/**
 * @brief Execute @oper{fully_connected_16} job.
 *
 * See @oper_ref{fully_connected_16} for more details about the
 * @oper{fully_connected_16} operator.
 *
 * An instance of the @oper{fully_connected_16} operator requires an initialized
 * plan and one or more jobs. See fully_connected_init() for more details.
 *
 * `Y` points to the output vector @tensor{y} with length @tensor_shape{M}. The
 * address supplied for `Y` should be the start address of the output image (for
 * any job being processed).
 *
 * `X` points to the input vector @tensor{x} with length @tensor_shape{N}. The
 * address supplied for `X` should be the start address of the input image (for
 * any job being processed).
 *
 * `W` points to the weight matrix @tensor{W} with shape @tensor_shape{M, N},
 * which correspond to the output and input vector respectively. The address
 * supplied for `W` should be the start address of the weight matrix (for any
 * job being processed).
 *
 * The memory layout of @tensor{W} is the standard memory layout for 2D tensors
 * (see @ref standard_layout).
 *
 * `BSO` points to an array of bias-scale-offset parameters required for this
 * convolution. See @ref bso_layout for details on the encoding of this array.
 * The address supplied for `BSO` should be the start address of the the array
 * (for any job being processed).
 *
 * `plan` points to the (initialized) plan associated with this instance of the
 * @oper{fully_connected_16} operator.
 *
 * `job` points to the (initialized) job to be performed with this call.
 *
 * @requires_word_alignment{Y,X,K,BSO}
 *
 * @param Y    [out]    The output vector @tensor{y}
 * @param W    [in]     The weight matrix @tensor{W}
 * @param X    [in]     The input vector @tensor{x}
 * @param BSO  [in]     The bias-scale-offset array
 * @param plan [in]     The @oper{fully_connected_16} plan to be processed
 * @param job  [in]     The @oper{fully_connected_16} job to be processed
 */
void fully_connected_16(int16_t* Y, const nn_tensor_t* W, const nn_tensor_t* X,
                        const nn_bso_block_t* BSO,
                        const nn_fully_connected_plan_t* plan,
                        const nn_fully_connected_job_t* job);

/**
 * @brief Execute @oper{argmax_16} job.
 *
 * See @oper_ref{argmax_16} for more details about the @oper{argmax_16}
 * operator.
 *
 * Unlike other operators, instances of @oper{argmax_16} do not require plans or
 * jobs and no initialization is necessary.
 *
 * `Y` points to the output @tensor{y}.
 *
 * `X` points to the input vector @tensor{x} with length @tensor_shape{N}.
 *
 * `N` is the length @math{N} of the input vector @tensor{x}.
 *
 * @requires_word_alignment{X}
 *
 * @param Y [out]   The output index @math{y}
 * @param X [in]    The input vector @tensor{x}
 * @param N [in]    The number of elements @math{N} of the input vector
 * @tensor{x}
 */
void argmax_16(int32_t* Y, const int16_t* X, const int32_t N);

/**
 * @brief Execute @oper{requantize_16_to_8} job.
 *
 * See @oper_ref{requantize_16_to_8} for more details about the
 * @oper{requantize_16_to_8} operator.
 *
 * An instance of the @oper{requantize_16_to_8} operator requires an job (but no
 * plan is required). See requantize_16_to_8_init() for more details.
 *
 * `Y` points to the output vector @tensor{y} with length @math{N}. The address
 * supplied for `Y` should be the start address of the output vector (for any
 * job being processed).
 *
 * `X` points to the input vector @tensor{x} with length @math{N}. The address
 * supplied for `X` should be the start address of the input vector (for any job
 * being processed).
 *
 * `job` points to the (initialized) @oper{requantize_16_to_8} job to be
 * performed with this call.
 *
 * @requires_word_alignment{Y,X}
 *
 * @param Y   [out]    The output vector @tensor{y}
 * @param X   [in]     The input vector @tensor{x}
 * @param job [in]     The @oper{requantize_16_to_8} job to be processed
 */
void requantize_16_to_8(int8_t* Y, const int16_t* X,
                        const nn_requantize_16_to_8_job_t* job);

/**
 * @brief Execute @oper{lookup8} job.
 *
 * See @oper_ref{lookup8} for more details about the @oper{lookup8} operator.
 *
 * Unlike other operators, instances of @oper{lookup8} do not require plans or
 * jobs and no initialization is necessary.
 *
 * `Y` points to the output vector @tensor{y} with length @math{N}.
 *
 * `X` points to the input vector @tensor{x} with length @math{N}.
 *
 * `lut` points to the look-up table @math{T} with shape @tensor_shape{256}.
 *
 * `N` is the length @math{N} of the input vector @tensor{x}.
 *
 * @requires_word_alignment{Y,X}
 *
 * @param Y      [out]  The output vector @tensor{y}
 * @param X      [in]   The input vector @tensor{x}
 * @param lut    [in]   Look-up table @tensor{T}
 * @param N      [in]   Length @math{N} of input and output vectors
 */
void lookup8(uint8_t* Y, const uint8_t* X, const uint8_t* lut,
             const unsigned N);

/**
 * @brief Execute @oper{pad_perpare} function.
 *
 * See @oper_ref{pad_perpare} for more details about the @oper{pad_perpare}
 * operator.
 *
 * Unlike other operators, instances of @oper{lookup8} do not require plans or
 * jobs and no initialization is necessary.
 *
 * `Y` points to the output vector @tensor{y} with length @math{N}.
 *
 * `X` points to the input vector @tensor{x} with length @math{N}.
 *
 * `lut` points to the look-up table @math{T} with shape @tensor_shape{256}.
 *
 * `N` is the length @math{N} of the input vector @tensor{x}.
 *
 * @requires_word_alignment{Y,X}
 *
 * @param plan             [out]  The output vector @tensor{y}
 * @param p                [in]   The input vector @tensor{x}
 * @param x                [in]   Look-up table @tensor{T}
 * @param bytes_per_pixel  [in]   Length @math{N} of input and output vectors
 */
void pad_perpare(nn_pad_plan_t* plan, const PaddingValues* p,
                 const nn_image_params_t* x, const unsigned bytes_per_pixel);

void pad_run(void* y, void* x, const nn_pad_plan_t* p);

void pad_ref(void* y, void* x, const PaddingValues* p,
             const nn_image_params_t* xp, const unsigned bytes_per_pixel);

#ifdef __XC__
}  // extern "C"
#endif

#endif  // NN_OPERATOR_H_
