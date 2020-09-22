

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
 * See @oper_ref{conv2d_deep} for more details about the @oper{conv2d_deep} operator.
 * 
 * An instance of the @oper{conv2d_deep} operator requires an initialized plan and one or more jobs. See 
 * conv2d_deep_init() for more details.
 * 
 * `Y` points to the output image @tensor{Y} with shape @tensor_shape{Y_h, Y_w, Y_c}. The address supplied for `Y` 
 * should be the start address of the output image (for any job being processed).
 * 
 * `X` points to the input image @tensor{X} with shape @tensor_shape{X_h, X_w, X_c}. The address supplied for `X` should 
 * be the start address of the input image (for any job being processed).
 * 
 * The memory layout of @tensor{Y} and @tensor{X} are the standard memory layout for image tensors (see @ref 
 * standard_layout).
 * 
 * `K` points to the kernel tensor @tensor{K} with shape @tensor_shape{Y_c, K_h, K_w, X_c}, which correspond to the 
 * output image channels, convolution window rows and columns, and the input image channels respectively. The address 
 * supplied for `K` should be the start address of the kernel tensor (for any job being processed).
 * 
 * The memory layout of @tensor{K} is the standard memory layout for 4D tensors (see @ref standard_layout).
 * 
 * `BSO` points to an array of bias-scale-offset parameters required for this convolution. See @ref bso_layout for 
 * details on the encoding of this array. The address supplied for `BSO` should be the start address of the the array 
 * (for any job being processed).
 * 
 * `plan` points to the (initialized) plan associated with this instance of the @oper{conv2d_deep} operator.
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
void conv2d_deep(
    nn_image_t* Y,
    const nn_image_t* X,
    const nn_tensor_t* K,
    const nn_bso_block_t* BSO,
    const nn_conv2d_deep_plan_t* plan,
    const nn_conv2d_deep_job_t* job);



/**
 * @brief Execute @oper{conv2d_shallowin} job.
 * 
 * See @oper_ref{conv2d_shallowin} for more details about the @oper{conv2d_shallowin} operator.
 * 
 * An instance of the @oper{conv2d_shallowin} operator requires an initialized plan and one or more jobs. See 
 * conv2d_shallowin_init() for more details.
 * 
 * `Y` points to the output image @tensor{Y} with shape @tensor_shape{Y_h, Y_w, Y_c}. The address supplied for `Y` 
 * should be the start address of the output image (for any job being processed).
 * 
 * `X` points to the input image @tensor{X} with shape @tensor_shape{X_h, X_w, X_c}. The address supplied for `X` should 
 * be the start address of the input image (for any job being processed).
 * 
 * The memory layout of @tensor{Y} and @tensor{X} are the standard memory layout for image tensors (see @ref 
 * standard_layout).
 * 
 * `K` points to the kernel tensor @tensor{K} encoded with shape @tensor_shape{Y_c, K_h, \hat{K_w}, X_c}, where 
 * @math{Y_c}, @math{K_h} and @math{X_c} correspond to the output image channels, convolution window rows and the input 
 * image channel count respectively. @math{\hat{K_w}} is the augmented convolution window width, which must be exactly
 * @math{32/X_c}, regardless of the convolution window width @math{K_w}. The address supplied for `K` should be the 
 * start address of the kernel tensor (for any job being processed).
 * 
 * The memory layout of @tensor{K} (with the modified 3rd dimension) is the standard memory layout for 4D tensors (see 
 * @ref standard_layout). Further, the coefficients for all elements @math{K\left[i,j,k,l\right]} where @math{k\geq K_w} 
 * must have the value 0.
 * 
 * `BSO` points to an array of bias-scale-offset parameters required for this convolution. See @ref bso_layout for 
 * details on the encoding of this array. The address supplied for `BSO` should be the start address of the the array 
 * (for any job being processed).
 * 
 * `plan` points to the (initialized) plan associated with this instance of the @oper{conv2d_shallowin} operator.
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
void conv2d_shallowin(
    nn_image_t* Y,
    const nn_image_t* X,
    const nn_tensor_t* K,
    const nn_bso_block_t* BSO,
    const nn_conv2d_shallowin_plan_t* plan,
    const nn_conv2d_shallowin_job_t* job);


/**
 * @brief Perform a 2D convolution of a shallow input image.
 * 
 * Perform a 2D convolution of kernel tensor @tensor{K} with input image @tensor{X}
 * to produce output image @tensor{Y}.
 *  
 * This function is optimized for input images that have 3 channels, but will work
 * with any number of input channels. This will use more memory than the TFLite 
 * reference implementation, but run much faster:
 * 
 * Additional memory: 1 patch worth (K_w * K_h * C_in) bytes + 32 bytes + some code
 * Performance gain: Depends on patch size vs # output channels, and padding, but approximately:
 *                   16x faster convolutions - PATCH_SIZE copy operations
 * 
 * @note multiples of 16 output channels will run fastest input channels not imporant, but a PATCH_SIZE 
 * that is a multiple of 32 will be the fastest, most memory effcient
 * 
  * `Y` points to the output image tensor @tensor{Y} with shape @tensor_shape{Y_h, Y_w, Y_c}, which 
 * correspond to the output image rows, columns and channels respectively. The dimensions of @tensor{Y} 
 * must be as specified when `plan` was initialized. The address supplied for `Y` should be the start 
 * address of the output image tensor, *not* the start address of the sub-tensor being computed by the 
 * current job.
 * 
 * `X` points to the input image tensor @tensor{X} with shape @tensor_shape{X_h, X_w, X_c}, which 
 * correspond to the input image rows, columns and channels respectively. The dimensions of @tensor{X} 
 * must be as specified when `plan` was initialized. The address supplied for `X` should be the start 
 * address of input image tensor, *not* the address at which the convolution window starts for the job 
 * being processed.
 * 
 * The memory layout of @tensor{Y} and @tensor{X} are the standard memory layout for image tensors (see @ref 
 * standard_layout).
 * 
 * `COL` points to a caller supplied buffer that will be used for the internal im2col() transformation. This buffer
 * needs to be word-aligned and a multiple of 32-bytes in length, no shorter than: (K_w * K_h * C_in) bytes. 
 * 
 * `K` points to the kernel tensor @tensor{K} with shape @tensor_shape{Y_c, K_h * K_w * X_c}, where @math{Y_c},
 * @math{K_h} @math{K_w} and @math{X_c} correspond to the output image channels, convolution window rows, 
 * convolution window columns, and the input image channel count respectively.
 * 
 * The memory layout of @tensor{K} is a row-major 2D matrix
 * 
 * `BSO` points to an array of bias-shifts-scale parameters required for this convolution. Each 
 * `nn_bso_block_t` in the array contains the bias-shifts-scale parameters for a single output channel group,
 * (@ttref{VPU_INT8_ACC_PERIOD} output channels). If @math{Y_c} is not a multiple of @ttref{VPU_INT8_ACC_PERIOD}, 
 * then the output channel tail ( the last @math{(Y_c mod 16)} output channels) also gets `nn_bso_block_t`, where
 * the entries corresponding to channels beyond @math{Y_c} are ignored. The address supplied for `BSO` should be
 * the start address of the the array, *not* the address of the `nn_bso_block_t` corresponding of the first output
 * channel of the job being processed.
 * 
 * `plan` points to the `nn_conv2d_im2col_plan_t` which was previously initialized with a call to 
 * `conv2d_im2col_init()`.
 * 
 * `job` points to the job to be performed in this call, which was previously initialized along-side `plan`. 
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
void conv2d_im2col(
    nn_image_t* Y,
    const nn_image_t* X,
    const nn_image_t* COL,
    const nn_tensor_t* K,
    const nn_bso_block_t* BSO,
    const nn_conv2d_im2col_plan_t* plan,
    const nn_conv2d_im2col_job_t* job);


/**
 * @brief Execute @oper{conv2d_1x1} job.
 * 
 * See @oper_ref{conv2d_1x1} for more details about the @oper{conv2d_1x1} operator.
 * 
 * An instance of the @oper{conv2d_1x1} operator requires a plan and one or more jobs, which are represented
 * by the `nn_conv2d_1x1_plan_t` and `nn_conv2d_1x1_job_t` structs. Before performing a 2D convolution using 
 * this function, a call must be made to conv2d_1x1_init() to initialize the plan and any jobs.
 * 
 * An instance of the @oper{conv2d_1x1} operator requires an initialized plan and one or more jobs. See 
 * conv2d_1x1_init() for more details.
 * 
 * `Y` points to the output image @tensor{Y} with shape @tensor_shape{Y_h, Y_w, Y_c}. The address supplied for `Y` 
 * should be the start address of the output image (for any job being processed).
 * 
 * `X` points to the input image @tensor{X} with shape @tensor_shape{X_h, X_w, X_c}. The address supplied for `X` should 
 * be the start address of the input image (for any job being processed).
 * 
 * The memory layout of @tensor{Y} and @tensor{X} are the standard memory layout for image tensors (see @ref 
 * standard_layout).
 * 
 * `K` points to the kernel tensor @tensor{K} with shape @tensor_shape{Y_c, X_c}, which correspond to the 
 * output image channels and input image channels respectively. The address supplied for `K` should be the start address 
 * of the kernel tensor (for any job being processed).
 * 
 * The memory layout of @tensor{K} is the standard memory layout for 2D tensors (see @ref standard_layout).
 * 
 * `BSO` points to an array of bias-scale-offset parameters required for this convolution. See @ref bso_layout for 
 * details on the encoding of this array. The address supplied for `BSO` should be the start address of the the array 
 * (for any job being processed).
 * 
 * `plan` points to the (initialized) plan associated with this instance of the @oper{conv2d_deep} operator.
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
void conv2d_1x1(
    nn_image_t* Y,
    const nn_image_t* X,
    const nn_tensor_t* K,
    const nn_bso_block_t* BSO,
    const nn_conv2d_1x1_plan_t* plan,
    const nn_conv2d_1x1_job_t* job);




/**
 * @brief Execute @oper{conv2d_depthwise} job.
 * 
 * See @oper_ref{conv2d_depthwise} for more details about the @oper{conv2d_depthwise} operator.
 * 
 * An instance of the @oper{conv2d_depthwise} operator requires an initialized plan and one or more jobs. See 
 * conv2d_depthwise_init() for more details.
 * 
 * `Y` points to the output image @tensor{Y} with shape @tensor_shape{Y_h, Y_w, X_c}. The address supplied for `Y` 
 * should be the start address of the output image (for any job being processed).
 * 
 * `X` points to the input image @tensor{X} with shape @tensor_shape{X_h, X_w, X_c}. The address supplied for `X` should 
 * be the start address of the input image (for any job being processed).
 * 
 * The memory layout of @tensor{Y} and @tensor{X} are the standard memory layout for image tensors (see @ref 
 * standard_layout).
 * 
 * `K` points to the kernel tensor @tensor{K} with shape @tensor_shape{K_h, K_w, X_c}, which correspond to the 
 * convolution window rows and columns and the input image channels respectively. The address supplied for `K` should be 
 * the start address of the kernel tensor (for any job being processed).
 * 
 * The memory layout of @tensor{K} is the standard memory layout for 3D tensors (see @ref standard_layout).
 * 
 * `BSO` points to an array of bias-scale-offset parameters required for this convolution. See @ref bso_layout for 
 * details on the encoding of this array. The address supplied for `BSO` should be the start address of the the array 
 * (for any job being processed).
 * 
 * `plan` points to the (initialized) plan associated with this instance of the @oper{conv2d_depthwise} operator.
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
void conv2d_depthwise(
    nn_image_t* Y,
    const nn_image_t* X,
    const nn_tensor_t* K,
    const nn_bso_block_t* BSO,
    const nn_conv2d_depthwise_plan_t* plan,
    const nn_conv2d_depthwise_job_t* job);

    

    
/**  
 * @brief Execute @oper{maxpool2d} job.
 * 
 * See @oper_ref{maxpool2d} for more details about the @oper{maxpool2d} operator.
 * 
 * An instance of the @oper{maxpool2d} operator requires an initialized plan and one or more jobs. See maxpool2d_init() 
 * for more details.
 * 
 * `Y` points to the output image @tensor{Y} with shape @tensor_shape{Y_h, Y_w, X_c}. The address supplied for `Y` 
 * should be the start address of the output image (for any job being processed).
 * 
 * `X` points to the input image @tensor{X} with shape @tensor_shape{X_h, X_w, X_c}. The address supplied for `X` should 
 * be the start address of the input image (for any job being processed).
 * 
 * The memory layout of @tensor{Y} and @tensor{X} are the standard memory layout for image tensors (see @ref 
 * standard_layout).
 * 
 * `plan` points to the (initialized) plan associated with this instance of the @oper{maxpool2d} operator.
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
void maxpool2d(
    nn_image_t* Y,
    const nn_image_t* X, 
    const nn_maxpool2d_plan_t* plan,
    const nn_pool2d_job_t* job);


/** 
 * @brief Execute @oper{avgpool2d} job.
 * 
 * See @oper_ref{avgpool2d} for more details about the @oper{avgpool2d} operator.
 * 
 * An instance of the @oper{avgpool2d} operator requires an initialized plan and one or more jobs. See avgpool2d_init() 
 * for more details.
 * 
 * `Y` points to the output image @tensor{Y} with shape @tensor_shape{Y_h, Y_w, X_c}. The address supplied for `Y`
 * should be the start address of the output image (for any job being processed).
 * 
 * `X` points to the input image @tensor{X} with shape @tensor_shape{X_h, X_w, X_c}. The address supplied for `X` should 
 * be the start address of the input image (for any job being processed).
 * 
 * The memory layout of @tensor{Y} and @tensor{X} are the standard memory layout for image tensors (see @ref 
 * standard_layout).
 * 
 * `plan` points to the (initialized) plan associated with this instance of the @oper{avgpool2d} operator.
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
static inline void avgpool2d(
    nn_image_t* Y,
    const nn_image_t* X, 
    const nn_avgpool2d_plan_t* plan,
    const nn_pool2d_job_t* job)
{
    switch(plan->impl){
        case AVGPOOL2D_2X2:
            avgpool2d_2x2(Y, X, plan, job);
            break;
        default:
            avgpool2d_gen(Y, X, plan, job);
            break;
    }
}

/** 
 * @brief Invoke a @oper{avgpool2d_global} job.
 * 
 * The @oper{avgpool2d_global} operator computes a scaled and biased sum of pixel values for each channel of an input
 * image, producing an 8-bit vector of outputs.
 * 
 * See @oper_ref{avgpool2d_global} for more details about the @oper{avgpool2d_global} operator, including the 
 * mathematical details of the operation performed.
 * 
 * @par Operator Plans and Jobs
 * 
 * Invoking an instance of @oper{avgpool2d_global} requires a plan and a job. The plan and one or more jobs may be 
 * initialized with the avgpool2d_global_init() function. Each job computes a contiguous subset of the output elements.
 * 
 * @par Parameter Details
 * 
 * `Y` points to the 8-bit output vector @tensor{y} with length @tensor_shape{X_c}.
 * 
 * `X` points to the 8-bit input image @tensor{X} with shape @tensor_shape{X_h, X_w, X_c}. The memory layout of 
 * @tensor{X} is the standard memory layout for image tensors (see @ref standard_layout).
 * 
 * `bias` is the 32-bit bias @math{b} with which the accumulators are initialized for each output. Note that the 
 * right-shift by @math{r} bits is applied after all accumulation. To add an absolute offset of @math{b_0} to each
 * result, then the value used for @math{b} should be @math{b_0 \cdot 2^r}.
 * 
 * `scale` is the 8-bit coefficient @math{s}. All pixel values are multiplied by @math{s} and added to the 32-bit 
 * accumulator.
 * 
 * `shift` is the (rounding) right-shift @math{r} applied to each 32-bit accumulator to yield an 8-bit result. Note 
 * that this is a saturating right-shift which will saturate to 8-bit bounds (see additional remarks below).
 * 
 * `plan` points to the (initialized) plan associated with this instance of the @oper{avgpool2d_global} operator.
 * 
 * `job` points to the (initialized) job to be performed with this call.
 * 
 * @par Parameter Constraints
 * 
 * The arguments `Y` and `X` must each point to a word-aligned address.
 * 
 * Due to memory alignment requirements, @math{X_c} must be a multiple of @math{4}, which forces all pixels to begin at
 * a word-aligned address.
 * 
 * @par Splitting the Workload
 * 
 * Jobs are used to split the work done by an instance of @oper{avgpool2d_global}. Each job computes a (contiguous)
 * subset of the elements of @tensor{y}. The elements to be computed by a job are specified when the jobs are 
 * initialized by avgpool2d_global_init().
 * 
 * @par Additional Remarks
 * 
 * The arguments `Y` and `X` should both point at the beginning of their respective objects, even if the job being 
 * invoked does not start at the beginning of the output vector.
 * 
 * By default this operator uses the standard 8-bit limits @math([-128, 127]) when applying saturation logic. Instead,
 * it can be configured to use symmetric saturation bounds @math([-127, 127]) by defining 
 * `CONFIG_SYMMETRIC_SATURATION_avgpool2d_global` appropriately. See @ref nn_config.h for more details. Note that this
 * configures _all_ instances of the @oper{avgpool2d_global} operator.
 * 
 * If @math{X_c} is not a multiple of @math{16}, this operator may read up to 12 bytes following the end of @tensor{X}. 
 * This is not ordinarily a problem. However, if the object to which `X` points is located very near the end of a valid 
 * memory address range, it is possible memory access exceptions may occur when this operator is invoked.
 * 
 * If necessary, this can be avoided by manually forcing a buffer region (no more than @math{12} bytes are necessary) 
 * following @tensor{X}. There are various ways this can be accomplished, including embedding these objects in larger 
 * structures.
 * 
 * @param [out] Y       The output vector @tensor{y}
 * @param [in]  X       The input image @tensor{X}
 * @param [in]  bias    Initial 32-bit accumulator value @math{b}. Shared by all channels.
 * @param [in]  scale   The factor @math{s} by which input pixel values are scaled.
 * @param [in]  shift   The right-shift @math{r} applied to the 32-bit accumulators to yield an 8-bit result.
 * @param [in]  plan    The @oper{avgpool2d_global} plan to be processed
 * @param [in]  job     The @oper{avgpool2d_global} job to be processed
 */
void avgpool2d_global(
    int8_t* Y,
    const int8_t* X, 
    const int32_t bias,
    const int8_t scale,
    const uint16_t shift,
    const nn_avgpool2d_global_plan_t* plan,
    const nn_avgpool2d_global_job_t* job);




/** 
 * @brief Invoke a @oper{fully_connected_8} job.
 * 
 * The @oper{fully_connected_8} operator performs a matrix-vector multiplication with an additional (per-output) scale 
 * and offset applied (@math{s\cdot\left(\bar{W}\bar{x}\right)+b}) to produce an 8-bit result.
 * 
 * See @oper_ref{fully_connected_8} for more details about the @oper{fully_connected_8} operator, including the 
 * mathematical details of the operation performed.
 * 
 * @par Operator Plans and Jobs
 * 
 * Invoking an instance of the @oper{fully_connected_8} operator requires no plan or job objects; no initialization is
 * required.
 * 
 * @par Parameter Details
 * 
 * `Y` points to the output vector @tensor{y} with length @tensor_shape{M}.
 * 
 * `X` points to the input vector @tensor{x} with length @tensor_shape{N}.
 * 
 * `W` points to the weight matrix @tensor{W} with shape @tensor_shape{M, N}, which correspond to the output and input
 * vector respectively.
 * 
 * The memory layout of @tensor{W} is the standard memory layout for 2D tensors (see @ref standard_layout).
 * 
 * `BSO` points to an array of bias-scale-offset parameters required for this convolution. See @ref bso_layout for 
 * details on the encoding of this array.
 * 
 * `N` is the number of input channels (i.e. the length of @tensor{x}).
 * 
 * `output_start` and `output_count` determine the range of outputs which are computed by this invocation of 
 * @oper{fully_connected_8}. To compute the entire output vector @tensor{y}, `output_start` and `output_count` should
 * be @math{0} and @math{M} respectively.
 * 
 * @par Parameter Constraints
 * 
 * The arguments `Y`, `W`, `X` and `BSO` must each point to a word-aligned address.
 * 
 * Due to memory alignment requirements, @math{N} must be a multiple of @math{4}. If necessary, the rows of @tensor{W} 
 * should be padded out with zeros at the end to satisfy this constraint.
 * 
 * In order to maintain alignment with `nn_bso_block_t` blocks, `output_start` must be a multiple of @math{16}.
 * 
 * @par Splitting the Workload
 * 
 * In some cases (e.g. parallelizing across multiple cores) it is desirable to only compute a subset of the output 
 * elements with a call to fully_connected_8(). The elements that will be computed and output by a call to 
 * fully_connected_8() are @math{y[s:s+c]}, where @math{s} and @math{c} are `output_start` and `output_count` 
 * respectively. Note that @math{y[s+c]} is *not* computed.
 * 
 * When splitting an instance of @oper{fully_connected_8} into multiple invocations it may be tempting to 
 * split the work evenly between invocations. However, the constraint that `output_start` be a multiple of @math{16} 
 * (see above) also suggests that `output_count` should be a multiple of @math{16} for each invocation. The exception to 
 * this is if @math{M \ne 0 \left(\text{mod } 16\right)}, in which case the invocation that processes the final elements 
 * of @tensor{y} needn't be a multiple of @math{16}.
 * 
 * @par Additional Remarks
 * 
 * `Y`, `X`, `W` and `BSO` should all point at the beginning of their respective objects, even if `output_start`
 * is not `0`. (Some advanced scenarios may require you to violate this.)
 * 
 * If @math{N} is not a multiple of @math{32}, then this operator may read memory before the start of @tensor{x} and 
 * @tensor{W}. This is not ordinarily a problem. However, if the objects to which `X` and `W` point are located very 
 * near the beginning of a valid memory address range, it is possible memory access exceptions may occur when this 
 * operator is invoked.
 * 
 * If necessary, this can be avoided by manually forcing a buffer region (no more than @math{28} bytes are necessary) 
 * prior to the start of @tensor{x} and @tensor{W}. There are various ways this can be accomplished, including embedding
 * these objects in larger structures.
 * 
 * @param [out] Y               The output vector @tensor{y}
 * @param [in]  W               The weight matrix @tensor{W}
 * @param [in]  X               The input vector @tensor{x}
 * @param [in]  BSO             The bias-scale-offset array
 * @param [in]  N               The number of input channels, @math{N}
 * @param [in]  output_start    The first output element to compute (index of @tensor{y})
 * @param [in]  output_count    The number of output elements to compute
 */
void fully_connected_8(
    int8_t* Y,
    const int8_t* W, 
    const int8_t* X, 
    const nn_bso_block_t* BSO,
    const channel_count_t N,
    const channel_count_t output_start,
    const channel_count_t output_count);




/** 
 * @brief Invoke a @oper{fully_connected_16} job.
 * 
 * The @oper{fully_connected_16} operator performs a matrix-vector multiplication with an additional (per-output) scale 
 * and offset applied (@math{s\cdot\left(\bar{W}\bar{x}\right)+b}) to produce a 16-bit result.
 * 
 * See @oper_ref{fully_connected_16} for more details about the @oper{fully_connected_16} operator, including the 
 * mathematical details of the operation performed.
 * 
 * @par Operator Plans and Jobs
 * 
 * Invoking an instance of the @oper{fully_connected_16} operator requires no plan or job objects; no initialization is
 * required.
 * 
 * @par Parameter Details
 * 
 * `Y` points to the output vector @tensor{y} with length @tensor_shape{M}.
 * 
 * `X` points to the input vector @tensor{x} with length @tensor_shape{N}.
 * 
 * `W` points to the weight matrix @tensor{W} with shape @tensor_shape{M, N}, which correspond to the output and input
 * vector respectively.
 * 
 * The memory layout of @tensor{W} is the standard memory layout for 2D tensors (see @ref standard_layout).
 * 
 * `BSO` points to an array of bias-scale-offset parameters required for this convolution. See @ref bso_layout for 
 * details on the encoding of this array.
 * 
 * `N` is the number of input channels (i.e. the length of @tensor{x}).
 * 
 * `output_start` and `output_count` determine the range of outputs which are computed by this invocation of 
 * @oper{fully_connected_16}. To compute the entire output vector @tensor{y}, `output_start` and `output_count` should
 * be @math{0} and @math{M} respectively.
 * 
 * @par Splitting the Workload
 * 
 * In some cases (e.g. parallelizing across multiple cores) it is desirable to only compute a subset of the output 
 * elements with a call to fully_connected_16(). The elements that will be computed and output by a call to 
 * fully_connected_16() are @math{y[s:s+c]}, where @math{s} and @math{c} are `output_start` and `output_count` 
 * respectively. Note that @math{y[s+c]} is *not* computed.
 * 
 * When splitting an instance of @oper{fully_connected_16} into multiple invocations it may be tempting to 
 * split the work evenly between invocations. However, the constraint that `output_start` be a multiple of @math{16} 
 * (see below) also suggests that `output_count` should be a multiple of @math{16} for each invocation. The exception to 
 * this is if @math{M \ne 0 \left(\text{mod } 16\right)}, in which case the invocation that processes the final elements 
 * of @tensor{y} needn't be a multiple of @math{16}.
 * 
 * @par Parameter Constraints
 * 
 * The arguments `Y`, `W`, `X` and `BSO` must each point to a word-aligned address.
 * 
 * Due to memory alignment requirements, @math{N} must be a multiple of @math{4}. If necessary, the rows of @tensor{W} 
 * should be padded out with zeros at the end to satisfy this constraint.
 * 
 * In order to maintain alignment with `nn_bso_block_t` blocks, `output_start` must be a multiple of @math{16}.
 * 
 * @par Additional Remarks
 * 
 * `Y`, `X`, `W` and `BSO` should all point at the beginning of their respective objects, even if `output_start`
 * is not `0`. (Some advanced scenarios may require you to violate this.)
 * 
 * If @math{N} is not a multiple of @math{32}, then this operator may read memory before the start of @tensor{x} and 
 * @tensor{W}. This is not ordinarily a problem. However, if the objects to which `X` and `W` point are located very 
 * near the beginning of a valid memory address range, it is possible memory access exceptions may occur when this 
 * operator is invoked.
 * 
 * If necessary, this can be avoided by manually forcing a buffer region (no more than @math{28} bytes are necessary) 
 * prior to the start of @tensor{x} and @tensor{W}. There are various ways this can be accomplished, including embedding
 * these objects in larger structures.
 * 
 * @param [out] Y               The output vector @tensor{y}
 * @param [in]  W               The weight matrix @tensor{W}
 * @param [in]  X               The input vector @tensor{x}
 * @param [in]  BSO             The bias-scale-offset array
 * @param [in]  N               The number of input channels, @math{N}
 * @param [in]  output_start    The first output element to compute (index of @tensor{y})
 * @param [in]  output_count    The number of output elements to compute
 */
void fully_connected_16(
    int16_t* Y,
    const int8_t* W, 
    const int8_t* X, 
    const nn_bso_block_t* BSO,
    const channel_count_t N,
    const channel_count_t output_start,
    const channel_count_t output_count);




/**  
 * @brief Execute @oper{argmax_16} job.
 * 
 * See @oper_ref{argmax_16} for more details about the @oper{argmax_16} operator.
 * 
 * Unlike other operators, instances of @oper{argmax_16} do not require plans or jobs and no initialization is 
 * necessary.
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
 * @param N [in]    The number of elements @math{N} of the input vector @tensor{x}
 */
void argmax_16(
    int32_t* Y,
    const int16_t* X,
    const int32_t N);


/** 
 * @brief Execute @oper{requantize_16_to_8} job.
 * 
 * See @oper_ref{requantize_16_to_8} for more details about the @oper{requantize_16_to_8} operator.
 * 
 * An instance of the @oper{requantize_16_to_8} operator requires an job (but no plan is required). See 
 * requantize_16_to_8_init() for more details.
 * 
 * `Y` points to the output vector @tensor{y} with length @math{N}. The address supplied for `Y` should be the start 
 * address of the output vector (for any job being processed).
 * 
 * `X` points to the input vector @tensor{x} with length @math{N}. The address supplied for `X` should be the start 
 * address of the input vector (for any job being processed).
 * 
 * `job` points to the (initialized) @oper{requantize_16_to_8} job to be performed with this call.
 * 
 * @requires_word_alignment{Y,X}
 *
 * @param Y   [out]    The output vector @tensor{y}
 * @param X   [in]     The input vector @tensor{x}
 * @param job [in]     The @oper{requantize_16_to_8} job to be processed
 */
void requantize_16_to_8(
    int8_t* Y,
    const int16_t* X,
    const nn_requantize_16_to_8_job_t* job);



/** 
 * @brief Execute @oper{lookup8} job.
 * 
 * See @oper_ref{lookup8} for more details about the @oper{lookup8} operator.
 * 
 * Unlike other operators, instances of @oper{lookup8} do not require plans or jobs and no initialization is
 * necessary.
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
void lookup8(
    uint8_t* Y,
    const uint8_t* X,
    const uint8_t* lut,
    const unsigned N);


/** 
 * @brief Execute @oper{bsign_8} job.
 * 
 * See @oper_ref{bsign_8} for more details about the @oper{requantize_16_to_8} operator.
 * 
 * An instance of the @oper{bsign_8} operator requires an job (but no plan is required). See 
 * bsign_8_init() for more details.
 * 
 * `Y` points to the output vector @tensor{y} with length @math{N}. The address supplied for `Y` should be the start 
 * address of the output vector (for any job being processed).
 * 
 * `X` points to the input vector @tensor{x} with length @math{N}. The address supplied for `X` should be the start 
 * address of the input vector (for any job being processed).
 * 
 * `job` points to the (initialized) @oper{bsign_8} job to be performed with this call.
 * 
 * @requires_word_alignment{Y,X}
 *
 * @param Y   [out]    The output vector @tensor{y}
 * @param X   [in]     The input vector @tensor{x}
 * @param plan [in]    The @oper{bsign_8} plan to be processed
 * @param job [in]     The @oper{bsign_8} job to be processed
 */
void bsign_8(
    uint32_t* Y,
    const int8_t* X,
    const nn_bsign_8_plan_t* plan,
    const nn_bsign_8_job_t* job);

void bsign_8_ref(
    uint32_t* Y,
    const int8_t* X,
    const nn_bsign_8_plan_t* plan,
    const nn_bsign_8_job_t* job);

// Binary Conv2D

#define CONV2D_OUTPUT_LENGTH(input_length, filter_size, dilation, stride)     \
  (((input_length - (filter_size + (filter_size - 1) * (dilation - 1)) + 1) + \
    stride - 1) /                                                             \
   stride)

#define CONV2D_INPUT_LENGTH(output_length, filter_size, dilation, stride)  (output_length * stride - (stride - 1) - 1  + (filter_size + (filter_size - 1) * (dilation - 1)))
    
/**  
 * @brief Execute @oper{bnn_reorder_threshold_tensor}.
 * 
 * This reorders the threshold tensor for efficient execution by bnn_conv2d_bin_out_asm. 
 * This is only inteneded for testing.
 * 
 * `thresh_reordered` points to the output threshold @tensor{thresh_reordered} .
 * 
 * `thresholds_ref` points to the input @tensor{thresholds_ref}.
 * 
 * `chans_out` is the number of output channels.
 * 
 * `receptive_field` the spatial area over which the kernel operates, i.e. (height x width).
 * 
 * @param thresh_reordered   [out]    The output @tensor{thresh_reordered}
 * @param thresholds_ref     [in]     The input @tensor{thresholds_ref}
 * @param chans_out          [in]     The number of output channels
 * @param receptive_field    [in]     The spatial area over which the kernel operates
 */
void bnn_reorder_threshold_tensor(const int32_t* thresh_reordered,
                                  const int32_t* thresholds_ref,
                                  const unsigned chans_out,
                                  const unsigned receptive_field);
    
/**  
 * @brief Execute @oper{bnn_reorder_kernel_tensor}.
 * 
 * This reorders the kernel tensor for efficient execution by bnn_conv2d_bin_out_asm. 
 * This is only intended for testing.
 * 
 * `K_p` points to the output kernel @tensor{K_p} .
 * 
 * `K_ref_p` points to the kernel input @tensor{K_ref_p}.
 * 
 * `k_height` is the kernel height.
 * 
 * `k_width` is the kernel width.
 * 
 * `chans_in` is the number of input channels.
 * 
 * `chans_out` is the number of output channels.    
 * 
 * @param K_p         [out]    The output @tensor{K_p}
 * @param K_ref_p     [in]     The input @tensor{K_ref_p}
 * @param k_height    [in]     The kernel height
 * @param k_width     [in]     The kernel width
 * @param chans_in    [in]     The number of input channels
 * @param chans_out   [in]     The number of output channels
 */
void bnn_reorder_kernel_tensor(const bnn_b256_t* K_p, const bnn_b256_t* K_ref_p,
                               const unsigned k_height, const unsigned k_width,
                               const unsigned chans_in,
                               const unsigned chans_out);

    
/**  
 * @brief Execute @oper{bnn_conv2d_bin_out_valid}.
 * 
 * This performs a binary conv2d on a rectangular sub-section of an input tensor X with 
 * kernel K.  
 * 
 * The tensor X_p represents a tensor of (x_full_height x x_full_width x X_channels)
 * The tensor K_p represents a tensor of (k_full_height x k_full_width x X_channels)
 * The tensor Y_p represents a tensor of (y_full_height x y_full_width x Y_channels)
 * 
 * 
 * @param Y             [out]    The output image @tensor{Y}
 * @param X             [in]     The input image @tensor{X}
 * @param K             [in]     The input kernel @tensor{K}
 * @param thresholds    [in]     The input thresholds @tensor{thresholds}
 * @param x             [in]     The parameters of the X image tensor
 * @param y             [in]     The parameters of the Y image tensor
 * @param k             [in]     The parameters of the K kernel tensor.
 * @param y_loc_x       [in]     The x coordinate of where the output will start writing from
 * @param y_loc_y       [in]     The y coordinate of where the output will start writing from
 * @param y_sub_width   [in]     The width of the output sub-image that will be computed
 * @param y_sub_height  [in]     The height of the output sub-image that will be computed
 */
void bnn_conv2d_bin_out_valid(bnn_b32_t* Y_p,
    const bnn_b256_t* X_p, 
    const bnn_b256_t* K_p, 
    const int32_t* thresholds_p,
    const nn_image_params_t* x,
    const nn_image_params_t* y,
    const nn_window_params_t* k, 

    const unsigned y_loc_x, const unsigned y_loc_y,
    const unsigned y_sub_width, const unsigned y_sub_height
);

/**  
 * @brief Execute @oper{bnn_conv2d_bin_out}.
 * 
 * This performs a binary conv2d on a rectangular sub-section of an input tensor X with 
 * a sub-section of kernel K and writes it to s sub-section of tensor Y.
 * 
 * The tensor X_p represents a tensor of (x_full_height x x_full_width x X_channels)
 * The tensor K_p represents a tensor of (k_full_height x k_full_width x X_channels)
 * The tensor Y_p represents a tensor of (y_full_height x y_full_width x Y_channels)
 * 
 * x_sub_height and x_sub_width will be infered by the parameters of y, x, k, y_loc_x, 
 * y_loc_y, y_sub_width, y_sub_height, k_loc_x, k_loc_y, k_sub_width and k_sub_height.
 * 
 * @param Y             [out]    The output image @tensor{Y}
 * @param X             [in]     The input image @tensor{X}
 * @param K             [in]     The input kernel @tensor{K}
 * @param thresholds    [in]     The input thresholds @tensor{thresholds}
 * @param x             [in]     The parameters of the X image tensor
 * @param y             [in]     The parameters of the Y image tensor
 * @param k             [in]     The parameters of the K kernel tensor.
 * @param y_loc_x       [in]     The x coordinate of where the output will start writing from
 * @param y_loc_y       [in]     The y coordinate of where the output will start writing from
 * @param y_sub_width   [in]     The width of the output sub-image that will be computed
 * @param y_sub_height  [in]     The height of the output sub-image that will be computed
 * @param x_loc_x       [in]     The x coordinate of where the input will start reading from
 * @param x_loc_y       [in]     The y coordinate of where the input will start reading from
 * @param k_loc_x       [in]     The x coordinate of where the kernel will start reading from
 * @param k_loc_y       [in]     The y coordinate of where the kernel will start reading from
 * @param k_sub_width   [in]     The width of the input sub-kernel that will be computed
 * @param k_sub_height  [in]     The height of the input sub-kernel that will be computed
 */
void bnn_conv2d_bin_out(bnn_b32_t* Y_p,
    const bnn_b256_t* X_p, const bnn_b256_t* K_p, const int32_t* thresholds_p,
    
    const nn_image_params_t* x, //The full image of x
    const nn_image_params_t* y, // the full image of y
    const nn_window_params_t* k, //the full kernel k
    
    const unsigned y_loc_x, const unsigned y_loc_y,
    const unsigned y_sub_width, const unsigned y_sub_height,

    const unsigned x_loc_x, const unsigned x_loc_y, 
    
    const unsigned k_loc_x, const unsigned k_loc_y, 
    const unsigned k_sub_width, const unsigned k_sub_height
);

/**
 * @brief Execute @oper{pad_prepare} function.
 *
 * `plan` points to the output vector @tensor{y} with length @math{N}.
 *
 * `p` struct describing the padding to be applied to the input tensor.
 *
 * `x` parameters describing the input tensor to be padded.
 *
 * `bytes_per_pixel` the bytes per pixel for tensor x.
 *
 * @param plan             [out]  The output vector @tensor{y}
 * @param p                [in]   The input vector @tensor{x}
 * @param x                [in]   Look-up table @tensor{T}
 * @param bytes_per_pixel  [in]   Length @math{N} of input and output vectors
 */
void pad_prepare(nn_pad_plan_t* plan, const padding_values_t* p,
                 const nn_image_params_t* x, const unsigned bytes_per_pixel);

/** 
 * @brief Execute @oper{pad_run} job.
 * 
 * See @oper_ref{pad_run} for more details about the @oper{requantize_16_to_8} operator.
 * 
 * `Y` points to the output vector @tensor{y}.
 * 
 * `X` points to the input vector @tensor{x}.
 * 
 * `plan` points to the (initialized) plan.
 * 
 * @requires_word_alignment{Y,X}
 *
 * @param y   [out]    The output vector @tensor{y}
 * @param x   [in]     The input vector @tensor{x}
 * @param plan [in]    The prameters describing how to pad.
 */
void pad_run(void* y, void* x, const nn_pad_plan_t* p, uint32_t pad_value);

void pad_ref(void* y, void* x, const padding_values_t* p,
             const nn_image_params_t* xp, const unsigned bytes_per_pixel, uint32_t pad_value);

#ifdef __XC__
} // extern "C"
#endif

#endif //NN_OPERATOR_H_
