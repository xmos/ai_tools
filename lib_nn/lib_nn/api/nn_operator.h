

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
 * @brief Execute a deep 2D convolution (@oper{conv2d_deep}) job.
 * 
 * See @oper_ref{conv2d_deep} for more details about the @oper{conv2d_deep} operator.
 * 
 * An instance of the @oper{conv2d_deep} operator requires a plan and one or more jobs, which are represented
 * by the `nn_conv2d_deep_plan_t` and `nn_conv2d_deep_job_t` structs. Before performing a 2D convolution using 
 * this function, a call must be made to conv2d_deep_init() to initialize the plan and any jobs.
 * 
 * `Y` points to the output image tensor @tensor{Y} with shape @tensor_shape{Y_h, Y_w, Y_c}, which correspond to
 * the output image rows, columns and channels respectively. The dimensions of @tensor{Y} must be as specified 
 * when `plan` and `job` were initialized. The address supplied for `Y` should be the start address of the output image
 * tensor, *not* the start address of the sub-tensor being computed by the current job.
 * 
 * `X` points to the input image tensor @tensor{X} with shape @tensor_shape{X_h, X_w, X_c}, which correspond to
 * the input image rows, columns and channels respectively. The dimensions of @tensor{X} must be as specified
 * when `plan` and `job` were initialized. The address supplied for `X` should be the start address of the input image tensor,
 * *not* the address at which the convolution window starts for the job being processed.
 * 
 * The memory layout of @tensor{Y} and @tensor{X} are the standard memory layout for image tensors (see @ref 
 * standard_layout).
 * 
 * `K` points to the kernel tensor @tensor{K} with shape @tensor_shape{Y_c, K_h, K_w, X_c}, which correspond to
 * the output image channels, convolution window rows and columns, and the input image channels respectively.
 * The dimensions of @tensor{K} must be as specified when `plan` and `job` were initialized. The address supplied for `K`
 * should be the start address of the kernel tensor.
 * 
 * The memory layout of @tensor{K} is the standard memory layout for 4D tensors (see @ref standard_layout).
 * 
 * `BSO` points to an array of bias-scale-offset parameters required for this convolution. Each `nn_bso_block_t` 
 * in the array contains the bias-scale-offset parameters for a single output channel group, (@ttref{VPU_INT8_ACC_PERIOD} 
 * output channels). If @math{Y_c} is not a multiple of @ttref{VPU_INT8_ACC_PERIOD}, then the output channel tail ( the 
 * last @math{(Y_c mod 16)} output channels) also gets a `nn_bso_block_t`, where the entries corresponding to channels 
 * beyond @math{Y_c} are ignored. The address supplied for `BSO` should be the start address of the the array, *not* 
 * the address of the `nn_bso_block_t` corresponding of the first output channel of the job being processed.
 * 
 * `plan` points to the `nn_conv2d_deep_plan_t` associated with this instance of the @oper{conv2d_deep} operator, previously
 * initialized with a call to conv2d_deep_init().
 * 
 * `job` points to the job to be performed with this call, previously initialized with `plan`. 
 * 
 * Note that a single call to this function processes only a *single job*. If multiple jobs were initialized,
 * performing the complete convolution requires multiple calls to this function. In such a case, the `Y`, `X`,
 * `K`, `BSO`, and `plan` pointers will be identical in each call, and the `job` pointer will be different with
 * each call.
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
 * @brief Execute a shallow-input 2D convolution (@oper{conv2d_shallowin}) job.
 * 
 * See @oper_ref{conv2d_shallowin} for more details about the @oper{conv2d_shallowin} operator.
 * 
 * An instance of the @oper{conv2d_shallowin} operator requires a plan and one or more jobs, which are represented
 * by the `nn_conv2d_shallowin_plan_t` and `nn_conv2d_shallowin_job_t` structs. Before performing a 2D convolution 
 * using this function, a call must be made to conv2d_shallowin_init() to initialize the plan and any jobs. 
 * 
 * `Y` points to the output image tensor @tensor{Y} with shape @tensor_shape{Y_h, Y_w, Y_c}, which 
 * correspond to the output image rows, columns and channels respectively. The dimensions of @tensor{Y} 
 * must be as specified when `plan` and `job` were initialized. The address supplied for `Y` should be the start 
 * address of the output image tensor, *not* the start address of the sub-tensor being computed by the 
 * current job.
 * 
 * `X` points to the input image tensor @tensor{X} with shape @tensor_shape{X_h, X_w, X_c}, which 
 * correspond to the input image rows, columns and channels respectively. The dimensions of @tensor{X} 
 * must be as specified when `plan` and `job` were initialized. The address supplied for `X` should be the start 
 * address of the input image tensor, *not* the address at which the convolution window starts for the job 
 * being processed.
 * 
 * The memory layout of @tensor{Y} and @tensor{X} are the standard memory layout for image tensors (see @ref 
 * standard_layout).
 * 
 * `K` points to the kernel tensor @tensor{K} with shape @tensor_shape{Y_c, K_h, \hat{K_w}, X_c}, where @math{Y_c},
 * @math{K_h} and @math{X_c} correspond to the output image channels, convolution window rows and the input image 
 * channel count respectively. @math{\hat{K_w}} is the augmented convolution window width, which must be exactly
 * @math{32/X_c}.The address supplied for `K` should be the start address of the kernel tensor.
 * 
 * The memory layout of @tensor{K} is the standard memory layout for 4D tensors (see @ref standard_layout). Further,
 * the coefficients for all elements @math{K\left[i,j,k,l\right]} where @math{k\geq K_w} must have the value 0.
 * 
 * `BSO` points to an array of bias-scale-offset parameters required for this convolution. Each `nn_bso_block_t` 
 * in the array contains the bias-scale-offset parameters for a single output channel group, (@ttref{VPU_INT8_ACC_PERIOD} 
 * output channels). If @math{Y_c} is not a multiple of @ttref{VPU_INT8_ACC_PERIOD}, then the output channel tail ( the 
 * last @math{(Y_c mod 16)} output channels) also gets a `nn_bso_block_t`, where the entries corresponding to channels 
 * beyond @math{Y_c} are ignored. The address supplied for `BSO` should be the start address of the the array, *not* 
 * the address of the `nn_bso_block_t` corresponding of the first output channel of the job being processed.
 * 
 * `plan` points to the `nn_conv2d_shallowin_plan_t` associated with this instance of the @oper{conv2d_shallowin} operator, 
 * previously initialized with a call to conv2d_shallowin_init().
 * 
 * `job` points to the job to be performed with this call, previously initialized with `plan`. 
 * 
 * Note that a single call to this function processes only a *single job*. If multiple jobs were initialized,
 * performing the complete convolution requires multiple calls to this function. In such a case, the `Y`, `X`,
 * `K`, `BSO`, and `plan` pointers will be identical in each call, and the `job` pointer will be different with
 * each call.
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
 * @brief Execute a specialized 2D convolution job using a 1x1 kernel (@oper{conv2d_1x1}).
 * 
 * See @oper_ref{conv2d_1x1} for more details about the @oper{conv2d_1x1} operator.
 * 
 * An instance of the @oper{conv2d_1x1} operator requires a plan and one or more jobs, which are represented
 * by the `nn_conv2d_1x1_plan_t` and `nn_conv2d_1x1_job_t` structs. Before performing a 2D convolution using 
 * this function, a call must be made to conv2d_1x1_init() to initialize the plan and any jobs.
 * 
 * `Y` points to the output image tensor @tensor{Y} with shape @tensor_shape{Y_h, Y_w, Y_c}, which correspond to
 * the output image rows, columns and channels respectively. The dimensions of @tensor{Y} must be as specified 
 * when `plan` and `job` were initialized. The address supplied for `Y` should be the start address of the output image
 * tensor, *not* the start address of the sub-tensor being computed by the current job.
 * 
 * `X` points to the input image tensor @tensor{X} with shape @tensor_shape{X_h, X_w, X_c}, which correspond to
 * the input image rows, columns and channels respectively. The dimensions of @tensor{X} must be as specified
 * when `plan` and `job` were initialized. The address supplied for `X` should be the start address of the input image tensor,
 * *not* the address at which the convolution window starts for the job being processed.
 * 
 * The memory layout of @tensor{Y} and @tensor{X} are the standard memory layout for image tensors (see @ref 
 * standard_layout).
 * 
 * `K` points to the kernel tensor @tensor{K} with shape @tensor_shape{Y_c, X_c}, which correspond to the output 
 * image channels and input image channels respectively. The dimensions of @tensor{K} must be as specified when 
 * `plan` was initialized. The address supplied for `K` should be the start address of the kernel tensor.
 * 
 * The memory layout of @tensor{K} is the standard memory layout for 2D tensors (see @ref standard_layout).
 * 
 * `BSO` points to an array of bias-scale-offset parameters required for this convolution. Each `nn_bso_block_t` 
 * in the array contains the bias-scale-offset parameters for a single output channel group, (@ttref{VPU_INT8_ACC_PERIOD} 
 * output channels). If @math{Y_c} is not a multiple of @ttref{VPU_INT8_ACC_PERIOD}, then the output channel tail ( the 
 * last @math{(Y_c mod 16)} output channels) also gets a `nn_bso_block_t`, where the entries corresponding to channels 
 * beyond @math{Y_c} are ignored. The address supplied for `BSO` should be the start address of the the array, *not* 
 * the address of the `nn_bso_block_t` corresponding of the first output channel of the job being processed.
 * 
 * `plan` points to the `nn_conv2d_1x1_plan_t` associated with this instance of the @oper{conv2d_1x1} operator, 
 * previously initialized with a call to conv2d_1x1_init().
 * 
 * `job` points to the job to be performed with this call, previously initialized with `plan`. 
 * 
 * Note that a single call to this function processes only a *single job*. If multiple jobs were initialized,
 * performing the complete convolution requires multiple calls to this function. In such a case, the `Y`, `X`,
 * `K`, `BSO`, and `plan` pointers will be identical in each call, and the `job` pointer will be different with
 * each call.
 * 
 * @requires_word_alignment{Y,X,K,BSO}
 * 
 * @note While the conv2d_deep() function is capable of handling convolutions with 1x1 kernels,
 *       the kernel tensor layout used by this function is NOT compatible with that required by 
 *       conv2d_deep().
 * 
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
 * @brief Execute a depthwise 2D convolution (@oper{conv2d_depthwise}) job.
 * 
 * See @oper_ref{conv2d_depthwise} for more details about the @oper{conv2d_depthwise} operator.
 * 
 * An instance of the @oper{conv2d_depthwise} operator requires a plan and one or more jobs, which are represented
 * by the `nn_conv2d_depthwise_plan_t` and `nn_conv2d_depthwise_job_t` structs. Before performing a 2D 
 * convolution using this function, a call must be made to conv2d_depthwise_init() to initialize the plan and 
 * any jobs.
 * 
 * `Y` points to the output image tensor @tensor{Y} with shape @tensor_shape{Y_h, Y_w, X_c}, which 
 * correspond to the output image rows, columns and channels respectively. The dimensions of @tensor{Y} 
 * must be as specified when `plan` and `job` were initialized. The address supplied for `Y` should be the start 
 * address of the output image tensor, *not* the start address of the sub-tensor being computed by the 
 * current job.
 * 
 * `X` points to the input image tensor @tensor{X} with shape @tensor_shape{X_h, X_w, X_c}, which 
 * correspond to the input image rows, columns and channels respectively. The dimensions of @tensor{X} 
 * must be as specified when `plan` and `job` were initialized. The address supplied for `X` should be the start 
 * address of input image tensor, *not* the address at which the convolution window starts for the job 
 * being processed.
 * 
 * The memory layout of @tensor{Y} and @tensor{X} are the standard memory layout for image tensors (see @ref 
 * standard_layout).
 * 
 * `K` points to the kernel tensor @tensor{K} with shape @tensor_shape{K_h, K_w, X_c}, which correspond to
 * the convolution window rows and columns, and the input image channels respectively. The dimensions of 
 * @tensor{K} must be as specified when `plan` and `job` were initialized. The address supplied for `K` should be the 
 * start address of the kernel tensor.
 * 
 * The memory layout of @tensor{K} is the standard memory layout for 3D tensors (see @ref standard_layout).
 * 
 * `BSO` points to an array of bias-scale-offset parameters required for this convolution. Each `nn_bso_block_t` 
 * in the array contains the bias-scale-offset parameters for a single output channel group, (@ttref{VPU_INT8_ACC_PERIOD} 
 * output channels). If @math{Y_c} is not a multiple of @ttref{VPU_INT8_ACC_PERIOD}, then the output channel tail ( the 
 * last @math{(Y_c mod 16)} output channels) also gets a `nn_bso_block_t`, where the entries corresponding to channels 
 * beyond @math{Y_c} are ignored. The address supplied for `BSO` should be the start address of the the array, *not* 
 * the address of the `nn_bso_block_t` corresponding of the first output channel of the job being processed.
 * 
 * `plan` points to the `nn_conv2d_depthwise_plan_t` associated with this instance of the @oper{conv2d_depthwise} operator, 
 * previously initialized with a call to conv2d_depthwise_init().
 * 
 * `job` points to the job to be performed with this call, previously initialized with `plan`. 
 * 
 * Note that a single call to this function processes only a *single job*. If multiple jobs were initialized,
 * performing the complete convolution requires multiple calls to this function. In such a case, the `Y`, `X`,
 * `K`, `BSO`, and `plan` pointers will be identical in each call, and the `job` pointer will be different with
 * each call.
 * 
 * @requires_word_alignment{Y,X,K,BSO}
 * 
 * @param Y    [out]    The output image tensor @tensor{Y}
 * @param X    [in]     The input image tensor @tensor{X}
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
 * @brief Execute a 2D max-pooling (@oper{maxpool2d}) job.
 * 
 * See @oper_ref{maxpool2d} for more details about the @oper{maxpool2d} operator.
 *
 * An instance of the @oper{maxpool2d} operator requires a plan and one or more jobs, which are 
 * represented by the `nn_maxpool2d_plan_t` and `nn_pool2d_job_t` structs. Before calling maxpool2d()
 * a call must be made to maxpool2d_init() to initialize the plan and any jobs. 
 * 
 * `Y` points to the output image tensor @tensor{Y} with shape @tensor_shape{Y_h, Y_w, X_c}, which correspond to
 * the output image rows, columns and channels respectively. The dimensions of @tensor{Y} must be as specified 
 * when `plan` and `job` were initialized. The address supplied for `Y` should be the start address of the output 
 * image tensor, *not* the start address of the sub-tensor being computed by the current job.
 * 
 * `X` points to the input image tensor @tensor{X} with shape @tensor_shape{X_h, X_w, X_c}, which correspond to
 * the input image rows, columns and channels respectively. The dimensions of @tensor{X} must be as specified
 * when `plan` and `job` were initialized. The address supplied for `X` should be the start address of the input 
 * image tensor, *not* the address at which the pooling window starts for the job being processed.
 * 
 * The memory layout of @tensor{Y} and @tensor{X} are the standard memory layout for image tensors (see @ref 
 * standard_layout).
 *  
 * `plan` points to the `nn_maxpool2d_plan_t` associated with this instance of the @oper{maxpool2d} operator, 
 * previously initialized with a call to maxpool2d_init().
 * 
 * `job` points to the `nn_pool2d_job_t` (previously initialized alongside `plan`) associated with the job to be 
 * processed by this call.
 * 
 * Note that a single call to this function processes only a *single job*. If multiple jobs were initialized,
 * performing the complete operation requires multiple calls to this function. In such a case, the `Y`, `X`,
 * and `plan` pointers will be identical in each call, and the `job` pointer will be different with each call.
 * 
 * @requires_word_alignment{Y,X}
 * 
 * @param Y    [out]    The output image tensor @tensor{Y}
 * @param X    [in]     The input image tensor @tensor{X}
 * @param plan [in]     The @oper{maxpool2d} plan to be processed
 * @param job  [in]     The @oper{maxpool2d} job to be processed
 *
 */
void maxpool2d(
    nn_image_t* Y,
    const nn_image_t* X, 
    const nn_maxpool2d_plan_t* plan,
    const nn_pool2d_job_t* job);


/** 
 * @brief Execute a 2D average-pooling (@oper{avgpool2d}) job.
 * 
 * See @oper_ref{avgpool2d} for more details about the @oper{avgpool2d} operator.
 * 
 * An instance of the @oper{avgpool2d} operator requires a plan and one or more jobs, which are 
 * represented by the `nn_avgpool2d_plan_t` and `nn_pool2d_job_t` structs. Before calling avgpool2d()
 * a call must be made to avgpool2d_init() to initialize the plan and any jobs. 
 * 
 * `Y` points to the output image tensor @tensor{Y} with shape @tensor_shape{Y_h, Y_w, X_c}, which correspond to
 * the output image rows, columns and channels respectively. The dimensions of @tensor{Y} must be as specified 
 * when `plan` and `job` were initialized. The address supplied for `Y` should be the start address of the output 
 * image tensor, *not* the start address of the sub-tensor being computed by the current job.
 * 
 * `X` points to the input image tensor @tensor{X} with shape @tensor_shape{X_h, X_w, X_c}, which correspond to
 * the input image rows, columns and channels respectively. The dimensions of @tensor{X} must be as specified
 * when `plan` and `job` were initialized. The address supplied for `X` should be the start address of the input 
 * image tensor, *not* the address at which the pooling window starts for the job being processed.
 * 
 * The memory layout of @tensor{Y} and @tensor{X} are the standard memory layout for image tensors (see @ref 
 * standard_layout).
 *  
 * `plan` points to the `nn_avgpool2d_plan_t` associated with this instance of the @oper{avgpool2d} operator, 
 * previously initialized with a call to avgpool2d_init().
 * 
 * `job` points to the `nn_pool2d_job_t` (previously initialized alongside `plan`) associated with the job to be 
 * processed by this call.
 * 
 * Note that a single call to this function processes only a *single job*. If multiple jobs were initialized,
 * performing the complete operation requires multiple calls to this function. In such a case, the `Y`, `X`,
 * and `plan` pointers will be identical in each call, and the `job` pointer will be different with each call.
 * 
 * @requires_word_alignment{Y,X}
 * 
 * @param Y    [out]    The output image tensor @tensor{Y}
 * @param X    [in]     The input image tensor @tensor{X}
 * @param plan [in]     The @oper{avgpool2d} plan to be processed
 * @param job  [in]     The @oper{avgpool2d} job to be processed
 * 
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
 * @brief Perform a 2D global average pooling job.
 * 
 * See @oper_ref{avgpool2d_global} for more details about the @oper{avgpool2d_global} operator.
 * 
 * An instance of the @oper{avgpool2d_global} operator requires a plan and one or more jobs, which are 
 * represented by the `nn_avgpool2d_global_plan_t` and `nn_avgpool2d_global_job_t` structs. Before 
 * calling avgpool2d_global() a call must be made to avgpool2d_global_init() to initialize the plan 
 * and any jobs.
 * 
 * `Y` points to the output vector @tensor{y} with shape @tensor_shape{X_c}, which corresponds to
 * the output image channels (equal to input channels). The address supplied for `Y` should be the start 
 * address of the output vector, *not* the start address of the channel being computed by the current 
 * job.
 * 
 * `X` points to the input image tensor @tensor{X} with shape @tensor_shape{X_h, X_w, X_c}, which correspond to
 * the input image rows, columns and channels respectively. The dimensions of @tensor{X} must be as specified
 * when `plan` and `job` were initialized. The address supplied for `X` should be the start address of the input 
 * image tensor, *not* the address at which the pooling window starts for the job being processed.
 * 
 * The memory layout of @tensor{X} is the standard memory layout for image tensors (see @ref standard_layout).
 *  
 * `plan` points to the `nn_avgpool2d_global_plan_t` associated with this instance of the @oper{avgpool2d_global} 
 * operator, previously initialized with a call to avgpool2d_global_init().
 * 
 * `job` points to the `nn_avgpool2d_global_job_t` (previously initialized alongside `plan`) associated with 
 * the job to be processed by this call.
 * 
 * Note that a single call to this function processes only a *single job*. If multiple jobs were initialized,
 * performing the complete operation requires multiple calls to this function. In such a case, the `Y`, `X`,
 * and `plan` pointers will be identical in each call, and the `job` pointer will be different with each call.
 * 
 * @requires_word_alignment{y,X}
 * 
 * @param Y    [out]    The output image tensor @tensor{Y}
 * @param X    [in]     The input image tensor @tensor{X}
 * @param bias [in]     Initial 32-bit accumulator value @math{B}. Shared by all channels.
 * @param plan [in]     The @oper{avgpool2d_global} plan to be processed
 * @param job  [in]     The @oper{avgpool2d_global} job to be processed
 */
void avgpool2d_global(
    int8_t* Y,
    const int8_t* X, 
    const int32_t bias,
    const nn_avgpool2d_global_plan_t* plan,
    const nn_avgpool2d_global_job_t* job);




/** 
 * @brief Perform a fully-connected job.
 * 
 * See @oper_ref{fully_connected_16} for more details about the @oper{fully_connected_16} operator.
 * 
 * An instance of the @oper{fully_connected_16} operator requires a plan and one or more 
 * jobs, which are represented by the `nn_fully_connected_plan_t` and `nn_fully_connected_job_t`
 * structs. Before calling fully_connected_16() a call must be made to fully_connected_init()
 * to initialize the plan and any jobs. 
 * 
 * `Y` points to the 16-bit output vector @tensor{y} with shape @tensor_shape{C_{out}}. The address
 * supplied for `Y` should be the start address of the output vector, *not* the address of the
 * first element to be output by the current job.
 * 
 * `W` points to the weight matrix @tensor{W} with shape @tensor_shape{C_{out}, C_{in}}, which correspond to
 * the output and input vector sizes respectively. The dimensions of @tensor{W} must be as specified when
 * `plan` and `job` were initialized. The address supplied for `W` should be the start address of the 
 * weight matrix, not the address corresponding to the first output index of the current job.
 * 
 * `X` points to the 8-bit input vector @tensor{x} with shape @tensor_shape{C_in}.
 * 
 * `BSO` points to an array of bias-scale-offset parameters required for this operation. Each `nn_bso_block_t` 
 * in the array contains the bias-scale-offset parameters for a single output channel group, (@ttref{VPU_INT8_ACC_PERIOD} 
 * output elements). If @math{C_{out}} is not a multiple of @ttref{VPU_INT8_ACC_PERIOD}, then the output channel tail ( the 
 * last @math{(C_{out} mod 16)} output elements) also gets a `nn_bso_block_t`, where the entries corresponding to channels 
 * beyond @math{C_{out}} are ignored. The address supplied for `BSO` should be the start address of the the array, *not* 
 * the address of the `nn_bso_block_t` corresponding of the first output channel of the job being processed.
 * 
 * `plan` points to the `nn_fully_connected_plan_t` associated with this instance of the @oper{fully_connected_16} 
 * operator, previously initialized with a call to fully_connected_init().
 * 
 * `job` points to the job to be performed with this call, previously initialized with `plan`. 
 * 
 * Note that a single call to this function processes only a *single job*. If multiple jobs were initialized,
 * performing the complete operation requires multiple calls to this function. In such a case, the `Y`, `X`,
 * `W`, `BSO`, and `plan` pointers will be identical in each call, and the `job` pointer will be different with
 * each call.
 * 
 * @requires_word_alignment{Y,W,X,BSO}
 * 
 * @note This function computes inner products of arbitrary length and is thus susceptible to accumulator
 *       saturation. See @ref inner_prod_sat for more information.
 * 
 * @note See @ref out_shift_scale for more details about how the offset and scales are applied to get 
 *       the final result.
 * 
 * 
 * @param Y    [out]    The output vector @tensor{y}
 * @param W    [in]     The weight matrix @tensor{W}
 * @param X    [in]     The input vector @tensor{x}
 * @param BSO  [in]     The bias-scale-offset array
 * @param plan [in]     The @oper{fully_connected_16} plan to be processed
 * @param job  [in]     The @oper{fully_connected_16} job to be processed
 */
void fully_connected_16(
    int16_t* Y,
    const nn_tensor_t* W, 
    const nn_tensor_t* X, 
    const nn_bso_block_t* BSO,
    const nn_fully_connected_plan_t* plan,
    const nn_fully_connected_job_t* job);




/**  
 * @brief Execute a 16-bit argmax (@oper{argmax_16}) job.
 * 
 * See @oper_ref{argmax_16} for more details about the @oper{argmax_16} operator.
 * 
 * @oper{argmax_16} is a simple operator that finds the 16-bit element of the input vector @tensor{x}
 * with the maximum value and sets the value of the output @math{y} to the index of that element.
 * 
 * Unlike most other operators, @oper{argmax_16} does not use a plan or jobs, and does not 
 * require any initialization.
 * 
 * `Y` points to the output index @math{y}.
 * 
 * `X` points to the input vector, flattened to have shape @tensor_shape{N}.
 * 
 * `length` is the size @math{N} of the input vector @tensor{x}, expressed in elements.
 *
 *  @param Y      [out]     The output index @math{y}
 *  @param X      [in]      The input vector @tensor{x}
 *  @param length [in]      The number of elements @math{N} of the input vector @tensor{x}
 */
void argmax_16(
    int32_t* Y,
    const int16_t* X,
    const int32_t length);


/** 
 * @brief Execute a bit depth reduction (@oper{requantize_16_to_8}) job.
 * 
 * See @oper_ref{requantize_16_to_8} for more details about the @oper{requantize_16_to_8} operator.
 * 
 * An instance of the @oper{requantize_16_to_8} operator requires one or more jobs, which are represented
 * by the `nn_requantize_16_to_8_job_t` struct. No plan is required for an instance of @oper{requantize_16_to_8}
 * as there are no parameters shared between jobs. Before calling requantize_16_to_8() a call must be made
 * to requantize_16_to_8_init() to initialize any jobs. 
 * 
 * `Y` points to the 8-bit output vector @tensor{y} with length @tensor_shape{N}.
 * 
 * `X` points to the 16-bit input vector @tensor{x} with length @tensor_shape{N}.
 * 
 * `job` points to the `nn_requantize_16_to_8_t` associated with this job, previously initialized 
 * with a call to requantize_16_to_8_init().
 * 
 * @note This function can safely operate in-place on a buffer if and only if there is only a single
 *       job to compute the entire output.
 *
 *  @param Y   [out]    The output vector @tensor{y}
 *  @param X   [in]     The input vector @tensor{x}
 *  @param job [in]     The @oper{requantize_16_to_8} job to be processed
 */
void requantize_16_to_8(
    int8_t* Y,
    const int16_t* X,
    const nn_requantize_16_to_8_job_t* job);



/** 
 * @brief Execute an 8-bit look-up (@oper{lookup8}) operation.
 * 
 * @oper{lookup8} is a simple operator that maps an 8-bit input vector @tensor{x} to an
 * 8-bit output vector @tensor{y} in an element-wise fashion using a look-up table @math{T}.
 * 
 * Unlike most other operators, @oper{lookup8} does not use a plan or jobs, and does not
 * require any initialization. 
 * 
 * `Y` points to the output vector @tensor{y} with length @tensor_shape{N}.
 * 
 * `X` points to the input vector @tensor{x} with length @tensor_shape{N}.
 * 
 * `lut` points to the start of the look-up table @math{T} with shape @tensor_shape{256}.
 * 
 * `length` is the size @math{N} of the input and output vectors (in elements).
 * 
 * @note This function can safely operate in-place on a buffer. To do the look-up in-place
 *        just pass the same address for both `X` and `Y`.
 * 
 * @todo Create jobs.
 * 
 * @param Y      [out]  The output vector @tensor{y}
 * @param X      [in]   The input vector @tensor{x}
 * @param lut    [in]   Look-up table @tensor{T}
 * @param length [in]   Length @math{N} of input and output vectors
 */
void lookup8(
    uint8_t* Y,
    const uint8_t* X,
    const uint8_t* lut,
    const unsigned length);




#ifdef __XC__
} // extern "C"
#endif

#endif //NN_OPERATOR_H_
