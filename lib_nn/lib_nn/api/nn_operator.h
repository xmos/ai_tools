

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
 * Perform a 2D convolution of kernel tensor @tensor{K} with input image @tensor{X}
 * to produce output image @tensor{Y}.
 *  
 * This function is optimized for shallow input images. A "shallow" image is one with a small number 
 * of input channels. With this function, the product of the input channel count and the 
 * convolution window width must be less than or equal to the byte-width of the vectors 
 * (@ttref{VPU_INT8_EPV}).
 * 
 * @math{X_c \cdot K_h \leq 32}
 * 
 * The output image may be deep. Both the input channel count @math{X_c} and the output channel count 
 * @math{Y_c} must be a multiple of 4.
 * 
 * This function also supports implied padding of the input image in which a specified value is used as 
 * the padding value.
 * 
 * An instance of the @oper{conv2d_shallowin} operator requires a plan and one or more jobs, which are represented
 * by the `nn_conv2d_shallowin_plan_t` and `nn_conv2d_shallowin_job_t` structs. Before performing a 2D convolution 
 * using this function, a call must be made to conv2d_shallowin_init() to initialize the plan and any jobs. 
 * Each job, together with the plan (which is shared by all jobs), contains the information required to compute 
 * a subset of the output image. 
 * 
 * The computation of the output image can be done with a single call, or may be divided among multiple calls. 
 * In either case, the particular work to be done by a given call is specified via a job. Each job specifies a 
 * rectangular region of the output image to be computed when that job is invoked with conv2d_shallowin().
 * 
 * Splitting the convolution into multiple jobs may serve several ends, including parallelization, returning
 * control to caller to service other application resources, or even to improve the performance of the 
 * operation itself.
 * ___
 * 
 * The operation performed is the following:
 * 
 * @f[
 *      V\left[r,c,p\right]=
 *          B_p+
 *          \sum_{w_r=0}^{K_h-1}\sum_{w_c=0}^{K_w-1}\sum_{k=0}^{X_c-1} 
 *          X\left[ w_{r0}+r\cdot w_{vert}+w_r,
 *                  w_{c0}+c\cdot w_{hori}+w_c,
 *                  k\right]\cdot K\left[p,w_r,w_c,k\right]\\\
 *   \\\  
 *       Y\left[r,c,p\right]= sat_{8}\left(\frac{\left(sat_{16}\left(\frac{V\left[r,c,p\right]}
 *              {2^{s_{1p}}}\right)\cdot s_{2p}\right)}{2^{s_{3p}}}\right)
 * @f]
 * 
 * where  
 * @tensor{V} is an intermediate value,  
 * @math{(r,c,p)} are the output row, column and channel,  
 * @math{(w_{vert},w_{hori})} are the vertical and horizontal strides of the convolution window,
 *      as provided to `conv2d_shallowin_init()`,  
 * @math{(w_{r0},w_{c0})} is the initial row and column of the convolution window,
 *      as provided to `conv2d_shallowin_init()`,  
 * @math{(B_i, s_{1p}, s_{2p}, s_{3p})} are the `bias`, `shift1`, `scale` and `shift2` values
 *      respectively) encoded in the `BSO` data, associated with output channel @math{i}, and  
 * @math{sat_8\left(\cdot\right)} and @math{sat_{16}\left(\cdot\right)} saturate their arguments 
 *      to the symmetric @math{8}- and @math{16}-bit bounds.
 * 
 * ___
 * 
 * The following diagram shows an example of a @math{3\times{}3} convolution window moving across 
 * an input image with shape @math{5\times{}7}, with vertical stride of @math{3} and a horizontal
 * stride of @math{2} to produce a @math{2\times{}4} output image. (Note: channel depth is not
 * shown)
 * 
 * @inlinecode
 *    _____                     _____                      _____                    _____   
 *   |O O O|P P P P P P     P P|O O O|P P P P      P P P P|O O O|P P    P P P P P P|O O O|  
 *   |O O O|X X X X X P     P X|O O O|X X X P      P X X X|O O O|X P    P X X X X X|O O O|
 *   |O_O_O|X X X X X P     P X|O_O_O|X X X P      P X X X|O_O_O|X P    P X X X X X|O_O_O|
 *    P X X X X X X X P     P X X X X X X X P      P X X X X X X X P    P X X X X X X X P
 *    P X X X X X X X P     P X X X X X X X P      P X X X X X X X P    P X X X X X X X P
 *    P X X X X X X X P     P X X X X X X X P      P X X X X X X X P    P X X X X X X X P
 *                                                                                            
 *        Y _ _ _               Y Y _ _                Y Y Y _             Y Y Y Y
 *        _ _ _ _               _ _ _ _                _ _ _ _             _ _ _ _
 *                                                                                          
 *                                                                                                
 *    P P P P P P P P P     P P P P P P P P P      P P P P P P P P P    P P P P P P P P P 
 *    P X X X X X X X P     P X X X X X X X P      P X X X X X X X P    P X X X X X X X P
 *    P_X_X X X X X X P     P X X_X_X X X X P      P X X X X_X_X X P    P X X X X X X_X_P
 *   |O O O|X X X X X P     P X|O O O|X X X P      P X X X|O O O|X P    P X X X X X|O O O| 
 *   |O O O|X X X X X P     P X|O O O|X X X P      P X X X|O O O|X P    P X X X X X|O O O| 
 *   |O_O_O|X X X X X P     P X|O_O_O|X X X P      P X X X|O_O_O|X P    P X X X X X|O_O_O| 
 *                                                                                            
 *        Y Y Y Y               Y Y Y Y                Y Y Y Y             Y Y Y Y
 *        Y _ _ _               Y Y _ _                Y Y Y _             Y Y Y Y  
 *  
 * 
 * @endinlinecode
 * 
 * The input, output, (implied) padding and window pixels are represented by `X`, `Y`, `P` 
 * and `O` respectively.
 * ___
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
 * The @oper{conv2d_1x1} operator is similar to @oper{conv2d_deep}, but is specialized for the case when
 * the the following conditions are met:
 * 
 * - The convolution window's spatial dimensions are @tensor_shape{1,1}
 * - The convolution window's strides are @tensor_shape{1,1}
 * - The convolution window's start position with respect to the input image is @tensor_shape{0,0}
 * - The spatial dimensions of the input and output images are the same.
 * - (Implied by the above conditions) No padding is required.
 * 
 * With those constraints, each pixel in the input image is mapped onto the corresponding pixel in
 * the output image, and the kernel tensor @tensor{K} is a matrix which transforms from input channels
 * to output channels.
 * 
 * An instance of the @oper{conv2d_1x1} operator requires a plan and one or more jobs, which are represented
 * by the `nn_conv2d_1x1_plan_t` and `nn_conv2d_1x1_job_t` structs. Before performing a 2D convolution using 
 * this function, a call must be made to conv2d_1x1_init() to initialize the plan and any jobs. Each job,
 * together with the plan (which is shared by all jobs), contains the information required to compute a subset of 
 * the output image. 
 * 
 * The computation of the output image can be done with a single call, or may be divided among multiple calls. 
 * In either case, the particular work to be done by a given call is specified via a job. Unlike the @oper{conv2d_deep} 
 * operator, the output pixels computed by a @oper{conv2d_1x1} job must be contiguous in memory, though each job may 
 * only compute a subset of the output channels.
 * 
 * Two instances of a @oper{conv2d_1x1} operator may only share the plan and jobs if the two instances share the same
 * hyperparameters. Hyperparameters for the @oper{conv2d_1x1} operator are the input and output image dimensions.
 * 
 * ___
 * 
 * The operation performed is the following:
 * 
 * @f[
 *      V\left[r,c,p\right]=
 *          B_p+\sum_{k=0}^{X_c-1} 
 *          X\left[ r,c,k\right]\cdot K\left[p,k\right]\\\
 *   \\\  
 *       Y\left[r,c,p\right]= sat_{8}\left(\frac{\left(sat_{16}\left(\frac{V\left[r,c,p\right]}
 *              {2^{s_{1p}}}\right)\cdot s_{2p}\right)}{2^{s_{3p}}}\right)
 * @f]
 * 
 * where  
 * @tensor{V} is an intermediate value,  
 * @math{(r,c,p)} are the output row, column and channel,  
 * @math{(w_{vert},w_{hori})} are the vertical and horizontal strides of the convolution window,
 *      as provided to `conv2d_shallowin_init()`,  
 * @math{(B_i, s_{1p}, s_{2p}, s_{3p})} are the `bias`, `shift1`, `scale` and `shift2` values
 *      respectively) encoded in the `BSO` data, associated with output channel @math{i}, and  
 * @math{sat_8\left(\cdot\right)} and @math{sat_{16}\left(\cdot\right)} saturate their arguments 
 *      to the symmetric @math{8}- and @math{16}-bit bounds.
 * 
 * ___
 * 
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
 * A depthwise 2D convolution is one in which the number of output channels is equal to the
 * number of input channels, and where the `k`th output channel receives no contributions from 
 * from any but the `k`th input channel.
 * 
 * In that way, the @oper{conv2d_depthwise} operator is similar to @oper{conv2d_deep}, but is specialized for the case
 * where the the following conditions are met:
 *  - The input and output images have the same channel count (@math{X_c = Y_c})
 *  - The kernel elements not along the channel-wise diagonal (first and final dimensions of the @oper{conv2d_deep} 
 *     kernel tensor) are all zero.
 * 
 * When the conditions above are met, the off-diagonal (channel-wise) elements can be omitted from the tensor,
 * and the operation can be performed more efficiently.
 * 
 * An instance of the @oper{conv2d_depthwise} operator requires a plan and one or more jobs, which are represented
 * by the `nn_conv2d_depthwise_plan_t` and `nn_conv2d_depthwise_job_t` structs. Before performing a 2D 
 * convolution using this function, a call must be made to conv2d_depthwise_init() to initialize the plan and 
 * any jobs. Each job, together with the plan (which is shared by all jobs), contains the information required 
 * to compute a subset of the output image.
 * 
 * The computation of the output image can be done with a single call, or may be divided among multiple calls. 
 * In either case, the particular work to be done by a given call is specified via a job. Each job specifies a 
 * rectangular region of the output image to be computed when that job is invoked with conv2d_depthwise().
 * 
 * Two instances of a @oper{conv2d_depthwise} operator may only share the plan and jobs if the two instances share 
 * the same hyperparameters. Hyperparameters for the @oper{conv2d_depthwise} operator are the input image dimensions, 
 * the output image dimensions, the zero point, and the convolution window's dimensions, stride and starting 
 * position.
 * 
 * @note In the future a "depth multiplier" may be supported. A depth multiplier is an integer which is the ratio
 * of output channel count to input channel count. With a depth multiplier of `N`, each input channel contributes
 * to `N` different output channels (though it is still true that no two input channels contribute to the same 
 * output channel). Currently, the depth multiplier has an implicit value of `1`.
 * 
 * ___
 * 
 * The operation performed is the following:
 * 
 * @f[
 *      V\left[r,c,p\right]=
 *          B_p+\sum_{w_r=0}^{K_h-1}\sum_{w_c=0}^{K_w-1} 
 *          X\left[ w_{r0}+r\cdot w_{vert}+w_r,
 *                  w_{c0}+c\cdot w_{hori}+w_c,
 *                  p\right]\cdot K\left[w_r,w_c,p\right]\\\
 *   \\\  
 *       Y\left[r,c,p\right]= sat_{8}\left(\frac{\left(sat_{16}\left(\frac{V\left[r,c,p\right]}
 *              {2^{s_{1p}}}\right)\cdot s_{2p}\right)}{2^{s_{3p}}}\right)
 * @f]
 * 
 * where  
 * @tensor{V} is an intermediate value,  
 * @math{(r,c,p)} are the output row, column and channel,  
 * @math{(w_{vert},w_{hori})} are the vertical and horizontal strides of the convolution window,
 *      as provided to `conv2d_shallowin_init()`,  
 * @math{(w_{r0},w_{c0})} is the initial row and column of the convolution window,
 *      as provided to `conv2d_shallowin_init()`,  
 * @math{(B_i, s_{1p}, s_{2p}, s_{3p})} are the `bias`, `shift1`, `scale` and `shift2` values
 *      respectively) encoded in the `BSO` data, associated with output channel @math{i}, and  
 * @math{sat_8\left(\cdot\right)} and @math{sat_{16}\left(\cdot\right)} saturate their arguments 
 *      to the symmetric @math{8}- and @math{16}-bit bounds.
 * 
 * ___
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
 * The @oper{maxpool2d} operator uses a rectangular 'pooling' window which slides across the spatial 
 * dimensions of an input image. Each pixel of the output image corresponds to a single location of
 * the pooling window relative to the input image. The resulting value of output channel `k` of some
 * output pixel is the maximum value of input channel `k` within the pooling window (at the location
 * corresponding to that output pixel).
 * 
 * An instance of the @oper{maxpool2d} operator requires a plan and one or more jobs, which are 
 * represented by the `nn_maxpool2d_plan_t` and `nn_pool2d_job_t` structs. Before calling maxpool2d()
 * a call must be made to maxpool2d_init() to initialize the plan and any jobs. Each job, together
 * with the plan (which is shared by all jobs), contains the information required to compute a 
 * subset of the output image.
 * 
 * The computation of the output image can be done with a single call, or may be divided among multiple
 * calls. In either case, the particular work to be done by a given call is specified via a job. Each
 * @oper{maxpool2d} job specifies a rectangular region of the output image to be computed when that
 * job is invoked with maxpool2d().
 * 
 * Two instances of the @oper{maxpool2d} operator may only share a plan or jobs if the two instances 
 * share the same hyperparameters. Hyperparameters for the @oper{maxpool2d} operator are the input
 * image dimensions, the output image dimensions, the pooling window dimensions, initial position 
 * and strides.
 * 
 * The input and output images must have the same channel count, which must be a multiple of `4`.
 * 
 * @oper{maxpool2d} does not support padding of the input image.
 * 
 * ___
 * 
 * The operation performed is the following:
 * 
 * @f[
 *      
 *      V_{r,c}\left[u,v,p \right] = X\left[
 *                                      W_{r0} + r\cdot W_{vert} + u,
 *                                      W_{c0} + c\cdot W_{hori} + v,
 *                                      p \right] \\\
 *      \text{ for } 0 \leq u \lt W_{height} \text{ and } 0 \leq v \lt W_{width} \\\
 *   \\\  
 *      Y\left[r,c,p\right]= max\left( \bar V_{r,c} \right)
 * 
 * @f]
 * 
 * where  
 * @tensor{V_{r,c}} is an intermediate tensor representing the portion of the input image within the pooling window,  
 * @math{(r,c,p)} are the output row, column and channel,  
 * @math{(W_{height}, W_{width})} are the height and width of the pooling window  
 * @math{(W_{vert},W_{hori})} are the vertical and horizontal strides of the pooling window, and   
 * @math{(W_{r0},W_{c0})} is the initial row and column of the pooling window.
 * ___
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
 * The @oper{avgpool2d} operator uses a rectangular 'pooling' window which slides across the spatial 
 * dimensions of an input image. Each pixel of the output image corresponds to a single location of
 * the pooling window relative to the input image. The resulting value of output channel `k` of some
 * output pixel is the average value of input channel `k` within the pooling window (at the location
 * corresponding to that output pixel).
 * 
 * An instance of the @oper{avgpool2d} operator requires a plan and one or more jobs, which are 
 * represented by the `nn_avgpool2d_plan_t` and `nn_pool2d_job_t` structs. Before calling avgpool2d()
 * a call must be made to avgpool2d_init() to initialize the plan and any jobs. Each job, together
 * with the plan (which is shared by all jobs), contains the information required to compute a 
 * subset of the output image.
 * 
 * The computation of the output image can be done with a single call, or may be divided among multiple
 * calls. In either case, the particular work to be done by a given call is specified via a job. Each
 * @oper{avgpool2d} job specifies a rectangular region of the output image to be computed when that
 * job is invoked with avgpool2d().
 * 
 * Two instances of the @oper{avgpool2d} operator may only share a plan or jobs if the two instances 
 * share the same hyperparameters. Hyperparameters for the @oper{avgpool2d} operator are the input
 * image dimensions, the output image dimensions, as well as the pooling window dimensions, initial 
 * position and strides.
 * 
 * The input and output images must have the same channel count, which must be a multiple of `4`.
 * 
 * @oper{avgpool2d} does not support padding of the input image.
 * 
 * ___
 * 
 * The operation performed is the following:
 * 
 * @f[
 *      
 *      Y\left[r,c,p \right] =  \frac{1}{W_{height}\cdot W_{width}} \cdot
 *                              \sum_{w_r=0}^{W_{height}-1}\sum_{w_c=0}^{W_{width}-1} 
 *                                  X\left[ W_{r0}+r\cdot W_{vert}+w_r,
 *                                  W_{c0}+c\cdot W_{hori}+w_c,
 *                                  p\right] \\\
 * 
 * 
 * @f]
 * 
 * where  
 * @math{(r,c,p)} are the output row, column and channel,  
 * @math{(W_{height}, W_{width})} are the height and width of the pooling window  
 * @math{(W_{vert},W_{hori})} are the vertical and horizontal strides of the pooling window, and   
 * @math{(W_{r0},W_{c0})} is the initial row and column of the pooling window.
 * ___
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
 * The @oper{avgpool2d_global} operator is like a specialized version of the @oper{avgpool2d} operator 
 * in which the pooling window is the size of the input image. The output image therefore has only a 
 * single pixel. Unlike @oper{avgpool2d}, @oper{avgpool2d_global} also applies a bias and scaling to
 * each of the output channels.
 * 
 * An instance of the @oper{avgpool2d} operator requires a plan and one or more jobs, which are 
 * represented by the `nn_avgpool2d_plan_t` and `nn_pool2d_job_t` structs. Before calling avgpool2d()
 * a call must be made to avgpool2d_init() to initialize the plan and any jobs. Each job, together
 * with the plan (which is shared by all jobs), contains the information required to compute a 
 * subset of the output image.
 * 
 * The computation of the output image can be done with a single call, or may be divided among multiple
 * calls. In either case, the particular work to be done by a given call is specified via a job. Each
 * @oper{avgpool2d} job specifies a rectangular region of the output image to be computed when that
 * job is invoked with avgpool2d().
 * 
 * Two instances of the @oper{avgpool2d} operator may only share a plan or jobs if the two instances 
 * share the same hyperparameters. Hyperparameters for the @oper{avgpool2d_global} operator are the input
 * image dimensions.
 * 
 * The input and output images must have the same channel count, which must be a multiple of `4`.
 * ___
 * 
 * The operation performed is the following:
 * 
 * @f[
 *      
 *      Y\left[p \right] =  \frac{
 *                              B + \sum_{r=0}^{X_h-1}\sum_{c=0}^{X_w-1}\left( 
 *                                  X\left[r,c,p \right]\right)}{X_h\cdot X_w} \\\
 * 
 * @f]
 * 
 * where  
 * @math{p} is the output channel,  
 * @math{B} is the bias,  
 * ___
 * 
 * `Y` points to the output image tensor @tensor{Y} with shape @tensor_shape{X_c}, which corresponds to
 * the output image channels (equal to input channels). The address supplied for `Y` should be the start 
 * address of the output image tensor, *not* the start address of the channel being computed by the current 
 * job.
 * 
 * `X` points to the input image tensor @tensor{X} with shape @tensor_shape{X_h, X_w, X_c}, which correspond to
 * the input image rows, columns and channels respectively. The dimensions of @tensor{X} must be as specified
 * when `plan` and `job` were initialized. The address supplied for `X` should be the start address of the input 
 * image tensor, *not* the address at which the pooling window starts for the job being processed.
 * 
 * The memory layout of @tensor{Y} and @tensor{X} are the standard memory layout for image tensors (see @ref 
 * standard_layout).
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
 * @requires_word_alignment{Y,X}
 * 
 * 
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
 * The @oper{fully_connected_16} operator is a matrix-vector multiplication with an 8-bit
 * input vector @tensor{x}, an 8-bit weight matrix @tensor{W} and a 16-bit output vector 
 * @tensor{y}.
 * 
 * An instance of the @oper{fully_connected_16} operator requires a plan and one or more 
 * jobs, which are represented by the `nn_fully_connected_plan_t` and `nn_fully_connected_job_t`
 * structs. Before calling fully_connected_16() a call must be made to fully_connected_init()
 * to initialize the plan and any jobs. Each job, together with the plan (which is shared by
 * all jobs), contains the information required to compute a subset of the output vector.
 * 
 * The computation of the output vector can be done with a single call, or may be divided 
 * among multiple calls. In either case, the particular work to be done by a given call is
 * specified via a job. Each @oper{fully_connected_16} job specifies a contiguous subsequence
 * of the output vector to be computed when that job is invoked with fully_connected_16().
 * 
 * Two instances of the @oper{fully_connected_16} operator may only share a plan or jobs if
 * the two instances share hte same hyperparameters. Hyperparameters for the 
 * @oper{fully_connected_16} operator are the input and output vector lengths (and thus
 * also the dimensions of the weight matrix.
 * 
 * ___
 * 
 * The operation performed is the following:
 * 
 * @f[
 * 
 *      v\left[p \right] = B_i + \sum_{r=0}^{C_{in}-1} \left( W[p,r] \cdot x[r] \right)\\\
 *   \\\  
 *      y\left[p \right] = sat_{16}\left(\frac{\left(sat_{16}\left(\frac{v\left[p \right]}
 *              {2^{s_{1p}}}\right)\cdot s_{2p}\right)}{2^{s_{3p}}}\right) \text{for } 0 \leq p \lt C_{out}
 * 
 * @f]
 * 
 * where  
 * @tensor{v} is an intermediate vector,  
 * @math{p} is the output index,  
 * @math{(B_i, s_{1i}, s_{2i}, s_{3i})} are the `bias`, `shift1`, `scale` and `shift2` values
 *      respectively encoded in the `BSO` data, associated with output channel @math{i}, and  
 * @math{sat_{16}\left(\cdot\right)} saturates its argument to the symmetric @math{16}-bit bounds.
 * 
 * ___
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
 * @oper{argmax_16} is a simple operator that finds the 16-bit element of the input vector @tensor{x}
 * with the maximum value and sets the value of the output @math{y} to the index of that element.
 * 
 * Unlike most other operators, @oper{argmax_16} does not use a plan or jobs, and does not 
 * require any initialization.
 * 
 * @oper{argmax_16} is defined with the assumption that the input is a vector @tensor{x} with only a 
 * single dimension. Because tensors are stored in contiguous blocks of memory, a multidimensional 
 * tensor can be implicitly flattened into a vector by simply taking the product of its dimensions 
 * as `length`.
 * 
 * The @oper{argmax_16} operator cannot be split into multiple jobs to be processed by multiple
 * cores simultaneously.
 * 
 * ___
 * 
 * The operation performed is the following:
 * 
 * @f[
 *      y \leftarrow argmax_{k}\{ x\left[k\right] \} \text{ for } 0 \leq k \lt N
 * @f]
 * 
 * where
 * @math{N} is the number of elements in the input vector, given by `length`.
 * 
 * ____
 * 
 * For example, if the input vector were the following 16-bit array:
 * 
 * @code
 *     int16_t input_vector[8] = { 0, -534, 1224, -32768, 16000, 8, 100, -32 };
 * @endcode
 * 
 * Then the result of an @oper{argmax_16} operation would be `4`, as `input_vector[4]` (`16000`) is the maximum element.
 * 
 * ____
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
 * The @oper{requantize_16_to_8} operator reduces the bit depth of the input vector @tensor{x} from
 * 16 bits to 8 bits. This is equivalent to dividing each element by @math{2^8} and rounding to the
 * nearest integer.
 * 
 * An instance of the @oper{requantize_16_to_8} operator requires one or more jobs, which are represented
 * by the `nn_requantize_16_to_8_job_t` struct. No plan is required for an instance of @oper{requantize_16_to_8}
 * as there are no parameters shared between jobs. Before calling requantize_16_to_8() a call must be made
 * to requantize_16_to_8_init() to initialize any jobs. Each job contains the information required to
 * compute a subset of the output vector.
 * 
 * The computation of the output vecto can be done with a single call, or may be divided among multiple 
 * calls. In either case, the particular work to be done by a given call is specified via a job. Each
 * @oper{requantize_16_to_8} job specifies a contiguous subsequence of the output vector to be computed
 * when that job is invoked with requantize_16_to_8().
 * 
 * ___
 * 
 * The operation performed is the following:
 * 
 * @f[
 *      y\left[k\right] \overset{8-bit}{\longleftarrow} x\left[k\right] \text{for } 0 \leq k \lt N
 * @f]
 * 
 * where
 * @math{N} is the number of elements in the input vector, given by `length`.
 * 
 * ____
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
 * @oper{lookup8} is defined with the assumption that the input and output are vectors @tensor{x} 
 * with only a single dimension. Because tensors are stored in contiguous blocks of memory, 
 * a multidimensional tensor can be implicitly flattened into a vector by simply taking the 
 * product of its dimensions as `length`.
 * 
 * The @oper{lookup8} operator cannot be split into multiple jobs to be processed by multiple
 * cores simultaneously.
 * 
 * ___
 * 
 * The operation performed is the following:
 * 
 * @f[
 *      y\left[k\right] = T\left[x\left[k\right]\right] \text{for } 0 \leq k \lt N
 * @f]
 * 
 * where
 * @math{N} is the number of elements in the input vector, given by `length`.
 * 
 * ____
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
