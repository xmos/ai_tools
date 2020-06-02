

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
 * 
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
 * ____
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
 * `BSO` points to an array of bias-shifts-scale parameters required for this convolution. Each 
 * `nn_bso_block_t` in the array contains the bias-shifts-scale parameters for a single output channel group,
 * (@ttref{VPU_INT8_ACC_PERIOD} output channels). If @math{Y_c} is not a multiple of @ttref{VPU_INT8_ACC_PERIOD}, 
 * then the output channel tail ( the last @math{(Y_c mod 16)} output channels) also gets `nn_bso_block_t`, where
 * the entries corresponding to channels beyond @math{Y_c} are ignored. The address supplied for `BSO` should be
 * the start address of the the array, *not* the address of the `nn_bso_block_t` corresponding of the first output
 * channel of the job being processed.
 * 
 * `plan` points to the `nn_conv2d_deep_plan_t` which was previously initialized with a call to `conv2d_deep_init()`.
 * 
 * `job` points to the job to be performed in this call, which was previously initialized along-side `plan`. 
 * 
 * Note that a single call to this function processes only a *single job*. If multiple jobs were initialized,
 * performing the complete convolution requires multiple calls to this function. In such a case, the `Y`, `X`,
 * `K`, `BSO`, and `plan` pointers will be identical in each call, and the `job` pointer will be different with
 * each call.
 * 
 * @requires_word_alignment{Y,X,K,BSO}
 * 
 * @param[out] Y        The output image @tensor{Y}
 * @param[in]  X        The input image @tensor{X}
 * @param[in]  K        The kernel tensor @tensor{K}
 * @param[in]  BSO      The bias-shifts-scale parameters
 * @param[in]  plan     The convolution plan
 * @param[in]  job      The convolution job
 */
void conv2d_deep(
    nn_image_t* Y,
    const nn_image_t* X,
    const nn_tensor_t* K,
    const nn_bso_block_t* BSO,
    const nn_conv2d_deep_plan_t* plan,
    const nn_conv2d_deep_job_t* job);



/**
 * @brief Perform a 2D convolution of a shallow input image.
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
 * Before performing a 2D convolution using this function, a call must be made to `conv2d_shallowin_init()`
 * to initialize a `nn_conv2d_shallowin_plan_t` struct and one or more `nn_conv2d_shallowin_job_t` structs. 
 * Each job, together with the plan shared by all jobs, contains the information required to compute a subset 
 * of the pixels in the output image. More specifically, each job corresponds to a rectangular subtensor 
 * of @tensor{Y}, which is to be computed by that job.
 * 
 * Splitting the convolution into multiple jobs may serve several ends, including parallelization, returning
 * control to caller to service other application resources, or even to improve the performance of the 
 * operation itself.
 * 
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
 * 
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
 * ____
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
 * `K` points to the kernel tensor @tensor{K} with shape @tensor_shape{Y_c, K_h, \hat{K_w}, X_c}, where @math{Y_c},
 * @math{K_h} and @math{X_c} correspond to the output image channels, convolution window rows and the input image 
 * channel count respectively. @math{\hat{K_w}} is the augmented convolution window width, which must be exactly
 * @math{32/X_c}.The address supplied for `K` should be the start address of the kernel tensor.
 * 
 * The memory layout of @tensor{K} is the standard memory layout for 4D tensors (see @ref standard_layout). Further,
 * the coefficients for all elements @math{K\left[i,j,k,l\right]} where @math{k\geq K_w} must have the value 0.
 * 
 * `BSO` points to an array of bias-shifts-scale parameters required for this convolution. Each 
 * `nn_bso_block_t` in the array contains the bias-shifts-scale parameters for a single output channel group,
 * (@ttref{VPU_INT8_ACC_PERIOD} output channels). If @math{Y_c} is not a multiple of @ttref{VPU_INT8_ACC_PERIOD}, 
 * then the output channel tail ( the last @math{(Y_c mod 16)} output channels) also gets `nn_bso_block_t`, where
 * the entries corresponding to channels beyond @math{Y_c} are ignored. The address supplied for `BSO` should be
 * the start address of the the array, *not* the address of the `nn_bso_block_t` corresponding of the first output
 * channel of the job being processed.
 * 
 * `plan` points to the `nn_conv2d_shallowin_plan_t` which was previously initialized with a call to 
 * `conv2d_shallowin_init()`.
 * 
 * `job` points to the job to be performed in this call, which was previously initialized along-side `plan`. 
 * 
 * Note that a single call to this function processes only a *single job*. If multiple jobs were initialized,
 * performing the complete convolution requires multiple calls to this function. In such a case, the `Y`, `X`,
 * `K`, `BSO`, and `plan` pointers will be identical in each call, and the `job` pointer will be different with
 * each call.
 * 
 * @requires_word_alignment{Y,X,K,BSO}
 * 
 * @param[out] Y        The output image @tensor{Y}
 * @param[in]  X        The input image @tensor{X}
 * @param[in]  K        The kernel tensor @tensor{K}
 * @param[in]  BSO      The bias-shifts-scale parameters
 * @param[in]  plan     The convolution plan
 * @param[in]  job      The convolution job
 */
void conv2d_shallowin(
    nn_image_t* Y,
    const nn_image_t* X,
    const nn_tensor_t* K,
    const nn_bso_block_t* BSO,
    const nn_conv2d_shallowin_plan_t* plan,
    const nn_conv2d_shallowin_job_t* job);




/**
 * Perform a 2D convolution using a 1x1 kernel over the input image.
 * 
 * 
 * ___
 * 
 * The operation performed is the following:
 * 
 * @f[
 *      V\left[r,c,p\right]=
 *          B_p+\sum_{k=0}^{X_c-1} 
 *          X\left[ r,c,k\right]\cdot K\left[p,k\right]\\\
 * 
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
 * ____
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
 * The bias-shifts-scale tensor `BSO` is layed out as specified in "Bias-Shifts-Scale Tensor Layout".
 * The accumulators for each output channel are seeded with the 32-bit biases encoded in `BSO`, and
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
    nn_image_t* Y,
    const nn_image_t* X,
    const nn_tensor_t* K,
    const nn_bso_block_t* BSO,
    const nn_conv2d_1x1_plan_t* plan,
    const nn_conv2d_1x1_job_t* job);




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
 * 
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
 * ____
 * 
 * `Y` points to the output image tensor @tensor{Y} with shape @tensor_shape{Y_h, Y_w, X_c}, which 
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
 * `K` points to the kernel tensor @tensor{K} with shape @tensor_shape{K_h, K_w, X_c}, which correspond to
 * the convolution window rows and columns, and the input image channels respectively. The dimensions of 
 * @tensor{K} must be as specified when `plan` was initialized. The address supplied for `K` should be the 
 * start address of the kernel tensor.
 * 
 * The memory layout of @tensor{K} is the standard memory layout for 3D tensors (see @ref standard_layout).
 * 
 * `BSO` points to an array of bias-shifts-scale parameters required for this convolution. Each 
 * `nn_bso_block_t` in the array contains the bias-shifts-scale parameters for a single output channel group,
 * (@ttref{VPU_INT8_ACC_PERIOD} output channels). If @math{X_c} is not a multiple of @ttref{VPU_INT8_ACC_PERIOD}, 
 * then the output channel tail ( the last @math{(X_c mod 16)} output channels) also gets `nn_bso_block_t`, where
 * the entries corresponding to channels beyond @math{X_c} are ignored. The address supplied for `BSO` should be
 * the start address of the the array, *not* the address of the `nn_bso_block_t` corresponding of the first output
 * channel of the job being processed.
 * 
 * `plan` points to the `nn_conv2d_depthwise_plan_t` which was previously initialized with a call to 
 * `conv2d_depthwise_init()`.
 * 
 * `job` points to the job to be performed in this call, which was previously initialized along-side `plan`. 
 * 
 * Note that a single call to this function processes only a *single job*. If multiple jobs were initialized,
 * performing the complete convolution requires multiple calls to this function. In such a case, the `Y`, `X`,
 * `K`, `BSO`, and `plan` pointers will be identical in each call, and the `job` pointer will be different with
 * each call.
 * 
 * Constraints:
 *  - `Y`, `X`, `K` and `BSO` must all point to word-aligned addresses.
 *  - @math{X_c} must be a multiple of 4.
 * 
 * \param Y     The output image.
 * \param X     The input image.
 * \param K     The kernel tensor.
 * \param BSO   The bias-shifts-scale tensor.
 * \param plan  The execution plan initialized by `conv2d_depthwise_init()`.
 * \param job   The (single) job to be performed.
 */
void conv2d_depthwise(
    nn_image_t* Y,
    const nn_image_t* X,
    const nn_tensor_t* K,
    const nn_bso_block_t* BSO,
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
    const nn_maxpool2d_plan_t* plan,
    const nn_maxpool2d_job_t* job);


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
    nn_image_t* Y,
    const nn_image_t* X, 
    const nn_avgpool2d_plan_t* plan)
{
    switch(plan->impl){
        case AVGPOOL2D_2X2:
            avgpool2d_2x2(Y, X, plan);
            break;
        default:
            avgpool2d_gen(Y, X, plan);
            break;
    }
}

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
    nn_image_t* Y,
    const nn_image_t* X, 
    const uint32_t x_height, 
    const uint32_t x_width,
    const channel_count_t x_chans,
    const int32_t  bias,
    const uint32_t shift,
    const uint32_t scale);




/** Generalized fully-connected layer with 16-bit outputs.
 * 
 * The logical operation performed is the following:
 * 
 *      Y[i] = ((dot(W[i][], X[] + bias[i]) >> shift[i]) * scale[i]) >> shift2[i];
 * 
 *   where
 *      W[i][] represents the `i`th row of the weight matrix
 *      dot(A,B) is the 32-bit inner product of 8-bit vectors A and B
 *      bias, shift and scale are encoded in `BSO`
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
 * `BSO` is the bias-scale-offset tensor with a shape of `(ceil(C_out/16), 5, 16)`. This tensor
 * encodes the bias, shift and scale for each output channel into a single linear block of memory,
 * allowing a more efficient implementation of this operator. The function `fc_boggle_BSO()` is
 * provided to simplify the layout of this tensor. Use of `fc_boggle_BSO` is not required, but
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
    const nn_tensor_t* W, 
    const nn_tensor_t* X, 
    const nn_bso_block_t* BSO,
    const nn_fully_connected_plan_t* plan,
    const nn_fully_connected_job_t* job);




/**  Determines the index of the largest element of a tensor.
 * 
 * ___
 * 
 * The operation performed is the following:
 * 
 * @f[
 *      C \leftarrow argmax_{k}\{ A\left[k\right] \} \text{ for } 0 \leq k \lt l
 * @f]
 * 
 * ____
 * 
 * `A` points to the input tensor @tensor{A} with shape @tensor_shape{l}.
 * 
 * `C` points to the output tensor with shape @tensor_shape{1}.
 * 
 * `length` is the size @math{l} of the input tensor @tensor{A} in elements.
 *
 *  \param  A       Tensor of shape (N) using a standard layout.
 *  \param  C       Output tensor of shape (1).
 *  \param  length  Number of elements in the input tensor A.
 */
void argmax_16(
    const int16_t* A,
    int32_t* C,
    const int32_t length);


/** Reduce the bit depth of a 16-bit vector to 8 bits
 * 
 * ___
 * 
 * The operation performed is the following:
 * 
 * @f[
 *      Y\left[k\right] \overset{8-bit}{\longleftarrow} X\left[k\right] \text{for } 0 \leq k \lt l
 * @f]
 * 
 * ____
 * 
 * `Y` points to the start of the output tensor @tensor{Y} with shape @tensor_shape{l}.
 * 
 * `X` points to the start of the input tensor @tensor{X} with shape @tensor_shape{l}.
 * 
 * `length` is the size of the input and output tensors @math{l}. If @tensor{X} and @tensor{Y} 
 * are multi-dimensional tensors (e.g. images), then @math{l} should be the product of the 
 * dimensions of the input tensors.
 * 
 * @note This function can safely operate in-place on a buffer. To do the look-up in-place
 *        just pass the same address for both `X` and `Y`.
 * 
 *
 * \param Y         Output tensor
 * \param X         Input tensor
 * \param length    Length of input and output tensors (in elements)
 */
void requantize_16_to_8(
    int8_t* Y,
    const int16_t* X,
    const nn_requantize_16_to_8_job_t* job);



/** Transform an input using a look-up table
 * 
 * This function transforms an 8-bit input tensor @tensor{X} into the 8-bit output tensor
 * @tensor{Y} using the look-up table @tensor{T}.
 * 
 * ___
 * 
 * The operation performed is the following:
 * 
 * @f[
 *      Y\left[k\right] = T\left[X\left[k\right]\right] \text{for } 0 \leq k \lt l
 * @f]
 * 
 * ____
 * 
 * `Y` points to the start of the output tensor @tensor{Y} with shape @tensor_shape{l}.
 * 
 * `X` points to the start of the input tensor @tensor{X} with shape @tensor_shape{l}.
 * 
 * `lut` points to the start of the look-up table @math{T} with shape @tensor_shape{256}.
 * 
 * `length` is the size of the input and output tensors @math{l}. If @tensor{X} and @tensor{Y} 
 * are multi-dimensional tensors (e.g. images), then @math{l} should be the product of the 
 * dimensions of the input tensors.
 * 
 * @note This function can safely operate in-place on a buffer. To do the look-up in-place
 *        just pass the same address for both `X` and `Y`.
 * 
 * \param Y         Output tensor
 * \param X         Input tensor
 * \param lut       Look-up table
 * \param length    Length of input and output tensors
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
 * \param dst   Destination address
 * \param src   Source address
 * \param size  Number of bytes to be copied
*/
void vpu_memcpy(
    void* dst,
    void* src,
    unsigned size);


#ifdef __XC__
} // extern "C"
#endif

#endif //NN_OPERATOR_H_
