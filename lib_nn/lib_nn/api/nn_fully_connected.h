#ifndef FULLY_CONNECTED_H_
#define FULLY_CONNECTED_H_

#include "nn_bso.h"

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
void fc_deepin_shallowout_16(
    const nn_tensor_t* W, 
    const int32_t* B,
    const nn_image_t* X, 
    int16_t* Y,
    const int32_t C_out, 
    const int32_t C_in,
    const uint16_t* shifts, 
    const int16_t* scales);

#endif // FULLY_CONNECTED_H_
