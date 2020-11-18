#ifndef LAYERS_H_
#define LAYERS_H_
#include "nn_types.h"
#include "nn_image.h"

/**
 * Describes the parameters needed for an @oper{add_elementwise} operator. @see add_elementwise().
 */
typedef struct {
    /**
     * The parameters that are applied to each input element. Those in `input[0]` are applied to elements of 
     * @tensor{x_0}, and `input[1]` are applied to elements of @tensor{x_1}.
     */
    struct {

        /**
         * `input[k].shr` is the arithmetic _right_-shift @math{s_k} applied to elements of @tensor{x_k}. This will 
         * usually be a negative value.
         */
        int16_t shr;

        /**
         * `input[k].multiplier` is the scale factor @math{m_k} applied to elements of @tensor{x_k}. This and @math{s_k}
         * are usually derived from the scale factor of the quantized vector @tensor{x_k}.
         */
        int16_t multiplier;
    } input[2];


    struct {
        /**
         * `output.bias` is the 32-bit bias @math{b} to which the scaled inputs are added.
         */
        int32_t bias;

        /**
         * `output.shr` is output the shift @math{s_{out}}, which is the number of bits the 32-bit accumulator is
         * right-shifted by to obtain a final result for each element.
         */
        uint8_t shr;
    } output;
} nn_add_params_t;

/**
 * @brief Invoke an @oper{add_elementwise} job.
 * 
 * The @oper{add_elementwise} operator adds together two quantized 8-bit input vectors, @tensor{x_0} and @tensor{x_1}
 * element-by-element to produce the output vector @tensor{y}. This function assumes that the input vectors and the 
 * output vector each require different quantization parameters.
 * 
 * In order to add together two quantized vectors, their quantization parameters must match. The contents of `params`
 * indicate how to do this.
 * 
 * @par Operation Performed
 * 
 * @f[
 * 
 *      v_i[k] \leftarrow m_i \cdot sat_{16}\!\left( floor\!\left( x_i[k] 
 *                         \cdot 2^{-s_i} \right) \right) \\
 *      
 *      y[k] \leftarrow   sat_{8}\!\left( round\!\left(sat_{32}\!\left(b + v_0[k] + v_1[k]\right) 
 *                          \cdot 2^{-s_{out}} \right) \right)
 * 
 * @f]
 * 
 * where
 * 
 * @par 
 * each @tensor{v_i} is an intermediate 32-bit vector representing the scaled contents of @tensor{x_i}, 
 * 
 * @par
 * @math{k} is the output index,
 * 
 * @par
 * @math{sat_{8}\!\left(\cdot\right)}, @math{sat_{16}\!\left(\cdot\right)}, and @math{sat_{32}\!\left(\cdot\right)} 
 *  saturate their arguments to @math{8}-, @math{16}- and @math{32}-bit bounds respectively, and
 * 
 * @par
 * the remaining parameters are as described below.
 * 
 * @par Parameter Details
 * 
 * `Y` points to the output vector @tensor{y} with shape @tensor_shape{N}.
 * 
 * `X0` and `X1` respectively point to the first and second input vectors @tensor{x_0} and @tensor{x_1}, each with shape 
 * @tensor_shape{N}.
 * 
 * `params` describes the parameters @math{s_i}, @math{m_i}, @math{b} and @math{s_{out}} which are applied for each
 * output element.
 * 
 * `elm_start` and `elm_count` together specify which output elements @math{y[k]} should be calculated by this 
 * invocation. Specifically, this invocation will calculate @math{y[k]} for which `elm_start` @math{\le k \lt} 
 * `(elm_start + elm_count)`.
 * 
 * @param[out]  Y           The output vector @tensor{y}
 * @param[in]   X0          The first input vector @tensor{x_0}
 * @param[in]   X1          The second input vector @tensor{x_1}
 * @param[in]   params      The scaling and bias parameters
 * @param[in]   elm_start   Index of first output element to be computed
 * @param[in]   elm_count   Number of output elements to be computed
 */
void add_elementwise(
    int8_t Y[],
    const int8_t X0[],
    const int8_t X1[],
    const nn_add_params_t* params,
    const unsigned elm_start,
    const unsigned elm_count);



/**  
 * @brief Invoke an @oper{argmax_16} job.
 * 
 * The @oper{argmax_16} operator invokes an argument maximization (@math{argmax_k\\{x[k]\\}}) function, which outputs 
 * the index @math{k} of the maximum element of the vector @tensor{x}. The function is applied to a 16-bit input vector
 * @tensor{x}.
 * 
 * @par Operation Performed
 * 
 * @f[
 *    y \leftarrow argmax_{k}\{ x\left[k\right] \} \text{ for } 0 \leq k \lt N
 * @f]
 * 
 * @par Hyperparameters
 * 
 * <table>
 * <tr><th>Symbol(s)        <th>From        <th>Description
 * <tr><td>@tensor_shape{N} <td>`N`         <td>The length of the input vector (in elements).
 * </table>
 * 
 * @par Data Parameters
 * 
 * <table>
 * <tr><th colspan="2">Symbol <th>Direction <th>Shape <th>From <th>Description
 * 
 * <tr><td colspan="2">@math{y}   <td>out <td><i>scalar</i> <td>`Y` 
 *      <td>The output index.
 * <tr><td colspan="2">@tensor{x} <td>in  <td>@math{(N)}    <td>`X` 
 *      <td>The input vector.
 * </table>
 *
 * @param[out]  Y   The output index @math{y}
 * @param[in]   X   The input vector @tensor{x}
 * @param[in]   N   The number of elements @math{N} of the input vector @tensor{x}
 */
void argmax_16(
    int32_t* Y,
    const int16_t* X,
    const int32_t N);

/** 
 * @brief Invoke a @oper{lookup8} job.
 * 
 * The @oper_ref{lookup8} operator transforms a vector of 8-bit inputs by interpreting each element as an index into
 * the LUT.
 * 
 * @par Operation Performed
 * 
 * @f[
*    y\left[k\right] = T\left[x\left[k\right]\right] \text{for } 0 \leq k \lt N
 * @f]
 * 
 * @par Hyperparameters
 * 
 * <table>
 * <tr><th>Symbol(s)        <th>From        <th>Description
 * <tr><td>@tensor_shape{N} <td>`N`         <td>The length of the input vector (in elements).
 * </table>
 * 
 * @par Data Parameters
 * 
 * <table>
 * <tr><th colspan="2">Symbol <th>Direction <th>Shape <th>From <th>Description
 * 
 * <tr><td colspan="2">@math{y}     <td>out <td><i>scalar</i> <td>`Y` 
 *      <td>The output vector.
 * <tr><td colspan="2">@tensor{x}   <td>in  <td>@math{(N)}    <td>`X` 
 *      <td>The input vector.
 * <tr><td colspan="2">@tensor{T}   <td>in  <td>@tensor_shape{256}  <td>`lut`
 *      <td>The look-up table.
 * </table>
 * 
 * @par Other Parameters
 * 
 * `elm_start` is the index of the first output element to be computed by this invocation.
 * 
 * `elm_count` is the number of output elements to be computed by this invocation.
 * 
 * @par Splitting the Workload
 * 
 * @todo Include information about how to split the work into multiple invocations (e.g. for parallelization), 
 *       particularly any counter-intuitive aspects.
 *
 * @param[out]  Y           The output vector @tensor{y}
 * @param[in]   X           The input vector @tensor{x}
 * @param[in]   lut         Look-up table @tensor{T}
 * @param[in]   elm_start   Index of first output element to be computed.
 * @param[in]   elm_count   Number of output elements to be computed.
 */
void lookup8(
    uint8_t* Y,
    const uint8_t* X,
    const uint8_t* lut,
    const unsigned elm_start,
    const unsigned elm_count);

/** 
 * @brief Invoke a @oper{requantize_16_to_8} job.
 * 
 * The @oper_ref{requantize_16_to_8} operator reduces a vector of 16-bit values down to a vector of 8-bit values.
 * 
 * @par Operation Performed
 * 
 * @f[
*    y\left[k\right] \overset{8-bit}{\longleftarrow} x\left[k\right] \text{for } 0 \leq k \lt N
 * @f]
 * 
 * @par Hyperparameters
 * 
 * <table>
 * <tr><th>Symbol(s)        <th>From        <th>Description
 * <tr><td>@tensor_shape{N} <td>`N`         <td>The length of the input vector (in elements).
 * </table>
 * 
 * @par Data Parameters
 * 
 * <table>
 * <tr><th colspan="2">Symbol <th>Direction <th>Shape <th>From <th>Description
 * 
 * <tr><td colspan="2">@math{y}     <td>out <td><i>scalar</i> <td>`Y` 
 *      <td>The output vector.
 * <tr><td colspan="2">@tensor{x}   <td>in  <td>@math{(N)}    <td>`X` 
 *      <td>The input vector.
 * </table>
 * 
 * @par Other Parameters
 * 
 * `elm_start` is the index of the first output element to be computed by this invocation.
 * 
 * `elm_count` is the number of output elements to be computed by this invocation.
 * 
 * @par Parameter Constraints
 * 
 * The arguments `Y` and `X` must each point to a word-aligned address.
 * 
 * Due to memory alignment requirements, `elm_start` must be a multiple of @math{2}.
 * 
 * @par Splitting the Workload
 * 
 * @todo Include information about how to split the work into multiple invocations (e.g. for parallelization), 
 *       particularly any counter-intuitive aspects.
 * 
 * @param[out]  Y           The output vector @tensor{y}
 * @param[in]   X           The input vector @tensor{x}
 * @param[in]   elm_start   Index of first output element to be computed.
 * @param[in]   elm_count   Number of output elements to be computed.
 */
void requantize_16_to_8(
    int8_t* Y,
    const int16_t* X,
    const unsigned elm_start,
    const unsigned elm_count);




/**
 * Struct represents the parameters needed by each `bsign_8()` job.
 * 
 * Values are set by `bsign_8_init()`.
 * 
 * @note This struct is intended to be opaque.
 */
typedef struct {
    mem_stride_t start;
    uint32_t length;
} nn_bsign_8_job_t;

/**
 * Struct represents the shared parameters required to execute a `bsign_8()` operation. 
 */
typedef struct {
    int8_t zero_point;
} nn_bsign_8_plan_t;

/**
 * @brief Initialize an instance of the @oper{bsign_8} operator.
 * 
 * See @oper_ref{bsign_8} for more details about the @oper{bsign_8} operator. To invoke a 
 * @oper{bsign_8} job, call bsign_8().
 * 
 * When bsign_8() is called, a job (`nn_bsign_8_job_t`) must be supplied to tell it how to do its 
 * work. This function initializes one or more jobs to be supplied in subsequent calls to bsign_8().
 * 
 * Each job computes a range of elements in the output vector (possibly the entire vector).
 * 
 * `jobs` points to an array of `nn_bsign_8_t` to be initialized. Each element represents one job. There 
 * should be `job_count` elements in the array.
 * 
 * `N` is the number of elements @math{N} in the input vector @tensor{x} and output vector @tensor{y}.
 * 
 * `job_count` indicates the number of jobs to be initialized (and thus the number of elements in the `jobs` array).
 * 
 * Unlike many other operators, @oper{bsign_8} will automatically divide the work to be done as evenly as 
 * possible between jobs.
 * 
 * @param plan      [out]  The plan to be initialized.
 * @param jobs      [out]   Array of jobs to be initialized.
 * @param N         [in]    The number of elements in the input. 
 * @param[in]  zero_point   The value @math{z_0} to be used for padding (for all channels)
 * @param job_count [in]    The number of jobs to be initialized.
 */
void bsign_8_init(
    nn_bsign_8_plan_t* plan,
    nn_bsign_8_job_t* jobs,
    const uint32_t N,
    const int8_t zero_point,
    const unsigned job_count);

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


/**
 * Struct represents the parameters needed by each `pad_run()` job.
 *
 * Values are set by `pad_prepare()`.
 *
 * @note This struct is intended to be opaque.
 */
typedef struct nn_pad_plan_t {
  unsigned top_pad_bytes;
  unsigned mid_loop_count;
  unsigned left_pad_bytes;
  unsigned mid_copy_bytes;
  unsigned right_pad_bytes;
  unsigned bottom_pad_bytes;
} nn_pad_plan_t;

// This is for the PaddingValues
// #include "tensorflow/lite/kernels/internal/types.h"

typedef struct padding_values_t {
  int16_t width;
  int16_t height;
  // offset is used for calculating "remaining" padding, for example, `width`
  // is 1 and `width_offset` is 1, so padding_left is 1 while padding_right is
  // 1 + 1 = 2.
  int16_t width_offset;
  // Same as width_offset except it's over the height dimension.
  int16_t height_offset;
} padding_values_t;

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

#endif //LAYERS_H_