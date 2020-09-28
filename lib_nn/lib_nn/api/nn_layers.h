#ifndef LAYERS_H_
#define LAYERS_H_
#include "nn_types.h"
#include "nn_image.h"

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
 * Struct represents the parameters needed by each `requantize_16_to_8()` job.
 * 
 * Values are set by `requantize_16_to_8_init()`.
 * 
 * @note This struct is intended to be opaque.
 */
typedef struct {
    mem_stride_t start;
    uint32_t length;
} nn_requantize_16_to_8_job_t;

/**
 * @brief Initialize an instance of the @oper{requantize_16_to_8} operator.
 * 
 * See @oper_ref{requantize_16_to_8} for more details about the @oper{requantize_16_to_8} operator. To invoke a 
 * @oper{requantize_16_to_8} job, call requantize_16_to_8().
 * 
 * When requantize_16_to_8() is called, a job (`nn_requantize_16_to_8_job_t`) must be supplied to tell it how to do its 
 * work. This function initializes one or more jobs to be supplied in subsequent calls to requantize_16_to_8().
 * 
 * Each job computes a range of elements in the output vector (possibly the entire vector).
 * 
 * `jobs` points to an array of `nn_fully_connected_job_t` to be initialized. Each element represents one job. There 
 * should be `job_count` elements in the array.
 * 
 * `N` is the number of elements @math{N} in the input vector @tensor{x} and output vector @tensor{y}.
 * 
 * `job_count` indicates the number of jobs to be initialized (and thus the number of elements in the `jobs` array).
 * 
 * Unlike many other operators, @oper{requantize_16_to_8} will automatically divide the work to be done as evenly as 
 * possible between jobs.
 * 
 * @param jobs      [out]   Array of jobs to be initialized.
 * @param N         [in]    The number of elements in the input (and output).
 * @param job_count [in]    The number of jobs to be initialized.
 */
void requantize_16_to_8_init(
    nn_requantize_16_to_8_job_t* jobs,
    const uint32_t N,
    const unsigned job_count);

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
void bsign_8_prepare(
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
 * bsign_8_prepare() for more details.
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
    bnn_b32_t* Y,
    const int8_t* X,
    const nn_bsign_8_plan_t* plan,
    const nn_bsign_8_job_t* job);

void bsign_8_ref(
    bnn_b32_t* Y,
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
