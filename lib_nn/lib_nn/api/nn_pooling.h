#ifndef POOLING_H_
#define POOLING_H_

#include "nn_window_params.h"
#include "nn_image.h"

/**
 * Enum identifies optimized assembly implementations for
 * the `avgpool2d()` function.
 */
typedef enum {
    AVGPOOL2D_DEFAULT = 0,    // General case, uses `avgpool2d_asm()`
    AVGPOOL2D_2X2     = 1,    //  Typical 2x2 average pool. Uses `avgpool2d_2x2_asm()`
} nn_avgpool2d_impl_t;

/**
 * Struct represents the parameters needed by each `avgpool2d()` job.
 * 
 * Values are set by `avgpool2d_init()`.
 * 
 * @note This struct is intended to be opaque.
 */
typedef struct {

    struct {
        uint32_t rows;
        uint32_t cols;
    } window;

    channel_count_t channels;

    int32_t shift;
    int32_t scale;

    nn_avgpool2d_impl_t impl;

} nn_avgpool2d_plan_t;




/**
 * Struct represents the parameters needed by a single `avgpool2d()` or `maxpool2d()` job.
 * 
 * Values are set by the corresponding initialization function.
 * 
 * @note This struct is intended to be opaque.
 */
typedef struct {

    struct {
        uint32_t rows;
        uint32_t cols;
        channel_count_t channels;
    } output;

    struct {
        struct {
            mem_stride_t start;
            mem_stride_t row;
            mem_stride_t cog;
        } X;

        struct {
            mem_stride_t row;
            mem_stride_t col;
        } window;

        struct {
            mem_stride_t start;
            mem_stride_t row;
            mem_stride_t cog;
        } Y;

    } stride;

} nn_pool2d_job_t;



/**
 * Flags used with maxpool2d_adv() for advanced scenarios.
 */
typedef enum {
    /** 
     * Placeholder flag used to indicate no other flags are needed.
     */
    MAXPOOL2D_FLAG_NONE = 0,
} nn_maxpool2d_flags_e;


/**
 * Flags used with avgpool2d_global_adv() for advanced scenarios.
 */
typedef enum {
    /** 
     * Placeholder flag used to indicate no other flags are needed.
     */
    AVGPOOL2D_GLOBAL_FLAG_NONE = 0,
} nn_avgpool2d_global_flags_e;

void avgpool2d_gen(
    nn_image_t* Y,
    const nn_image_t* X, 
    const nn_avgpool2d_plan_t* plan,
    const nn_pool2d_job_t* job);
    
void avgpool2d_2x2(
    nn_image_t* Y,
    const nn_image_t* X, 
    const nn_avgpool2d_plan_t* plan,
    const nn_pool2d_job_t* job);




// /**
//  * @brief Initialize an instance of the @oper{maxpool2d} operator.
//  * 
//  * See @oper_ref{maxpool2d} for more details about the @oper{maxpool2d} operator. To invoke a @oper{maxpool2d} job, call 
//  * maxpool2d().
//  * 
//  * When maxpool2d() is called, a plan (`nn_maxpool2d_plan_t`) and a job (`nn_pool2d_job_t`) must be supplied to tell it 
//  * how to do its work. This function initializes that plan and one or more jobs to be supplied in subsequent calls to 
//  * maxpool2d().
//  * 
//  * A plan contains information shared by all jobs of an instance of @oper{maxpool2d}. Each job computes a rectangular 
//  * sub-tensor of the output image (possibly the entire image).
//  * 
//  * `plan` points to the plan to be initialized. It need only be initialized once for many calls to maxpool2d().
//  * 
//  * `jobs` points to an array of `nn_pool2d_job_t` to be initialized. Each element represents one job. There should be 
//  * `job_count` elements in the array.
//  * 
//  * `x_params` points to the image parameters for the instance's input image @tensor{X}.
//  * 
//  * `y_params` points to the image parameters for the instance's output image @tensor{Y}.
//  * 
//  * `window_config` points to a `nn_window_params_t` struct containing the instance's @math{W_h}, @math{W_w}, 
//  * @math{W_{vert}}, @math{W_{hori}}, @math{W_{r0}} and @math{W_{c0}} hyperparameters (see @ref 
//  * maxpool2d_hyperparameters) which describe the relationship between the input image, the pooling window and the 
//  * output image.
//  * 
//  * `window_config->shape` specified @math{W_w} and @math{W_h}, the height and width of the pooling window. 
//  * 
//  * `window_config->start` specifies @math{W_{r0}} and @math{W_{c0}}, the starting row and column of the pooling window 
//  * in @tensor{X}'s coordinate space. For example, a `start` value of `(0,0)` indicates that the top-left pixel of the 
//  * output image has the pooling window aligned with the top-left corner of the input image, with no implied padding at 
//  * the top or left sides of the input image.
//  * 
//  * `window_config->stride.horizontal` specifies @math{W_{vert}} and @math{W_{hori}}, the vertical and horizontal strides 
//  * of the pooling window. The strides describe the number of pixels the pooling window moves (across the input image) 
//  * with each pixel in the output image.
//  * 
//  * `job_params` points to either an array of `nn_window_op_job_params_t` structs or else is `NULL`. A `job_params` value 
//  * of `NULL` indicates that there will only be a single job which computes the entire output image. If `job_params` is 
//  * `NULL`, then `job_count` must be `1`. If `job_params` is not `NULL`, it must point to an array of `job_count` 
//  * `nn_window_op_job_params_t` elements.
//  * 
//  * In particular, job `k` will compute the output elements @math{Y[r,c,p]} for which:
//  * @inlinecode
//  *     job_params[k].start.rows <= r < job_params[k].start.rows + job_params[k].size.rows
//  *     job_params[k].start.cols <= c < job_params[k].start.cols + job_params[k].size.cols
//  *     job_params[k].start.channels <= p < job_params[k].start.channels + job_params[k].size.channels
//  * @endinlinecode
//  * 
//  * `job_count` indicates the number of jobs to be initialized (and thus the number of elements in the `jobs` array), as 
//  * well the number of elements in the `job_params` array if it is not `NULL`.
//  * 
//  * 
//  * @param plan          [out]   The plan to be initialized.
//  * @param jobs          [out]   Array of jobs to be initialized.
//  * @param x_params      [in]    Parameters describing the shape of each input image tensor @tensor{X}.
//  * @param y_params      [in]    Parameters describing the shape of each output image tensor @tensor{Y}
//  * @param window_config [in]    Pooling window configuration.
//  * @param job_params    [in]    An array of `nn_window_op_job_params_t` structs, or NULL
//  * @param job_count     [in]    The number of jobs to be initialized.
//  */
// void maxpool2d_init(
//     nn_maxpool2d_plan_t* plan,
//     nn_pool2d_job_t* jobs,
//     const nn_image_params_t* x_params,
//     const nn_image_params_t* y_params,
//     const nn_window_params_t* window_config,
//     const nn_window_op_job_params_t* job_params,
//     const unsigned job_count);


/**
 * @brief Initialize an instance of the @oper{avgpool2d} operator.
 * 
 * See @oper_ref{avgpool2d} for more details about the @oper{avgpool2d} operator. To invoke a @oper{avgpool2d} job, call 
 * avgpool2d().
 * 
 * When avgpool2d() is called, a plan (`nn_avgpool2d_plan_t`) and a job (`nn_pool2d_job_t`) must be supplied to tell it 
 * how to do its work. This function initializes that plan and one or more jobs to be supplied in subsequent calls to 
 * avgpool2d().
 * 
 * A plan contains information shared by all jobs of an instance of @oper{avgpool2d}. Each job computes a rectangular 
 * sub-tensor of the output image (possibly the entire image).
 * 
 * `plan` points to the plan to be initialized. It need only be initialized once for many calls to avgpool2d().
 * 
 * `jobs` points to an array of `nn_pool2d_job_t` to be initialized. Each element represents one job. There should be 
 * `job_count` elements in the array.
 * 
 * `x_params` points to the image parameters for the instance's input image @tensor{X}.
 * 
 * `y_params` points to the image parameters for the instance's output image @tensor{Y}.
 * 
 * `window_config` points to a `nn_window_params_t` struct containing the instance's @math{W_h}, @math{W_w}, 
 * @math{W_{vert}), @math{W_{hori}), @math{W_{r0}) and @math{W_{c0}} hyperparameters (see @ref 
 * avgpool2d_hyperparameters) which describe the relationship between the input image, the pooling window and the 
 * output image.
 * 
 * `window_config->shape` specified @math{W_w} and @math{W_h}, the height and width of the pooling window. 
 * 
 * `window_config->start` specifies @math{W_{r0}} and @math{W_{c0}}, the starting row and column of the pooling window 
 * in @tensor{X}'s coordinate space. For example, a `start` value of `(0,0)` indicates that the top-left pixel of the 
 * output image has the pooling window aligned with the top-left corner of the input image, with no implied padding at 
 * the top or left sides of the input image.
 * 
 * `window_config->stride.horizontal` specifies @math{W_{vert}} and @math{W_{hori}}, the vertical and horizontal strides 
 * of the pooling window. The strides describe the number of pixels the pooling window moves (across the input image) 
 * with each pixel in the output image.
 * 
 * `job_params` points to either an array of `nn_window_op_job_params_t` structs or else is `NULL`. A `job_params` value 
 * of `NULL` indicates that there will only be a single job which computes the entire output image. If `job_params` is 
 * `NULL`, then `job_count` must be `1`. If `job_params` is not `NULL`, it must point to an array of `job_count` 
 * `nn_window_op_job_params_t` elements.
 * 
 * In particular, job `k` will compute the output elements @math{Y[r,c,p]} for which:
 * @inlinecode
 *     job_params[k].start.rows <= r < job_params[k].start.rows + job_params[k].size.rows
 *     job_params[k].start.cols <= c < job_params[k].start.cols + job_params[k].size.cols
 *     job_params[k].start.channels <= p < job_params[k].start.channels + job_params[k].size.channels
 * @endinlinecode
 * 
 * `job_count` indicates the number of jobs to be initialized (and thus the number of elements in the `jobs` array), as 
 * well the number of elements in the `job_params` array if it is not `NULL`.
 * 
 * @param plan          [out]   The plan to be initialized.
 * @param jobs          [out]   Array of jobs to be initialized.
 * @param x_params      [in]    Parameters describing the shape of each input image tensor @tensor{X}.
 * @param y_params      [in]    Parameters describing the shape of each output image tensor @tensor{Y}
 * @param window_config [in]    Pooling window configuration.
 * @param job_params    [in]    An array of `nn_window_op_job_params_t` structs, or NULL
 * @param job_count     [in]    The number of jobs to be initialized.
 */
void avgpool2d_init(
    nn_avgpool2d_plan_t* plan,
    nn_pool2d_job_t* jobs,
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_window_params_t* window_config,
    const nn_window_op_job_params_t* job_params,
    const unsigned job_count);




// /**
//  * @brief Initialize an instance of the @oper{avgpool2d_global} operator.
//  * 
//  * See @oper_ref{avgpool2d_global} for more details about the @oper{avgpool2d_global} operator. To invoke a 
//  * @oper{avgpool2d_global} job, call avgpool2d_global().
//  * 
//  * When avgpool2d_global() is called, a plan (`nn_avgpool2d_global_plan_t`) and a job (`nn_avgpool2d_global_job_t`) must 
//  * be supplied to tell it how to do its work. This function initializes that plan and one or more jobs to be supplied in 
//  * subsequent calls to avgpool2d_global().
//  * 
//  * A plan contains information shared by all jobs of an instance of @oper{avgpool2d_global}. Each job computes a range
//  * of channels in the output vector (possibly the entire vector).
//  * 
//  * `plan` points to the plan to be initialized. It need only be initialized once for many calls to avgpool2d_global().
//  * 
//  * `jobs` points to an array of `nn_avgpool2d_global_job_t` to be initialized. Each element represents one job. There 
//  * should be `job_count` elements in the array.
//  * 
//  * `x_params` points to the image parameters for the instance's input (and output) image @tensor{X}.
//  * 
//  * `job_params` points to either an array of `nn_avgpool2d_global_job_params_t` structs or else is `NULL`. A 
//  * `job_params` value of `NULL` indicates that there will only be a single job which computes the entire output image. 
//  * If `job_params` is `NULL`, then `job_count` must be `1`. If `job_params` is not `NULL`, it must point to an array 
//  * of `job_count` `nn_avgpool2d_global_job_params_t` elements.
//  * 
//  * In particular, job `k` will compute the output elements @math{y[p]} for which:
//  * @inlinecode
//  *     job_params[k].start_channel <= p < job_params[k].start_channel + job_params[k].out.channels
//  * @endinlinecode
//  * 
//  * `job_count` indicates the number of jobs to be initialized (and thus the number of elements in the `jobs` array), as 
//  * well the number of elements in the `job_params` array if it is not `NULL`.
//  * 
//  * @param plan          [out]   The plan to be initialized.
//  * @param jobs          [out]   Array of jobs to be initialized.
//  * @param x_params      [in]    Parameters describing the shape of each input (and output) image tensor @tensor{X}.
//  * @param job_params    [in]    An array of `nn_avgpool2d_global_job_params_t` structs, or NULL
//  * @param job_count     [in]    The number of jobs to be initialized.
//  */
// void avgpool2d_global_init(
//     nn_avgpool2d_global_plan_t* plan,
//     nn_avgpool2d_global_job_t* jobs,
//     const nn_image_params_t* x_params,
//     const nn_avgpool2d_global_job_params_t* job_params,
//     const unsigned job_count);


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
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_window_params_t* window_config);
    
void maxpool2d_adv(
    nn_image_t* Y,
    const nn_image_t* X, 
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_window_params_t* window_config,
    const nn_window_op_job_params_t* job_params,
    const nn_maxpool2d_flags_e flags);


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
    nn_image_t* Y,
    const nn_image_t* X, 
    const int32_t bias,
    const int8_t scale,
    const uint16_t shift,
    const nn_image_params_t* x_params);

void avgpool2d_global_adv(
    nn_image_t* Y,
    const nn_image_t* X, 
    const int32_t bias,
    const int8_t scale,
    const uint16_t shift,
    const nn_image_params_t* x_params,
    const unsigned chan_start,
    const unsigned chan_count,
    const nn_avgpool2d_global_flags_e flags);

#endif //POOLING_H_