#ifndef POOLING_H_
#define POOLING_H_

#include "nn_window_params.h"
#include "nn_image.h"




/**
 * Flags used with maxpool2d_ext() for advanced scenarios.
 */
typedef enum {
    /** 
     * Placeholder flag used to indicate no other flags are needed.
     */
    MAXPOOL2D_FLAG_NONE = 0,
} nn_maxpool2d_flags_e;


/**
 * Flags used with avgpool2d_ext() for advanced scenarios.
 */
typedef enum {
    /** 
     * Placeholder flag used to indicate no other flags are needed.
     */
    AVGPOOL2D_FLAG_NONE = 0,
} nn_avgpool2d_flags_e;


/**
 * Flags used with avgpool2d_global_ext() for advanced scenarios.
 */
typedef enum {
    /** 
     * Placeholder flag used to indicate no other flags are needed.
     */
    AVGPOOL2D_GLOBAL_FLAG_NONE = 0,
} nn_avgpool2d_global_flags_e;





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
 * @brief Invoke @oper{maxpool2d} job.
 * 
 * See @oper_ref{maxpool2d} for more details about the @oper{maxpool2d} operator.
 * 
 * @par Parameter Details
 * 
 * `Y` points to the output image @tensor{Y} with shape @tensor_shape{Y_h, Y_w, X_c}.
 * 
 * `X` points to the input image @tensor{X} with shape @tensor_shape{X_h, X_w, X_c}.
 * 
 * The memory layout of @tensor{Y} and @tensor{X} are the standard memory layout for image tensors (see @ref 
 * standard_layout).
 * 
 * `x_params` points to the image parameters describing the shape of the input image @tensor{X}. The size of each of
 * @tensor{X}'s dimensions, @math{X_h}, @math{X_w}, and @math{X_c} correspond to `x_params->height`, `x_params->width`,
 * and `x_params->channels` respectively.
 * 
 * `y_params` points to the image parameters describing the shape of the output image @tensor{Y}. The size of each of
 * @tensor{Y}'s dimensions, @math{Y_h}, @math{Y_w}, and @math{X_c} correspond to `y_params->height`, `y_params->width`,
 * and `x_params->channels` respectively.
 * 
 * `pooling_window` points to a `nn_window_params_t` struct containing the instance's @math{W_h}, @math{W_w}, 
 * @math{W_{vert}}, @math{W_{hori}}, @math{W_{r0}} and @math{W_{c0}} hyperparameters (see @ref 
 * maxpool2d_hyperparameters) which describe the spacial relationship between the input image, the pooling window 
 * window and the output image. `pooling_window->dilation` is ignored.
 * 
 * `pooling_window->shape` specifies @math{W_h} and @math{W_w}, the height and width of the pooling window. 
 * 
 * `pooling_window->start` specifies @math{W_{r0}} and @math{W_{c0}}, the starting row and column of the pooling 
 * window in @tensor{X}'s coordinate space. For example, a `start` value of `(0,0)` indicates that the top-left pixel of 
 * the output image has the pooling window aligned with the top-left corner of the input image, with no implied padding 
 * at the top or left sides of the input image. A `start` value of `(1,1)`, on the other hand, indicates that the 
 * top-left pixel of the output image has the pooling window shifted one pixel right and one pixel down relative to the
 * top-left corner of the input image.
 * 
 * `pooling_window->stride.horizontal` specifies @math{W_{vert}} and @math{W_{hori}}, the vertical and horizontal 
 * strides of the pooling window. The strides describe the number of pixels the pooling window moves (across the input 
 * image) with each pixel in the output image.
 * 
 * @par Parameter Constraints
 * 
 * The arguments `Y` and `X` must each point to a word-aligned address.
 * 
 * The input and output images must have the same number of channels. As such, it is required that 
 * `y_params->channels == x_params->channels`.
 * 
 * Due to memory alignment requirements, @math{X_c} must be a multiple of @math{4}, which forces all 
 * pixels to begin at word-aligned addresses.
 * 
 * Padding is not supported by this operator.
 * 
 * @par Splitting the Workload
 * 
 * See maxpool2d_ext() for more advanced scenarios which allow the the work to be split across multiple invocations 
 * (which can be parallelized across cores).
 * 
 * @par Additional Remarks
 * 
 * Internally, maxpool2d() calls maxpool2d_ext() with a `job_params` argument that computes the entire output image, and 
 * with no flags set. For more advanced scenarios, use maxpool2d_ext().
 * 
 * By default this operator uses the standard 8-bit limits @math([-128, 127]) when applying saturation logic. Instead,
 * it can be configured to use symmetric saturation bounds @math([-127, 127]) by defining 
 * `CONFIG_SYMMETRIC_SATURATION_maxpool2d` appropriately. See @ref nn_config.h for more details. Note that this
 * configures _all_ instances of the @oper{maxpool2d} operator.
 * 
 * If @math{X_c} is not a multiple of @math{32}, this operator may read up to 28 bytes following the end of @tensor{X}. 
 * This is not ordinarily a problem. However, if the object to which `X` points is located very near the end of a valid 
 * memory address range, it is possible memory access exceptions may occur when this operator is invoked.
 * 
 * If necessary, this can be avoided by manually forcing a buffer region (no more than @math{28} bytes are necessary) 
 * following @tensor{X}. There are various ways this can be accomplished, including embedding these objects in larger 
 * structures.
 * 
 * @param[out]  Y               The output image @tensor{Y}
 * @param[in]   X               The input image @tensor{X}
 * @param[in]   x_params        Parameters describing the shape of input image tensor @tensor{X}
 * @param[in]   y_params        Parameters describing the shape of output image tensor @tensor{Y}
 * @param[in]   pooling_window  Parameters describing the relationship between the pooling window, the input image,
 *                              and the output image
 */
void maxpool2d(
    nn_image_t* Y,
    const nn_image_t* X, 
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_window_params_t* pooling_window);

/**  
 * @brief Invoke @oper{maxpool2d} job.
 * 
 * See @oper_ref{maxpool2d} for more details about the @oper{maxpool2d} operator.
 * 
 * @par Parameter Details
 * 
 * `Y` points to the output image @tensor{Y} with shape @tensor_shape{Y_h, Y_w, X_c}.
 * 
 * `X` points to the input image @tensor{X} with shape @tensor_shape{X_h, X_w, X_c}.
 * 
 * The memory layout of @tensor{Y} and @tensor{X} are the standard memory layout for image tensors (see @ref 
 * standard_layout).
 * 
 * `x_params` points to the image parameters describing the shape of the input image @tensor{X}. The size of each of
 * @tensor{X}'s dimensions, @math{X_h}, @math{X_w}, and @math{X_c} correspond to `x_params->height`, `x_params->width`,
 * and `x_params->channels` respectively.
 * 
 * `y_params` points to the image parameters describing the shape of the output image @tensor{Y}. The size of each of
 * @tensor{Y}'s dimensions, @math{Y_h}, @math{Y_w}, and @math{X_c} correspond to `y_params->height`, `y_params->width`,
 * and `x_params->channels` respectively.
 * 
 * `pooling_window` points to a `nn_window_params_t` struct containing the instance's @math{W_h}, @math{W_w}, 
 * @math{W_{vert}}, @math{W_{hori}}, @math{W_{r0}} and @math{W_{c0}} hyperparameters (see @ref 
 * maxpool2d_hyperparameters) which describe the spacial relationship between the input image, the pooling window 
 * window and the output image. `pooling_window->dilation` is ignored.
 * 
 * `pooling_window->shape` specifies @math{W_h} and @math{W_w}, the height and width of the pooling window. 
 * 
 * `pooling_window->start` specifies @math{W_{r0}} and @math{W_{c0}}, the starting row and column of the pooling 
 * window in @tensor{X}'s coordinate space. For example, a `start` value of `(0,0)` indicates that the top-left pixel of 
 * the output image has the pooling window aligned with the top-left corner of the input image, with no implied padding 
 * at the top or left sides of the input image. A `start` value of `(1,1)`, on the other hand, indicates that the 
 * top-left pixel of the output image has the pooling window shifted one pixel right and one pixel down relative to the
 * top-left corner of the input image.
 * 
 * `pooling_window->stride.horizontal` specifies @math{W_{vert}} and @math{W_{hori}}, the vertical and horizontal 
 * strides of the pooling window. The strides describe the number of pixels the pooling window moves (across the input 
 * image) with each pixel in the output image.
 * 
 * `job_params` describes which elements of the output image will be computed by this invocation. This invocation 
 * computes the output elements @math{Y[r,c,p]} for which:
 * @inlinecode
 *     job_params->start.rows <= r < job_params->start.rows + job_params->size.rows
 *     job_params->start.cols <= c < job_params->start.cols + job_params->size.cols
 *     job_params->start.channels <= p < job_params->start.channels + job_params->size.channels
 * @endinlinecode
 * 
 * `flags` is a collection of flags which modify the behavior of @oper{maxpool2d}. See `nn_maxpool2d_flags_e` for a 
 * description of each flag. `MAXPOOL2D_FLAG_NONE`(0) can be used for default behavior.
 * 
 * @par Parameter Constraints
 * 
 * The arguments `Y` and `X` must each point to a word-aligned address.
 * 
 * The input and output images must have the same number of channels. As such, it is required that 
 * `y_params->channels == x_params->channels`.
 * 
 * Due to memory alignment requirements, @math{X_c} must be a multiple of @math{4}, which forces all 
 * pixels to begin at word-aligned addresses.
 * 
 * Padding is not supported by this operator.
 * 
 * @par Splitting the Workload
 * 
 * @todo Include information about how to split the work into multiple invocations (e.g. for parallelization), 
 *       particularly any counter-intuitive aspects.
 * 
 * @par Additional Remarks
 * 
 * Internally, maxpool2d() calls maxpool2d_ext() with a `job_params` argument that computes the entire output image, and 
 * with no flags set. For more advanced scenarios, use maxpool2d_ext().
 * 
 * By default this operator uses the standard 8-bit limits @math([-128, 127]) when applying saturation logic. Instead,
 * it can be configured to use symmetric saturation bounds @math([-127, 127]) by defining 
 * `CONFIG_SYMMETRIC_SATURATION_maxpool2d` appropriately. See @ref nn_config.h for more details. Note that this
 * configures _all_ instances of the @oper{maxpool2d} operator.
 * 
 * If @math{X_c} is not a multiple of @math{32}, this operator may read up to 28 bytes following the end of @tensor{X}. 
 * This is not ordinarily a problem. However, if the object to which `X` points is located very near the end of a valid 
 * memory address range, it is possible memory access exceptions may occur when this operator is invoked.
 * 
 * If necessary, this can be avoided by manually forcing a buffer region (no more than @math{28} bytes are necessary) 
 * following @tensor{X}. There are various ways this can be accomplished, including embedding these objects in larger 
 * structures.
 * 
 * @param[out]  Y               The output image @tensor{Y}
 * @param[in]   X               The input image @tensor{X}
 * @param[in]   x_params        Parameters describing the shape of input image tensor @tensor{X}
 * @param[in]   y_params        Parameters describing the shape of output image tensor @tensor{Y}
 * @param[in]   pooling_window  Parameters describing the relationship between the pooling window, the input image,
 *                              and the output image
 * @param[in]   job_params      Indicates which output elements will be computed by this invocation
 * @param[in]   flags           Flags which modify the behavior of maxpool2d_ext()
 */ 
void maxpool2d_ext(
    nn_image_t* Y,
    const nn_image_t* X, 
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_window_params_t* window_config,
    const nn_window_op_job_params_t* job_params,
    const nn_maxpool2d_flags_e flags);


/** 
 * @brief Invoke @oper{avgpool2d} job.
 * 
 * See @oper_ref{avgpool2d} for more details about the @oper{avgpool2d} operator.
 * 
 * @par Parameter Details
 * 
 * `Y` points to the output image @tensor{Y} with shape @tensor_shape{Y_h, Y_w, X_c}.
 * 
 * `X` points to the input image @tensor{X} with shape @tensor_shape{X_h, X_w, X_c}.
 * 
 * The memory layout of @tensor{Y} and @tensor{X} are the standard memory layout for image tensors (see @ref 
 * standard_layout).
 * 
 * `x_params` points to the image parameters describing the shape of the input image @tensor{X}. The size of each of
 * @tensor{X}'s dimensions, @math{X_h}, @math{X_w}, and @math{X_c} correspond to `x_params->height`, `x_params->width`,
 * and `x_params->channels` respectively.
 * 
 * `y_params` points to the image parameters describing the shape of the output image @tensor{Y}. The size of each of
 * @tensor{Y}'s dimensions, @math{Y_h}, @math{Y_w}, and @math{X_c} correspond to `y_params->height`, `y_params->width`,
 * and `x_params->channels` respectively.
 * 
 * `pooling_window` points to a `nn_window_params_t` struct containing the instance's @math{W_h}, @math{W_w}, 
 * @math{W_{vert}}, @math{W_{hori}}, @math{W_{r0}} and @math{W_{c0}} hyperparameters (see @ref 
 * avgpool2d_hyperparameters) which describe the spacial relationship between the input image, the pooling window 
 * window and the output image. `pooling_window->dilation` is ignored.
 * 
 * `pooling_window->shape` specifies @math{W_h} and @math{W_w}, the height and width of the pooling window. 
 * 
 * `pooling_window->start` specifies @math{W_{r0}} and @math{W_{c0}}, the starting row and column of the pooling 
 * window in @tensor{X}'s coordinate space. For example, a `start` value of `(0,0)` indicates that the top-left pixel of 
 * the output image has the pooling window aligned with the top-left corner of the input image, with no implied padding 
 * at the top or left sides of the input image. A `start` value of `(1,1)`, on the other hand, indicates that the 
 * top-left pixel of the output image has the pooling window shifted one pixel right and one pixel down relative to the
 * top-left corner of the input image.
 * 
 * `pooling_window->stride.horizontal` specifies @math{W_{vert}} and @math{W_{hori}}, the vertical and horizontal 
 * strides of the pooling window. The strides describe the number of pixels the pooling window moves (across the input 
 * image) with each pixel in the output image.
 * 
 * @par Parameter Constraints
 * 
 * The arguments `Y` and `X` must each point to a word-aligned address.
 * 
 * The input and output images must have the same number of channels. As such, it is required that 
 * `y_params->channels == x_params->channels`.
 * 
 * Due to memory alignment requirements, @math{X_c} must be a multiple of @math{4}, which forces all 
 * pixels to begin at word-aligned addresses.
 * 
 * Padding is not supported by this operator.
 * 
 * @par Splitting the Workload
 * 
 * See avgpool2d_ext() for more advanced scenarios which allow the the work to be split across multiple invocations 
 * (which can be parallelized across cores).
 * 
 * @par Additional Remarks
 * 
 * Internally, avgpool2d() calls avgpool2d_ext() with a `job_params` argument that computes the entire output image, and 
 * with no flags set. For more advanced scenarios, use avgpool2d_ext().
 * 
 * By default this operator uses the standard 8-bit limits @math([-128, 127]) when applying saturation logic. Instead,
 * it can be configured to use symmetric saturation bounds @math([-127, 127]) by defining 
 * `CONFIG_SYMMETRIC_SATURATION_avgpool2d` appropriately. See @ref nn_config.h for more details. Note that this
 * configures _all_ instances of the @oper{avgpool2d} operator.
 * 
 * If @math{X_c} is not a multiple of @math{16}, this operator may read up to 12 bytes following the end of @tensor{X}. 
 * This is not ordinarily a problem. However, if the object to which `X` points is located very near the end of a valid 
 * memory address range, it is possible memory access exceptions may occur when this operator is invoked.
 * 
 * If necessary, this can be avoided by manually forcing a buffer region (no more than @math{12} bytes are necessary) 
 * following @tensor{X}. There are various ways this can be accomplished, including embedding these objects in larger 
 * structures.
 * 
 * @param[out]  Y               The output image @tensor{Y}
 * @param[in]   X               The input image @tensor{X}
 * @param[in]   x_params        Parameters describing the shape of input image tensor @tensor{X}
 * @param[in]   y_params        Parameters describing the shape of output image tensor @tensor{Y}
 * @param[in]   pooling_window  Parameters describing the relationship between the pooling window, the input image,
 *                              and the output image
 */
void avgpool2d(
    int8_t* Y,
    const int8_t* X, 
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_window_params_t* pooling_window);

/** 
 * @brief Invoke @oper{avgpool2d} job.
 * 
 * See @oper_ref{avgpool2d} for more details about the @oper{avgpool2d} operator.
 * 
 * @par Parameter Details
 * 
 * `Y` points to the output image @tensor{Y} with shape @tensor_shape{Y_h, Y_w, X_c}.
 * 
 * `X` points to the input image @tensor{X} with shape @tensor_shape{X_h, X_w, X_c}.
 * 
 * The memory layout of @tensor{Y} and @tensor{X} are the standard memory layout for image tensors (see @ref 
 * standard_layout).
 * 
 * `x_params` points to the image parameters describing the shape of the input image @tensor{X}. The size of each of
 * @tensor{X}'s dimensions, @math{X_h}, @math{X_w}, and @math{X_c} correspond to `x_params->height`, `x_params->width`,
 * and `x_params->channels` respectively.
 * 
 * `y_params` points to the image parameters describing the shape of the output image @tensor{Y}. The size of each of
 * @tensor{Y}'s dimensions, @math{Y_h}, @math{Y_w}, and @math{X_c} correspond to `y_params->height`, `y_params->width`,
 * and `x_params->channels` respectively.
 * 
 * `pooling_window` points to a `nn_window_params_t` struct containing the instance's @math{W_h}, @math{W_w}, 
 * @math{W_{vert}}, @math{W_{hori}}, @math{W_{r0}} and @math{W_{c0}} hyperparameters (see @ref 
 * avgpool2d_hyperparameters) which describe the spacial relationship between the input image, the pooling window 
 * window and the output image. `pooling_window->dilation` is ignored.
 * 
 * `pooling_window->shape` specifies @math{W_h} and @math{W_w}, the height and width of the pooling window. 
 * 
 * `pooling_window->start` specifies @math{W_{r0}} and @math{W_{c0}}, the starting row and column of the pooling 
 * window in @tensor{X}'s coordinate space. For example, a `start` value of `(0,0)` indicates that the top-left pixel of 
 * the output image has the pooling window aligned with the top-left corner of the input image, with no implied padding 
 * at the top or left sides of the input image. A `start` value of `(1,1)`, on the other hand, indicates that the 
 * top-left pixel of the output image has the pooling window shifted one pixel right and one pixel down relative to the
 * top-left corner of the input image.
 * 
 * `pooling_window->stride.horizontal` specifies @math{W_{vert}} and @math{W_{hori}}, the vertical and horizontal 
 * strides of the pooling window. The strides describe the number of pixels the pooling window moves (across the input 
 * image) with each pixel in the output image.
 * 
 * `job_params` describes which elements of the output image will be computed by this invocation. This invocation 
 * computes the output elements @math{Y[r,c,p]} for which:
 * @inlinecode
 *     job_params->start.rows <= r < job_params->start.rows + job_params->size.rows
 *     job_params->start.cols <= c < job_params->start.cols + job_params->size.cols
 *     job_params->start.channels <= p < job_params->start.channels + job_params->size.channels
 * @endinlinecode
 * 
 * `flags` is a collection of flags which modify the behavior of @oper{avgpool2d}. See `nn_avgpool2d_flags_e` for a 
 * description of each flag. `AVGPOOL2D_FLAG_NONE`(0) can be used for default behavior.
 * 
 * @par Parameter Constraints
 * 
 * The arguments `Y` and `X` must each point to a word-aligned address.
 * 
 * The input and output images must have the same number of channels. As such, it is required that 
 * `y_params->channels == x_params->channels`.
 * 
 * Due to memory alignment requirements, @math{X_c} must be a multiple of @math{4}, which forces all 
 * pixels to begin at word-aligned addresses.
 * 
 * Padding is not supported by this operator.
 * 
 * @par Splitting the Workload
 * 
 * @todo Include information about how to split the work into multiple invocations (e.g. for parallelization), 
 *       particularly any counter-intuitive aspects.
 * 
 * @par Additional Remarks
 * 
 * Internally, avgpool2d() calls avgpool2d_ext() with a `job_params` argument that computes the entire output image, and 
 * with no flags set. For more advanced scenarios, use avgpool2d_ext().
 * 
 * By default this operator uses the standard 8-bit limits @math([-128, 127]) when applying saturation logic. Instead,
 * it can be configured to use symmetric saturation bounds @math([-127, 127]) by defining 
 * `CONFIG_SYMMETRIC_SATURATION_avgpool2d` appropriately. See @ref nn_config.h for more details. Note that this
 * configures _all_ instances of the @oper{avgpool2d} operator.
 * 
 * If @math{X_c} is not a multiple of @math{16}, this operator may read up to 12 bytes following the end of @tensor{X}. 
 * This is not ordinarily a problem. However, if the object to which `X` points is located very near the end of a valid 
 * memory address range, it is possible memory access exceptions may occur when this operator is invoked.
 * 
 * If necessary, this can be avoided by manually forcing a buffer region (no more than @math{12} bytes are necessary) 
 * following @tensor{X}. There are various ways this can be accomplished, including embedding these objects in larger 
 * structures.
 * 
 * @param[out]  Y               The output image @tensor{Y}
 * @param[in]   X               The input image @tensor{X}
 * @param[in]   x_params        Parameters describing the shape of input image tensor @tensor{X}
 * @param[in]   y_params        Parameters describing the shape of output image tensor @tensor{Y}
 * @param[in]   pooling_window  Parameters describing the relationship between the pooling window, the input image,
 *                              and the output image
 * @param[in]   job_params      Indicates which output elements will be computed by this invocation
 * @param[in]   flags           Flags which modify the behavior of avgpool2d_ext()
 */
void avgpool2d_ext(
    int8_t* Y,
    const int8_t* X, 
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_window_params_t* pooling_window,
    const nn_window_op_job_params_t* job_params,
    const nn_avgpool2d_flags_e flags);

/** 
 * @brief Invoke a @oper{avgpool2d_global} job.
 * 
 * The @oper{avgpool2d_global} operator computes a scaled and biased sum of pixel values for each channel of an input
 * image, producing an 8-bit vector of outputs.
 * 
 * See @oper_ref{avgpool2d_global} for more details about the @oper{avgpool2d_global} operator, including the 
 * mathematical details of the operation performed.
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
 * `x_params` points to the image parameters describing the shape of the input image @tensor{X}. The size of each of
 * @tensor{X}'s dimensions, @math{X_h}, @math{X_w}, and @math{X_c} correspond to `x_params->height`, `x_params->width`,
 * and `x_params->channels` respectively.
 * 
 * @par Parameter Constraints
 * 
 * The arguments `Y` and `X` must each point to a word-aligned address.
 * 
 * The input and output images must have the same number of channels. As such, it is required that 
 * `y_params->channels == x_params->channels`.
 * 
 * Due to memory alignment requirements, @math{X_c} must be a multiple of @math{4}, which forces all pixels to begin at
 * a word-aligned address.
 * 
 * Padding is not supported by this operator.
 * 
 * @par Splitting the Workload
 * 
 * See avgpool2d_ext() for more advanced scenarios which allow the the work to be split across multiple invocations 
 * (which can be parallelized across cores).
 * 
 * @par Additional Remarks
 * 
 * Internally, avgpool2d() calls avgpool2d_ext() with a `job_params` argument that computes the entire output image, and 
 * with no flags set. For more advanced scenarios, use avgpool2d_ext().
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
 * @param[out]  Y           The output vector @tensor{y}
 * @param[in]   X           The input image @tensor{X}
 * @param[in]   bias        Initial 32-bit accumulator value @math{b}. Shared by all channels.
 * @param[in]   scale       The factor @math{s} by which input pixel values are scaled.
 * @param[in]   shift       The right-shift @math{r} applied to the 32-bit accumulators to yield an 8-bit result.
 * @param[in]   x_params    Parameters describing the shape of input image tensor @tensor{X}
 */
void avgpool2d_global(
    nn_image_t* Y,
    const nn_image_t* X, 
    const int32_t bias,
    const int8_t scale,
    const uint16_t shift,
    const nn_image_params_t* x_params);

/** 
 * @brief Invoke a @oper{avgpool2d_global} job.
 * 
 * The @oper{avgpool2d_global} operator computes a scaled and biased sum of pixel values for each channel of an input
 * image, producing an 8-bit vector of outputs.
 * 
 * See @oper_ref{avgpool2d_global} for more details about the @oper{avgpool2d_global} operator, including the 
 * mathematical details of the operation performed.
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
 * `x_params` points to the image parameters describing the shape of the input image @tensor{X}. The size of each of
 * @tensor{X}'s dimensions, @math{X_h}, @math{X_w}, and @math{X_c} correspond to `x_params->height`, `x_params->width`,
 * and `x_params->channels` respectively.
 * 
 * `chan_start` is the first output channel to be computed by this invocation.
 * 
 * `chan_count` is the number of channels to be computed by this invocation.
 * 
 * `flags` is a collection of flags which modify the behavior of @oper{avgpool2d_global}. See `nn_avgpool2d_flags_e` for 
 * a description of each flag. `AVGPOOL2D_GLOBAL_FLAG_NONE`(0) can be used for default behavior.
 * 
 * @par Parameter Constraints
 * 
 * The arguments `Y` and `X` must each point to a word-aligned address.
 * 
 * The input and output images must have the same number of channels. As such, it is required that 
 * `y_params->channels == x_params->channels`.
 * 
 * Due to memory alignment requirements, @math{X_c} must be a multiple of @math{4}, which forces all pixels to begin at
 * a word-aligned address.
 * 
 * Padding is not supported by this operator.
 * 
 * @par Splitting the Workload
 * 
 * @todo Include information about how to split the work into multiple invocations (e.g. for parallelization), 
 *       particularly any counter-intuitive aspects.
 * 
 * @par Additional Remarks
 * 
 * Internally, avgpool2d() calls avgpool2d_ext() with a `job_params` argument that computes the entire output image, and 
 * with no flags set. For more advanced scenarios, use avgpool2d_ext().
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
 * @param[out]  Y           The output vector @tensor{y}
 * @param[in]   X           The input image @tensor{X}
 * @param[in]   bias        Initial 32-bit accumulator value @math{b}. Shared by all channels.
 * @param[in]   scale       The factor @math{s} by which input pixel values are scaled.
 * @param[in]   shift       The right-shift @math{r} applied to the 32-bit accumulators to yield an 8-bit result.
 * @param[in]   x_params    Parameters describing the shape of input image tensor @tensor{X}
 */
void avgpool2d_global_ext(
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
