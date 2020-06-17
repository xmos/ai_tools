

#ifndef NN_OP_INIT_H_
#define NN_OP_INIT_H_

#include "nn_types.h"
#include "nn_op_structs.h"

#include <stdint.h>

#include "xs3_vpu.h"

#ifdef __XC__
extern "C" {
#endif




/** 
 * @brief Prepare an execution plan for a 2D deep convolution.
 * 
 * When `conv2d_deep()` is called, a plan (`nn_conv2d_deep_plan_t`) and a
 * job (`nn_conv2d_deep_job_t`) must be supplied to tell it how to do its work. This 
 * function initializes that plan and one or more jobs to be supplied in subsequent calls
 * to `conv2d_deep()`.
 * 
 * A plan contains information shared by all jobs. A job, when provided to `conv2d_deep()`,
 * computes a rectangular sub-tensor of the output image (possibly the entire image).
 * 
 * `plan` is a pointer to the execution plan to be initialized. It need only be 
 * initialized once for many calls to `conv2d_deep()`.
 * 
 * `jobs` is a pointer, supplied by the caller to an array of `nn_conv2d_deep_job_t` 
 * structs which will be initialized by this function. `job_count` jobs will be 
 * initialized.
 * 
 * `x_params` is a pointer to image parameters for an input image @tensor{X} that will be
 * passed to `conv2d_deep()` in subsequent calls.
 * 
 * `y_params` is a pointer to image parameters for an output image @tensor{Y} that will be
 * computed by subsequent calls to `conv2d_deep()`.
 * 
 * `job_params` points to either an array of `nn_conv2d_job_params_t` structs or else
 * is `NULL`. A `job_params` value of  `NULL` indicates that there will only be a single
 * job which computes the entire output image. If `job_params` is `NULL`, then `job_count` 
 * must be `1`. If `job_params` is not `NULL`, it must point to an array containing 
 * `job_count` `nn_conv2d_job_params_t` elements.
 * 
 * It is the callers responsibility to ensure that the supplied list of job params
 * collectively computes the entire output image. It is also the caller's responsibility
 * to ensure that the supplied list of jobs does not include duplicate calculation of
 * outputs.
 * 
 * `conv_window` points to a `nn_conv2d_window_params_t` struct which describes the 
 * relationship between the input image, the convolution window and the output image.
 * `conv_window->shape` describes the height and width of the convolution window. 
 * 
 * `conv_window->start` specifies where the top-left cell of the convolution window is
 * placed, relative to the top-left pixel of the input image, for the top-left pixel of
 * the output image. For example, a `start` value of `(0,0)` indicates that the top-left 
 * pixel of the output image has the convolution window aligned with the top-left corner
 * of the input image, with no implied padding a the top or left side of the input image.
 * 
 * `conv_window->stride.horizontal` indicates how many pixels to the right the convolution
 * window moves (across the input image) for each pixel moved to the right in the output image. 
 * `conv_window->stride.vertical` indicates how many pixels downwards the convolution
 * window moves (across the input image) for each pixel moved downwards in the output image.
 * 
 * `zero_point` specifies the value associated with the (implied) padding space around the input
 * image. For any output pixel whereupon the corresponding convolution window location
 * in the input image extends beyond the bounds of the input image, those coefficients
 * in the convolution window which are in the padding are multiplied by `zero_point`
 * rather than by values from the input image. All input channels currently share a
 * common zero-point value.
 * 
 * `job_count` indicates the number of elements in the `jobs` array that is supplied 
 * by the user, as well the number of elements in the `job_params` array if it is not
 * `NULL`.
 * 
 * Constraints:
 *  - @math{X_c} (i.e. `x_params->channels`) and @math{Y_c} (i.e. `y_params->channels`) must 
 *    each be a multiple of `4`.
 * 
 * 
 * @param[out] plan         The plan to be initialized
 * @param[out] jobs         Array of jobs to be initialized (length: `job_count`)
 * @param[in]  x_params     Parameters describing the shape of each input image tensor @tensor{X}
 * @param[in]  y_params     Parameters describing the shape of each output image tensor @tensor{K}
 * @param[in]  job_params   Array with configuration parameters for each job, or `NULL`
 * @param[in]  conv_window  Parameters describing the relationship between the convolution window, the
 *                          input image and hte output image
 * @param[in]  zero_point   The value to be used (for all channels) for padding
 * @param[in]  job_count    The number of jobs to initialize
 */
void conv2d_deep_init(
    nn_conv2d_deep_plan_t* plan,
    nn_conv2d_deep_job_t* jobs,
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_conv2d_job_params_t* job_params,
    const nn_conv2d_window_params_t* conv_window,
    const int8_t zero_point,
    const unsigned job_count);



void conv2d_shallowin_init(
    nn_conv2d_shallowin_plan_t* plan,
    nn_conv2d_shallowin_job_t* jobs,
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_conv2d_job_params_t* job_params,
    const nn_conv2d_window_params_t* conv_window,
    const int8_t zero_point,
    const unsigned job_count);








/** Compute an execution plan for a 2D convolution with a 1x1 kernel.
 * 
 * The output `plan` is the execution plan to be computed. It is passed to the `conv2d_1x1()`
 * function when performing the convolution.
 * 
 * `x` contains the parameters for the input image `X` to `conv2d_1x1()`.
 * 
 * `y` contains the parameters for the output image `Y` to be computed by `conv2d_1x1()`.
 * 
 * The `start_row`, `start_col` and `out_pixels` parameters can be used to limit which
 * output pixels are computed. This can be used to parallelize work across cores. The call
 * to `conv2d_1x1()` using this plan will compute `out_pixels` output pixels, beginning at
 * the specified row and column of the output image. With this scheme, the image is effectively
 * flattened by concatenating rows of the input image, and the pixels computed will be a
 * contiguous subsequence of pixels in the flattened image.
 * 
 * \param plan          Output. Execution plan to be computed.
 * \param x             `conv2d_1x1()` input image parameters.
 * \param y             `conv2d_1x1()` output image parameters.
 * \param start_row     Row of output image `Y` at which to begin computing outputs.
 * \param start_col     Column of output image `Y` at which to begin computing outputs.
 * \param out_pixels    Numbers of output pixels to compute.
 */
void conv2d_1x1_init(
    nn_conv2d_1x1_plan_t* plan,
    nn_conv2d_1x1_job_t* jobs,
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_conv2d_1x1_job_params_t* job_params,
    const unsigned job_count);



/** Compute an execution plan for a 2D depthwise convolution
 * 
 * When `conv2d_depthwise()` is called, a plan (`nn_conv2d_depthwise_plan_t`) and a
 * job (`nn_conv2d_depthwise_job_t`) must be supplied to tell it how to do its work. This 
 * function initializes that plan and one or more jobs to be supplied in subsequent calls
 * to `conv2d_depthwise()`.
 * 
 * A plan contains information shared by all jobs. A job, when provided to `conv2d_depthwise()`,
 * computes a rectangular region of the output image (possibly the entire image).
 * 
 * `plan` is a pointer to the execution plan to be initialized. It need only be 
 * initialized once for many calls to `conv2d_depthwise()`.
 * 
 * `jobs` is a pointer, supplied by the caller to an array of `nn_conv2d_depthwise_job_t` 
 * structs which will be initialized by this function. `job_count` jobs will be 
 * initialized.
 * 
 * `x_params` is a pointer to image parameters for the input image `X` that will be
 * passed to `conv2d_depthwise()` in subsequent calls.
 * 
 * `y_params` is a pointer to image parameters for the output image `Y` that will be
 * computed by subsequent calls to `conv2d_depthwise()`.
 * 
 * `job_params` points to either an array of `nn_conv2d_job_params_t` structs or else
 * is `NULL`. A `job_params` value of  `NULL` indicates that there will only be a single
 * job which computes the entire output image. If `job_params` is `NULL`, then `job_count` 
 * must be 1. If `job_params` is not `NULL`, it must point to an array containing 
 * `job_count` `nn_conv2d_job_params_t` elements.
 * 
 * It is the callers responsibility to ensure that the supplied list of job params
 * collectively computes the entire output image. It is also the caller's responsibility
 * to ensure that the supplied list of jobs does not include duplicate calculation of
 * outputs.
 * 
 * The `window_start_row`, `window_start_col`, `K_h`, `K_w`, `v_stride` and `h_stride` 
 * parameters, together with the dimensions of the input and output images, collectively 
 * define how the coordinate spaces of the input and output images relate to one another
 * via the convolution window.
 * 
 * `window_start_row` and `window_start_col` indicate the row and column (respectively) 
 * in the coordinate space of the input image, at which the convolution window starts.
 * These values may be negative if, for example, the convolution should start centered
 * on the top-left pixel of the input image, putting part of the convolution window
 * in the padding around the input image.
 * 
 * `K_h` and `K_w` are the height and width of the convolution window, specified in 
 * pixels.
 * 
 * `v_stride` and `h_stride` are the vertical and horizontal strides of the convolution
 * window, specified in pixels. A movement one pixel to the right in the output image
 * corresponds to a translation of the convolution window by `h_stride` pixels to the 
 * right across the input image. Likewise, each movement one pixel down in the output
 * image corresponds to a translation of the convolution window by `v_stride` pixels
 * downwards across the input image.
 * 
 * `zero_point` specifies the value associated with the padding space around the input
 * image. For any output pixel whereupon the corresponding convolution window location
 * in the input image extends beyond the bounds of the input image, those coefficients
 * in the convolution window which are in the padding are multiplied by `zero_point`
 * rather than from values in the input image. All input channels currently share a
 * common zero-point value.
 * 
 * `job_count` indicates the number of elements in the `jobs` array that is supplied 
 * by the user, as well the number of elements in the `job_params` array if it is not
 * `NULL`.
 * 
 * Constraints:
 *  - The input and output images must have a multiple of 4 channels. (i.e. 
 *      `x_params->channels` and `y_params->channels` must be a multiple of 4.)
 *  - There must always be at least one pixel of the convolution window within the 
 *      input image.
 * 
 * @param plan              The plan to be initialized.
 * @param jobs              Array of jobs to be initialized (length: `job_count`).
 * @param x_params          `conv2d_depthwise()` input image parameters.
 * @param y_params          `conv2d_depthwise()` output image parameters.
 * @param job_params        Array with configuration parameters for each job, or NULL.
 * @param window_start_row  The row at which the convolution window starts.
 * @param window_start_col  The column at which the convolution window starts.
 * @param K_h               The height of the convolution window (in pixels).
 * @param K_w               The width of the convolution window (in pixels).
 * @param v_stride          The vertical stride of the convolution window (in pixels).
 * @param h_stride          The horizontal stride of the convolution window (in pixels).
 * @param zero_point        The zero-point value of the input image.
 * @param job_count         The number of jobs to initialize.
 */
void conv2d_depthwise_init(
    nn_conv2d_depthwise_plan_t* plan,
    nn_conv2d_depthwise_job_t* jobs,
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_conv2d_job_params_t* job_params,
    const int window_start_row,
    const int window_start_col,
    const unsigned K_h,
    const unsigned K_w,
    const int v_stride,
    const int h_stride,
    const int8_t zero_point,
    const unsigned job_count);





/**
 * Initialize the plan and jobs required by the maxpool2d() function.
 * 
 * An instance of a `maxpool2d` operator is described by a plan and one or more jobs.
 * This function is used to initialize the plan and any jobs, and need only be called once
 * per instance of the `maxpool2d` operator.
 * 
 * Once initialized, a single instance of `maxpool2d` has its hyperparameters fixed. The hyperparameters
 * of the `maxpool2d` operator are those that capture the geometry of the input image, output image,
 * and the mapping of the former to the latter. This includes the dimensions of the input and output 
 * images, the pooling window dimensions, the pooling window strides, and the pooling window start
 * position with respect to the input image.
 * 
 * `maxpool2d` requires that the input and output channel counts be the same, and because every pixel must
 * start on a word-aligned boundary, the channel count must be a multiple of 4.
 * 
 * `plan` points to a `nn_maxpool2d_plan_t` struct, which will be populated with the info needed
 * by all jobs to implement the instance of maxpool2d().
 * 
 * `jobs` points to an array of one or more `nn_pool2d_job_t` structs, each of which will be populated
 * with the information necessary to compute a portion of the output image.
 * 
 * `x_params` and `y_params` are descriptions of the input and output images for this instance of `maxpool2d`.
 * 
 * The `window_config` parameter describes the pooling window, and how input pixels get mapped to pixels
 * in the output image.
 * 
 * `job_params` is either an array of `nn_window_op_params_t` of length `job_count`, or is `NULL`. If `NULL`,  
 * `job_count` must be 1, and the job `jobs` points to will be initialized to compute the entire output image.
 * If `job_count` is not `NULL`, each of the jobs in `jobs` will be initialized according to the corresponding
 * element in `job_params`.
 * 
 * `job_count` is the number of jobs to be initialized in the `jobs` array.
 * 
 * @param plan [out]            The plan to be initialized.
 * @param jobs [out]            Array of jobs to be initialized.
 * @param x_params [in]         Parameters describing the shape of each input image tensor @tensor{X}.
 * @param y_params [in]         Parameters describing the shape of each output image tensor @tensor{Y}
 * @param window_config [in]    Pooling window configuration.
 * @param job_params [in]       An array of `nn_window_op_job_params_t` structs, or NULL
 * @param job_count [in]        The number of jobs to be initialized.
 */
void maxpool2d_init(
    nn_maxpool2d_plan_t* plan,
    nn_pool2d_job_t* jobs,
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_window_params_t* window_config,
    const nn_window_op_job_params_t* job_params,
    const unsigned job_count);


/**
 * Initialize the plan and jobs required by the avgpool2d() function.
 * 
 * An instance of an `avgpool2d` operator is described by a plan and one or more jobs.
 * This function is used to initialize the plan and any jobs, and need only be called once
 * per instance of the `avgpool2d` operator.
 * 
 * Once initialized, a single instance of `avgpool2d` has its hyperparameters fixed. The hyperparameters
 * of the `avgpool2d` operator are those that capture the geometry of the input image, output image,
 * and the mapping of the former to the latter. This includes the dimensions of the input and output 
 * images, the pooling window dimensions, the pooling window strides, and the pooling window start
 * position with respect to the input image.
 * 
 * `avgpool2d` requires that the input and output channel counts be the same, and because every pixel must
 * start on a word-aligned boundary, the channel count must be a multiple of 4.
 * 
 * `plan` points to a `nn_avgpool2d_plan_t` struct, which will be populated with the info needed
 * by all jobs to implement the instance of avgpool2d().
 * 
 * `jobs` points to an array of one or more `nn_pool2d_job_t` structs, each of which will be populated
 * with the information necessary to compute a portion of the output image.
 * 
 * `x_params` and `y_params` are descriptions of the input and output images for this instance of `avgpool2d`.
 * 
 * The `window_config` parameter describes the pooling window, and how input pixels get mapped to pixels
 * in the output image.
 * 
 * `job_params` is either an array of `nn_window_op_params_t` of length `job_count`, or is `NULL`. If `NULL`,  
 * `job_count` must be 1, and the job `jobs` points to will be initialized to compute the entire output image.
 * If `job_count` is not `NULL`, each of the jobs in `jobs` will be initialized according to the corresponding
 * element in `job_params`.
 * 
 * `job_count` is the number of jobs to be initialized in the `jobs` array.
 * 
 * NOTE: If this average pool describes a global average pool, in which the entire input image contributes to
 *       a single output pixel, then the `avgpool2d_global` operator should be used instead of `avgpool2d`, 
 *       (In that case, `avgpool2d_global_init()` can be used to initialize it.)
 * 
 * @param plan [out]            The plan to be initialized.
 * @param jobs [out]            Array of jobs to be initialized.
 * @param x_params [in]         Parameters describing the shape of each input image tensor @tensor{X}.
 * @param y_params [in]         Parameters describing the shape of each output image tensor @tensor{Y}
 * @param window_config [in]    Pooling window configuration.
 * @param job_params [in]       An array of `nn_window_op_job_params_t` structs, or NULL
 * @param job_count [in]        The number of jobs to be initialized.
 */
void avgpool2d_init(
    nn_avgpool2d_plan_t* plan,
    nn_pool2d_job_t* jobs,
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_window_params_t* window_config,
    const nn_window_op_job_params_t* job_params,
    const unsigned job_count);




/**
 * Obtain the shift and scale parameters required by the `avgpool2d_global()` function.
 * 
 * The `x_height` and `x_width` inputs are the height and width of the image to be input
 * into the `avgpool2d_global()` function.
 * 
 * The `shift` and `scale` values will be set by this function, and should be passed to
 * the `shift` and `scale` inputs to `avgpool2d_global()`.
 * 
 * @param shift [out]   The shift parameter required by `avgpool2d_global()`
 * @param scale [out]   The scale parameter required by `avgpool2d_global()`
 * @param x_height [in] The height of the image to be input to `avgpool2d_global()`
 * @param x_width [in]  The width of the image to be input to `avgpool2d_global()`
 */
void avgpool2d_global_init(
    nn_avgpool2d_global_plan_t* plan,
    nn_avgpool2d_global_job_t* jobs,
    const nn_image_params_t* x_params,
    const nn_avgpool2d_global_job_params_t* job_params,
    const unsigned job_count);




/** 
 * Initialize the plan and jobs required by the fully_connected_16() function.
 * 
 * An instance of a `fully_connected_16` operator is described by a plan and one or more
 * jobs. This function is used to initialize the plan and any jobs, and need only be called once
 * per instance of the `fully_connected_16` operator.
 * 
 * Once initialized, a single instance of the `fully_connected_16` has its hyperparameters fixed.
 * The hyperparameters of the `fully_connected_16` operator are the numbers of input channels and
 * output channels.
 * 
 * `plan` points to a `nn_fully_connected_plan_t` struct, which will be populated with the info needed
 * by all jobs to implement the instance of `fully_connected_16`.
 * 
 * `jobs` points to an array of one or more `nn_fully_connected_job_t` structs, each of which will be
 * populated with the information necessary to compute a contiguous subset of the output channels.
 * 
 * `C_in` and `C_out` are the number of input and output elements respectively.
 * 
 * `job_params` is either an array of `nn_fully_connected_job_params_t` of length `job_count`, or is `NULL`. 
 * If `NULL`, `job_count` must be 1, and the job `jobs` points to will be initialized to compute the entire 
 * output image. If `job_count` is not `NULL`, each of the jobs in `jobs` will be initialized according to 
 * the corresponding element in `job_params`.
 * 
 * `job_count` is the number of jobs to be initialized in the `jobs` array.
 * 
 * @param plan [out]        The plan to be initialized.
 * @param jobs [out]        Array of jobs to be initialized.
 * @param C_in [in]         The number of input elements.
 * @param C_out [in]        The number of output elements.
 * @param job_params [in]   An array of `nn_fully_connected_job_params_t` structs, or `NULL`.
 * @param job_count [in]    The number of jobs to be initialized.
 */
void fully_connected_init(
    nn_fully_connected_plan_t* plan,
    nn_fully_connected_job_t* jobs,
    const channel_count_t C_in,
    const channel_count_t C_out,
    const nn_fully_connected_job_params_t* job_params,
    const unsigned job_count);


/**
 * Initialize the parameters required by the `requantize_16_to_8()` function.
 * 
 * This function initializes one or more jobs, represented by `nn_requantize_16_to_8_job_t` structs. Each job computes a 
 * contiguous subset of the operators's output channels.
 * 
 * Unlike most operators, no common parameters are required between jobs of `requantize_16_to_8()`, and so no structure
 * representing a plan is required.
 * 
 * Also unlike most operators, this function makes its own decision about how the work is to be divided up between the jobs.
 * 
 * `jobs` points to an array of `nn_requantize_16_to_8_job_t` structs to be initialized by the call. The number of elements
 * in the array should be `job_count`.
 * 
 * `length` is the number of elements in the input (and output).
 * 
 * `job_count` is the number of that the work is to be split between.
 * 
 * @param jobs [out]        Array of jobs, to be initialized by this call.
 * @param length [in]       The number of elements in the input (and output).
 * @param job_count [in]    The number of jobs.
 */
void requantize_16_to_8_init(
    nn_requantize_16_to_8_job_t* jobs,
    const uint32_t length,
    unsigned job_count);


#ifdef __XC__
}   //extern "C"
#endif

#endif //NN_OP_INIT_H_
