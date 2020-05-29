

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




/** Compute an execution plan for a 16-bit fully-connected layer.
 * 
 * The output `plan` is the execution plan to be computed.
 * 
 * `C_in` is the number of input elements.
 * 
 * `C_out` is the number of output elements.
 * 
 */
void fully_connected_init(
    nn_fully_connected_plan_t* plan,
    nn_fully_connected_job_t* jobs,
    const channel_count_t C_in,
    const channel_count_t C_out,
    const nn_fully_connected_job_params_t* job_params,
    const unsigned job_count);





/**
 * Compute the execution plan required by the `maxpool2d()` function.
 * 
 * `maxpool2d()` requires an execution plan (represented by a `nn_window_op_plan_t` struct)
 * to do its job. This function computes that execution plan based on the behavior specified
 * in `config`. The execution plan can be reused, so it need only be computed once at 
 * initialization time.
 * 
 * The `x` and `y` inputs describe the input and output images which will be used
 * when `avgpool2d() is called.
 * 
 * The `config` parameter describes the behavior of the operation, including pool window dimensions,
 * stride lengths, input and output start positions, and output pixel counts. See the 
 * `nn_maxpool_config_t` documentation for more details.
 * 
 * `maxpool2d()` requires that *every pixel* in the input and output images start at a word-aligned 
 * address, which means the input and output channel counts must be a multiple of 4.
 * 
 * \param plan      Output. Parameters used by `maxpool2d()`
 * \param x         Parameters of the image to be input to `maxpool2d()`
 * \param y         Parameters of the iamge to be output from `maxpool2d()`
 * \param config    `nn_maxpool_config_t` describing the behavior of the maxpool2d operation.
 */
void maxpool2d_init(
    nn_maxpool2d_plan_t* plan,
    nn_maxpool2d_job_t* jobs,
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_conv2d_window_params_t* window_params,
    const nn_conv2d_job_params_t* job_params,
    const unsigned job_count);


/**
 * Initialize the parameters required by the `avgpool2d()` function.
 * 
 * This function sets the values in `pool` to those needed by the `avgpool2d()` function. 
 * This need only be called once during initialization time.
 * 
 * The `x` and `y` inputs describe the input and output images which will be used
 * when `avgpool2d() is called.
 * 
 * The `config` parameter describes which output pixel values will be written, and how they are mapped
 * from pixels in the input image.
 * 
 * `avgpool2d()` requires that each pixel in both the input and output images start at a word-aligned 
 * address, thus the number of channels in the input and output images must both be multiples of 4 
 * (though the input and output images need not have the same number of channels).
 * 
 * NOTE: See the documentation for `nn_window_op_config_t` for details about what its fields mean.
 * 
 * NOTE: If this average pool describes the standard 2x2 average pool with a 2x2 stride across the entire
 *       input image, then the `avgpool2d_2x2()` function should be used instead, which is optimized 
 *       for that common scenario. (In that case, `avgpool2d_2x2_init()` should be used instead of this)
 * 
 * NOTE: If this average pool describes a global average pool, then the `avgpool2d_global` function
 *       should be used instead, which is optimized for that scenario. (In that case, `avgpool2d_global_init()`
 *       should be used instead of this.
 * 
 * \param pool      Output. Parameters used by `avgpool2d()`
 * \param x         Parameters of the image to be input to `avgpool2d()`
 * \param y         Parameters of the image to be output from `avgpool2d()`
 * \param config    Configuration struct specifying desired behavior.
 */
void avgpool2d_init(
    nn_avgpool2d_plan_t* pool,
    const nn_image_params_t* x,
    const nn_image_params_t* y,
    const nn_window_op_config_t* config);



/**
 * Initialize the supplied `nn_avgpool2d_plan_t` for use with `avgpool2d_2x2()`.
 * 
 * `avgpool2d_2x2_asm()` is an optimized implementation of the `avgpool2d()` function for when the pooling 
 * window dimensions are 2x2 and both horizontal and vertical strides are 2.
 * 
 * The `x` and `y` parameters describe the shape of the input (`X[][][]`) and output (`Y[][][]`) images 
 * respectively. When `avgpool2d_2x2_asm()` is called, input and output tensors must have the shapes specified 
 * in `x` and `y`.
 * 
 * The `x_start`, `y_start`, `out_rows`, `out_cols` and `out_chans` parameters collectively define sub-tensors 
 * in the input and output images, and can be used to bound the output to only a region of the output image. 
 * This may be useful for parallelization.
 * 
 * More specifically, the subtensor in the input starts at `X[x_start->rows][x_start->cols][x_start->channels]`, 
 * and extends to (including) `X[x_start->rows + 2*out_rows - 1][x_start->cols + 2*out_cols - 1][x_start->channels + out_chans - 1]`,
 * and the subtensor in the output starts at `Y[y_start->rows][y_start->cols][y_start->channels]` and extends to 
 * (including) `Y[y_start->rows + out_rows - 1][y_start->cols + out_cols -1][y_start->channels + out_chans -1]`.
 * 
 * The operation performed is:
 * \code
 *      Y[y_start->rows + i][y_start->cols + j][y_start->channels + k] <- 
 *         (X[x_start->rows + 2*i    ][x_start->cols + 2*j    ][x_start->channels + k]
 *        + X[x_start->rows + 2*i    ][x_start->cols + 2*j + 1][x_start->channels + k]
 *        + X[x_start->rows + 2*i + 1][x_start->cols + 2*j    ][x_start->channels + k]
 *        + X[x_start->rows + 2*i + 1][x_start->cols + 2*j + 1][x_start->channels + k]
 *        + 2) >> 2;
 *  \endcode
 *          where `0 <= i < out_rows`,  `0 <= j < out_cols`, and  `0 <= k < out_chans`
 * 
 * To consume an entire input image with this operator, use (given the input image parameters `x`):
 * \code
 *  x_start->rows = x_start->cols = x_start->channels = 0;
 *  y_start->rows = y_start->cols = y_start->channels = 0;
 *  out_rows = y->height = x->height/2;
 *  out_cols = y->width = x->width/2;
 *  out_chans = y->channels = x->channels;
 * \endcode
 * 
 * NOTE: A sub-tensor does *not* imply a contiguous region of memory. It is only contiguous (and rectangular)
 *       in the conceptual 3-dimensional tensor.
 * 
 * NOTE: The `plan` generated by `avgpool2d_2x2_init()` is compatible with both `avgpool2d()`, but plans 
 *       generated by `avgpool2d_init()` are not necessarily compatible with `avgpool2d_2x2()`.
 * 
 * \param plan      `nn_avgpool2d_plan_t` struct to be initialized.
 * \param x         Parameters of the image to be input to `avgpool2d_2x2()`
 * \param y         Parameters of the image to be output from `avgpool2d_2x2()`
 * \param x_start   Initial position for the window, relative to the input image
 * \param y_start   Initial position for output, relative to the output image
 * \param out_rows  The number of rows to output
 * \param out_cols  The number of columns to output
 * \param out_chans The number of channels to output
 */
void avgpool2d_2x2_init(
    nn_avgpool2d_plan_t* plan,
    const nn_image_params_t* x,
    const nn_image_params_t* y,
    const nn_image_vect_t* x_start,
    const nn_image_vect_t* y_start,
    const unsigned out_rows,
    const unsigned out_cols,
    const unsigned out_chans);


/**
 * Obtain the shift and scale parameters required by the `avgpool2d_global()` function.
 * 
 * The `x_height` and `x_width` inputs are the height and width of the image to be input
 * into the `avgpool2d_global()` function.
 * 
 * The `shift` and `scale` values will be set by this function, and should be passed to
 * the `shift` and `scale` inputs to `avgpool2d_global()`.
 * 
 * \param shift     Output. The shift parameter required by `avgpool2d_global()`
 * \param scale     Output. The scale parameter required by `avgpool2d_global()`
 * \param x_height  The height of the image to be input to `avgpool2d_global()`
 * \param x_width   The width of the image to be input to `avgpool2d_global()`
 */
void avgpool2d_global_init(
    uint32_t* shift,
    uint32_t* scale,
    const uint32_t x_height,
    const uint32_t x_width);


#ifdef __XC__
}   //extern "C"
#endif

#endif //NN_OP_INIT_H_
