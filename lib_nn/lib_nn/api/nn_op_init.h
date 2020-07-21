

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
 * @brief Initialize an instance of the @oper{conv2d_deep} operator.
 * 
 * See @oper_ref{conv2d_deep} for more details about the @oper{conv2d_deep} operator. To invoke a @oper{conv2d_deep} 
 * job, call conv2d_deep().
 * 
 * When conv2d_deep() is called, a plan (`nn_conv2d_deep_plan_t`) and a job (`nn_conv2d_deep_job_t`) must be supplied 
 * to tell it how to do its work. This function initializes that plan and one or more jobs to be supplied in subsequent 
 * calls to conv2d_deep().
 * 
 * A plan contains information shared by all jobs of an instance of @oper{conv2d_deep}. Each job computes a rectangular 
 * sub-tensor of the output image (possibly the entire image).
 * 
 * `plan` points to the plan to be initialized. It need only be initialized once for many calls to conv2d_deep().
 * 
 * `jobs` points to an array of `nn_conv2d_deep_job_t` to be initialized. Each element represents one job. There should 
 * be `job_count` elements in the array.
 * 
 * `x_params` points to the image parameters for the instance's input image @tensor{X}.
 * 
 * `y_params` points to the image parameters for the instance's output image @tensor{Y}.
 * 
 * `job_params` points to either an array of `nn_conv2d_job_params_t` structs or else is `NULL`. A `job_params` value of  
 * `NULL` indicates that there will only be a single job which computes the entire output image. If `job_params` is 
 * `NULL`, then `job_count` must be `1`. If `job_params` is not `NULL`, it must point to an array of `job_count` 
 * `nn_conv2d_job_params_t` elements.
 * 
 * In particular, job `k` will compute the output elements @math{Y[r,c,p]} for which:
 * @inlinecode
 *     job_params[k].start.rows <= r < job_params[k].start.rows + job_params[k].size.rows
 *     job_params[k].start.cols <= c < job_params[k].start.cols + job_params[k].size.cols
 *     job_params[k].start.channels <= p < job_params[k].start.channels + job_params[k].size.channels
 * @endinlinecode
 * 
 * If multiple jobs are specified, it is the user's responsibility to ensure that the supplied list of job params 
 * collectively computes the entire output image (no gaps) and does not compute any output values redundantly (no 
 * overlap).
 * 
 * `conv_window` points to a `nn_window_params_t` struct containing the instance's @math{K_h}, @math{K_w}, 
 * @math{W_{vert}), @math{W_{hori}), @math{W_{r0}) and @math{W_{c0}} hyperparameters (see @ref 
 * conv2d_deep_hyperparameters) which describe the relationship between the input image, the convolution window and the 
 * output image.
 * 
 * `conv_window->shape` specified @math{K_w} and @math{K_h}, the height and width of the convolution window. 
 * 
 * `conv_window->start` specifies @math{W_{r0}} and @math{W_{c0}}, the starting row and column of the convolution window 
 * in @tensor{X}'s coordinate space. For example, a `start` value of `(0,0)` indicates that the top-left pixel of the 
 * output image has the convolution window aligned with the top-left corner of the input image, with no implied padding 
 * at the top or left sides of the input image.
 * 
 * `conv_window->stride.horizontal` specifies @math{W_{vert}} and @math{W_{hori}}, the vertical and horizontal strides 
 * of the convolution window. The strides describe the number of pixels the convolution window moves (across the input 
 * image) with each pixel in the output image.
 * 
 * `zero_point` specifies @math{z_0}, the value associated with the (implied) padding space around the input image. For 
 * any output pixel whereupon the corresponding convolution window location in the input image extends beyond the bounds 
 * of the input image, those coefficients in the convolution window which are in the padding are multiplied by 
 * @math{z_0} rather than by values from the input image. All input channels currently share a common zero-point value.
 * 
 * `job_count` indicates the number of jobs to be initialized (and thus the number of elements in the `jobs` array), as 
 * well the number of elements in the `job_params` array if it is not `NULL`.
 * 
 * 
 * @param[out] plan         The plan to be initialized
 * @param[out] jobs         Array of jobs to be initialized
 * @param[in]  x_params     Parameters describing the shape of input image tensor @tensor{X}
 * @param[in]  y_params     Parameters describing the shape of output image tensor @tensor{Y}
 * @param[in]  job_params   Array with configuration parameters for each job, or `NULL`
 * @param[in]  conv_window  Parameters describing the relationship between the convolution window, the input image and 
 *                          the output image
 * @param[in]  zero_point   The value @math{z_0} to be used for padding (for all channels)
 * @param[in]  job_count    The number of jobs to initialize
 */
void conv2d_deep_init(
    nn_conv2d_deep_plan_t* plan,
    nn_conv2d_deep_job_t* jobs,
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_conv2d_job_params_t* job_params,
    const nn_window_params_t* conv_window,
    const int8_t zero_point,
    const unsigned job_count);



/** 
 * @brief Initialize an instance of the @oper{conv2d_shallowin} operator.
 * 
 * See @oper_ref{conv2d_shallowin} for more details about the @oper{conv2d_shallowin} operator. To invoke a 
 * @oper{conv2d_shallowin} job, call conv2d_shallowin().
 * 
 * When conv2d_shallowin() is called, a plan (`nn_conv2d_shallowin_plan_t`) and a job (`nn_conv2d_shallowin_job_t`) must 
 * be supplied to tell it how to do its work. This function initializes that plan and one or more jobs to be supplied in 
 * subsequent calls to conv2d_shallowin().
 * 
 * A plan contains information shared by all jobs of an instance of @oper{conv2d_shallowin}. Each job computes a 
 * rectangular sub-tensor of the output image (possibly the entire image).
 * 
 * `plan` points to the plan to be initialized. It need only be initialized once for many calls to conv2d_shallowin().
 * 
 * `jobs` points to an array of `nn_conv2d_shallowin_job_t` to be initialized. Each element represents one job. There 
 * should be `job_count` elements in the array.
 * 
 * `x_params` points to the image parameters for the instance's input image @tensor{X}.
 * 
 * `y_params` points to the image parameters for the instance's output image @tensor{Y}.
 * 
 * `job_params` points to either an array of `nn_conv2d_job_params_t` structs or else is `NULL`. A `job_params` value of 
 * `NULL` indicates that there will only be a single job which computes the entire output image. If `job_params` is 
 * `NULL`, then `job_count` must be `1`. If `job_params` is not `NULL`, it must point to an array of `job_count` 
 * `nn_conv2d_job_params_t` elements.
 * 
 * In particular, job `k` will compute the output elements @math{Y[r,c,p]} for which:
 * @inlinecode
 *     job_params[k].start.rows <= r < job_params[k].start.rows + job_params[k].size.rows
 *     job_params[k].start.cols <= c < job_params[k].start.cols + job_params[k].size.cols
 *     job_params[k].start.channels <= p < job_params[k].start.channels + job_params[k].size.channels
 * @endinlinecode
 * 
 * If multiple jobs are specified, it is the user's responsibility to ensure that the supplied list of job params 
 * collectively computes the entire output image (no gaps) and does not compute any output values redundantly (no 
 * overlap).
 * 
 * `conv_window` points to a `nn_window_params_t` struct containing the instance's @math{K_h}, @math{K_w}, 
 * @math{W_{vert}}, @math{W_{hori}}, @math{W_{r0}} and @math{W_{c0}} hyperparameters (see @ref 
 * conv2d_shallowin_hyperparameters) which describe the relationship between the input image, the convolution window and 
 * the output image.
 * 
 * `conv_window->shape` specified @math{K_w} and @math{K_h}, the height and width of the convolution window. 
 * 
 * `conv_window->start` specifies @math{W_{r0}} and @math{W_{c0}}, the starting row and column of the convolution window 
 * in @tensor{X}'s coordinate space. For example, a `start` value of `(0,0)` indicates that the top-left pixel of the 
 * output image has the convolution window aligned with the top-left corner of the input image, with no implied padding 
 * at the top or left sides of the input image.
 * 
 * `conv_window->stride.horizontal` specifies @math{W_{vert}} and @math{W_{hori}}, the vertical and horizontal strides 
 * of the convolution window. The strides describe the number of pixels the convolution window moves (across the input 
 * image) with each pixel in the output image.
 * 
 * `zero_point` specifies @math{z_0}, the value associated with the (implied) padding space around the input image. For 
 * any output pixel whereupon the corresponding convolution window location in the input image extends beyond the bounds 
 * of the input image, those coefficients in the convolution window which are in the padding are multiplied by 
 * @math{z_0} rather than by values from the input image. All input channels currently share a common zero-point value.
 * 
 * `job_count` indicates the number of jobs to be initialized (and thus the number of elements in the `jobs` array), as 
 * well the number of elements in the `job_params` array if it is not `NULL`.
 * 
 * 
 * @param[out] plan         The plan to be initialized
 * @param[out] jobs         Array of jobs to be initialized
 * @param[in]  x_params     Parameters describing the shape of input image tensor @tensor{X}
 * @param[in]  y_params     Parameters describing the shape of output image tensor @tensor{Y}
 * @param[in]  job_params   Array with configuration parameters for each job, or `NULL`
 * @param[in]  conv_window  Parameters describing the relationship between the convolution window, the input image and 
 *                          the output image
 * @param[in]  zero_point   The value @math{z_0} to be used for padding (for all channels)
 * @param[in]  job_count    The number of jobs to initialize
 */
void conv2d_shallowin_init(
    nn_conv2d_shallowin_plan_t* plan,
    nn_conv2d_shallowin_job_t* jobs,
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_conv2d_job_params_t* job_params,
    const nn_window_params_t* conv_window,
    const int8_t zero_point,
    const unsigned job_count);


/** 
 * @brief Prepare an execution plan for a 2D shallow im2col convolution.
 * 
 * When `conv2d_im2col()` is called, a plan (`nn_conv2d_im2col_plan_t`) and a
 * job (`nn_conv2d_im2col_job_t`) must be supplied to tell it how to do its work. This 
 * function initializes that plan and one or more jobs to be supplied in subsequent calls
 * to `conv2d_im2col()`.
 * 
 * A plan contains information shared by all jobs. A job, when provided to `conv2d_im2col()`,
 * computes a rectangular sub-tensor of the output image (possibly the entire image).
 * 
 * `plan` is a pointer to the execution plan to be initialized. It need only be 
 * initialized once for many calls to `conv2d_im2col()`.
 * 
 * `jobs` is a pointer, supplied by the caller to an array of `nn_conv2d_im2col_job_t` 
 * structs which will be initialized by this function. `job_count` jobs will be 
 * initialized.
 * 
 * `x_params` is a pointer to image parameters for an input image @tensor{X} that will be
 * passed to `conv2d_im2col()` in subsequent calls.
 * 
 * `y_params` is a pointer to image parameters for an output image @tensor{Y} that will be
 * computed by subsequent calls to `conv2d_im2col()`.
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
 * 
 * @param[out] plan         The plan to be initialized
 * @param[out] jobs         Array of jobs to be initialized (length: `job_count`)
 * @param[in]  x_params     Parameters describing the shape of each input image tensor @tensor{X}
 * @param[in]  y_params     Parameters describing the shape of each output image tensor @tensor{K}
 * @param[in]  job_params   Array with configuration parameters for each job, or `NULL`
 * @param[in]  conv_window  Parameters describing the relationship between the convolution window, the
 *                          input image and the output image
 * @param[in]  zero_point   The value to be used (for all channels) for padding
 * @param[in]  job_count    The number of jobs to initialize
 */
void conv2d_im2col_init(
    nn_conv2d_im2col_plan_t* plan,
    nn_conv2d_im2col_job_t* jobs,
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_conv2d_job_params_t* job_params,
    const nn_window_params_t* conv_window,
    const int8_t zero_point,
    const unsigned job_count);




/** 
 * @brief Initialize an instance of the @oper{conv2d_1x1} operator.
 * 
 * See @oper_ref{conv2d_1x1} for more details about the @oper{conv2d_1x1} operator. To invoke a @oper{conv2d_1x1} job, 
 * call conv2d_1x1().
 * 
 * When conv2d_1x1() is called, a plan (`nn_conv2d_1x1_plan_t`) and a job (`nn_conv2d_1x1_job_t`) must be supplied 
 * to tell it how to do its work. This function initializes that plan and one or more jobs to be supplied in subsequent 
 * calls to conv2d_1x1().
 * 
 * A plan contains information shared by all jobs of an instance of @oper{conv2d_1x1}. Each job computes a range of the
 * output channels for a contiguous sequence of pixels (when the image is flattened row by row), possibly including the
 * entire output image.
 * 
 * `plan` points to the plan to be initialized. It need only be initialized once for many calls to conv2d_1x1().
 * 
 * `jobs` points to an array of `nn_conv2d_1x1_job_t` to be initialized. Each element represents one job. There should 
 * be `job_count` elements in the array.
 * 
 * `x_params` points to the image parameters for the instance's input image @tensor{X}.
 * 
 * `y_params` points to the image parameters for the instance's output image @tensor{Y}.
 * 
 * `job_params` points to either an array of `nn_conv2d_1x1_job_params_t` structs or else is `NULL`. A `job_params` 
 * value of `NULL` indicates that there will only be a single job which computes the entire output image. If 
 * `job_params` is `NULL`, then `job_count` must be `1`. If `job_params` is not `NULL`, it must point to an array of 
 * `job_count` `nn_conv2d_1x1_job_params_t` elements.
 * 
 * In particular, job `k` will start computing outputs at row `job_params[k].start.rows` and column 
 * `job_params[k].start.cols` of the output image. It will compute outputs for `job_params[k].size.pixels` pixels, 
 * increasing the column index until the end of the row, then moving to column 0 of the following row. The job will
 * compute out channels `job_params[k].start.channels` through `job_params[k].size.channels - 1` (inclusive).
 * 
 * If multiple jobs are specified, it is the user's responsibility to ensure that the supplied list of job params 
 * collectively computes the entire output image (no gaps) and does not compute any output values redundantly (no 
 * overlap).
 * 
 * `job_count` indicates the number of jobs to be initialized (and thus the number of elements in the `jobs` array), as 
 * well the number of elements in the `job_params` array if it is not `NULL`.
 * 
 * @param[out] plan         The plan to be initialized
 * @param[out] jobs         Array of jobs to be initialized
 * @param[in]  x_params     Parameters describing the shape of input image tensor @tensor{X}
 * @param[in]  y_params     Parameters describing the shape of output image tensor @tensor{Y}
 * @param[in]  job_params   Array with configuration parameters for each job, or `NULL`
 * @param[in]  job_count    The number of jobs to initialize
 */
void conv2d_1x1_init(
    nn_conv2d_1x1_plan_t* plan,
    nn_conv2d_1x1_job_t* jobs,
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_conv2d_1x1_job_params_t* job_params,
    const unsigned job_count);



/** 
 * @brief Initialize an instance of the @oper{conv2d_depthwise} operator.
 * 
 * See @oper_ref{conv2d_depthwise} for more details about the @oper{conv2d_depthwise} operator. To invoke a 
 * @oper{conv2d_depthwise} job, call conv2d_depthwise().
 * 
 * When conv2d_depthwise() is called, a plan (`nn_conv2d_depthwise_plan_t`) and a job (`nn_conv2d_depthwise_job_t`) must 
 * be supplied to tell it how to do its work. This function initializes that plan and one or more jobs to be supplied in 
 * subsequent calls to conv2d_depthwise().
 * 
 * A plan contains information shared by all jobs of an instance of @oper{conv2d_depthwise}. Each job computes a 
 * rectangular sub-tensor of the output image (possibly the entire image).
 * 
 * `plan` points to the plan to be initialized. It need only be initialized once for many calls to conv2d_depthwise().
 * 
 * `jobs` points to an array of `nn_conv2d_depthwise_job_t` to be initialized. Each element represents one job. There 
 * should be `job_count` elements in the array.
 * 
 * `x_params` points to the image parameters for the instance's input image @tensor{X}.
 * 
 * `y_params` points to the image parameters for the instance's output image @tensor{Y}.
 * 
 * `job_params` points to either an array of `nn_conv2d_job_params_t` structs or else is `NULL`. A `job_params` value of  
 * `NULL` indicates that there will only be a single job which computes the entire output image. If `job_params` is 
 * `NULL`, then `job_count` must be `1`. If `job_params` is not `NULL`, it must point to an array of `job_count` 
 * `nn_conv2d_job_params_t` elements.
 * 
 * In particular, job `k` will compute the output elements @math{Y[r,c,p]} for which:
 * @inlinecode
 *     job_params[k].start.rows <= r < job_params[k].start.rows + job_params[k].size.rows
 *     job_params[k].start.cols <= c < job_params[k].start.cols + job_params[k].size.cols
 *     job_params[k].start.channels <= p < job_params[k].start.channels + job_params[k].size.channels
 * @endinlinecode
 * 
 * If multiple jobs are specified, it is the user's responsibility to ensure that the supplied list of job params 
 * collectively computes the entire output image (no gaps) and does not compute any output values redundantly (no 
 * overlap).
 * 
 * `conv_window` points to a `nn_window_params_t` struct containing the instance's @math{K_h}, @math{K_w}, 
 * @math{W_{vert}), @math{W_{hori}), @math{W_{r0}) and @math{W_{c0}} hyperparameters (see @ref 
 * conv2d_depthwise_hyperparameters) which describe the relationship between the input image, the convolution window and 
 * the output image.
 * 
 * `conv_window->shape` specified @math{K_w} and @math{K_h}, the height and width of the convolution window. 
 * 
 * `conv_window->start` specifies @math{W_{r0}} and @math{W_{c0}}, the starting row and column of the convolution window 
 * in @tensor{X}'s coordinate space. For example, a `start` value of `(0,0)` indicates that the top-left pixel of the 
 * output image has the convolution window aligned with the top-left corner of the input image, with no implied padding 
 * at the top or left sides of the input image.
 * 
 * `conv_window->stride.horizontal` specifies @math{W_{vert}} and @math{W_{hori}}, the vertical and horizontal strides 
 * of the convolution window. The strides describe the number of pixels the convolution window moves (across the input 
 * image) with each pixel in the output image.
 * 
 * `zero_point` specifies @math{z_0}, the value associated with the (implied) padding space around the input image. For 
 * any output pixel whereupon the corresponding convolution window location in the input image extends beyond the bounds 
 * of the input image, those coefficients in the convolution window which are in the padding are multiplied by 
 * @math{z_0} rather than by values from the input image. All input channels currently share a common zero-point value.
 * 
 * `job_count` indicates the number of jobs to be initialized (and thus the number of elements in the `jobs` array), as 
 * well the number of elements in the `job_params` array if it is not `NULL`.
 * 
 * Constraints:
 *  - The input and output images must have a multiple of 4 channels. (i.e. `x_params->channels` and 
 *      `y_params->channels` must be a multiple of 4.)
 *  - There must be at least one pixel of the convolution window within the input image for every output pixel.
 * 
 * 
 * @param[out] plan         The plan to be initialized
 * @param[out] jobs         Array of jobs to be initialized
 * @param[in]  x_params     Parameters describing the shape of input image tensor @tensor{X}
 * @param[in]  y_params     Parameters describing the shape of output image tensor @tensor{Y}
 * @param[in]  job_params   Array with configuration parameters for each job, or `NULL`
 * @param[in]  conv_window  Parameters describing the relationship between the convolution window, the input image and 
 *                          the output image
 * @param[in]  zero_point   The value @math{z_0} to be used for padding (for all channels)
 * @param[in]  job_count    The number of jobs to initialize
 */
void conv2d_depthwise_init(
    nn_conv2d_depthwise_plan_t* plan,
    nn_conv2d_depthwise_job_t* jobs,
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_conv2d_job_params_t* job_params,
    const nn_window_params_t* conv_window,
    const int8_t zero_point,
    const unsigned job_count);





/**
 * @brief Initialize an instance of the @oper{maxpool2d} operator.
 * 
 * See @oper_ref{maxpool2d} for more details about the @oper{maxpool2d} operator. To invoke a @oper{maxpool2d} job, call 
 * maxpool2d().
 * 
 * When maxpool2d() is called, a plan (`nn_maxpool2d_plan_t`) and a job (`nn_pool2d_job_t`) must be supplied to tell it 
 * how to do its work. This function initializes that plan and one or more jobs to be supplied in subsequent calls to 
 * maxpool2d().
 * 
 * A plan contains information shared by all jobs of an instance of @oper{maxpool2d}. Each job computes a rectangular 
 * sub-tensor of the output image (possibly the entire image).
 * 
 * `plan` points to the plan to be initialized. It need only be initialized once for many calls to maxpool2d().
 * 
 * `jobs` points to an array of `nn_pool2d_job_t` to be initialized. Each element represents one job. There should be 
 * `job_count` elements in the array.
 * 
 * `x_params` points to the image parameters for the instance's input image @tensor{X}.
 * 
 * `y_params` points to the image parameters for the instance's output image @tensor{Y}.
 * 
 * `window_config` points to a `nn_window_params_t` struct containing the instance's @math{W_h}, @math{W_w}, 
 * @math{W_{vert}}, @math{W_{hori}}, @math{W_{r0}} and @math{W_{c0}} hyperparameters (see @ref 
 * maxpool2d_hyperparameters) which describe the relationship between the input image, the pooling window and the 
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
 * 
 * @param plan          [out]   The plan to be initialized.
 * @param jobs          [out]   Array of jobs to be initialized.
 * @param x_params      [in]    Parameters describing the shape of each input image tensor @tensor{X}.
 * @param y_params      [in]    Parameters describing the shape of each output image tensor @tensor{Y}
 * @param window_config [in]    Pooling window configuration.
 * @param job_params    [in]    An array of `nn_window_op_job_params_t` structs, or NULL
 * @param job_count     [in]    The number of jobs to be initialized.
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




/**
 * @brief Initialize an instance of the @oper{avgpool2d_global} operator.
 * 
 * See @oper_ref{avgpool2d_global} for more details about the @oper{avgpool2d_global} operator. To invoke a 
 * @oper{avgpool2d_global} job, call avgpool2d_global().
 * 
 * When avgpool2d_global() is called, a plan (`nn_avgpool2d_global_plan_t`) and a job (`nn_avgpool2d_global_job_t`) must 
 * be supplied to tell it how to do its work. This function initializes that plan and one or more jobs to be supplied in 
 * subsequent calls to avgpool2d_global().
 * 
 * A plan contains information shared by all jobs of an instance of @oper{avgpool2d_global}. Each job computes a range
 * of channels in the output vector (possibly the entire vector).
 * 
 * `plan` points to the plan to be initialized. It need only be initialized once for many calls to avgpool2d_global().
 * 
 * `jobs` points to an array of `nn_avgpool2d_global_job_t` to be initialized. Each element represents one job. There 
 * should be `job_count` elements in the array.
 * 
 * `x_params` points to the image parameters for the instance's input (and output) image @tensor{X}.
 * 
 * `job_params` points to either an array of `nn_avgpool2d_global_job_params_t` structs or else is `NULL`. A 
 * `job_params` value of `NULL` indicates that there will only be a single job which computes the entire output image. 
 * If `job_params` is `NULL`, then `job_count` must be `1`. If `job_params` is not `NULL`, it must point to an array 
 * of `job_count` `nn_avgpool2d_global_job_params_t` elements.
 * 
 * In particular, job `k` will compute the output elements @math{y[p]} for which:
 * @inlinecode
 *     job_params[k].start_channel <= p < job_params[k].start_channel + job_params[k].out.channels
 * @endinlinecode
 * 
 * `job_count` indicates the number of jobs to be initialized (and thus the number of elements in the `jobs` array), as 
 * well the number of elements in the `job_params` array if it is not `NULL`.
 * 
 * @param plan          [out]   The plan to be initialized.
 * @param jobs          [out]   Array of jobs to be initialized.
 * @param x_params      [in]    Parameters describing the shape of each input (and output) image tensor @tensor{X}.
 * @param job_params    [in]    An array of `nn_avgpool2d_global_job_params_t` structs, or NULL
 * @param job_count     [in]    The number of jobs to be initialized.
 */
void avgpool2d_global_init(
    nn_avgpool2d_global_plan_t* plan,
    nn_avgpool2d_global_job_t* jobs,
    const nn_image_params_t* x_params,
    const nn_avgpool2d_global_job_params_t* job_params,
    const unsigned job_count);




/** 
 * @brief Initialize an instance of the @oper{fully_connected_16} operator.
 * 
 * See @oper_ref{fully_connected_16} for more details about the @oper{fully_connected_16} operator. To invoke a 
 * @oper{fully_connected_16} job, call fully_connected_16().
 * 
 * When fully_connected_16() is called, a plan (`nn_fully_connected_plan_t`) and a job (`nn_fully_connected_job_t`) must 
 * be supplied to tell it how to do its work. This function initializes that plan and one or more jobs to be supplied in 
 * subsequent calls to fully_connected_16().
 * 
 * A plan contains information shared by all jobs of an instance of @oper{fully_connected_16}. Each job computes a 
 * range of elements in the output vector (possibly the entire vector).
 * 
 * `plan` points to the plan to be initialized. It need only be initialized once for many calls to fully_connected_16().
 * 
 * `jobs` points to an array of `nn_fully_connected_job_t` to be initialized. Each element represents one job. There 
 * should be `job_count` elements in the array.
 * 
 * `N` is the number of elements in the input vector @tensor{x}.
 * 
 * `M` is the number of elements in the output vector @tensor{y}.
 * 
 * `job_params` points to either an array of `nn_fully_connected_job_params_t` structs or else is `NULL`. A `job_params` 
 * value of `NULL` indicates that there will only be a single job which computes the entire output image. If 
 * `job_params` is `NULL`, then `job_count` must be `1`. If `job_params` is not `NULL`, it must point to an array of 
 * `job_count` `nn_fully_connected_job_params_t` elements.
 * 
 * In particular, job `k` will compute the output elements @math{y[p]} for which:
 * @inlinecode
 *     job_params[k].start_channel <= p < job_params[k].start_channel + job_params[k].out.channels
 * @endinlinecode
 * 
 * `job_count` indicates the number of jobs to be initialized (and thus the number of elements in the `jobs` array), as 
 * well the number of elements in the `job_params` array if it is not `NULL`.
 * 
 * @param plan       [out]  The plan to be initialized.
 * @param jobs       [out]  Array of jobs to be initialized.
 * @param N          [in]   The number of input elements @math{N}.
 * @param M          [in]   The number of output elements @math{M}.
 * @param job_params [in]   An array of `nn_fully_connected_job_params_t` structs, or `NULL`.
 * @param job_count  [in]   The number of jobs to be initialized.
 */
void fully_connected_init(
    nn_fully_connected_plan_t* plan,
    nn_fully_connected_job_t* jobs,
    const channel_count_t N,
    const channel_count_t M,
    const nn_fully_connected_job_params_t* job_params,
    const unsigned job_count);


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
 * Unlike most other operators, @oper{requantize_16_to_8} will automatically divide the work to be done as evenly as 
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


void bsign_8_init(
    nn_bsign_8_job_t* jobs,
    const uint32_t N,
    const unsigned job_count);


#ifdef __XC__
}   //extern "C"
#endif

#endif //NN_OP_INIT_H_
