
#include "nn_conv2d_structs.h"
#include "nn_bso.h"
#include "nn_conv2d_int8_structs.h"


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



// /** 
//  * @brief Initialize an instance of the @oper{conv2d_depthwise} operator.
//  * 
//  * See @oper_ref{conv2d_depthwise} for more details about the @oper{conv2d_depthwise} operator. To invoke a 
//  * @oper{conv2d_depthwise} job, call conv2d_depthwise().
//  * 
//  * When conv2d_depthwise() is called, a plan (`nn_conv2d_depthwise_plan_t`) and a job (`nn_conv2d_depthwise_job_t`) must 
//  * be supplied to tell it how to do its work. This function initializes that plan and one or more jobs to be supplied in 
//  * subsequent calls to conv2d_depthwise().
//  * 
//  * A plan contains information shared by all jobs of an instance of @oper{conv2d_depthwise}. Each job computes a 
//  * rectangular sub-tensor of the output image (possibly the entire image).
//  * 
//  * `plan` points to the plan to be initialized. It need only be initialized once for many calls to conv2d_depthwise().
//  * 
//  * `jobs` points to an array of `nn_conv2d_depthwise_job_t` to be initialized. Each element represents one job. There 
//  * should be `job_count` elements in the array.
//  * 
//  * `x_params` points to the image parameters for the instance's input image @tensor{X}.
//  * 
//  * `y_params` points to the image parameters for the instance's output image @tensor{Y}.
//  * 
//  * `job_params` points to either an array of `nn_conv2d_job_params_t` structs or else is `NULL`. A `job_params` value of  
//  * `NULL` indicates that there will only be a single job which computes the entire output image. If `job_params` is 
//  * `NULL`, then `job_count` must be `1`. If `job_params` is not `NULL`, it must point to an array of `job_count` 
//  * `nn_conv2d_job_params_t` elements.
//  * 
//  * In particular, job `k` will compute the output elements @math{Y[r,c,p]} for which:
//  * @inlinecode
//  *     job_params[k].start.rows <= r < job_params[k].start.rows + job_params[k].size.rows
//  *     job_params[k].start.cols <= c < job_params[k].start.cols + job_params[k].size.cols
//  *     job_params[k].start.channels <= p < job_params[k].start.channels + job_params[k].size.channels
//  * @endinlinecode
//  * 
//  * If multiple jobs are specified, it is the user's responsibility to ensure that the supplied list of job params 
//  * collectively computes the entire output image (no gaps) and does not compute any output values redundantly (no 
//  * overlap).
//  * 
//  * `conv_window` points to a `nn_window_params_t` struct containing the instance's @math{K_h}, @math{K_w}, 
//  * @math{W_{vert}), @math{W_{hori}), @math{W_{r0}) and @math{W_{c0}} hyperparameters (see @ref 
//  * conv2d_depthwise_hyperparameters) which describe the relationship between the input image, the convolution window and 
//  * the output image.
//  * 
//  * `conv_window->shape` specified @math{K_w} and @math{K_h}, the height and width of the convolution window. 
//  * 
//  * `conv_window->start` specifies @math{W_{r0}} and @math{W_{c0}}, the starting row and column of the convolution window 
//  * in @tensor{X}'s coordinate space. For example, a `start` value of `(0,0)` indicates that the top-left pixel of the 
//  * output image has the convolution window aligned with the top-left corner of the input image, with no implied padding 
//  * at the top or left sides of the input image.
//  * 
//  * `conv_window->stride.horizontal` specifies @math{W_{vert}} and @math{W_{hori}}, the vertical and horizontal strides 
//  * of the convolution window. The strides describe the number of pixels the convolution window moves (across the input 
//  * image) with each pixel in the output image.
//  * 
//  * `zero_point` specifies @math{z_0}, the value associated with the (implied) padding space around the input image. For 
//  * any output pixel whereupon the corresponding convolution window location in the input image extends beyond the bounds 
//  * of the input image, those coefficients in the convolution window which are in the padding are multiplied by 
//  * @math{z_0} rather than by values from the input image. All input channels currently share a common zero-point value.
//  * 
//  * `job_count` indicates the number of jobs to be initialized (and thus the number of elements in the `jobs` array), as 
//  * well the number of elements in the `job_params` array if it is not `NULL`.
//  * 
//  * Constraints:
//  *  - The input and output images must have a multiple of 4 channels. (i.e. `x_params->channels` and 
//  *      `y_params->channels` must be a multiple of 4.)
//  *  - There must be at least one pixel of the convolution window within the input image for every output pixel.
//  * 
//  * 
//  * @param[out] plan         The plan to be initialized
//  * @param[out] jobs         Array of jobs to be initialized
//  * @param[in]  x_params     Parameters describing the shape of input image tensor @tensor{X}
//  * @param[in]  y_params     Parameters describing the shape of output image tensor @tensor{Y}
//  * @param[in]  job_params   Array with configuration parameters for each job, or `NULL`
//  * @param[in]  conv_window  Parameters describing the relationship between the convolution window, the input image and 
//  *                          the output image
//  * @param[in]  zero_point   The value @math{z_0} to be used for padding (for all channels)
//  * @param[in]  job_count    The number of jobs to initialize
//  */
// void conv2d_depthwise_prepare(
//     nn_conv2d_depthwise_plan_t* plan,
//     nn_conv2d_depthwise_job_t* job,
//     const nn_image_params_t* x_params,
//     const nn_image_params_t* y_params,
//     const nn_window_params_t* conv_window,
//     const nn_conv2d_job_params_t* job_params);




/**
 * @brief Execute @oper{conv2d_deep} job.
 * 
 * See @oper_ref{conv2d_deep} for more details about the @oper{conv2d_deep} operator.
 * 
 * An instance of the @oper{conv2d_deep} operator requires an initialized plan and one or more jobs. See 
 * conv2d_deep_init() for more details.
 * 
 * `Y` points to the output image @tensor{Y} with shape @tensor_shape{Y_h, Y_w, Y_c}. The address supplied for `Y` 
 * should be the start address of the output image (for any job being processed).
 * 
 * `X` points to the input image @tensor{X} with shape @tensor_shape{X_h, X_w, X_c}. The address supplied for `X` should 
 * be the start address of the input image (for any job being processed).
 * 
 * The memory layout of @tensor{Y} and @tensor{X} are the standard memory layout for image tensors (see @ref 
 * standard_layout).
 * 
 * `K` points to the kernel tensor @tensor{K} with shape @tensor_shape{Y_c, K_h, K_w, X_c}, which correspond to the 
 * output image channels, convolution window rows and columns, and the input image channels respectively. The address 
 * supplied for `K` should be the start address of the kernel tensor (for any job being processed).
 * 
 * The memory layout of @tensor{K} is the standard memory layout for 4D tensors (see @ref standard_layout).
 * 
 * `BSO` points to an array of bias-scale-offset parameters required for this convolution. See @ref bso_layout for 
 * details on the encoding of this array. The address supplied for `BSO` should be the start address of the the array 
 * (for any job being processed).
 * 
 * `plan` points to the (initialized) plan associated with this instance of the @oper{conv2d_deep} operator.
 * 
 * `job` points to the (initialized) job to be performed with this call.
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
 * @brief Execute @oper{conv2d_shallowin} job.
 * 
 * See @oper_ref{conv2d_shallowin} for more details about the @oper{conv2d_shallowin} operator.
 * 
 * An instance of the @oper{conv2d_shallowin} operator requires an initialized plan and one or more jobs. See 
 * conv2d_shallowin_init() for more details.
 * 
 * `Y` points to the output image @tensor{Y} with shape @tensor_shape{Y_h, Y_w, Y_c}. The address supplied for `Y` 
 * should be the start address of the output image (for any job being processed).
 * 
 * `X` points to the input image @tensor{X} with shape @tensor_shape{X_h, X_w, X_c}. The address supplied for `X` should 
 * be the start address of the input image (for any job being processed).
 * 
 * The memory layout of @tensor{Y} and @tensor{X} are the standard memory layout for image tensors (see @ref 
 * standard_layout).
 * 
 * `K` points to the kernel tensor @tensor{K} encoded with shape @tensor_shape{Y_c, K_h, \hat{K_w}, X_c}, where 
 * @math{Y_c}, @math{K_h} and @math{X_c} correspond to the output image channels, convolution window rows and the input 
 * image channel count respectively. @math{\hat{K_w}} is the augmented convolution window width, which must be exactly
 * @math{32/X_c}, regardless of the convolution window width @math{K_w}. The address supplied for `K` should be the 
 * start address of the kernel tensor (for any job being processed).
 * 
 * The memory layout of @tensor{K} (with the modified 3rd dimension) is the standard memory layout for 4D tensors (see 
 * @ref standard_layout). Further, the coefficients for all elements @math{K\left[i,j,k,l\right]} where @math{k\geq K_w} 
 * must have the value 0.
 * 
 * `BSO` points to an array of bias-scale-offset parameters required for this convolution. See @ref bso_layout for 
 * details on the encoding of this array. The address supplied for `BSO` should be the start address of the the array 
 * (for any job being processed).
 * 
 * `plan` points to the (initialized) plan associated with this instance of the @oper{conv2d_shallowin} operator.
 * 
 * `job` points to the (initialized) job to be performed with this call.
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
 * @brief Perform a 2D convolution of a shallow input image.
 * 
 * Perform a 2D convolution of kernel tensor @tensor{K} with input image @tensor{X}
 * to produce output image @tensor{Y}.
 *  
 * This function is optimized for input images that have 3 channels, but will work
 * with any number of input channels. This will use more memory than the TFLite 
 * reference implementation, but run much faster:
 * 
 * Additional memory: 1 patch worth (K_w * K_h * C_in) bytes + 32 bytes + some code
 * Performance gain: Depends on patch size vs # output channels, and padding, but approximately:
 *                   16x faster convolutions - PATCH_SIZE copy operations
 * 
 * @note multiples of 16 output channels will run fastest input channels not imporant, but a PATCH_SIZE 
 * that is a multiple of 32 will be the fastest, most memory effcient
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
 * `COL` points to a caller supplied buffer that will be used for the internal im2col() transformation. This buffer
 * needs to be word-aligned and a multiple of 32-bytes in length, no shorter than: (K_w * K_h * C_in) bytes. 
 * 
 * `K` points to the kernel tensor @tensor{K} with shape @tensor_shape{Y_c, K_h * K_w * X_c}, where @math{Y_c},
 * @math{K_h} @math{K_w} and @math{X_c} correspond to the output image channels, convolution window rows, 
 * convolution window columns, and the input image channel count respectively.
 * 
 * The memory layout of @tensor{K} is a row-major 2D matrix
 * 
 * `BSO` points to an array of bias-shifts-scale parameters required for this convolution. Each 
 * `nn_bso_block_t` in the array contains the bias-shifts-scale parameters for a single output channel group,
 * (@ttref{VPU_INT8_ACC_PERIOD} output channels). If @math{Y_c} is not a multiple of @ttref{VPU_INT8_ACC_PERIOD}, 
 * then the output channel tail ( the last @math{(Y_c mod 16)} output channels) also gets `nn_bso_block_t`, where
 * the entries corresponding to channels beyond @math{Y_c} are ignored. The address supplied for `BSO` should be
 * the start address of the the array, *not* the address of the `nn_bso_block_t` corresponding of the first output
 * channel of the job being processed.
 * 
 * `plan` points to the `nn_conv2d_im2col_plan_t` which was previously initialized with a call to 
 * `conv2d_im2col_init()`.
 * 
 * `job` points to the job to be performed in this call, which was previously initialized along-side `plan`. 
 * 
 * @requires_word_alignment{Y,X,COL,K,BSO}
 * 
 * @param[out] Y        The output image @tensor{Y}
 * @param[in]  X        The input image @tensor{X}
 * @param[in]  COL      Scratch space for im2col (multiple of 32 words >= |K|)
 * @param[in]  K        The kernel tensor @tensor{K}
 * @param[in]  BSO      The bias-shifts-scale parameters
 * @param[in]  plan     The convolution plan
 * @param[in]  job      The convolution job
 */
void conv2d_im2col(
    nn_image_t* Y,
    const nn_image_t* X,
    const nn_image_t* COL,
    const nn_tensor_t* K,
    const nn_bso_block_t* BSO,
    const nn_conv2d_im2col_plan_t* plan,
    const nn_conv2d_im2col_job_t* job);


/**
 * @brief Execute @oper{conv2d_1x1} job.
 * 
 * See @oper_ref{conv2d_1x1} for more details about the @oper{conv2d_1x1} operator.
 * 
 * An instance of the @oper{conv2d_1x1} operator requires a plan and one or more jobs, which are represented
 * by the `nn_conv2d_1x1_plan_t` and `nn_conv2d_1x1_job_t` structs. Before performing a 2D convolution using 
 * this function, a call must be made to conv2d_1x1_init() to initialize the plan and any jobs.
 * 
 * An instance of the @oper{conv2d_1x1} operator requires an initialized plan and one or more jobs. See 
 * conv2d_1x1_init() for more details.
 * 
 * `Y` points to the output image @tensor{Y} with shape @tensor_shape{Y_h, Y_w, Y_c}. The address supplied for `Y` 
 * should be the start address of the output image (for any job being processed).
 * 
 * `X` points to the input image @tensor{X} with shape @tensor_shape{X_h, X_w, X_c}. The address supplied for `X` should 
 * be the start address of the input image (for any job being processed).
 * 
 * The memory layout of @tensor{Y} and @tensor{X} are the standard memory layout for image tensors (see @ref 
 * standard_layout).
 * 
 * `K` points to the kernel tensor @tensor{K} with shape @tensor_shape{Y_c, X_c}, which correspond to the 
 * output image channels and input image channels respectively. The address supplied for `K` should be the start address 
 * of the kernel tensor (for any job being processed).
 * 
 * The memory layout of @tensor{K} is the standard memory layout for 2D tensors (see @ref standard_layout).
 * 
 * `BSO` points to an array of bias-scale-offset parameters required for this convolution. See @ref bso_layout for 
 * details on the encoding of this array. The address supplied for `BSO` should be the start address of the the array 
 * (for any job being processed).
 * 
 * `plan` points to the (initialized) plan associated with this instance of the @oper{conv2d_deep} operator.
 * 
 * `job` points to the (initialized) job to be performed with this call.
 * 
 * @requires_word_alignment{Y,X,K,BSO} 
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
 * @brief Execute @oper{conv2d_depthwise} job.
 * 
 * See @oper_ref{conv2d_depthwise} for more details about the @oper{conv2d_depthwise} operator.
 * 
 * An instance of the @oper{conv2d_depthwise} operator requires an initialized plan and one or more jobs. See 
 * conv2d_depthwise_init() for more details.
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
 * `K` points to the kernel tensor @tensor{K} with shape @tensor_shape{K_h, K_w, X_c}, which correspond to the 
 * convolution window rows and columns and the input image channels respectively. The address supplied for `K` should be 
 * the start address of the kernel tensor (for any job being processed).
 * 
 * The memory layout of @tensor{K} is the standard memory layout for 3D tensors (see @ref standard_layout).
 * 
 * `BSO` points to an array of bias-scale-offset parameters required for this convolution. See @ref bso_layout for 
 * details on the encoding of this array. The address supplied for `BSO` should be the start address of the the array 
 * (for any job being processed).
 * 
 * `plan` points to the (initialized) plan associated with this instance of the @oper{conv2d_depthwise} operator.
 * 
 * `job` points to the (initialized) job to be performed with this call.
 * 
 * @requires_word_alignment{Y,X,K,BSO}
 * 
 * @param Y    [out]    The output image @tensor{Y}
 * @param X    [in]     The input image @tensor{X}
 * @param K    [in]     The kernel tensor @tensor{K}
 * @param BSO  [in]     The bias-scale-offset array
 * @param plan [in]     The @oper{conv2d_depthwise} plan to be processed
 * @param job  [in]     The @oper{conv2d_depthwise} job to be processed
 */
void conv2d_depthwise(
    int8_t* Y,
    const int8_t* X,
    const int8_t* K,
    const nn_bso_block_t* BSO,
    const int8_t zero_point,
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_window_params_t* conv_window);

/**
 * @brief Execute @oper{conv2d_depthwise} job.
 * 
 * See @oper_ref{conv2d_depthwise} for more details about the @oper{conv2d_depthwise} operator.
 * 
 * An instance of the @oper{conv2d_depthwise} operator requires an initialized plan and one or more jobs. See 
 * conv2d_depthwise_init() for more details.
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
 * `K` points to the kernel tensor @tensor{K} with shape @tensor_shape{K_h, K_w, X_c}, which correspond to the 
 * convolution window rows and columns and the input image channels respectively. The address supplied for `K` should be 
 * the start address of the kernel tensor (for any job being processed).
 * 
 * The memory layout of @tensor{K} is the standard memory layout for 3D tensors (see @ref standard_layout).
 * 
 * `BSO` points to an array of bias-scale-offset parameters required for this convolution. See @ref bso_layout for 
 * details on the encoding of this array. The address supplied for `BSO` should be the start address of the the array 
 * (for any job being processed).
 * 
 * `plan` points to the (initialized) plan associated with this instance of the @oper{conv2d_depthwise} operator.
 * 
 * `job` points to the (initialized) job to be performed with this call.
 * 
 * @requires_word_alignment{Y,X,K,BSO}
 * 
 * @param Y    [out]    The output image @tensor{Y}
 * @param X    [in]     The input image @tensor{X}
 * @param K    [in]     The kernel tensor @tensor{K}
 * @param BSO  [in]     The bias-scale-offset array
 * @param plan [in]     The @oper{conv2d_depthwise} plan to be processed
 * @param job  [in]     The @oper{conv2d_depthwise} job to be processed
 */
void conv2d_depthwise_adv(
    int8_t* Y,
    const int8_t* X,
    const int8_t* K,
    const nn_bso_block_t* BSO,
    const int8_t zero_point,
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_window_params_t* conv_window,
    const nn_window_op_job_params_t* job_params,
    const nn_conv2d_depthwise_adv_t* adv);


    
/** 
 * Compute a row of output pixels obtained through depthwise 2D convolution.
 * 
 * Consider the following:
 * 
 *    _____                   _____                    _____                  _____   
 *   |O O O|                 |O O O|                  |O O O|                |O O O|  
 *   |O O O|X X X X X       X|O O O|X X X        X X X|O O O|X      X X X X X|O O O|
 *   |O_O_O|X X X X X       X|O_O_O|X X X        X X X|O_O_O|X      X X X X X|O_O_O|
 *      X X X X X X X       X X X X X X X        X X X X X X X      X X X X X X X 
 *      X X X X X X X       X X X X X X X        X X X X X X X      X X X X X X X 
 *      X X X X X X X       X X X X X X X        X X X X X X X      X X X X X X X 
 *           
 *        Y _ _ _             Y Y _ _              Y Y Y _           Y Y Y Y
 *        _ _ _ _             _ _ _ _              _ _ _ _           _ _ _ _
 * 
 * The above shows a 3x3 convolution kernel being scanned horizontally across (centered
 * on) the top row of a 5x7 input image. The convolution window shifts, with a horizontal
 * stride of 2 pixels each time to produce 4 adjacent output pixels. (The example does
 * not indicate the number of image channels.)
 * 
 * In this example, the top row of the convolution window is in padding the whole time, 
 * while the padding on the left and right edges changes.
 * 
 * This example corresponds to calling `nn_compute_patch_depthwise_padded_asm()` 4 times
 * with appropriate adjustments made to its `Y`, `X`, `pad_l`, and `pad_r` arguments
 * each time. That is what this function does, only more efficiently. To perform a full
 * depthwise convolution, this function can be called multiple times, with `Y`, `X`, 
 * `pad_t`,  `pad_b` appropriately adjusted each time.
 * 
 * In the documentation that follows, "input image patch" or just "patch" refers to the
 * region of the input image which is inside the convolution window.
 * 
 *  
 * `Y` is the memory location at which to write the first output values. This address
 * will be incremented by `Y_c` bytes following each of the `out_pixels` outputs.
 * 
 * `X` points to the location in (or near) the input image where the convolution window
 * will start. The input image at which `X` points has `K_c` channels. If the top-left
 * spatial cell of the convolution window is in padding at its first location, then
 * the `X` pointer may point outside the actual input image's memory. Specifically, from
 * the example above, if the input image is defined as
 * \code
 *      int8_t input_image[5][7][K_c];
 * \endcode
 * then `X = &input_image[-1][-1][0]`. If the appropriate padding values are provided,
 * this function will never perform a read from such an invalid location.
 * 
 * `K` is the convolution kernel, and has shape `(K_h, K_w, K_c)`.
 * 
 * `BSO` are the biases shifts and scale which are applied to the convolution. See _________
 * 
 * `K_h` is the height of the convolution window in pixels.
 * 
 * `K_w` is the width of the convolution window in pixels.
 * 
 * `pad_t` and `pad_b` are the number of rows of padding at the top and bottom of the
 * convolution window respectively. As the window only strides horizontally, this padding 
 * will not change during a single invocation of this function. These values are 
 * non-negative. In the example above, `pad_t` is `1`, because the top row of the 
 * convolution kernel is in padding, and `pad_b` is `0` (NOT `-3`) because none of the
 * convolution window is below the input image.
 * 
 * `pad_l_initial` and `pad_r_inital` are the possibly-negative number of columns of
 * padding on the left and right sides of the convolution window respectively at the 
 * _first_ window location. Unlike the top and bottom padding, the padding at the left
 * or right may change with each window stride. In order to correctly and efficiently
 * track this padding change, negative values may need to be supplied. For example,
 * in the case above, `pad_l_initial` is `1`, because the left-most column of the
 * convolution window is in padding, and `pad_r_initial` is `-5`, because there are
 * `5` image pixels between the right edge of the convolution window and the right
 * edge of the input image. The relation `pad_l_initial + pad_r_initial == K_w - X_w` 
 * should always be true.
 * 
 * `X_c` is the number channels in the input image.
 * 
 * `K_c` is the number channels in the weight tensor.
 * 
 * `x_row_stride` is the number of _bytes_ between end of one row of an input image patch
 * and the start of the next row of the input image patch. In the example above, `x_row_stride`
 * would be `(7-3)*K_c`, because the input image is `7` pixels wide, the convolution
 * window is `3` pixels wide, and there are `K_c` bytes per pixel.
 * 
 * `window_hstride` is the number of _bytes_ by which the convolution window will be
 * moved to the right for each output pixel. In the example above, the horizontal 
 * stride of the convolution window is 2 pixels, and so `window_hstride` would be
 * `2*K_c`, because each pixel of the input image is `K_c` bytes.
 * 
 * `Y_c` is the number of channels in the output image, or equivalently, the number of
 * bytes per pixel of the output image. The pointer `Y` is incremented by `Y_c` for
 * each output written. (`chans_to_write` bytes are written at each location.)
 * 
 * `out_pixels` is the number of output pixels to write.
 * 
 * `chans_to_write` is the number of output channels to be written. 
 * 
 * `zero_point_vec` is a vector of `chans_to_write` zero-point values for each channel.
 * 
 * Constraints:
 *  - `Y`, `X`, `K`, and `BSO` must all point to word-aligned memory addresses.
 *  - `0 <= chans_to_write <= 16`  (though 0 is an expensive no-op)
 *  - `K_c` and `Y_c` must be a multiple of 4.
 * 
 * \param Y                 Address at which to write first outputs
 * \param X                 Address in input image of the initial convolution window location
 * \param K                 Convolution kernel
 * \param BSO               Bias, shifts and scale tensor
 * \param K_h               Convolution window height
 * \param K_w               Convolution window width
 * \param pad_t             Number of rows of padding at the top of the convolution window
 * \param pad_l_initial     Number of columns of padding at the left of the convolution window in its first location
 * \param pad_b             Number of rows of padding at the bottom of the convolution window
 * \param pad_r_initial     Number of columns of padding at the right of the convolution window in its first location
 * \param X_c               Number of channels in the input image
 * \param K_c               Number of channels in the kernel tensor
 * \param x_row_stride      Number of bytes between the end of a patch row and the start of the next
 * \param window_hstride    Number of bytes between subsequent patches of the input image
 * \param Y_c               Number of channels in the output image
 * \param out_pixels        Number of pixels to be written
 * \param chans_to_write    Number of channels to write in each output pixel
 * \param zero_point_vec    Vector of zero-points for each channel
 */
void nn_conv2d_hstrip_depthwise_padded(
    int8_t* Y,
    const int8_t* X, 
    const int8_t* K,
    const nn_bso_block_t* BSO,
    const unsigned K_h,
    const unsigned K_w,
    const int32_t pad_t,
    const int32_t pad_l_initial,
    const int32_t pad_b,
    const int32_t pad_r_initial,
    const int32_t X_c,
    const int32_t K_c,
    const int32_t x_row_stride,
    const int32_t window_hstride,
    const int32_t Y_c,
    const unsigned out_pixels,
    const unsigned chans_to_write,
    const int8_t* zero_point_vec);


/** 
 * Compute a row of output pixels obtained through depthwise 2D convolution. (padding
 * not supported)
 * 
 * Consider the following:
 * 
 *    _____                    _____                    _____   
 *   |O O O|X X X X        X X|O O O|X X        X X X X|O O O|  
 *   |O O O|X X X X        X X|O O O|X X        X X X X|O O O|  
 *   |O_O_O|X X X X        X X|O_O_O|X X        X X X X|O_O_O|  
 *    X X X X X X X        X X X X X X X        X X X X X X X   
 *    X X X X X X X        X X X X X X X        X X X X X X X   
 *                                                              
 *        Y _ _                 Y Y _               Y Y Y        
 *        _ _ _                 _ _ _               _ _ _
 * 
 * The above shows a 3x3 convolution kernel being scanned horizontally across the top 
 * row of a 5x7 input image. The convolution window shifts, with a horizontal
 * stride of 2 pixels each time to produce 3 adjacent output pixels. (The example does
 * not indicate the number of image channels.)
 * 
 * In this example, the convolution window never extends beyond the bounds of the input
 * image. Padding is not supported by this function. If padding is needed, use 
 * `nn_conv2d_hstrip_depthwise_padded_asm()` instead.
 * 
 * This example corresponds to calling `nn_compute_patch_depthwise_asm()` 4 times
 * with appropriate adjustments made to its `Y` and `X` arguments each time. 
 * That is what this function does. To perform a full depthwise convolution, this 
 * function can be called multiple times, with `Y` and `X` appropriately adjusted each 
 * time.
 * 
 * In the documentation that follows, "input image patch" or just "patch" refers to the
 * region of the input image which is inside the convolution window.
 * 
 *  
 * `Y` is the memory location at which to write the first output values. This address
 * will be incremented by `Y_c` bytes following each of the `out_pixels` outputs.
 * 
 * `X` points to the location in the input image where the convolution window
 * will start. The input image at which `X` points has `K_c` channels. 
 * 
 * `K` is the convolution kernel, and has shape `(K_h, K_w, K_c)`.
 * 
 * `BSO` are the biases shifts and scale which are applied to the convolution. See _________
 * 
 * `K_h` is the height of the convolution window in pixels.
 * 
 * `K_w` is the width of the convolution window in pixels.
 * 
 * `X_c` is the number channels in the input image.
 * 
 * `K_c` is the number channels in the weight tensor.
 * 
 * `x_row_stride` is the number of _bytes_ between end of one row of an input image patch
 * and the start of the next row of the input image patch. In the example above, `x_row_stride`
 * would be `(7-3)*K_c`, because the input image is `7` pixels wide, the convolution
 * window is `3` pixels wide, and there are `K_c` bytes per pixel.
 * 
 * `window_hstride` is the number of _bytes_ by which the convolution window will be
 * moved to the right for each output pixel. In the example above, the horizontal 
 * stride of the convolution window is 2 pixels, and so `window_hstride` would be
 * `2*K_c`, because each pixel of the input image is `K_c` bytes.
 * 
 * `Y_c` is the number of channels in the output image, or equivalently, the number of
 * bytes per pixel of the output image. The pointer `Y` is incremented by `Y_c` for
 * each output written. (`chans_to_write` bytes are written at each location.)
 * 
 * `out_pixels` is the number of output pixels to write.
 * 
 * `chans_to_write` is the number of output channels to be written. 
 * 
 * Constraints:
 *  - `Y`, `X`, `K`, and `BSO` must all point to word-aligned memory addresses.
 *  - `0 <= chans_to_write <= 16`  (although 0 is an expensive no-op)
 *  - `K_c` and `Y_c` must be a multiple of 4.
 * 
 * \param Y                 Address at which to write first outputs
 * \param X                 Address in input image of the initial convolution window location
 * \param K                 Convolution kernel
 * \param BSO               Bias, shifts and scale tensor
 * \param K_h               Convolution window height
 * \param K_w               Convolution window width
 * \param X_c               Number of channels in the input image
 * \param K_c               Number of channels in the kernel tensor
 * \param x_row_stride      Number of bytes between the end of a patch row and the start of the next
 * \param window_hstride    Number of bytes between subsequent patches of the input image
 * \param Y_c               Number of channels in the output image
 * \param out_pixels        Number of pixels to be written
 * \param chans_to_write    Number of channels to write in each output pixel
 */
void nn_conv2d_hstrip_depthwise(
    int8_t* Y,
    const int8_t* X, 
    const int8_t* K,
    const nn_bso_block_t* BSO,
    const unsigned K_h,
    const unsigned K_w,
    const int32_t X_c,
    const int32_t K_c,
    const int32_t x_row_stride,
    const int32_t window_hstride,
    const int32_t Y_c,
    const unsigned out_pixels,
    const unsigned chans_to_write);


/**
 * @brief Compute a row of output pixels for a deep 2D convolution (Full @ttref{VPU_INT8_ACC_PERIOD} channels) 
 * 
 * A deep 2D convolution is one in which each output channel is a function of every input channel. This function
 * is intended to be invoked multiple times to do a full 2D convolution. Each invocation will compute 
 * @ttref{VPU_INT8_ACC_PERIOD} channels on each of `out_cols` pixels in a single row of @tensor{Y}.
 * 
 * `y` points to output tensor @tensor{Y} with shape @tensor_shape{Y_h, Y_w, Y_c}. @math{Y_w} must
 * be at least `out_cols`, and @math{Y_c} must be at least @ttref{VPU_INT8_ACC_PERIOD}. Rather than pointing
 * at the start of @tensor{Y}, the `y` passed to this function should point to the first channel to be output
 * of the first pixel to be output. The address to which `y` points will be the first memory written.
 * 
 * `x` points to the input tensor @tensor{X} with shape @tensor_shape{X_h, X_w, X_c}. Rather than pointing
 * at the start of @tensor{X}, the `x` passed to this function should point to the beginning of the first 
 * pixel of the convolution window. If the convolution window starts in padding, then this *may point
 * outside of @tensor{X}.
 * 
 * For example, the diagram below shows a `3x3` convolution being applied to a `4x4` image. The input image pixels
 * are marked with `X`'s, and the (implied) padding is marked with `P`'s. In this case, the convolution window 
 * starts in the padding at `(-1, -1, 0)` in @vector{X}'s coordinate space (i.e. @math{X[-1][-1]}) as is indicated 
 * by `P*` in the diagram. If `X_image` points to the start of the input image, then the argument `x` should 
 * be `(nn_image_t*) &X_image[-1][-1][0]`.
 * 
 * @inlinecode
 *  _ _ _      
 * |P*P P|P P P
 * |P X X|X X P
 * |P_X_X|X X P
 *  P X X X X P
 *  P X X X X P
 *  P P P P P P  
 *   
 * @endinlinecode
 * 
 * `k` points to the kernel tensor @tensor{K} with shape @tensor_shape{Y_c, K_h, K_w, X_c}. @math{Y_c} and 
 * @math{X_c} are the channel counts of the output and input images respectively. @math{K_h} and @math{K_w}
 * are the spatial height and width of the 2D convolution window respectively, expressed in pixels. Rather than
 * pointing at the start of @tensor{K}, the `k` passed to this function should point to the beginning of the
 * <b>final</b> channel of the output channel group being processed in the call.
 * 
 * For example, if @tensor{K} has shape @tensor_shape{48,5,5,32}, defined as `nn_tensor_t kernel[48][5][5][32]`,
 * then to process the output channel group (output channels `16` through `31`), then the address passed to this
 * call as `k` should be `&kernel[31][0][0][0]`. The `KERNEL_4D_COG_LAST_CHAN_START()` macro can be used to 
 * simplify this. The memory layout of @tensor{K} is the standard memory layout for 4D tensors (see 
 * @ref standard_layout).
 * 
 * `bso` points to the `nn_bso_block_t` describing the biases, shifts and scale for the current group of output 
 * channels. `bso` can be initialized with `nn_standard_BSO_layout()`. See @ref bso_layout for more information.
 * 
 * `K_height` and `K_width` are the height and width respectively of the 2D convolution window.
 * 
 * `K_hori_stride` is the horizontal stride (in pixels) taken by the 2D convolution window between between
 * output pixels.
 * 
 * `C_in` (or @math{X_c}) is the number of channels in the input image @tensor{X}.
 * 
 * `pad_t` and `pad_b` are the number of rows of padding at the top and bottom of the convolution window 
 * respectively. Because this function computes a horizontal strip of output pixels, the padding at the
 * top and bottom of the convolution window does not change during one invocation. These values are non-negative, 
 * and so 'negative' amounts of top or bottom padding should be conveyed as zeros. See example below.
 * 
 * `pad_l_initial` and `pad_r_initial` are the number of columns of padding at the left and right sides of the 
 * convolution window at the begining of the operation. Because this function computes a horizontal strip of output
 * pixels, the amount of left and right padding may change during an invocation. Unlike `pad_t` and `pad_b`, 
 * `pad_l_initial` and `pad_r_initial` indicate the *signed* padding, where gaps between the convolution window
 * and the edge of the input image may be negative, as in the following example.
 * 
 * @inlinecode
 *            pad_r_initial = -3
 *         |-----|  
 *    _____       _
 *   |W W W|      | pad_t = max(1,0) = 1
 *   |W W W|X X X ̅
 *   |W W W|X X X
 *   |W_W_W|X X X _ pad_b = max(-2,0) = 0
 *      X X X X X |
 *      X X X X X |
 *   |-|          ̅
 *     pad_l_initial = 1    
 * @endinlinecode
 * 
 * The diagram above shows a `4x3` convolution window over a `5x5` image. In this case, `pad_t` is `1`, because the
 * top row of the convolution window is outside the input image. `pad_b` is `0`, because the bottom of the convolution
 * window is 2 pixels above the bottom of the image. `pad_l_initial` is `1`, because the left column of the 
 * convolution window is outside the input image. And `pad_r_initial` is `-3`, because if the convolution window
 * moves to the right 3 pixels, the edge of the convolution window will coincide with the edge of the input image.
 * Note also that `pad_r_initial` is *independent of* `K_hori_stride`; `pad_r_initial` is expressed as *input image*
 * pixels, not *output image* pixels.
 * 
 * `x_v_stride` is the memory stride required to move a pointer from (immediately after) the right edge of the 
 * convolution window to (immediately after) the left edge of the convolution window on the following row (note: 
 * "immediately after" because the convolution window's right and left edges are effectively *between* memory 
 * addresses). This is equivalent to @math{\left(X_w-K_w\right)\cdot X_c}.
 * 
 * `k_cout_stride` is the memory stride required to move a pointer from the beginning of one output channel of 
 * @tensor{K} to the output channel with index one *lower*, i.e. stride required to decrement the output channel
 * by one. This is equivalent to @math{-\left(K_h\cdot K_w\cdot X_c\right)}.
 * 
 * `y_h_stride` is the memory stride required to move one pixel to the right in @tensor{Y}. This is equivalent 
 * to @math{Y_c}.
 * 
 * `out_cols` is the number of output pixels for each of which @ttref{VPU_INT8_ACC_PERIOD} channels are to be computed.
 * 
 * `zero_point_vec` is a pointer to an `int8_t` buffer with length @ttref{VPU_INT8_EPV}. This is supplied by
 * the user as a vector to save time (it allows the same vector to be reused, rather than copying a single
 * `int8_t` value several times).
 * 
 * @requires_word_alignment{y,x,k,bso}
 * 
 * @par Requirements and Constraints
 *   - `C_in` must be a multiple of `4`.
 *   - `y_h_stride` must be a multiple of `4`.
 * 
 * @param[out] y                    Pointer to output image @tensor{Y}
 * @param[in]  x                    Pointer to input image @tensor{X}
 * @param[in]  k                    The kernel tensor @tensor{K}
 * @param[in]  bso                  The bias-scale-offset parameters
 * @param[in]  K_height             Kernel height @math{K_h} (in pixels)
 * @param[in]  K_width              Kernel width @math{K_w} (in pixels)
 * @param[in]  K_hori_stride        Horizontal stride of the convolution window (in pixels)
 * @param[in]  C_in                 Number of input channels, @math{X_c}
 * @param[in]  pad_t                (unsigned) Number of rows of padding at the top of the convolution window
 * @param[in]  pad_b                (unsigned) Number or rows of padding at the bottom of the convolution window
 * @param[in]  pad_l_initial        (signed) Number of columns of padding at the left of the convolution window (at 
 *                                  its initial location)
 * @param[in]  pad_r_initial        (signed) Number of columns of padding at the right of the convolution window (at
 *                                  its initial location)
 * @param[in]  x_v_stride           Stride to move from the right edge of a patch to the left-most pixel 
 *                                  of the following row (in bytes)
 * @param[in]  k_cout_stride        
 * @param[in]  y_h_stride           Stride to move the output pointer from one pixel to the next (in bytes)
 * @param[in]  out_cols             Number of output pixels to write.
 * @param[in]  zero_point_vec       Vector containing `VPU_INT8_EPV` copies of the zero point value.
 */
void nn_conv2d_hstrip_deep_padded(
        nn_image_t* Y,
        const nn_image_t* X,
        const nn_tensor_t* K,    
        const nn_bso_block_t* BSO,
        const unsigned K_height,
        const unsigned K_width,
        const unsigned K_hori_stride,
        const channel_count_t C_in,
        const unsigned pad_t,
        const unsigned pad_b,
        const int pad_l_initial,
        const int pad_r_initial,
        const mem_stride_t x_v_stride,
        const mem_stride_t k_cout_stride,
        const mem_stride_t y_h_stride,
        const unsigned out_cols,
        const int8_t* zero_point_vec);


/**
 * @brief Compute a row of output pixels for a deep 2D convolution (Full @ttref{VPU_INT8_ACC_PERIOD} channels) 
 * 
 * A deep 2D convolution is one in which each output channel is a function of every input channel. This function
 * is intended to be invoked multiple times to do a full 2D convolution. Each invocation will compute 
 * @ttref{VPU_INT8_ACC_PERIOD} channels on each of `out_cols` pixels in a single row of @tensor{Y}.
 * 
 * `y` points to output tensor @tensor{Y} with shape @tensor_shape{Y_h, Y_w, Y_c}. @math{Y_w} must
 * be at least `out_cols`, and @math{Y_c} must be at least @ttref{VPU_INT8_ACC_PERIOD}. Rather than pointing
 * at the start of @tensor{Y}, the `y` passed to this function should point to the first channel to be output
 * of the first pixel to be output. The address to which `y` points will be the first memory written.
 * 
 * `x` points to the input tensor @tensor{X} with shape @tensor_shape{X_h, X_w, X_c}. Rather than pointing
 * at the start of @tensor{X}, the `x` passed to this function should point to the beginning of the first 
 * pixel of the convolution window. If the convolution window starts in padding, then this *may point
 * outside of @tensor{X}.
 * 
 * For example, the diagram below shows a `3x3` convolution being applied to a `4x4` image. The input image pixels
 * are marked with `X`'s, and the (implied) padding is marked with `P`'s. In this case, the convolution window 
 * starts in the padding at `(-1, -1, 0)` in @vector{X}'s coordinate space (i.e. @math{X[-1][-1]}) as is indicated 
 * by `P*` in the diagram. If `X_image` points to the start of the input image, then the argument `x` should 
 * be `(nn_image_t*) &X_image[-1][-1][0]`.
 * 
 * @inlinecode
 *  _ _ _      
 * |P*P P|P P P
 * |P X X|X X P
 * |P_X_X|X X P
 *  P X X X X P
 *  P X X X X P
 *  P P P P P P  
 *   
 * @endinlinecode
 * 
 * `k` points to the kernel tensor @tensor{K} with shape @tensor_shape{Y_c, K_h, K_w, X_c}. @math{Y_c} and 
 * @math{X_c} are the channel counts of the output and input images respectively. @math{K_h} and @math{K_w}
 * are the spatial height and width of the 2D convolution window respectively, expressed in pixels. Rather than
 * pointing at the start of @tensor{K}, the `k` passed to this function should point to the beginning of the
 * <b>final</b> channel of the output channel group being processed in the call.
 * 
 * For example, if @tensor{K} has shape @tensor_shape{60,5,5,32}, defined as `nn_tensor_t kernel[60][5][5][32]`,
 * then to process the output channel tail (output channels `48` through `59`), then the address passed to this
 * call as `k` should be `&kernel[59][0][0][0]`. The `KERNEL_4D_COG_LAST_CHAN_START()` macro can be used to 
 * simplify this. The memory layout of @tensor{K} is the standard memory layout for 4D tensors (see 
 * @ref standard_layout).
 * 
 * `bso` points to the `nn_bso_block_t` describing the biases, shifts and scale for the current group of output 
 * channels. `bso` can be initialized with `nn_standard_BSO_layout()`. See @ref bso_layout for more information.
 * 
 * `K_height` and `K_width` are the height and width respectively of the 2D convolution window.
 * 
 * `K_hori_stride` is the horizontal stride (in pixels) taken by the 2D convolution window between between
 * output pixels.
 * 
 * `C_in` (or @math{X_c}) is the number of channels in the input image @tensor{X}.
 * 
 * `pad_t` and `pad_b` are the number of rows of padding at the top and bottom of the convolution window 
 * respectively. Because this function computes a horizontal strip of output pixels, the padding at the
 * top and bottom of the convolution window does not change during one invocation. These values are non-negative, 
 * and so 'negative' amounts of top or bottom padding should be conveyed as zeros. See example below.
 * 
 * `pad_l_initial` and `pad_r_initial` are the number of columns of padding at the left and right sides of the 
 * convolution window at the begining of the operation. Because this function computes a horizontal strip of output
 * pixels, the amount of left and right padding may change during an invocation. Unlike `pad_t` and `pad_b`, 
 * `pad_l_initial` and `pad_r_initial` indicate the *signed* padding, where gaps between the convolution window
 * and the edge of the input image may be negative, as in the following example.
 * 
 * @inlinecode
 *            pad_r_initial = -3
 *         |-----|  
 *    _____       _
 *   |W W W|      | pad_t = max(1,0) = 1
 *   |W W W|X X X ̅
 *   |W W W|X X X
 *   |W_W_W|X X X _ pad_b = max(-2,0) = 0
 *      X X X X X |
 *      X X X X X |
 *   |-|          ̅
 *     pad_l_initial = 1    
 * @endinlinecode
 * 
 * The diagram above shows a `4x3` convolution window over a `5x5` image. In this case, `pad_t` is `1`, because the
 * top row of the convolution window is outside the input image. `pad_b` is `0`, because the bottom of the convolution
 * window is 2 pixels above the bottom of the image. `pad_l_initial` is `1`, because the left column of the 
 * convolution window is outside the input image. And `pad_r_initial` is `-3`, because if the convolution window
 * moves to the right 3 pixels, the edge of the convolution window will coincide with the edge of the input image.
 * Note also that `pad_r_initial` is *independent of* `K_hori_stride`; `pad_r_initial` is expressed as *input image*
 * pixels, not *output image* pixels.
 * 
 * `x_v_stride` is the memory stride required to move a pointer from (immediately after) the right edge of the 
 * convolution window to (immediately after) the left edge of the convolution window on the following row (note: 
 * "immediately after" because the convolution window's right and left edges are effectively *between* memory 
 * addresses). This is equivalent to @math{\left(X_w-K_w\right)\cdot X_c}.
 * 
 * `k_cout_stride` is the memory stride required to move a pointer from the beginning of one output channel of 
 * @tensor{K} to the output channel with index one *lower*, i.e. stride required to decrement the output channel
 * by one. This is equivalent to @math{-\left(K_h\cdot K_w\cdot X_c\right)}.
 * 
 * `y_h_stride` is the memory stride required to move one pixel to the right in @tensor{Y}. This is equivalent 
 * to @math{Y_c}.
 * 
 * `out_cols` is the number of output pixels for each of which @ttref{VPU_INT8_ACC_PERIOD} channels are to be computed.
 * 
 * `zero_point_vec` is a pointer to an `int8_t` buffer with length @ttref{VPU_INT8_EPV}. This is supplied by
 * the user as a vector to save time (it allows the same vector to be reused, rather than copying a single
 * `int8_t` value several times).
 * 
 * `C_out_tail` is the tail of the output channel count. The value given should be @math{left(Y_c mod 16\right}.
 * 
 * @requires_word_alignment{y,x,k,bso}
 * 
 * @par Requirements and Constraints
 *   - `C_in` must be a multiple of `4`.
 *   - `y_h_stride` must be a multiple of `4`.
 *   - `C_out_tail` must be a multiple of `4`.
 * 
 * @param[out] y                    Pointer to output image @tensor{Y}
 * @param[in]  x                    Pointer to input image @tensor{X}
 * @param[in]  k                    The kernel tensor @tensor{K}
 * @param[in]  bso                  The bias-scale-offset parameters
 * @param[in]  K_height             Kernel height @math{K_h} (in pixels)
 * @param[in]  K_width              Kernel width @math{K_w} (in pixels)
 * @param[in]  K_hori_stride        Horizontal stride of the convolution window (in pixels)
 * @param[in]  C_in                 Number of input channels, @math{X_c}
 * @param[in]  pad_t                (unsigned) Number of rows of padding at the top of the convolution window
 * @param[in]  pad_b                (unsigned) Number or rows of padding at the bottom of the convolution window
 * @param[in]  pad_l_initial        (signed) Number of columns of padding at the left of the convolution window (at 
 *                                  its initial location)
 * @param[in]  pad_r_initial        (signed) Number of columns of padding at the right of the convolution window (at
 *                                  its initial location)
 * @param[in]  x_v_stride           Stride to move from the right edge of a patch to the left-most pixel 
 *                                  of the following row (in bytes)
 * @param[in]  k_cout_stride        
 * @param[in]  y_h_stride           Stride to move the output pointer from one pixel to the next (in bytes)
 * @param[in]  out_cols             Number of output pixels to write.
 * @param[in]  zero_point_vec       Vector containing `VPU_INT8_EPV` copies of the zero point value.
 * @param[in]  C_out_tail           The tail of the output channel count.
 */
void nn_conv2d_hstrip_tail_deep_padded(
        nn_image_t* Y,
        const nn_image_t* X,
        const nn_tensor_t* K,    
        const nn_bso_block_t* BSO,
        const unsigned K_height,
        const unsigned K_width,
        const unsigned K_hori_stride,
        const channel_count_t C_in,
        const unsigned pad_t,
        const unsigned pad_b,
        const int pad_l_initial,
        const int pad_r_initial,
        const mem_stride_t x_v_stride,
        const mem_stride_t k_cout_stride,
        const mem_stride_t y_h_stride,
        const unsigned out_cols,
        const int8_t* zero_point_vec,
        const channel_count_t C_out_tail);

void nn_conv2d_hstrip_deep(
        nn_image_t* Y,
        const nn_image_t* X,
        const nn_tensor_t* K,    
        const nn_bso_block_t* BSO,
        const unsigned K_height,
        const unsigned K_width,
        const unsigned K_hori_stride,
        const channel_count_t C_in,
        const mem_stride_t x_v_stride,
        const mem_stride_t k_cout_stride,
        const mem_stride_t y_h_stride,
        const unsigned out_cols);

void nn_conv2d_hstrip_tail_deep(
        nn_image_t* Y,
        const nn_image_t* X,
        const nn_tensor_t* K,    
        const nn_bso_block_t* BSO,
        const unsigned K_height,
        const unsigned K_width,
        const unsigned K_hori_stride,
        const channel_count_t C_in,
        const mem_stride_t x_v_stride,
        const mem_stride_t k_cout_stride,
        const mem_stride_t y_h_stride,
        const unsigned out_cols,
        const channel_count_t C_out_tail);

void nn_conv2d_hstrip_shallowin_padded(
        nn_image_t* Y,
        const nn_image_t* X,
        const nn_tensor_t* K,
        const nn_bso_block_t* BSO,
        const unsigned K_h,
        const unsigned K_h_stride,
        const channel_count_t C_in,
        const unsigned pad_t,
        const unsigned pad_b,
        const int pad_l_initial,
        const int pad_r_initial,
        const mem_stride_t x_v_stride,
        const mem_stride_t y_h_stride,
        const unsigned out_cols,
        const int8_t* zero_point_vec);



void nn_conv2d_hstrip_shallowin(
        nn_image_t* Y,
        const nn_image_t* X,
        const nn_tensor_t* K,
        const nn_bso_block_t* BSO,
        const unsigned K_h,
        const unsigned K_h_stride,
        const channel_count_t C_in,
        const mem_stride_t x_v_stride,
        const mem_stride_t y_h_stride,
        const unsigned out_cols);



void nn_conv2d_hstrip_tail_shallowin_padded(
        nn_image_t* Y,
        const nn_image_t* X,
        const nn_tensor_t* K,
        const nn_bso_block_t* BSO,
        const unsigned K_h,
        const unsigned K_h_stride,
        const channel_count_t C_in,
        const unsigned pad_t,
        const unsigned pad_b,
        const int pad_l_initial,
        const int pad_r_initial,
        const mem_stride_t x_v_stride,
        const mem_stride_t y_h_stride,
        const unsigned out_cols,
        const int8_t* zero_point_vec,
        const channel_count_t C_out_tail);



void nn_conv2d_hstrip_tail_shallowin(
        nn_image_t* Y,
        const nn_image_t* X,
        const nn_tensor_t* K,
        const nn_bso_block_t* BSO,
        const unsigned K_h,
        const unsigned K_h_stride,
        const channel_count_t C_in,
        const mem_stride_t x_v_stride,
        const mem_stride_t y_h_stride,
        const unsigned out_cols,
        const channel_count_t C_out_tail);


    