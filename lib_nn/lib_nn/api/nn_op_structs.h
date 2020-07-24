

#ifndef NN_OP_STRUCTS_H_
#define NN_OP_STRUCTS_H_

#include "nn_operator.h"
#include "xs3_vpu.h"
#include "nn_bin_types.h"

#ifdef __XC__
extern "C" {
#endif

/**
 * This struct represents an indexing vector for an image.
 */
typedef struct {
  /** Number of image pixel rows */
  int32_t rows;
  /** Number of image pixel columns */
  int32_t cols;
  /** Number of image pixel channels */
  int32_t channels;
} nn_image_vect_t;

/**
 * Macro returns the number of `nn_bso_block_t`s required for `OUT_CHANS` output
 * channels. This is the same as the number output channel groups, rounded up.
 *
 * @param[in] OUT_CHANS     Number of output channels
 *
 * @return  Number of required `nn_bso_block_t`.
 */
#define BSO_BLOCK_COUNT(OUT_CHANS) \
  ((OUT_CHANS + (VPU_INT8_VLMACC_ELMS - 1)) >> VPU_INT8_VLMACC_ELMS_LOG2)

/**
 * Represents the Bias, shifts and scale for a single output channel group.
 *
 */
typedef struct {
  /**
   * Contains the upper 16-bits of output channel bias for an operator for (up
   * to) 16 channels.
   *
   * The full 32-bit bias for an output channel corresponding to index `k` is:
   *
   * @math{ B_{hi}[k]\cdot 2^{16} + B_{lo}[k] } where @math{ B_{hi}[k] } is
   * `bias_hi[k]` interpreted as a signed 16-bit integer, and @math{B_{lo}[k]}
   * is `bias_lo[k]` interpreted as an unsigned 16-bit integer.
   */
  data16_t bias_hi[VPU_INT8_ACC_PERIOD];

  /**
   * Contains the lower 16-bits of output channel bias for an operator for (up
   * to) 16 channels.
   *
   * The full bias for an output channel corresponding to index `k` is:
   *
   * @math{ B_{hi}[k]\cdot 2^{16} + B_{lo}[k] } where @math{ B_{hi}[k] } is
   * `bias_hi[k]` interpreted as a signed 16-bit integer, and @math{B_{lo}[k]}
   * is `bias_lo[k]` interpreted as an unsigned 16-bit integer.
   */
  data16_t bias_lo[VPU_INT8_ACC_PERIOD];

  /**
   * Contains the first shift value for an operator for (up to) 16 channels.
   *
   * After accumulating all weights and input data, the channel corresponding to
   * index `k` is first divided by @math{ 2^{s_1[k]} }, where @math{s_1[k]} is
   * `shift1[k]`.
   */
  data16_t shift1[VPU_INT8_ACC_PERIOD];

  /**
   * Contains the scale value for an operator for (up to) 16 channels.
   *
   * After applying the first shift, the result of that is multiplied by
   * `scale[k]`.
   *
   */
  data16_t scale[VPU_INT8_ACC_PERIOD];

  /**
   * `offset_scale[k]` and `offset[k]` are multiplied together and added to the
   * result of applying the scale.
   */
  data16_t offset_scale[VPU_INT8_ACC_PERIOD];

  /**
   * `offset_scale[k]` and `offset[k]` are multiplied together and added to the
   * result of applying the scale.
   */
  data16_t offset[VPU_INT8_ACC_PERIOD];

  /**
   * Contains the second shift value for an operator for (up to) 16 channels.
   *
   * After the offset and offset scale are added, the channel corresponding to
   * index `k` is divided by @math{ 2^{s_2[k]} }, where @math{s_2[k]} is
   * `shift2[k]`.
   */
  data16_t shift2[VPU_INT8_ACC_PERIOD];

} nn_bso_block_t;

/**
 * Describes the relationship between the convolution window and the
 * input image.
 */
typedef struct {
  /** The shape of the convolution window */
  struct {
    /** Height of the convolution window in pixels */
    unsigned height;
    /** Width of the convolution window in pixels */
    unsigned width;
  } shape;

  /**
   * The initial position of the convolution window, relative to the input
   * image.
   *
   * The position given by this pair indicates where the top-left pixel of the
   * convolution window begins relative to the top-left pixel of the input
   * image.
   *
   * If this pair is, for example, `(0, 0)`, then the convolution window starts
   * at the top left of the input image and involves no top or left padding.
   */
  struct {
    /** Row offset of convolution window inital position */
    int row;
    /** Column offset of convolution window inital position */
    int column;
  } start;

  /**
   * The strides of the convolution window. These are the number of (input
   * image) pixels that the convolution window moves down and right for each
   * pixel moved down or right in the output image.
   */
  struct {
    /** Vertical stride of the convolution window. */
    int vertical;
    /** Horizontal stride of the convolution window */
    int horizontal;
  } stride;
} nn_window_params_t;

/**
 * Some of the functions in this API can have their work split into
 * multiple parts, each called a "job". This is useful, for example,
 * for parallelizing a computation across multiple cores, or to reduce
 * the delay between which the calling function can service some other
 * resource.
 *
 * This struct contains the parameters required to specify how work
 * can be split according to the region of the output image in which
 * the job operates.
 *
 * For an output image `Y` with shape `(Y_height, Y_width, Y_chans)`,
 * the value of this struct indicates that a particular job should
 * compute the output values in the rectangular region
 *   `Y[  start.rows     : (start.rows + size.rows),
 *        start.cols     : (start.cols + size.cols),
 *        start.channels : (start.channels + size.channels) ]`
 */
typedef struct {
  /**
   * Indices in an output image at which to begin producing output.
   *
   * Typically channels must be a multiple of 4.
   */
  nn_image_vect_t start;

  /**
   * The number of rows, columns and channels of output to produce.
   *
   * Typically channels must be a multiple of 4.
   */
  nn_image_vect_t size;

} nn_window_op_job_params_t;

/**
 * Struct represents the parameters needed by all `fully_connected_16()` jobs.
 *
 * Values are set by `fully_connected_16_init()`.
 *
 * @note This struct is intended to be opaque.
 */
typedef struct {
  struct {
    channel_count_t X;
  } channels;
} nn_fully_connected_plan_t;

/**
 * Struct represents the parameters needed by a single `fully_connected_16()`
 * job.
 *
 * Values are set by `fully_connected_16_init()`.
 *
 * @note This struct is intended to be opaque.
 */
typedef struct {
  struct {
    struct {
      mem_stride_t Y;
      mem_stride_t W;
      mem_stride_t BSO;
    } start;
  } stride;

  struct {
    channel_count_t channels;
  } output;
} nn_fully_connected_job_t;

/**
 * Struct represents the job initialization information required by
 * `fully_connected_16_init()`.
 *
 * `fully_connected_16()` job computes a contiguous subset of the output
 * channels.
 *
 * @note When splitting a `fully_connected_16()` into multiple jobs, jobs that
 * compute less than 16 output channels will often be *less* efficient than a
 * full 16 channels.
 */
typedef struct {
  /**
   * The first output channel to be computed by the job. Must be a multiple of
   * `4`.
   */
  uint32_t start_channel;

  /**
   * The number of output channels to be computed by the job. Does not have to
   * be a multiple of 4, however, because the `start_channel` for each job must
   * be a multiple of 4, this value can only be a non-multiple of 4 for the last
   * job.
   */
  channel_count_t out_channels;
} nn_fully_connected_job_params_t;

/**
 * Enum identifies optimized assembly implementations for
 * the `avgpool2d()` function.
 */
typedef enum {
  AVGPOOL2D_DEFAULT = 0,  // General case, uses `avgpool2d_asm()`
  AVGPOOL2D_2X2 = 1,  //  Typical 2x2 average pool. Uses `avgpool2d_2x2_asm()`
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
 * Struct represents the parameters needed by each @oper{avgpool2d_global} job.
 *
 * Values are set by avgpool2d_global_init().
 *
 * @note This struct is intended to be opaque.
 */
typedef struct {
  struct {
    uint32_t pixels;
    channel_count_t channels;
  } X;
  uint32_t shift;
  uint32_t scale;
} nn_avgpool2d_global_plan_t;

/**
 * Struct represents the parameters needed by a single @oper{avgpool2d_global}
 * job.
 *
 * Values are set by avgpool2d_global_init().
 *
 * @note This struct is intended to be opaque.
 */
typedef struct {
  mem_stride_t start_stride;
  channel_count_t out_channels;
} nn_avgpool2d_global_job_t;

/**
 * Struct represents the job initialization information required by
 * avgpool2d_global_init().
 *
 * @oper{avgpool2d_global} job computes a contiguous subset of the output
 * channels.
 *
 */
typedef struct {
  /**
   * The first output channel to be computed by the job. Must be a multiple of
   * `4`.
   */
  channel_count_t start_channel;

  /**
   * The number of output channels to be computed by the job. Does not have to
   * be a multiple of 4, however, because the `start_channel` for each job must
   * be a multiple of 4, this value can only be a non-multiple of 4 for the last
   * job.
   */
  channel_count_t out_channels;
} nn_avgpool2d_global_job_params_t;

/**
 * Struct represents the parameters needed by a single `avgpool2d()` or
 * `maxpool2d()` job.
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
 * Struct represents the parameters needed by each `maxpool2d()` job.
 *
 * Values are set by `maxpool2d_init()`.
 *
 * @note This struct is intended to be opaque.
 */
typedef struct {
  struct {
    uint32_t rows;
    uint32_t cols;
  } window;

  struct {
    channel_count_t X;
    channel_count_t Y;
  } channels;

} nn_maxpool2d_plan_t;

/**
 * Struct represents the parameters needed by each `bnn_conv2d()` job.
 *
 * Values are set by `bnn_conv2d_init()`.
 *
 * @note This struct is intended to be opaque.
 */
typedef struct {
  unsigned outer_x_h_step;
  unsigned output_channel_loop_counter;
  void * threshold_p;
  unsigned inner_x_v_step;
  unsigned k_v_step;
  unsigned inner_x_h_step;
  unsigned k_h_step;
  int outer_x_v_step;
  unsigned y_v_step;
  unsigned k_height_loop_counter;
  unsigned k_width_loop_counter;
  unsigned x_height_loop_counter;
  unsigned x_width_loop_counter;
  unsigned input_channel_loop_counter;
} nn_bnn_conv2d_bin_out_asm_plan_t;

/**
 * Struct represents the parameters needed by each `bnn_conv2d()` job.
 *
 * Values are set by `bnn_conv2d_init()`.
 *
 * @note This struct is intended to be opaque.
 */
typedef struct {
  unsigned y_dims[3];     // out_height, out_width, out_channels
  unsigned x_dims[3];     // in_height, in_width, in_channels
  unsigned k_dims[2];     // kernel_height, kernel_width
  unsigned start_loc[2];  // start_height, start_width
  unsigned stride[2];     // stride_height, stride_width
  // int8_t clamp_min;
  // int8_t clamp_max;
} nn_bnn_conv2d_bin_out_plan_t;

// /**
//  * Struct represents the parameters needed by each `bnn_conv2d()` job.
//  *
//  * Values are set by `bnn_conv2d_init()`.
//  *
//  * @note This struct is intended to be opaque.
//  */
// typedef struct {
//   unsigned y_dims[3];     // out_height, out_width, out_channels
//   unsigned x_dims[3];     // in_height, in_width, in_channels
//   unsigned k_dims[2];     // kernel_height, kernel_width
//   unsigned start_loc[2];  // start_height, start_width
//   unsigned stride[2];     // stride_height, stride_width
// } nn_bnn_conv2d_bin_out_ref_plan_t;

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
 * This struct describes the basic parameters for an image tensor
 */
typedef struct {
  /**
   * Height of an image (in pixels)
   */
  uint32_t height;
  /**
   * Width of the image (in pixels)
   */
  uint32_t width;
  /**
   * Number of channels per pixel
   */
  channel_count_t channels;
} nn_image_params_t;

#ifdef __XC__
}  // extern "C"
#endif

#endif  // NN_OP_STRUCTS_H_