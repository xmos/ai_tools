

#ifndef NN_OP_STRUCTS_H_
#define NN_OP_STRUCTS_H_


#include "nn_operator.h"
#include "xs3_vpu.h"

#ifdef __XC__
extern "C" {
#endif




/**
 * This struct represents an indexing vector for an image.
 */
typedef struct {
    int32_t rows;
    int32_t cols;
    int32_t channels;
} nn_image_vect_t;


typedef struct {
    int32_t rows;
    int32_t cols;
} nn_index_vector2d_t;

/**
 * Macro returns the number of `nn_bso_block_t`s required for `OUT_CHANS` output channels. This is
 * the same as the number output channel groups, rounded up.
 * 
 * @param[in] OUT_CHANS     Number of output channels
 * 
 * @return  Number of required `nn_bso_block_t`.
 */
#define BSO_BLOCK_COUNT(OUT_CHANS) ((OUT_CHANS+(VPU_INT8_VLMACC_ELMS-1))>>VPU_INT8_VLMACC_ELMS_LOG2)

/**
 * Represents the Bias, shifts and scale for a single output channel group.
 */
typedef struct {
    data16_t bias_hi[VPU_INT8_ACC_PERIOD];
    data16_t bias_lo[VPU_INT8_ACC_PERIOD];
    data16_t shift1[VPU_INT8_ACC_PERIOD];
    data16_t scale[VPU_INT8_ACC_PERIOD];
    data16_t offset_scale[VPU_INT8_ACC_PERIOD];
    data16_t offset[VPU_INT8_ACC_PERIOD];
    data16_t shift2[VPU_INT8_ACC_PERIOD];
} nn_bso_block_t;




typedef struct {
    struct {
        channel_count_t X;
    } channels;
} nn_fully_connected_plan_t;

/**
 * 
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


typedef struct {
    uint32_t start_channel;
    channel_count_t out_channels;
} nn_fully_connected_job_params_t;




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
     * The initial position of the convolution window, relative to the input image.
     * 
     * The position given by this pair indicates where the top-left pixel of the convolution
     * window begins relative to the top-left pixel of the input image. 
     * 
     * If this pair is, for example, `(0, 0)`, then the convolution window starts at the top 
     * left of the input image and involves no top or left padding.
     */
    struct {
        /** Row offset of convolution window inital position */
        int row;
        /** Column offset of convolution window inital position */
        int column;
    } start;

    /**
     * The strides of the convolution window. These are the number of (input image) pixels that
     * the convolution window moves down and right for each pixel moved down or right in the
     * output image.
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
 * Enum identifies optimized assembly implementations for
 * the `avgpool2d()` function.
 */
typedef enum {
    AVGPOOL2D_DEFAULT = 0,    // General case, uses `avgpool2d_asm()`
    AVGPOOL2D_2X2     = 1,    //  Typical 2x2 average pool. Uses `avgpool2d_2x2_asm()`
} nn_avgpool2d_impl_t;

/**
 * Struct represents the parameters needed by the avgpool2d() funciton.
 * Values are set by avgpool2d_init().
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
 * 
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
 * 
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
 * 
 */
typedef nn_pool2d_job_t nn_maxpool2d_job_t;




/**
 * 
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
}   //extern "C"
#endif

#endif //NN_OP_STRUCTS_H_