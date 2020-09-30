#ifndef NN_CONV2D_STRUCTS_H_
#define NN_CONV2D_STRUCTS_H_

#include "nn_types.h"
#include "nn_window_params.h"

#ifdef __XC__
extern "C" {
#endif

/**
 * Macro to get the address of the start of the last output channel of the `COG`<sup>th</sup> output
 * channel group of 4D kernel tensor `KRN`.
 * 
 * @param KRN[in]   4D kernel tensor
 * @param COG[in]   Output channel group
 * 
 * @return  Address of start of last output channel of `COG`<sup>th</sup> output channel group.
 */
#define KERNEL_4D_COG_LAST_CHAN_START(KRN, COG)    ((nn_tensor_t*) &(KRN[(VPU_INT8_ACC_PERIOD*(COG))+(VPU_INT8_ACC_PERIOD-1)][0][0][0]))


typedef nn_window_op_job_params_t nn_conv2d_job_params_t;

/**
 * Struct represents the shared parameters required to execute a `conv2d_deep()` operation. 
 */
typedef struct {

    struct {
        struct {
            mem_stride_t row;
        } X;

        // struct {
        //     mem_stride_t col;
        // } window;

        struct {
            mem_stride_t cout;
        } K;

    } stride;

    struct {
        struct {
            unsigned height;
            unsigned width;
        } shape;

        struct {
            int vertical;
            int horizontal;
        } stride;
    } window;

    struct {
        uint32_t X;
        uint32_t Y;
    } channels;

    int32_t zero_point;

} nn_conv2d_deep_plan_t;

/**
 * Struct represents the job-specific parameters required to execute a `conv2d_deep()` operation. 
 */
typedef struct {

    struct {
        struct {
            int32_t X;
            int32_t Y;
            int32_t K;
            int32_t BSO;
        } start;

        struct {
            int32_t Y;
        } chan_group;

        struct {
            int32_t window;
            int32_t Y;
        } row;
    } stride;

    struct {
        unsigned rows;
        unsigned cols;
        unsigned channels;
    } output;

    struct {
        int32_t top;
        int32_t left;
        int32_t bottom;
        int32_t right;
    } init_padding;
} nn_conv2d_deep_job_t;




/**
 * Struct represents the shared parameters required to execute a `conv2d_deep()` operation. 
 */
typedef struct {

    struct {
        struct {
            mem_stride_t row;
        } X;

    } stride;

    struct {
        struct {
            unsigned height;
            unsigned width;
        } shape;

        struct {
            int vertical;
            int horizontal;
        } stride;
    } window;

    struct {
        uint32_t X;
        uint32_t Y;
    } channels;

    int32_t zero_point;

} nn_conv2d_shallowin_plan_t;

/**
 * Struct represents the job-specific parameters required to execute a `conv2d_deep()` operation. 
 */
typedef struct {

    struct {
        struct {
            int32_t X;
            int32_t Y;
            int32_t K;
            int32_t BSO;
        } start;

        struct {
            int32_t Y;
        } chan_group;

        struct {
            int32_t window;
            int32_t Y;
        } row;
    } stride;

    struct {
        unsigned rows;
        unsigned cols;
        unsigned channels;
    } output;

    struct {
        int32_t top;
        int32_t left;
        int32_t bottom;
        int32_t right;
    } init_padding;
} nn_conv2d_shallowin_job_t;


/**
 * Struct represents the shared parameters required to execute a `conv2d_im2col()` operation. 
 */
typedef struct {

    struct {
        struct {
            mem_stride_t row;
        } X;

    } stride;

    struct {
        struct {
            unsigned height;
            unsigned width;
            unsigned len_col;
            unsigned kernel_row_elements;
        } shape;

        struct {
            int vertical;
            int horizontal;
        } stride;
    } window;

    struct {
        uint32_t X;
        uint32_t Y;
    } channels;

    int32_t zero_point;

} nn_conv2d_im2col_plan_t;

/**
 * Struct represents the job-specific parameters required to execute a `conv2d_im2col()` operation. 
 */
typedef struct {

    struct {
        struct {
            int32_t X;
            int32_t Y;
            int32_t K;
            int32_t BSO;
        } start;

        struct {
            int32_t Y;
            int32_t K;
        } chan_group;

        struct {
            int32_t Y;
        } col;

        struct {
            int32_t window;
            int32_t Y;
            int32_t K;
        } row;
    } stride;

    struct {
        unsigned rows;
        unsigned cols;
        unsigned channels;
    } output;

    struct {
        int32_t top;
        int32_t left;
        int32_t bottom;
        int32_t right;
    } init_padding;
} nn_conv2d_im2col_job_t;


/**
 * Struct represents the shared parameters required to execute a `conv2d_1x1()` operation. 
 */
typedef struct {
    
    struct {
        uint32_t X;
        uint32_t Y;
    } channels;

} nn_conv2d_1x1_plan_t;

/**
 * Struct represents the job-specific parameters required to execute a `conv2d_1x1()` operation. 
 */
typedef struct {

    struct {
        int32_t X;
        int32_t Y;
        int32_t K;
        int32_t BSO;
    } start;

    struct {
        unsigned pixels;
        unsigned channels;
    } output;

} nn_conv2d_1x1_job_t;


/**

 */
typedef struct {
    /** 
     * Indices in an output image at which to begin producing output.
     * 
     * Typically channels must be a multiple of 16.
     */
    nn_image_vect_t start;

    /**
     * The number of pixels and channels of output to produce.
     * 
     * Whereas in `nn_conv2d_job_params_t` a number of rows and columns of the output image can be specified to 
     * be computed, in `conv2d_1x1()` a number of pixels is specified instead. Starting at the start position,
     * a job will continue computing outputs until the end of the row of the output image, and then move to the
     * first column of the following row.
     * 
     * Typically channels must be a multiple of 4.
     */
    struct {
        uint32_t pixels;
        uint32_t channels;
    } size;

} nn_conv2d_1x1_job_params_t;


/**
 * Flags used with conv2d_depthwise_adv() for advanced scenarios.
 */
typedef enum {
    /** 
     * If non-zero, this flag signals to conv2d_depthwise_adv() that the supplied kernel weight tensor (@tensor{K}) and 
     * the BSO tensor are slices, rather than the full tensors.
     * 
     * If this is set, then conv2d_depthwise_adv() will treat @tensor{K} and BSO as if they contain _only the 
     * necessary channels for the job being invoked_.
     * 
     * When loading data from flash or external RAM, this can be used to decrease peak SRAM usage.
     * 
     * For example, if a @oper{conv2d_depthwise} instance performs a 5x3 convolution on an input image with 40 channels,
     * normally the shape of @tensor{K} would normally be @math{(5,3,40)}, and the shape of BSO would normally
     * be @math{(3)} for any job invoked.
     * 
     * However, if a particular job is calculating only output channels 16 through 31 (inclusive), then if this flag is 
     * set, conv2d_depthwise_adv() will interpret the supplied argument `K` as pointing to @math{\bar K[:,:,16:32]} 
     * (using Python-style array slicing notation), and the supplied `BSO` as pointing to @math{BSO[2]} (i.e. the
     * second `nn_bso_block_t`, corresponding to channels 16 through 31.)
     * 
     * @note Each block of the BSO tensor always corresponds to 16 channels. Even if a job invocation is only computing
     *       4 output channels, a full 16-channel `nn_bso_block_t` must be provided.
     */
    CONV2D_DEPTHWISE_FLAG_SLICED_K = (1<<0),
} nn_conv2d_depthwise_flags_e;

#ifdef __XC__
}   //extern "C"
#endif

#endif //NN_CONV2D_STRUCTS_H_
