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
 * Struct represents the shared parameters required to execute a 
 * `conv2d_depthwise()` operation. 
 */
typedef struct {

    struct {
        struct {
            int32_t row;
        } X;

        struct {
            int32_t col;
        } window;

    } stride;

    struct {
        unsigned height;
        unsigned width;
        int vstride; //TODO: get rid of this
    } kernel;

    struct {
        uint32_t X;
        uint32_t Y;
    } channels;

    int32_t zero_point;

} nn_conv2d_depthwise_plan_t;

/**
 * Struct represents the job-specific parameters required to execute a 
 * `conv2d_depthwise()` operation. 
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
            int32_t X;
            int32_t Y;
        } chan_group;

        struct {
            int32_t window;
            int32_t Y;
        } row;

        int32_t k_channels;

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
        unsigned unpadded;
    } init_padding;
} nn_conv2d_depthwise_job_t;




#ifdef __XC__
}   //extern "C"
#endif

#endif //NN_CONV2D_STRUCTS_H_
