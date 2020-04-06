

#ifndef NN_CONV2D_STRUCTS_H_
#define NN_CONV2D_STRUCTS_H_


#include "nn_operator.h"
#include "xs3_vpu.h"
#include "nn_op_structs.h"

#ifdef __XC__
extern "C" {
#endif

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
} nn_conv2d_window_params_t;





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

} nn_conv2d_job_params_t;




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
            int32_t BSS;
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
* 
*/
typedef struct {

    struct {

        struct {
            int32_t X;
            int32_t Y;
            int32_t K;
        } start_offset;

        data16_t* biases;

    } init;

    struct {
        int32_t K;
    } cout_group_incr;

    struct {
        unsigned rows;
        unsigned cols;


        //Bytes the pointer needs to be incremented by to move to handle moving to a new
        //  output image row
        struct {
            int32_t X;
            int32_t Y;
        } row_incr;
    } output;



    struct {
        uint32_t pad_mask;

        unsigned rows;

        struct {
            int32_t X;
            int32_t K;
        } row_incr;


    } patch;

} nn_conv2d_sido_block_params_t;

/**
*
*/
typedef struct {

    unsigned block_count;

    unsigned chans_in;
    unsigned chans_out;
    unsigned C_in_groups;
    unsigned C_out_groups;
    int32_t zero_point;

    nn_conv2d_sido_block_params_t* blocks;

} nn_conv2d_sido_params_t;



typedef struct {
    uint32_t X_height;
    uint32_t X_width;
    uint32_t K_h;
    uint32_t K_w;
    uint32_t C_in;
    uint32_t C_out;
    padding_mode_t pad_mode;
    int8_t zero_point;
} nn_conv2d_init_params_t;



typedef struct {
    uint32_t top;
    uint32_t left;
    uint32_t rows;
    uint32_t cols;
} nn_conv2d_region_params_t;





typedef struct {
    struct {
        int32_t X;
        int32_t Y;
        int32_t K;
    } start_stride;
    
    struct {
        int32_t Y;
        int32_t K;
    } cog_stride;

    struct {
        int32_t body;
        int32_t tail;
    } cig_stride;

    uint32_t pix_count;
    uint32_t C_in;
    uint32_t C_out;

} nn_conv2d_1x1_plan_t;


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
            int32_t BSS;
        } start;

        struct {
            int32_t X;
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
        unsigned unpadded;
    } init_padding;
} nn_conv2d_depthwise_job_t;




#ifdef __XC__
}   //extern "C"
#endif

#endif //NN_CONV2D_STRUCTS_H_