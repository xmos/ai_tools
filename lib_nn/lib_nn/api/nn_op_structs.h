

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
 * Macro returns the number of `nn_bss_block_t`s required for `OUT_CHANS` output channels. This is
 * the same as the number output channel groups, rounded up.
 * 
 * @param[in] OUT_CHANS     Number of output channels
 * 
 * @return  Number of required `nn_bss_block_t`.
 */
#define BSS_BLOCK_COUNT(OUT_CHANS) ((OUT_CHANS+(VPU_INT8_VLMACC_ELMS-1))>>VPU_INT8_VLMACC_ELMS_LOG2)

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
} nn_bss_block_t;





typedef enum {
    FC16_DEFAULT    = 0,
} nn_fc16_tail_strat_t;

typedef struct {
    int32_t c_in;
    int32_t c_out;
    int32_t cig_end_stride;
    nn_fc16_tail_strat_t tail_strat;
} nn_fully_connected_plan_t;




/**
 * This struct describes the mapping of an input window to outputs in operators where each output pixel 
 * is computed over a 2D input winow of pixels in the input image, and where each output pixel corresponds
 * to a different input window location on the input image.
 * 
 * These are provided to functions such as `avgpool2d_init()` and `maxpool2d_init()` at initialization 
 * time to compute an execution plan for when their corresponding operations are performed.
 * 
 * This struct can be used to describe a wide variety of mappings between output pixels and input pixels.
 * The pseudo-code below describes in detail the model for how the various fields in this configuration 
 * are used.
 * 
 * A helper function `nn_window_op_config_simple()` is provided to initialize a `nn_window_op_config_t` 
 *  for typical behaviors.
 * 
 * \code
 * 
 *  const nn_image_params_t* x_img_params;    //X tensor dimensions
 *  const nn_image_params_t* y_img_params;    //Y tensor dimensions
 *  const nn_window_op_config_t* config;      //config model
 *  
 *  //Input image
 *  int8_t X[x_img_params->height][x_img_params->width][x_img_params->channels];
 *  //Output image
 *  int8_t Y[y_img_params->height][y_img_params->width][y_img_params->channels];
 *  
 *  // Input window is evaluated once for each channel and each output pixel.
 * 
 *  for(unsigned out_row = 0; out_row < config->output.shape.height; out_row++) {
 *      for(unsigned out_col = 0; out_col < config->output.shape.width; out_col++) {
 *          for(unsigned out_chan = 0; out_chan < config->output.shape.channels; out_chan++){
 * 
 *              // The values from X added to input_window here will be those that contribute 
 *              //  to the next output.
 *              int8_t input_window[config->window.shape.height][config->window.shape.width];
 * 
 *              for(unsigned win_row = 0; win_row < config->window.shape.height; win_row++){
 *                  for(unsigned win_col = 0; win_col < config->window.shape.width; win_col++){
 *                      
 *                      int x_row  = config->window.start.rows                            // the horizontal strides CAN (but usually won't) contribute to the row
 *                                  + out_row * config->window.outer_stride.vertical.rows + out_col * config->window.outer_stride.horizontal.rows
 *                                  + win_row * config->window.inner_stride.vertical.rows + win_col * config->window.inner_stride.horizontal.rows;
 * 
 *                      int x_col  = config->window.start.cols // the vertical strides CAN (but usually won't) contribute to the col
 *                                  + out_row * config->window.outer_stride.vertical.cols + out_col * config->window.outer_stride.horizontal.cols
 *                                  + win_row * config->window.inner_stride.vertical.cols + win_col * config->window.inner_stride.horizontal.cols;
 * 
 *                      int x_chan = config->window.start.channels
 *                                  + out_row * config->window.outer_stride.vertical.channels + out_col * config->window.outer_stride.horizontal.channels
 *                                  + win_row * config->window.inner_stride.vertical.channels + win_col * config->window.inner_stride.horizontal.channels
 
 *                      input_window[win_row][win_col] = X[x_row][x_col][x_chan];
 * 
 *                  }
 *              }
 *              
 *              unsigned y_row  = config->output.start.rows     + out_row * config->output.stride.vertical.rows     + out_col * config->output.stride.horizontal.rows;
 *              unsigned y_col  = config->output.start.cols     + out_row * config->output.stride.vertical.cols     + out_col * config->output.stride.horizontal.cols;
 *              unsigned y_chan = config->output.start.channels + out_row * config->output.stride.vertical.channels + out_col * config->output.stride.horizontal.channels
 *                                + out_chan;
 *              
 *              Y[y_row][y_col][y_chan] =  compute_output(input_window, config->window.shape.height, config->window.shape.width);
 *          }
 *      }
 *  }
 * \endcode
 * 
 * NOTE: The fields in this struct are named with the assumption that typical behavior is desired.
 *       e.g. rectangular input window and rectangular output region.
 * 
 * NOTE: The model makes the mapping of inputs to outputs look complicated. However, under 
 *       normal circumstances, each of the following will be true:
 *          `window.outer_stride.vertical.cols == 0`
 *          `window.outer_stride.horizontal.rows == 0`
 *          `window.outer_stride.vertical.channels == 0`
 *          `window.outer_stride.horizontal.channels == 0`
 *          `window.inner_stride.vertical.cols == 0`
 *          `window.inner_stride.horizontal.rows == 0`
 *          `window.inner_stride.vertical.channels == 0`
 *          `window.inner_stride.horizontal.channels == 0`
 *          `output.stride.vertical.cols == 0`
 *          `output.stride.horizontal.rows == 0`
 *          `output.stride.vertical.channels == 0`
 *          `output.stride.horizontal.channels == 0`
 *       When each of those is true, the model greatly simplifies, such that vertical iterations
 *       only interact with rows, and horizontal iterations only interact with cols.
 * 
 *       Additionally, the following will also usually be true:
 *          `window.inner_stride.vertical.rows == 1`
 *          `window.inner_stride.horizontal.cols == 1`
 *          `output.stride.vertical.rows == 1`
 *          `output.stride.horizontal.rows == 1`
 *       When these are also true, the operator behaves in the usual way, where outputs form
 *       a rectangualr sub-tensor in the output image, and where each pixel is mapped from
 *       a rectangular sub-tensor in the input image.
 * 
 * NOTE: In the descriptions for fields, vectors are described as a 3-tuple `[r, c, ch]`, 
 * where 3 elements correspond to index changes in a 3-dimensional image tensor.
 *       `r` is a row delta, `c` is a column delta and `ch` is a channel delta.
 */
typedef struct {

    struct {

        /**
         * Vector (in output image coordinate space) to the first output pixel to be written.
         * 
         * Typically this is `[0, 0, 0]` if the whole output image is to be computed.
         */
        nn_image_vect_t start;

        /**
         * The shape of the output region.
         * 
         * NOTE: This is *not* necessarily the shape of the output image.
         */
        struct {
           /**
            * The number of rows of pixels to be output.
            */
            unsigned height;

            /**
             * The number of columns of pixels to be computed for each output row.
             */
            unsigned width;

            /**
             * The number of channels to compute for each output pixel.
             */
            unsigned channels;
        } shape;

        struct {

            /**
             * Vector (in output image coordinate space) describing the vertical stride to move from one
             * row of the output image to the next.
             * i.e. the vector from the first pixel of the first row of the output region to
             * the first pixel of the second row of the output region.
             * 
             * Applied when the input window makes a vertical outer stride.
             * 
             * Typically this is [1, 0, 0]. Using other values is an advanced usage and is not recommended.
             */
            nn_image_vect_t vertical;
            
            /**
             * Vector (in output image coordinate space) describing the horizontal stride to move from one
             * column of the output image to the next.
             * i.e. the vector from the first pixel of the first row of the output region to
             * the second pixel of the first row of the output region.
             * 
             * Applied when the input window makes a horizontal outer stride.
             * 
             * Typically this is [0, 1, 0]. Using other values is an advanced usage and is not recommended.
             */
            nn_image_vect_t horizontal;
        } stride;

    } output; 

    struct {
        /** 
         * Vector (in input image coorindate space) describing the initial position of the input window.
         * i.e. the top-left pixel of the initial input window.
        */
        nn_image_vect_t start;

        struct {
            /**
             * This is the number of rows in the input window which contribute to a single
             * output pixel. The window's inner vertical stride is applied `height-1` times for each
             * output pixel.
             * 
             * Typically, this is equivalent to the height of the desired input window (in pixels).
             */
            uint32_t height;

            /**
             * This is the number of columns in the input window which contribute to a single
             * output pixel. The input window's inner horizontal stride is applied `width-1` times
             * for each vertical stride of the input window.
             * 
             * Typically, this is equivalent to the width of the desired pooling window (in pixels).
             */
            uint32_t width;
        } shape;

        /**
         * These are the strides of window corresponding to changes in the output pixel
         * location.
         */
        struct {
            /**
             * Vector (in input image coordinate space) describing the desired vertical
             * stride of the input window.
             * 
             * This is applied at the same time as the vertical stride of the output.
             * 
             * Typically this is `[v, 0, 0]`, where `v` is the desired vertical window
             * stride (in pixels). Using other values is an advanced usage and is not
             * recommended.
             */
            nn_image_vect_t vertical;

            /**
             * Vector (in input image coordinate space) describing the desired horizontal
             * stride of the input window.
             * 
             * This is applied at the same time as the horizontal stride of the output.
             * 
             * Typically this is `[0, h, 0]`, where `h` is the desired horizontal window
             * stride (in pixels). Using other values is an advanced usage and is not
             * recommended.
             */
            nn_image_vect_t horizontal;
        } outer_stride;

        /**
         * These are the strides within an input window (relative to a fixed input window position) 
         *  of the input pixels which comprise a single input window.
         */
        struct {
            /** 
             * Vector (in input image coordinate space) describing the vertical stride to
             * move from one row of the input window to the next.
             * i.e. the vector from the first pixel of the first row of the input window to
             * the first pixel of the second row of the input window.
             * 
             * Typically this is `[1, 0, 0]`. Using other values is an advanced usage and is 
             * not recommended.
             */
            nn_image_vect_t vertical;

            /**
             * Vector (in input image coordinate space) describing the horizontal stride to
             * move from one column of the input window to the next.
             * i.e. the vector from the first pixel of the first row of the input window to
             * the second pixel of the first row of the input window.
             * 
             * Typically this is `[0, 1, 0]`. Using other values is an advanced usage and is 
             * not recommended.
             */
            nn_image_vect_t horizontal;
        } inner_stride;
    } window;

} nn_window_op_config_t;

/**
 * This struct represents the details of an execution plan for a typical windowed operator 
 * (`avgpool2d()`, `maxpool2d()`). 
 * The strides here are the actual address deltas that need to be added to pointer addresses
 * while performing the operation. This plan assumes the following loop structure:
 * 
 *      - for each channel output group (chan_grp_stride)
 *          - for each output row (vertical outer stride)
 *              - for each output col (horizontal outer stride)
 *                  - for each window row (vertical inner stride)
 *                      - for each window col (horizontal inner stride)
 * 
 * A plan can be computed using `nn_window_op_init()`, but this is usually called indirectly
 * through an operator's init method.
 */
typedef struct {

    struct {
        uint32_t rows;
        uint32_t cols;
        uint32_t channels;
    } output;

    struct {
        uint32_t rows;
        uint32_t cols;
    } window;

    struct {
        int32_t x;
        int32_t y;
    } start_stride;

    struct {
        struct {
            int32_t x;
        } vertical;

        struct {
            int32_t x;
        } horizontal;
    } inner_stride;

    struct {
        struct {
            int32_t x;
            int32_t y;
        } vertical;
        
        struct {
            int32_t x;
            int32_t y;
        } horizontal;
    } outer_stride;

    struct {
        int32_t x;
        int32_t y;
    } chan_grp_stride;

} nn_window_op_plan_t;


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

    nn_window_op_plan_t window;

    int32_t shift;
    int32_t scale;

    nn_avgpool2d_impl_t impl;

} nn_avgpool2d_plan_t;


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
    uint32_t channels;
} nn_image_params_t;




#ifdef __XC__
}   //extern "C"
#endif

#endif //NN_OP_STRUCTS_H_