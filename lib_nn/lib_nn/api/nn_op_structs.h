

#ifndef NN_OP_STRUCTS_H_
#define NN_OP_STRUCTS_H_


#include "nn_operator.h"
#include "xs3_vpu.h"

#ifdef __XC__
extern "C" {
#endif


/**

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
        //VLMACCR *groups* per row (16 VLMACCRs in a group)
        unsigned maccs_per_row;

        unsigned rows;

        struct {
            int32_t X;
            int32_t K;
        } row_incr;

    } patch;

} nn_conv2d_dido_block_params_t;

typedef struct {

    unsigned block_count;

    unsigned chans_in;
    unsigned chans_out;
    unsigned C_in_groups;
    unsigned C_out_groups;
    int32_t zero_point;

    nn_conv2d_dido_block_params_t* blocks;

} nn_conv2d_dido_params_t;

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


/**
 * Struct represents the parameters needed by the avgpool2d() funciton.
 * Values are set by avgpool2d_init().
 */
typedef struct {

    uint32_t out_rows;
    uint32_t out_cols;   
    uint32_t out_chans;

    uint32_t W_h;
    uint32_t W_w;

    int32_t hstride_incr_x;
    int32_t vstride_incr_x;
    int32_t vstride_incr_y;

    int32_t shift;
    int32_t scale;

    int32_t chan_incr_x;
    int32_t win_col_incr_x;
    int32_t win_row_incr_x;

    int32_t start_incr_x;
    int32_t start_incr_y;

    unsigned special_case;

} nn_avgpool_params_t;

/**
 * This struct describes the basic parameters for an image tensor
 */
typedef struct {
    uint32_t height;
    uint32_t width;
    uint32_t channels;
} nn_image_params_t;

/**
 * This struct represents an indexing vector for an image.
 */
typedef struct {
    int32_t rows;
    int32_t cols;
    int32_t channels;
} nn_image_vect_t;


/**
 * This struct describes the relationship between
 * an input image and an output image in a windowed
 * image operator (e.g. convolutions, pooling)
 */
typedef struct {
    
    struct {
        /** Index of the input image at which to begin */
        nn_image_vect_t X;
        /** Index of the output image at which to begin */
        nn_image_vect_t Y;
    } start;
    
    struct {
        /** Height of the window in pixels */
        uint32_t height;
        /** Width of the window in pixels */
        uint32_t width;

        /** Horizontal stride of the window in pixels */
        int32_t hstride;
        /** Vertical stride of the window in pixels */
        int32_t vstride;

        /** Number of positions at which the window will be applied in the horizontal direction */
        int32_t hcount;
        /** Number of positions at which the window will be applied in the vertical direction */
        int32_t vcount;
    } window;

} nn_window_map_t;

#define NN_WINDOW_MAP_DEFAULT()     { { {0,0,0},{0,0,0}}, {0, 0, 0, 0, 0, 0} }


#ifdef __XC__
}   //extern "C"
#endif

#endif //NN_OP_STRUCTS_H_