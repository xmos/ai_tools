

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


typedef enum {
    FC16_DEFAULT = 0,
    FC16_ROW_WISE = 1,
} nn_fc16_tail_strat_t;

typedef struct {
    int32_t c_in;
    int32_t c_out;
    int32_t cig_end_stride;
    nn_fc16_tail_strat_t tail_strat;
} nn_fully_connected_plan_t;



#ifdef __XC__
}   //extern "C"
#endif

#endif //NN_OP_STRUCTS_H_