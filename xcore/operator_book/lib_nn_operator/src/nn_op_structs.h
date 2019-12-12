

#ifndef NN_OP_STRUCTS_H_
#define NN_OP_STRUCTS_H_


#include "nn_operator.h"

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

        uint32_t padding_cells;

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

#ifdef __XC__
}   //extern "C"
#endif

#endif //NN_OP_STRUCTS_H_