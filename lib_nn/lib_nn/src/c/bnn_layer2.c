#include "nn_operator.h"
#include "nn_op_structs.h"

void bnn_conv2d_bin_out_valid(bnn_b32_t* Y_p,
    const bnn_b256_t* X_p, const bnn_b256_t* K_p, const int32_t* thresholds_p,

    const nn_image_params_t* x,
    const nn_image_params_t* y,
    const nn_window_params_t* k, 

    const unsigned y_loc_x, const unsigned y_loc_y,
    const unsigned y_sub_width, const unsigned y_sub_height
){  
    unsigned x_loc_x = y_loc_x*k->stride.horizontal;
    unsigned x_loc_y = y_loc_y*k->stride.vertical;
    
    unsigned k_loc_x = 0;
    unsigned k_loc_y = 0;
    unsigned k_sub_width = k->shape.width;
    unsigned k_sub_height = k->shape.height;

    bnn_conv2d_bin_out(Y_p, X_p, K_p, thresholds_p,
        x,  y,  k, 
        y_loc_x, y_loc_y, y_sub_width, y_sub_height,
        x_loc_x,  x_loc_y,  
        k_loc_x,  k_loc_y, k_sub_width,  k_sub_height);
}

void bnn_conv2d_bin_out_padded(bnn_b32_t* Y_p,
    const bnn_b256_t* X_p, const bnn_b256_t* K_p, const int32_t* thresholds_p,

    const nn_image_params_t* x, const nn_image_params_t* y,
    const nn_window_params_t* k, 

    const unsigned y_loc_x, const unsigned y_loc_y,
    const unsigned y_sub_width, const unsigned y_sub_height,
    
    PaddingValues *pv
){

    //TODO


}


void bnn_conv2d_int8_out_valid(int8_t* Y_p,
    const bnn_b256_t* X_p, const bnn_b256_t* K_p, 
    
    const int16_t* post_activation_multiplier_q, 
    const int16_t* post_activation_bias_q,
    const int accu_shr,
    const int final_shr,

    const nn_image_params_t* x,
    const nn_image_params_t* y,
    const nn_window_params_t* k, 

    const unsigned y_loc_x, const unsigned y_loc_y,
    const unsigned y_sub_width, const unsigned y_sub_height
){  
    unsigned x_loc_x = y_loc_x*k->stride.horizontal;
    unsigned x_loc_y = y_loc_y*k->stride.vertical;
    
    unsigned k_loc_x = 0;
    unsigned k_loc_y = 0;
    unsigned k_sub_width = k->shape.width;
    unsigned k_sub_height = k->shape.height;

    bnn_conv2d_int8_out(Y_p, X_p, K_p, post_activation_multiplier_q,
        post_activation_bias_q, accu_shr, final_shr,
        x,  y,  k, 
        y_loc_x, y_loc_y, y_sub_width, y_sub_height,
        x_loc_x,  x_loc_y,  
        k_loc_x,  k_loc_y, k_sub_width,  k_sub_height);
}
