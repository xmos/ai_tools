#include "nn_operator.h"

void bconv2d_bin_DI_valid(bnn_b32_t* Y_p,
    const bnn_b256_t* X_p, const bnn_b256_t* K_p, const int32_t* thresholds_p,

    const nn_image_params_t* x,
    const nn_image_params_t* y,
    const nn_window_params_t* k, 

    const unsigned y_loc_x, const unsigned y_loc_y,
    const unsigned y_sub_width, const unsigned y_sub_height
){  
    unsigned x_loc_x = y_loc_x*k->stride.horizontal;
    unsigned x_loc_y = y_loc_y*k->stride.vertical;

    bconv2d_bin_DI(Y_p, X_p, K_p, thresholds_p,
        x,  y, k, 
        y_loc_x, y_loc_y,
        y_sub_width, y_sub_height,
        x_loc_x,  x_loc_y);
}

void bconv2d_bin_valid(bnn_b32_t* Y_p,
    const bnn_b32_t* X_p, const bnn_b32_t* K_p, const int32_t* thresholds_p,
    bnn_b32_t * data_scratch, 

    const nn_image_params_t* x,
    const nn_image_params_t* y,
    const nn_window_params_t* k, 

    const unsigned y_loc_x, const unsigned y_loc_y,
    const unsigned y_sub_width, const unsigned y_sub_height
){  
    unsigned x_loc_x = y_loc_x*k->stride.horizontal;
    unsigned x_loc_y = y_loc_y*k->stride.vertical;

    bconv2d_bin(Y_p, X_p, K_p, thresholds_p, data_scratch,
        x,  y, k, 
        y_loc_x, y_loc_y,
        y_sub_width, y_sub_height,
        x_loc_x,  x_loc_y);
}

void bconv2d_int8_DIDO_valid(int8_t* Y_p,
    const bnn_b256_t* X_p, const bnn_b256_t* K_p, 
    
    const int16_t* post_activation_multiplier_q, 
    const int16_t* post_activation_bias_q,
    const int accu_shr,
    const int16_t bias_multiplier,
    const int final_shr,

    const nn_image_params_t* x,
    const nn_image_params_t* y,
    const nn_window_params_t* k, 

    const unsigned y_loc_x, const unsigned y_loc_y,
    const unsigned y_sub_width, const unsigned y_sub_height
){  
    unsigned x_loc_x = y_loc_x*k->stride.horizontal;
    unsigned x_loc_y = y_loc_y*k->stride.vertical;

    bconv2d_int8_DIDO(Y_p, X_p, K_p, 

        post_activation_multiplier_q,
        post_activation_bias_q, 
        accu_shr, bias_multiplier, final_shr,
        x,  y,  k, 
        y_loc_x, y_loc_y,
        y_sub_width, y_sub_height, 
        x_loc_x,  x_loc_y);
}

void bconv2d_int8_valid(int8_t* Y_p,
    const bnn_b32_t* X_p, const bnn_b32_t* K_p, 
    
    const int16_t* post_activation_multiplier_q, 
    const int16_t* post_activation_bias_q,
    const int accu_shr,
    const int16_t bias_multiplier,
    const int final_shr,

    bnn_b32_t * data_scratch,

    const nn_image_params_t* x,
    const nn_image_params_t* y,
    const nn_window_params_t* k, 

    const unsigned y_loc_x, const unsigned y_loc_y,
    const unsigned y_sub_width, const unsigned y_sub_height
){  
    unsigned x_loc_x = y_loc_x*k->stride.horizontal;
    unsigned x_loc_y = y_loc_y*k->stride.vertical;

    bconv2d_int8(Y_p, X_p, K_p, 

        post_activation_multiplier_q,
        post_activation_bias_q, 
        accu_shr, bias_multiplier, final_shr,
        data_scratch,
        x,  y,  k, 
        y_loc_x, y_loc_y,
        y_sub_width, y_sub_height, 
        x_loc_x,  x_loc_y);
}
