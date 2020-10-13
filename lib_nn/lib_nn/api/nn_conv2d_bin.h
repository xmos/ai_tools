
#include "nn_conv2d_structs.h"
#include "nn_binary_structs.h"
// Binary Conv2D
  
/**  
 * @brief Execute @oper{bnn_reorder_threshold_tensor}.
 * 
 * This reorders the threshold tensor for efficient execution by bnn_conv2d_bin_out_asm. 
 * This is only inteneded for testing.
 * 
 * `thresh_reordered` points to the output threshold @tensor{thresh_reordered} .
 * 
 * `thresholds_ref` points to the input @tensor{thresholds_ref}.
 * 
 * `chans_out` is the number of output channels.
 * 
 * `receptive_field` the spatial area over which the kernel operates, i.e. (height x width).
 * 
 * @param thresh_reordered   [out]    The output @tensor{thresh_reordered}
 * @param thresholds_ref     [in]     The input @tensor{thresholds_ref}
 * @param chans_out          [in]     The number of output channels
 * @param receptive_field    [in]     The spatial area over which the kernel operates
 * @param chan_overlaps      [in]     The overlap between one channel and the next //FIXME
 */
void bnn_reorder_threshold_tensor(int32_t* thresh_boggled,
                                  const int32_t* thresholds_ref,
                                  const unsigned chans_out,
                                  const unsigned receptive_field,
                                  int *chan_overlaps) ;
    
/**  
 * @brief Execute @oper{bnn_reorder_multiplier_and_bias_tensors}.
 * 
 * This reorders the post_activation_multiplier and post_activation_bias tensors 
 * for efficient execution by bnn_conv2d_int8_out_asm. 
 * This is only inteneded for testing.
 * 
 * `post_activation_multiplier_q_reordered` points to the output threshold @tensor{post_activation_multiplier_q_reordered} .
 * 
 * `post_activation_multiplier_q` points to the input @tensor{post_activation_multiplier_q}.
 * 
 * `post_activation_bias_q_reordered` points to the output threshold @tensor{post_activation_bias_q_reordered} .
 * 
 * `post_activation_bias_q` points to the input @tensor{post_activation_bias_q}.
 * 
 * `chans_out` is the number of output channels.
 * 
 * 
 * @param post_activation_multiplier_q_reordered   [out]    The output @tensor{post_activation_multiplier_q_reordered}
 * @param post_activation_multiplier_q             [in]     The input @tensor{post_activation_multiplier_q}
 * @param post_activation_bias_q_reordered         [out]    The output @tensor{post_activation_bias_q_reordered}
 * @param post_activation_bias_q                   [in]     The input @tensor{post_activation_bias_q}
 * @param chans_out                                [in]     The number of output channels
 */
void bnn_reorder_multiplier_and_bias_tensors(
                                  int16_t* post_activation_multiplier_q_reordered,
                                  const int16_t* post_activation_multiplier_q,
                                  int16_t* post_activation_bias_q_reordered,
                                  const int16_t* post_activation_bias_q,
                                  const unsigned chans_out);

/**  
 * @brief Execute @oper{bnn_reorder_kernel_tensor}.
 * 
 * This reorders the kernel tensor for efficient execution by bnn_conv2d_bin_out_asm. 
 * This is only intended for testing.
 * 
 * `K_p` points to the output kernel @tensor{K_p} .
 * 
 * `K_ref_p` points to the kernel input @tensor{K_ref_p}.
 * 
 * `k_height` is the kernel height.
 * 
 * `k_width` is the kernel width.
 * 
 * `chans_in` is the number of input channels.
 * 
 * `chans_out` is the number of output channels.    
 * 
 * @param K_p         [out]    The output @tensor{K_p}
 * @param K_ref_p     [in]     The input @tensor{K_ref_p}
 * @param k_height    [in]     The kernel height
 * @param k_width     [in]     The kernel width
 * @param chans_in    [in]     The number of input channels
 * @param chans_out   [in]     The number of output channels
 */
void bnn_reorder_kernel_tensor(bnn_b32_t* K_p, const bnn_b32_t* K_ref_p,
                               const unsigned k_height, const unsigned k_width,
                               const unsigned chans_in,
                               const unsigned chans_out, 
                               int * chan_overlaps) ;


/**  
 * @brief Execute @oper{bnn_reorder_int8_kernel_tensor}.
 * 
 * This reorders the kernel tensor for efficient execution by bnn_conv2d_int8_out_asm. 
 * This is only intended for testing.
 * 
 * `K_p` points to the output kernel @tensor{K_p} .
 * 
 * `K_ref_p` points to the kernel input @tensor{K_ref_p}.
 * 
 * `k_height` is the kernel height.
 * 
 * `k_width` is the kernel width.
 * 
 * `chans_in` is the number of input channels.
 * 
 * `chans_out` is the number of output channels.    
 * 
 * @param K_p         [out]    The output @tensor{K_p}
 * @param K_ref_p     [in]     The input @tensor{K_ref_p}
 * @param k_height    [in]     The kernel height
 * @param k_width     [in]     The kernel width
 * @param chans_in    [in]     The number of input channels
 * @param chans_out   [in]     The number of output channels
 */                          
void bnn_reorder_int8_kernel_tensor(bnn_b256_t* K_p, const bnn_b256_t* K_ref_p,
                               const unsigned k_height, const unsigned k_width,
                               const unsigned chans_in,
                               const unsigned chans_out);

/**  
 * @brief Execute @oper{bnn_conv2d_int8_out_valid}.
 * 
 * This performs a binary conv2d on a rectangular sub-section of an input tensor X with 
 * kernel K.  
 * 
 * After the convolution has been computed the accumulator is multiplied and biased. The 
 * following illustrates the operation applied to each output channel:
 * 
 * channel_output = ashr(ashr(ashr(accumulator, accu_shr) * post_activation_multiplier_q[ch], 14) + 
 *                      post_activation_bias_q[ch], final_shr)
 * 
 * where ashr is an arithemetic shift right.
 * 
 * The tensor X_p represents a tensor of (x_full_height x x_full_width x X_channels)
 * The tensor K_p represents a tensor of (k_full_height x k_full_width x X_channels)
 * The tensor Y_p represents a tensor of (y_full_height x y_full_width x Y_channels)
 * 
 * 
 * @param Y             [out]    The output image @tensor{Y}
 * @param X             [in]     The input image @tensor{X}
 * @param K             [in]     The input kernel @tensor{K}
 * @param post_activation_multiplier_q  [in] The quantised post-acvtivation multiplier tensor
 * @param post_activation_bias_q        [in] The quantised post-acvtivation bias tensor
 * @param accu_shr      [in]     The amount to shift the accumulator right by before multiplying
 * @param final_shr     [in]     The amount to shift the result right by after the bias has been added
 * @param x             [in]     The parameters of the X image tensor
 * @param y             [in]     The parameters of the Y image tensor
 * @param k             [in]     The parameters of the K kernel tensor
 * @param y_h_loc       [in]     The x coordinate(horizontal) of where the output will start writing from
 * @param y_v_loc       [in]     The y coordinate(vertical) of where the output will start writing from
 * @param y_sub_width   [in]     The width of the output sub-image that will be computed
 * @param y_sub_height  [in]     The height of the output sub-image that will be computed
 */
void bnn_conv2d_int8_out_valid(int8_t* Y_p,
    const bnn_b256_t* X_p, const bnn_b256_t* K_p, 
    
    const int16_t* post_activation_multiplier_q, 
    const int16_t* post_activation_bias_q,
    const int accu_shr,
    const int final_shr,

    const nn_image_params_t* x,
    const nn_image_params_t* y,
    const nn_window_params_t* k, 

    const unsigned y_h_loc, const unsigned y_v_loc,
    const unsigned y_sub_width, const unsigned y_sub_height
);


void bnn_conv2d_int8_out_SISO_valid(int8_t* Y_p,
    const bnn_b32_t* X_p, const bnn_b32_t* K_p, 
    
    const int16_t* post_activation_multiplier_q, 
    const int16_t* post_activation_bias_q,
    const int accu_shr,
    const int final_shr,

    int *chan_overlaps,
    bnn_b32_t * data_scratch,

    const nn_image_params_t* x,
    const nn_image_params_t* y,
    const nn_window_params_t* k, 

    const unsigned y_loc_x, const unsigned y_loc_y,
    const unsigned y_sub_width, const unsigned y_sub_height
);

/**  
 * @brief Execute @oper{bnn_conv2d_bin_out_valid}.
 * 
 * This performs a binary conv2d on a rectangular sub-section of an input tensor X with 
 * kernel K.  
 * 
 * The tensor X_p represents a tensor of (x_full_height x x_full_width x X_channels)
 * The tensor K_p represents a tensor of (k_full_height x k_full_width x X_channels)
 * The tensor Y_p represents a tensor of (y_full_height x y_full_width x Y_channels)
 * 
 * 
 * @param Y             [out]    The output image @tensor{Y}
 * @param X             [in]     The input image @tensor{X}
 * @param K             [in]     The input kernel @tensor{K}
 * @param thresholds    [in]     The input thresholds @tensor{thresholds}
 * @param x             [in]     The parameters of the X image tensor
 * @param y             [in]     The parameters of the Y image tensor
 * @param k             [in]     The parameters of the K kernel tensor.
 * @param y_h_loc       [in]     The x coordinate(horizontal) of where the output will start writing from
 * @param y_v_loc       [in]     The y coordinate(vertical) of where the output will start writing from
 * @param y_sub_width   [in]     The width of the output sub-image that will be computed
 * @param y_sub_height  [in]     The height of the output sub-image that will be computed
 */
void bnn_conv2d_bin_out_valid(bnn_b32_t* Y_p,
    const bnn_b256_t* X_p, 
    const bnn_b256_t* K_p, 
    const int32_t* thresholds_p,
    const nn_image_params_t* x,
    const nn_image_params_t* y,
    const nn_window_params_t* k, 

    const unsigned y_h_loc, const unsigned y_v_loc,
    const unsigned y_sub_width, const unsigned y_sub_height
);

/**  
 * @brief Execute @oper{bnn_conv2d_bin_out_SISO_valid}.
 * 
 * This performs a binary conv2d on a rectangular sub-section of an input tensor X with 
 * kernel K. 
 * 
 * The tensor X_p represents a tensor of (x_full_height x x_full_width x X_channels)
 * The tensor K_p represents a tensor of (k_full_height x k_full_width x X_channels)
 * The tensor Y_p represents a tensor of (y_full_height x y_full_width x Y_channels)
 * 
 * 
 * @param Y             [out]    The output image @tensor{Y}
 * @param X             [in]     The input image @tensor{X}
 * @param K             [in]     The input kernel @tensor{K}
 * @param thresholds    [in]     The input thresholds @tensor{thresholds}
 * @param x             [in]     The parameters of the X image tensor
 * @param y             [in]     The parameters of the Y image tensor
 * @param k             [in]     The parameters of the K kernel tensor.
 * @param y_h_loc       [in]     The x coordinate(horizontal) of where the output will start writing from
 * @param y_v_loc       [in]     The y coordinate(vertical) of where the output will start writing from
 * @param y_sub_width   [in]     The width of the output sub-image that will be computed
 * @param y_sub_height  [in]     The height of the output sub-image that will be computed
 */
void bnn_conv2d_bin_out_SISO_valid(bnn_b32_t* Y_p,
    const bnn_b32_t* X_p, const bnn_b32_t* K_p, const int32_t* thresholds_p,
    bnn_b32_t * data_scratch, 

    const nn_image_params_t* x,
    const nn_image_params_t* y,
    const nn_window_params_t* k, 

    const unsigned y_loc_x, const unsigned y_loc_y,
    const unsigned y_sub_width, const unsigned y_sub_height
);

/**  
 * @brief Execute @oper{bnn_conv2d_bin_out}.
 * 
 * This performs a binary conv2d on a rectangular sub-section of an input tensor X with 
 * a sub-section of kernel K and writes it to s sub-section of tensor Y.
 * 
 * The tensor X_p represents a tensor of (x_full_height x x_full_width x X_channels)
 * The tensor K_p represents a tensor of (k_full_height x k_full_width x X_channels)
 * The tensor Y_p represents a tensor of (y_full_height x y_full_width x Y_channels)
 * 
 * x_sub_height and x_sub_width will be infered by the parameters of y, x, k, y_h_loc, 
 * y_v_loc, y_sub_width, y_sub_height, k_h_loc, k_v_loc, k_sub_width and k_sub_height.
 * 
 * @param Y             [out]    The output image @tensor{Y}
 * @param X             [in]     The input image @tensor{X}
 * @param K             [in]     The input kernel @tensor{K}
 * @param thresholds    [in]     The input thresholds @tensor{thresholds}
 * @param x             [in]     The parameters of the X image tensor
 * @param y             [in]     The parameters of the Y image tensor
 * @param k             [in]     The parameters of the K kernel tensor.
 * @param y_h_loc       [in]     The x coordinate(horizontal) of where the output will start writing from
 * @param y_v_loc       [in]     The y coordinate(vertical) of where the output will start writing from
 * @param y_sub_width   [in]     The width of the output sub-image that will be computed
 * @param y_sub_height  [in]     The height of the output sub-image that will be computed
 * @param x_h_loc       [in]     The x coordinate(horizontal) of where the input will start reading from
 * @param x_v_loc       [in]     The y coordinate(vertical) of where the input will start reading from
 * @param k_h_loc       [in]     The x coordinate(horizontal) of where the kernel will start reading from
 * @param k_v_loc       [in]     The y coordinate(vertical) of where the kernel will start reading from
 * @param k_sub_width   [in]     The width of the input sub-kernel that will be computed
 * @param k_sub_height  [in]     The height of the input sub-kernel that will be computed
 */
void bnn_conv2d_bin_out(bnn_b32_t* Y_p,
    const bnn_b256_t* X_p, const bnn_b256_t* K_p, const int32_t* thresholds_p,
    
    const nn_image_params_t* x, //The full image of x
    const nn_image_params_t* y, // the full image of y
    const nn_window_params_t* k, //the full kernel k
    
    const unsigned y_h_loc, const unsigned y_v_loc,
    const unsigned y_sub_width, const unsigned y_sub_height,

    const unsigned x_h_loc, const unsigned x_v_loc, 
    
    const unsigned k_h_loc, const unsigned k_v_loc, 
    const unsigned k_sub_width, const unsigned k_sub_height
);

/**  
 * @brief Execute @oper{bnn_conv2d_bin_out_SISO}.
 * 
 * Shallow input, shallow output version, i.e. it supports multiples of 32 channels in and
 * multiples of 32 channels out.
 * 
 * This performs a binary conv2d on a rectangular sub-section of an input tensor X with 
 * a sub-section of kernel K and writes it to s sub-section of tensor Y.
 * It requires a scratch tensor of k_full_height x k_full_width x Y_channels/32 + 7
 * 32 bit words.
 * 
 * The tensor X_p represents a tensor of (x_full_height x x_full_width x X_channels)
 * The tensor K_p represents a tensor of (k_full_height x k_full_width x X_channels)
 * The tensor Y_p represents a tensor of (y_full_height x y_full_width x Y_channels)
 * 
 * x_sub_height and x_sub_width will be infered by the parameters of y, x, k, y_h_loc, 
 * y_v_loc, y_sub_width, y_sub_height, k_h_loc, k_v_loc, k_sub_width and k_sub_height.
 * 
 * @param Y             [out]    The output image @tensor{Y}
 * @param X             [in]     The input image @tensor{X}
 * @param K             [in]     The input kernel @tensor{K}
 * @param thresholds    [in]     The input thresholds @tensor{thresholds}
 * @param data_scratch  [in]     A scratch tensor used for the patch to col process
 * @param x             [in]     The parameters of the X image tensor
 * @param y             [in]     The parameters of the Y image tensor
 * @param k             [in]     The parameters of the K kernel tensor.
 * @param y_h_loc       [in]     The x coordinate(horizontal) of where the output will start writing from
 * @param y_v_loc       [in]     The y coordinate(vertical) of where the output will start writing from
 * @param y_sub_width   [in]     The width of the output sub-image that will be computed
 * @param y_sub_height  [in]     The height of the output sub-image that will be computed
 * @param x_h_loc       [in]     The x coordinate(horizontal) of where the input will start reading from
 * @param x_v_loc       [in]     The y coordinate(vertical) of where the input will start reading from
 * @param k_h_loc       [in]     The x coordinate(horizontal) of where the kernel will start reading from
 * @param k_v_loc       [in]     The y coordinate(vertical) of where the kernel will start reading from
 * @param k_sub_width   [in]     The width of the input sub-kernel that will be computed
 * @param k_sub_height  [in]     The height of the input sub-kernel that will be computed
 */
void bnn_conv2d_bin_out_SISO(bnn_b32_t* Y_p,
    const bnn_b32_t* X_p, const bnn_b32_t* K_p, const int32_t* thresholds_p,
    bnn_b32_t * data_scratch, 
    const nn_image_params_t* x, //The full image of x
    const nn_image_params_t* y, // the full image of y
    const nn_window_params_t* k, //the full kernel k
    
    const unsigned y_loc_x, const unsigned y_loc_y,
    const unsigned y_sub_width, const unsigned y_sub_height,

    const unsigned x_loc_x, const unsigned x_loc_y, 
    
    const unsigned k_loc_x, const unsigned k_loc_y, 
    const unsigned k_sub_width, const unsigned k_sub_height
);

/**  
 * @brief Execute @oper{bnn_conv2d_int8_out}.
 * 
 * This performs a binary conv2d on a rectangular sub-section of an input tensor X with 
 * a sub-section of kernel K and writes it to s sub-section of tensor Y.
 * 
 * After the convolution has been computed the accumulator is multiplied and biased. The 
 * following illustrates the operation applied to each output channel:
 * 
 * channel_output = ashr(ashr(ashr(accumulator, accu_shr) * post_activation_multiplier_q[ch], 14) + 
 *                      post_activation_bias_q[ch], final_shr)
 * 
 * where ashr is an arithemetic shift right.
 * 
 * The tensor X_p represents a tensor of (x_full_height x x_full_width x X_channels)
 * The tensor K_p represents a tensor of (k_full_height x k_full_width x X_channels)
 * The tensor Y_p represents a tensor of (y_full_height x y_full_width x Y_channels)
 * 
 * x_sub_height and x_sub_width will be infered by the parameters of y, x, k, y_h_loc, 
 * y_v_loc, y_sub_width, y_sub_height, k_h_loc, k_v_loc, k_sub_width and k_sub_height.
 * 
 * @param Y             [out]    The output image @tensor{Y}
 * @param X             [in]     The input image @tensor{X}
 * @param K             [in]     The input kernel @tensor{K}
 * @param post_activation_multiplier  [in] The quantised post-acvtivation multiplier tensor
 * @param post_activation_bias        [in] The quantised post-acvtivation bias tensor
 * @param accu_shr      [in]     The amount to shift the accumulator right by before multiplying
 * @param final_shr     [in]     The amount to shift the result right by after the bias has been added
 * @param x             [in]     The parameters of the X image tensor
 * @param y             [in]     The parameters of the Y image tensor
 * @param k             [in]     The parameters of the K kernel tensor.
 * @param y_h_loc       [in]     The x coordinate(horizontal) of where the output will start writing from
 * @param y_v_loc       [in]     The y coordinate(vertical) of where the output will start writing from
 * @param y_sub_width   [in]     The width of the output sub-image that will be computed
 * @param y_sub_height  [in]     The height of the output sub-image that will be computed
 * @param x_h_loc       [in]     The x coordinate(horizontal) of where the input will start reading from
 * @param x_v_loc       [in]     The y coordinate(vertical) of where the input will start reading from
 * @param k_h_loc       [in]     The x coordinate(horizontal) of where the kernel will start reading from
 * @param k_v_loc       [in]     The y coordinate(vertical) of where the kernel will start reading from
 * @param k_sub_width   [in]     The width of the input sub-kernel that will be computed
 * @param k_sub_height  [in]     The height of the input sub-kernel that will be computed
 */
void bnn_conv2d_int8_out(int8_t* Y_p,
    const bnn_b256_t* X_p, const bnn_b256_t* K_p, 
    
    const int16_t* post_activation_multiplier, 
    const int16_t* post_activation_bias,
    const int accu_shr,
    const int final_shr,
    
    const nn_image_params_t* x, //The full image of x
    const nn_image_params_t* y, // the full image of y
    const nn_window_params_t* k, //the full kernel k
    
    const unsigned y_h_loc, const unsigned y_v_loc,
    const unsigned y_sub_width, const unsigned y_sub_height,

    const unsigned x_h_loc, const unsigned x_v_loc, 
    
    const unsigned k_h_loc, const unsigned k_v_loc, 
    const unsigned k_sub_width, const unsigned k_sub_height
) ;

void bnn_conv2d_int8_out_SISO(int8_t* Y_p,
    const bnn_b32_t* X_p, const bnn_b32_t* K_p, 
    
    const int16_t* post_activation_multiplier_q, 
    const int16_t* post_activation_bias_q,
    const int accu_shr,
    const int final_shr,

    int *chan_overlaps,
    bnn_b32_t * data_scratch,
    
    const nn_image_params_t* x, //The full image of x
    const nn_image_params_t* y, // the full image of y
    const nn_window_params_t* k, //the full kernel k
    
    const unsigned y_loc_x, const unsigned y_loc_y,
    const unsigned y_sub_width, const unsigned y_sub_height,

    const unsigned x_loc_x, const unsigned x_loc_y, 
    
    const unsigned k_loc_x, const unsigned k_loc_y, 
    const unsigned k_sub_width, const unsigned k_sub_height
) ;