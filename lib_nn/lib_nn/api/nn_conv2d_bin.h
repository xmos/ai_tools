
#include "nn_conv2d_structs.h"
#include "nn_binary_structs.h"
// Binary Conv2D

#define BCONV2D_BIN_DI_INPUT_CH_INCREMENT  (XS3_VPU_VREG_WIDTH_BITS)
#define BCONV2D_BIN_DI_OUTPUT_CH_INCREMENT (8*sizeof(int32_t))
#define BCONV2D_BIN_INPUT_CH_INCREMENT     (8*sizeof(int32_t))
#define BCONV2D_BIN_OUTPUT_CH_INCREMENT    (8*sizeof(int32_t))

#define BCONV2D_INT8_DIDO_INPUT_CH_INCREMENT  (XS3_VPU_VREG_WIDTH_BITS)
#define BCONV2D_INT8_DIDO_OUTPUT_CH_INCREMENT (VPU_INT16_ACC_SIZE)
#define BCONV2D_INT8_INPUT_CH_INCREMENT       (8*sizeof(int32_t))
#define BCONV2D_INT8_OUTPUT_CH_INCREMENT      (sizeof(int32_t))


/**
 * Reference implementation of the post accumulation activation.
 * 
 * @param vpu_acc                       [in]     The output of the accumulator 
 * @param ch                            [in]     The channel to apply the post activation to
 * @param post_activation_multiplier_q  [in]     Array of post activation multipliers
 * @param post_activation_bias_q        [in]     Array of post activation biases
 * @param accu_shr                      [in]     The amount to arithemetic shift the vpu_acc right by.
 * @param bias_multipler                [in]     The amount to multiply the post_activation_bias_q by.
 * @param final_shr                     [in]     The final shift right of the product to normalise the output.
 * 
 */
int8_t bnn_post_activation_reference(
              const int32_t vpu_acc,
              const unsigned ch,
              const int16_t * post_activation_multiplier_q,
              const int16_t* post_activation_bias_q,
              const int accu_shr,
              const int16_t bias_multipler,
              const int final_shr);
/**
 * Reference implementation of activation quantisation. 
 */
void bnn_quantise_activation(
               int16_t * post_activation_multiplier_q,
               int16_t* post_activation_bias_q,

               float* post_activation_multiplier,
               float* post_activation_bias, 

               unsigned chans_out,

               int32_t clamp_low,
               int32_t clamp_high,

               int *accu_shr,
               int16_t *bias_multipler,
               int *final_shr,

               int32_t receptive_volume, 
               int * chan_overlaps
);
  
/**  
 * @brief Execute @oper{bnn_reorder_threshold_tensor}.
 * 
 * This reorders the threshold tensor for efficient execution by bconv2d_bin_DI_impl. 
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
 * @param receptive_field    [in]     The spatial area over which the kernel operates, i.e. 
 *                                    kernel height * kernel width * input channel count.           
 * @param chan_overlaps      [in]     The overlap between one channel and the next //FIXME
 */
void bnn_reorder_threshold_tensor(int32_t* thresh_boggled,
                                  const int32_t* thresholds_ref,
                                  const unsigned chans_out,
                                  const unsigned receptive_field,
                                  int *chan_overlaps) ;
 
/**  
 * @brief Execute @oper{bnn_reorder_kernel_tensor}.
 * 
 * This reorders the kernel tensor for efficient execution by bconv2d_bin_DI_impl. 
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
 * @param chan_overlaps   [in]     Array of the overlap between one channel and the next
 */
void bnn_reorder_kernel_tensor(bnn_b32_t* K_p, const bnn_b32_t* K_ref_p,
                               const unsigned k_height, const unsigned k_width,
                               const unsigned chans_in,
                               const unsigned chans_out, 
                               int * chan_overlaps) ;


/**  
 * @brief Execute @oper{bnn_reorder_int8_kernel_tensor}.
 * 
 * This reorders the kernel tensor for efficient execution by bconv2d_int8_DIDO_asm. 
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
void bnn_reorder_int8_kernel_tensor(bnn_b32_t* K_p, const bnn_b32_t* K_ref_p,
                               const unsigned k_height, const unsigned k_width,
                               const unsigned chans_in,
                               const unsigned chans_out, 
                               int * chan_overlaps) ;//TODO

/**  
 * @brief Execute @oper{bconv2d_int8_DIDO_valid}.
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

    const unsigned y_h_loc, const unsigned y_v_loc,
    const unsigned y_sub_width, const unsigned y_sub_height
);


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
);

/**  
 * @brief Execute @oper{bconv2d_bin_DI_valid}.
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
void bconv2d_bin_DI_valid(bnn_b32_t* Y_p,
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
 * @brief Execute @oper{bconv2d_bin_valid}.
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
void bconv2d_bin_valid(bnn_b32_t* Y_p,
    const bnn_b32_t* X_p, const bnn_b32_t* K_p, const int32_t* thresholds_p,
    bnn_b32_t * data_scratch, 

    const nn_image_params_t* x,
    const nn_image_params_t* y,
    const nn_window_params_t* k, 

    const unsigned y_loc_x, const unsigned y_loc_y,
    const unsigned y_sub_width, const unsigned y_sub_height
);

/**  
 * @brief Execute @oper{bconv2d_bin_DI}.
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
void bconv2d_bin_DI(bnn_b32_t* Y_p,
    const bnn_b256_t* X_p, const bnn_b256_t* K_p, const int32_t* thresholds_p,
    
    const nn_image_params_t* x, //The full image of x
    const nn_image_params_t* y, // the full image of y
    const nn_window_params_t* k, //the full kernel k
    
    const unsigned y_h_loc, const unsigned y_v_loc,
    const unsigned y_sub_width, const unsigned y_sub_height,

    const unsigned x_h_loc, const unsigned x_v_loc
);

/**  
 * @brief Execute @oper{bconv2d_bin}.
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
void bconv2d_bin(bnn_b32_t* Y_p,
    const bnn_b32_t* X_p, const bnn_b32_t* K_p, const int32_t* thresholds_p,
    bnn_b32_t * data_scratch, 
    const nn_image_params_t* x, //The full image of x
    const nn_image_params_t* y, // the full image of y
    const nn_window_params_t* k, //the full kernel k
    
    const unsigned y_loc_x, const unsigned y_loc_y,
    const unsigned y_sub_width, const unsigned y_sub_height,

    const unsigned x_loc_x, const unsigned x_loc_y
);

/**  
 * @brief Execute @oper{bconv2d_int8_DIDO}.
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
 */
void bconv2d_int8_DIDO(int8_t* Y_p,
    const bnn_b256_t* X_p, const bnn_b256_t* K_p, 
    
    const int16_t* post_activation_multiplier, 
    const int16_t* post_activation_bias,
    const int accu_shr,
    const int16_t bias_multipler,
    const int final_shr,
    
    const nn_image_params_t* x, //The full image of x
    const nn_image_params_t* y, // the full image of y
    const nn_window_params_t* k, //the full kernel k
    
    const unsigned y_h_loc, const unsigned y_v_loc,
    const unsigned y_sub_width, const unsigned y_sub_height,

    const unsigned x_h_loc, const unsigned x_v_loc
) ;

void bconv2d_int8(int8_t* Y_p,
    const bnn_b32_t* X_p, const bnn_b32_t* K_p, 
    
    const int16_t* post_activation_multiplier_q, 
    const int16_t* post_activation_bias_q,
    const int accu_shr,
    const int16_t bias_multipler,
    const int final_shr,

    bnn_b32_t * data_scratch,
    
    const nn_image_params_t* x, //The full image of x
    const nn_image_params_t* y, // the full image of y
    const nn_window_params_t* k, //the full kernel k
    
    const unsigned y_loc_x, const unsigned y_loc_y,
    const unsigned y_sub_width, const unsigned y_sub_height,

    const unsigned x_loc_x, const unsigned x_loc_y
) ;