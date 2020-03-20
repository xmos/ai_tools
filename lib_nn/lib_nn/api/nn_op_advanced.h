

#ifndef NN_OP_ADVANCED_H_
#define NN_OP_ADVANCED_H_

#include "nn_types.h"
#include "nn_op_structs.h"

#include <stdint.h>

#include "xs3_vpu.h"

#ifdef __XC__
extern "C" {
#endif


#if defined(__XS3A__)

/**
 * Compute adjusted biases for a depthwise kernel, taking into account only the padding at the top
 * or bottom of a window.
 * 
 * When applying a convolution operation to an image, often the kernel is positioned such that not
 * all spatial cells of the kernel overlap with the input image, as in the following example, where
 * the 3x3 kernel is centered on the top-left pixel of a 5x5 input image:
 * 
 *    _____
 *   |O O O|    
 *   |O O O|X X X 
 *   |O_O_O|X X X 
 *      X X X X X  
 *      X X X X X  
 *      X X X X X  
 * 
 * In this case, the top row and left column of the kernel window are implicitly in the image's
 * padding. As the kernel window is scanned across the top row of image pixels (as in the following),
 * note that the same kernel rows at the top of the window are always in padding, whereas the number
 * of columns in padding at the left and right sides of the window change (the reverse would be true
 * if we were scanning the window vertically).
 * 
 *    _____               _____               _____                _____                _____       
 *   |O O O|             |O O O|             |O O O|              |O O O|              |O O O|      
 *   |O O O|X X X        |O O O|X X         X|O O O|X          X X|O O O|         X X X|O O O|
 *   |O_O_O|X X X        |O_O_O|X X         X|O_O_O|X          X X|O_O_O|         X X X|O_O_O|
 *      X X X X X         X X X X X         X X X X X          X X X X X          X X X X X 
 *      X X X X X         X X X X X         X X X X X          X X X X X          X X X X X 
 *      X X X X X         X X X X X         X X X X X          X X X X X          X X X X X 
 * 
 * In this situation, the top row of the kernel window is contributing a constant amount to the
 * output accumulator at each position. Rather than re-compute that constant amount for each
 * position of the window, the biases used to seed the accumulator can be adjusted once for
 * the entire row of output pixels, and then only the padding on the left and right sides of
 * the image would need to be accounted for while computing each output pixel.
 * 
 * This function computes the adjustments for a depthwise convolution. A depthwise 2D convolution
 * is one in which the number of output channels is the same as the number of input channels, and
 * where the `k`th output channel receives contributions from only the `k`th input channel. 
 * 
 * The adjusted biases are written to `adj_biases`. `adj_biases` has shape `(2, 16)`, where
 * `adj_biases[0,k]` is the upper 16 bits of the bias for channel `k`, and `adj_biases[1,k]`
 * is the lower 16 bits of the bias for channel `k`.
 * 
 * `K` is the depthwise kernel tensor with shape `(K_h, K_w, K_c)`. The cells of `K` in the 
 * top or bottom padding will be multiplied by the zero-point value and accumulated into 
 * the original biases.
 * 
 * `bias_hi_lo` has shape `(2, 16)`, with the same interpretation as `adj_biases`, and 
 * contains the original biases to be adjusted.
 * 
 * `zero_point_vec` has shape `(16)`, where the signed 8-bit value at `zero_point_vec[k]` is
 * the zero-point for the channel `k`.
 * 
 * `K_h` is the height (first dimension) of the kernel tensor `K`.
 * 
 * `K_w` is the width (second dimension) of the kernel tensor `K`.
 * 
 * `K_c` is the number channels (third dimension) of the kernel tensor `K`.
 * 
 * `pad_t` is the number of rows at the top of the kernel window which are in padding.
 * 
 * `pad_b` is the number of rows at the bottom of the kernel window which are in padding.
 * 
 * 
 * \param adj_biases        Output. Adjusted Biases
 * \param K                 Kernel tensor
 * \param bias_hi_lo        Original bias values
 * \param zero_point_vec    Vector of channel zero-points
 * \param K_h               Height of the kernel window
 * \param K_w               Width of the kernel window
 * \param K_c               Channel count of the kernel window
 * \param pad_t             Number of padded rows at the top of the kernel window
 * \param pad_b             Number of padded rows at the bottom of the kernel window
 */
void nn_compute_hstrip_depthwise_bias_adj_asm(
    data16_t* adj_biases,
    const int8_t* K,
    const data16_t* bias_hi_lo,
    const int8_t* zero_point_vec,
    const unsigned K_h,
    const unsigned K_w,
    const unsigned K_c,
    const unsigned pad_t,
    const unsigned pad_b);

/** 
 * Apply a depthwise convolution to an input image for a single output pixel (padding
 * not supported).
 * 
 * A depthwise 2D convolution is one in which the number of output channels is 
 * the same as the number of input channels, and where the `k`th output channel 
 * receives contributions from only the `k`th input channel. 
 * 
 * A "patch" is the windowed region of the input image that gets multiplied by the 
 * convolution kernel to produce the output.
 * 
 * This function computes up to 16 channels of output and requires that the entire
 * convolution window be within the bounds of the input image (i.e. no padding).
 * 
 * \code
 *      acc[k]  <--  bias[k] + sum(X[i:i+K_h, j:j+K_w, k] * K[:,:,k])
 *      Y[k]    <--  ((acc[k]  >> shift1[k]) * scale[k]) >> shift2[k]
 * \endcode
 * 
 * If the number of channels in the input image is larger than 16, this function can
 * be called multiple times with `X`, `K` and `Y` each properly incremented, and with
 * `chans_to_write` adjusted if necessary. For example, if `K_c` is 24, the second call
 * to `nn_compute_patch_depthwise_asm` should have `Y`, `X` and `K` incremented by 16
 * bytes, and a `chans_to_write` value of 8.
 * 
 * `Y` is the memory location at which to write the output values. 
 * 
 * `X` is a pointer to the beginning of the input patch. The input image at which `X` 
 * points has `K_c` channels.
 * 
 * `K` is the convolution kernel, and has shape `(K_h, K_w, K_c)`.
 * 
 * `BSS` are the biases shifts and scale which are applied to the convolution. See _________
 * 
 * `K_h` is the height (first dimension) of both the kernel tensor `K` and the image
 * patch (though the input image's height may be larger).
 * 
 * `K_w` is the width (second dimension) of both the kernel tensor `K` and the image
 * patch (though the input image's width may be larger).
 * 
 * `K_c` is the number channels (third dimension) of both the kernel tensor `K` and
 * the input image.
 * 
 * `x_row_stride` is the number of bytes between the end of one row of the image patch
 * and the start of the next row of the patch. If the input image's width is
 * `X_width`, then  `x_row_stride` should be set to `(X_width-K_w) * K_c`.
 * 
 * `chans_to_write` is the number of output channels to be written. 
 * 
 * Constraints:
 *  - `Y`, `X`, `K`, and `BSS` must all point to word-aligned memory addresses.
 *  - `0 <= chans_to_write <= 16`  (though 0 is an expensive no-op)
 *  - `K_c` must be a multiple of 4.
 * 
 * \param Y                 Location at which to write outputs
 * \param X                 Pointer to start of patch within input image
 * \param K                 Convolution kernel
 * \param BSS               Bias, shifts and scale tensor
 * \param K_h               Height of `K`
 * \param K_w               Width of `K`
 * \param K_c               Number of channels in both `X` and `K`
 * \param x_row_stride      Number of bytes between the end of a patch row to the start of the next
 * \param chans_to_write    Number of channels to be written
 */
void nn_compute_patch_depthwise_asm(
    int8_t* Y,
    const int8_t* X, 
    const int8_t* K,
    const nn_bss_block_t* BSS,
    const unsigned K_h,
    const unsigned K_w,
    const int32_t K_c,
    const int32_t x_row_stride,
    const unsigned chans_to_write);

    
/** 
 * Apply a depthwise convolution to an input image for a single output pixel.
 * 
 * A depthwise 2D convolution is one in which the number of output channels is 
 * the same as the number of input channels, and where the `k`th output channel 
 * receives contributions from only the `k`th input channel. 
 * 
 * A "patch" is the windowed region of the input image that gets multiplied by the 
 * convolution kernel to produce the output.
 * 
 * This function computes up to 16 channels of output and supports convolutions
 * in which the input image patch includes padding.
 * 
 * \code
 *      V[0:pad_t, :, k]            = zero_point_vec[k]
 *      V[K_h-pad_b:K_h, :, k]      = zero_point_vec[k]
 *      V[:, 0:pad_l, k]            = zero_point_vec[k]
 *      V[:, K_w-pad_r:K_w, k]      = zero_point_vec[k]
 *      V[ pad_t:K_h-pad_b, pad_l:K_w-pad_r, k ] = X[pad_t:K_h-pad_b, pad_l:K_w-pad_r, k]
 *      acc[k]  =  bias[k] + sum(V[:.:, k] * K[:,:,k])
 *      Y[k]    <--  ((acc[k]  >> shift1[k]) * scale[k]) >> shift2[k]
 * \endcode
 * 
 * If the number of channels in the input image is larger than 16, this function can
 * be called multiple times with `X`, `K` and `Y` each properly incremented, and with
 * `chans_to_write` adjusted if necessary. For example, if `K_c` is 24, the second call
 * to `nn_compute_patch_depthwise_asm` should have `Y`, `X` and `K` incremented by 16
 * bytes, and a `chans_to_write` value of 8.
 * 
 * `Y` is the memory location at which to write the output values.
 * 
 * `X` is a pointer to the beginning of the input patch. The input image at which `X` 
 * points has `K_c` channels.
 * 
 * `K` is the convolution kernel, and has shape `(K_h, K_w, K_c)`. 
 * 
 * `BSS` are the biases shifts and scale which are applied to the convolution. See _________
 * 
 * `K_h` is the height (first dimension) of both the kernel tensor `K` and the image
 * patch (though the input image's height may be larger).
 * 
 * `K_w` is the width (second dimension) of both the kernel tensor `K` and the image
 * patch (though the input image's width may be larger).
 * 
 * `pad_t` is the number of rows of padding at the top of the patch.
 * 
 * `pad_l` is the number of columns of padding at the left of the patch.
 * 
 * `pad_b` is the number of rows of padding at the bottom of the patch.
 * 
 * `pad_r` is the number of columns of padding at the right of the patch.
 * 
 * `K_c` is the number channels (third dimension) of both the kernel tensor `K` and
 * the input image.
 * 
 * `x_row_stride` is the number of bytes between the end of one row of the image patch
 * and the start of the next row of the patch. If the input image's width is
 * `X_width`, then  `x_row_stride` should be set to `(X_width-K_w) * K_c`.
 * 
 * `chans_to_write` is the number of output channels to be written. 
 * 
 * `zero_point_vec` is a vector of `chans_to_write` zero-point values for each channel.
 * 
 * Constraints:
 *  - `Y`, `X`, `K`, and `BSS` must all point to word-aligned memory addresses.
 *  - `0 <= chans_to_write <= 16`  (though 0 is an expensive no-op)
 *  - `K_c` must be a multiple of 4.
 * 
 * \param Y                 Location at which to write outputs
 * \param X                 Pointer to start of patch within input image
 * \param K                 Convolution kernel
 * \param BSS               Bias, shifts and scale tensor
 * \param K_h               Height of `K`
 * \param K_w               Width of `K`
 * \param pad_t             Number of rows of padding at the top of the patch
 * \param pad_l             Number of columns of padding at the left of the patch
 * \param pad_b             Number of rows of padding at the bottom of the patch
 * \param pad_r             Number of columns of padding at the right of the patch
 * \param K_c               Number of channels in both `X` and `K`
 * \param x_row_stride      Number of bytes between the end of a patch row to the start of the next
 * \param chans_to_write    Number of channels to be written
 * \param zero_point_vec    Vector of zero-points for each channel
 */
void nn_compute_patch_depthwise_padded_asm(
    int8_t* Y,
    const int8_t* X, 
    const int8_t* K,
    const nn_bss_block_t* BSS,
    const unsigned K_h,
    const unsigned K_w,
    const unsigned pad_t,
    const unsigned pad_l,
    const unsigned pad_b,
    const unsigned pad_r,
    const int32_t K_c,
    const int32_t x_row_stride,
    const unsigned chans_to_write,
    const int8_t* zero_point_vec);
    
/** 
 * Compute a row of output pixels obtained through depthwise 2D convolution.
 * 
 * Consider the following:
 * 
 *    _____                   _____                    _____                  _____   
 *   |O O O|                 |O O O|                  |O O O|                |O O O|  
 *   |O O O|X X X X X       X|O O O|X X X        X X X|O O O|X      X X X X X|O O O|
 *   |O_O_O|X X X X X       X|O_O_O|X X X        X X X|O_O_O|X      X X X X X|O_O_O|
 *      X X X X X X X       X X X X X X X        X X X X X X X      X X X X X X X 
 *      X X X X X X X       X X X X X X X        X X X X X X X      X X X X X X X 
 *      X X X X X X X       X X X X X X X        X X X X X X X      X X X X X X X 
 *           
 *        Y _ _ _             Y Y _ _              Y Y Y _           Y Y Y Y
 *        _ _ _ _             _ _ _ _              _ _ _ _           _ _ _ _
 * 
 * The above shows a 3x3 convolution kernel being scanned horizontally across (centered
 * on) the top row of a 5x7 input image. The convolution window shifts, with a horizontal
 * stride of 2 pixels each time to produce 4 adjacent output pixels. (The example does
 * not indicate the number of image channels.)
 * 
 * In this example, the top row of the convolution window is in padding the whole time, 
 * while the padding on the left and right edges changes.
 * 
 * This example corresponds to calling `nn_compute_patch_depthwise_padded_asm()` 4 times
 * with appropriate adjustments made to its `Y`, `X`, `pad_l`, and `pad_r` arguments
 * each time. That is what this function does, only more efficiently. To perform a full
 * depthwise convolution, this function can be called multiple times, with `Y`, `X`, 
 * `pad_t`,  `pad_b` appropriately adjusted each time.
 * 
 * In the documentation that follows, "input image patch" or just "patch" refers to the
 * region of the input image which is inside the convolution window.
 * 
 *  
 * `Y` is the memory location at which to write the first output values. This address
 * will be incremented by `Y_c` bytes following each of the `out_pixels` outputs.
 * 
 * `X` points to the location in (or near) the input image where the convolution window
 * will start. The input image at which `X` points has `K_c` channels. If the top-left
 * spatial cell of the convolution window is in padding at its first location, then
 * the `X` pointer may point outside the actual input image's memory. Specifically, from
 * the example above, if the input image is defined as
 * \code
 *      int8_t input_image[5][7][K_c];
 * \endcode
 * then `X = &input_image[-1][-1][0]`. If the appropriate padding values are provided,
 * this function will never perform a read from such an invalid location.
 * 
 * `K` is the convolution kernel, and has shape `(K_h, K_w, K_c)`.
 * 
 * `BSS` are the biases shifts and scale which are applied to the convolution. See _________
 * 
 * `K_h` is the height of the convolution window in pixels.
 * 
 * `K_w` is the width of the convolution window in pixels.
 * 
 * `pad_t` and `pad_b` are the number of rows of padding at the top and bottom of the
 * convolution window respectively. As the window only strides horizontally, this padding 
 * will not change during a single invocation of this function. These values are 
 * non-negative. In the example above, `pad_t` is `1`, because the top row of the 
 * convolution kernel is in padding, and `pad_b` is `0` (NOT `-3`) because none of the
 * convolution window is below the input image.
 * 
 * `pad_l_initial` and `pad_r_inital` are the possibly-negative number of columns of
 * padding on the left and right sides of the convolution window respectively at the 
 * _first_ window location. Unlike the top and bottom padding, the padding at the left
 * or right may change with each window stride. In order to correctly and efficiently
 * track this padding change, negative values may need to be supplied. For example,
 * in the case above, `pad_l_initial` is `1`, because the left-most column of the
 * convolution window is in padding, and `pad_r_initial` is `-5`, because there are
 * `5` image pixels between the right edge of the convolution window and the right
 * edge of the input image. `pad_l_initial`, `pad_r_initial` and `K_w` should always
 * add up to the width of the input image (in pixels).
 * 
 * `K_c` is the number channels in the input image.
 * 
 * `x_row_stride` is the number of _bytes_ between end of one row of an input image patch
 * and the start of the next row of the input image patch. In the example above, `x_row_stride`
 * would be `(7-3)*K_c`, because the input image is `7` pixels wide, the convolution
 * window is `3` pixels wide, and there are `K_c` bytes per pixel.
 * 
 * `window_hstride` is the number of _bytes_ by which the convolution window will be
 * moved to the right for each output pixel. In the example above, the horizontal 
 * stride of the convolution window is 2 pixels, and so `window_hstride` would be
 * `2*K_c`, because each pixel of the input image is `K_c` bytes.
 * 
 * `Y_c` is the number of channels in the output image, or equivalently, the number of
 * bytes per pixel of the output image. The pointer `Y` is incremented by `Y_c` for
 * each output written. (`chans_to_write` bytes are written at each location.)
 * 
 * `out_pixels` is the number of output pixels to write.
 * 
 * `chans_to_write` is the number of output channels to be written. 
 * 
 * `zero_point_vec` is a vector of `chans_to_write` zero-point values for each channel.
 * 
 * Constraints:
 *  - `Y`, `X`, `K`, and `BSS` must all point to word-aligned memory addresses.
 *  - `0 <= chans_to_write <= 16`  (though 0 is an expensive no-op)
 *  - `K_c` and `Y_c` must be a multiple of 4.
 * 
 * \param Y                 Address at which to write first outputs
 * \param X                 Address in input image of the initial convolution window location
 * \param K                 Convolution kernel
 * \param BSS               Bias, shifts and scale tensor
 * \param K_h               Convolution window height
 * \param K_w               Convolution window width
 * \param pad_t             Number of rows of padding at the top of the convolution window
 * \param pad_l_initial     Number of columns of padding at the left of the convolution window in its first location
 * \param pad_b             Number of rows of padding at the bottom of the convolution window
 * \param pad_r_initial     Number of columns of padding at the right of the convolution window in its first location
 * \param K_c               Number of channels in the input image and the kernel tensor
 * \param x_row_stride      Number of bytes between the end of a patch row and the start of the next
 * \param window_hstride    Number of bytes between subsequent patches of the input image
 * \param Y_c               Number of channels in the output image
 * \param out_pixels        Number of pixels to be written
 * \param chans_to_write    Number of channels to write in each output pixel
 * \param zero_point_vec    Vector of zero-points for each channel
 */
void nn_compute_hstrip_depthwise_padded_asm(
    int8_t* Y,
    const int8_t* X, 
    const int8_t* K,
    const nn_bss_block_t* BSS,
    const unsigned K_h,
    const unsigned K_w,
    const int32_t pad_t,
    const int32_t pad_l_initial,
    const int32_t pad_b,
    const int32_t pad_r_initial,
    const int32_t K_c,
    const int32_t x_row_stride,
    const int32_t window_hstride,
    const int32_t Y_c,
    const unsigned out_pixels,
    const unsigned chans_to_write,
    const int8_t* zero_point_vec);


/** 
 * Compute a row of output pixels obtained through depthwise 2D convolution. (padding
 * not supported)
 * 
 * Consider the following:
 * 
 *    _____                    _____                    _____   
 *   |O O O|X X X X        X X|O O O|X X        X X X X|O O O|  
 *   |O O O|X X X X        X X|O O O|X X        X X X X|O O O|  
 *   |O_O_O|X X X X        X X|O_O_O|X X        X X X X|O_O_O|  
 *    X X X X X X X        X X X X X X X        X X X X X X X   
 *    X X X X X X X        X X X X X X X        X X X X X X X   
 *                                                              
 *        Y _ _                 Y Y _               Y Y Y        
 *        _ _ _                 _ _ _               _ _ _
 * 
 * The above shows a 3x3 convolution kernel being scanned horizontally across the top 
 * row of a 5x7 input image. The convolution window shifts, with a horizontal
 * stride of 2 pixels each time to produce 3 adjacent output pixels. (The example does
 * not indicate the number of image channels.)
 * 
 * In this example, the convolution window never extends beyond the bounds of the input
 * image. Padding is not supported by this function. If padding is needed, use 
 * `nn_compute_hstrip_depthwise_padded_asm()` instead.
 * 
 * This example corresponds to calling `nn_compute_patch_depthwise_asm()` 4 times
 * with appropriate adjustments made to its `Y` and `X` arguments each time. 
 * That is what this function does. To perform a full depthwise convolution, this 
 * function can be called multiple times, with `Y` and `X` appropriately adjusted each 
 * time.
 * 
 * In the documentation that follows, "input image patch" or just "patch" refers to the
 * region of the input image which is inside the convolution window.
 * 
 *  
 * `Y` is the memory location at which to write the first output values. This address
 * will be incremented by `Y_c` bytes following each of the `out_pixels` outputs.
 * 
 * `X` points to the location in the input image where the convolution window
 * will start. The input image at which `X` points has `K_c` channels. 
 * 
 * `K` is the convolution kernel, and has shape `(K_h, K_w, K_c)`.
 * 
 * `BSS` are the biases shifts and scale which are applied to the convolution. See _________
 * 
 * `K_h` is the height of the convolution window in pixels.
 * 
 * `K_w` is the width of the convolution window in pixels.
 * 
 * `K_c` is the number channels in the input image.
 * 
 * `x_row_stride` is the number of _bytes_ between end of one row of an input image patch
 * and the start of the next row of the input image patch. In the example above, `x_row_stride`
 * would be `(7-3)*K_c`, because the input image is `7` pixels wide, the convolution
 * window is `3` pixels wide, and there are `K_c` bytes per pixel.
 * 
 * `window_hstride` is the number of _bytes_ by which the convolution window will be
 * moved to the right for each output pixel. In the example above, the horizontal 
 * stride of the convolution window is 2 pixels, and so `window_hstride` would be
 * `2*K_c`, because each pixel of the input image is `K_c` bytes.
 * 
 * `Y_c` is the number of channels in the output image, or equivalently, the number of
 * bytes per pixel of the output image. The pointer `Y` is incremented by `Y_c` for
 * each output written. (`chans_to_write` bytes are written at each location.)
 * 
 * `out_pixels` is the number of output pixels to write.
 * 
 * `chans_to_write` is the number of output channels to be written. 
 * 
 * Constraints:
 *  - `Y`, `X`, `K`, and `BSS` must all point to word-aligned memory addresses.
 *  - `0 <= chans_to_write <= 16`  (although 0 is an expensive no-op)
 *  - `K_c` and `Y_c` must be a multiple of 4.
 * 
 * \param Y                 Address at which to write first outputs
 * \param X                 Address in input image of the initial convolution window location
 * \param K                 Convolution kernel
 * \param BSS               Bias, shifts and scale tensor
 * \param K_h               Convolution window height
 * \param K_w               Convolution window width
 * \param K_c               Number of channels in the input image and the kernel tensor
 * \param x_row_stride      Number of bytes between the end of a patch row and the start of the next
 * \param window_hstride    Number of bytes between subsequent patches of the input image
 * \param Y_c               Number of channels in the output image
 * \param out_pixels        Number of pixels to be written
 * \param chans_to_write    Number of channels to write in each output pixel
 */
void nn_compute_hstrip_depthwise_asm(
    int8_t* Y,
    const int8_t* X, 
    const int8_t* K,
    const nn_bss_block_t* BSS,
    const unsigned K_h,
    const unsigned K_w,
    const int32_t K_c,
    const int32_t x_row_stride,
    const int32_t window_hstride,
    const int32_t Y_c,
    const unsigned out_pixels,
    const unsigned chans_to_write);

#endif //defined(__XS3A__)

#ifdef __XC__
}   //extern "C"
#endif

#endif //NN_OP_ADVANCED_H_
