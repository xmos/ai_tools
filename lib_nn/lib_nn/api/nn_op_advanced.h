

#ifndef NN_OP_ADVANCED_H_
#define NN_OP_ADVANCED_H_

#include "xs3_vpu.h"


#ifdef __XC__
extern "C" {
#endif


/**
 * Macro to get the address of the start of the last output channel of the `COG`<sup>th</sup> output
 * channel group of 4D kernel tensor `KRN`.
 * 
 * @param KRN[in]   4D kernel tensor
 * @param COG[in]   Output channel group
 * 
 * @return  Address of start of last output channel of `COG`<sup>th</sup> output channel group.
 */
#define KERNEL_4D_COG_LAST_CHAN_START(KRN, COG)    ((nn_tensor_t*) &(KRN[(VPU_INT8_ACC_PERIOD*(COG))+(VPU_INT8_ACC_PERIOD-1)][0][0][0]))



void avgpool2d_gen(
    nn_image_t* Y,
    const nn_image_t* X, 
    const nn_avgpool2d_plan_t* plan);
    
void avgpool2d_2x2(
    nn_image_t* Y,
    const nn_image_t* X, 
    const nn_avgpool2d_plan_t* plan);



/**  Fully connected layer for "deep" input and "shallow" output tensors.
 *
 *  Number of inputs must be divisible by 32. No activation is applied (i.e. linear).
 *
 * Weight tensor `W` is a 2D matrix with shape (C_out, C_in) in standard layout.
 *
 * Bias tensor `B` has shape (C_out), and is in standard layout.
 * 
 * Input tensor `X` has shape (C_in), and is in standard layout.
 * 
 * Output tensor `Y` has shape (C_out), and will be in standard layout.
 *
 * The `shifts` tensor has shape (C_out) and is in standard layout. The 32-bit accumulant
 * is arithmetically right-shifted by this number of bits, with rounding to the nearest integer.
 * Saturation to 16-bit bounds is applied immediately after the shift-and-round.
 * 
 * The `scales` tensor has shape (C_out) and is in standard layout. The scales can be
 * interpreted as signed Q1.14 fixed-point values.
 * 
 * Each output `Y[i]` is computed as
 *
 *  Y[i] = ( ( B[i] + sum(W[i,:] * X[:]) ) >> shifts[i] ) * scales[i]
 *
 *
 *  \param  W       Weight tensor
 *  \param  B       Bias tensor
 *  \param  X       Input tensor
 *  \param  Y       Output tensor
 *  \param  C_out   Number of output channels
 *  \param  C_in    Number of input channels, must be divisible by 32.
 *  \param  shifts  Shift tensor
 *  \param  scales  Scale tensor
 */
void fc_deepin_shallowout_16(
    const nn_tensor_t* W, 
    const int32_t* B,
    const nn_image_t* X, 
    int16_t* Y,
    const int32_t C_out, 
    const int32_t C_in,
    const uint16_t* shifts, 
    const int16_t* scales);


    
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
 * `BSO` are the biases shifts and scale which are applied to the convolution. See _________
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
 * edge of the input image. The relation `pad_l_initial + pad_r_initial == K_w - X_w` 
 * should always be true.
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
 *  - `Y`, `X`, `K`, and `BSO` must all point to word-aligned memory addresses.
 *  - `0 <= chans_to_write <= 16`  (though 0 is an expensive no-op)
 *  - `K_c` and `Y_c` must be a multiple of 4.
 * 
 * \param Y                 Address at which to write first outputs
 * \param X                 Address in input image of the initial convolution window location
 * \param K                 Convolution kernel
 * \param BSO               Bias, shifts and scale tensor
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
void nn_conv2d_hstrip_depthwise_padded(
    int8_t* Y,
    const int8_t* X, 
    const int8_t* K,
    const nn_bso_block_t* BSO,
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
 * `nn_conv2d_hstrip_depthwise_padded_asm()` instead.
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
 * `BSO` are the biases shifts and scale which are applied to the convolution. See _________
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
 *  - `Y`, `X`, `K`, and `BSO` must all point to word-aligned memory addresses.
 *  - `0 <= chans_to_write <= 16`  (although 0 is an expensive no-op)
 *  - `K_c` and `Y_c` must be a multiple of 4.
 * 
 * \param Y                 Address at which to write first outputs
 * \param X                 Address in input image of the initial convolution window location
 * \param K                 Convolution kernel
 * \param BSO               Bias, shifts and scale tensor
 * \param K_h               Convolution window height
 * \param K_w               Convolution window width
 * \param K_c               Number of channels in the input image and the kernel tensor
 * \param x_row_stride      Number of bytes between the end of a patch row and the start of the next
 * \param window_hstride    Number of bytes between subsequent patches of the input image
 * \param Y_c               Number of channels in the output image
 * \param out_pixels        Number of pixels to be written
 * \param chans_to_write    Number of channels to write in each output pixel
 */
void nn_conv2d_hstrip_depthwise(
    int8_t* Y,
    const int8_t* X, 
    const int8_t* K,
    const nn_bso_block_t* BSO,
    const unsigned K_h,
    const unsigned K_w,
    const int32_t K_c,
    const int32_t x_row_stride,
    const int32_t window_hstride,
    const int32_t Y_c,
    const unsigned out_pixels,
    const unsigned chans_to_write);


/**
 * @brief Compute a row of output pixels for a deep 2D convolution (Full @ttref{VPU_INT8_ACC_PERIOD} channels) 
 * 
 * A deep 2D convolution is one in which each output channel is a function of every input channel. This function
 * is intended to be invoked multiple times to do a full 2D convolution. Each invocation will compute 
 * @ttref{VPU_INT8_ACC_PERIOD} channels on each of `out_cols` pixels in a single row of @tensor{Y}.
 * 
 * `y` points to output tensor @tensor{Y} with shape @tensor_shape{Y_h, Y_w, Y_c}. @math{Y_w} must
 * be at least `out_cols`, and @math{Y_c} must be at least @ttref{VPU_INT8_ACC_PERIOD}. Rather than pointing
 * at the start of @tensor{Y}, the `y` passed to this function should point to the first channel to be output
 * of the first pixel to be output. The address to which `y` points will be the first memory written.
 * 
 * `x` points to the input tensor @tensor{X} with shape @tensor_shape{X_h, X_w, X_c}. Rather than pointing
 * at the start of @tensor{X}, the `x` passed to this function should point to the beginning of the first 
 * pixel of the convolution window. If the convolution window starts in padding, then this *may point
 * outside of @tensor{X}.
 * 
 * For example, the diagram below shows a `3x3` convolution being applied to a `4x4` image. The input image pixels
 * are marked with `X`'s, and the (implied) padding is marked with `P`'s. In this case, the convolution window 
 * starts in the padding at `(-1, -1, 0)` in @vector{X}'s coordinate space (i.e. @math{X[-1][-1]}) as is indicated 
 * by `P*` in the diagram. If `X_image` points to the start of the input image, then the argument `x` should 
 * be `(nn_image_t*) &X_image[-1][-1][0]`.
 * 
 * @inlinecode
 *  _ _ _      
 * |P*P P|P P P
 * |P X X|X X P
 * |P_X_X|X X P
 *  P X X X X P
 *  P X X X X P
 *  P P P P P P  
 *   
 * @endinlinecode
 * 
 * `k` points to the kernel tensor @tensor{K} with shape @tensor_shape{Y_c, K_h, K_w, X_c}. @math{Y_c} and 
 * @math{X_c} are the channel counts of the output and input images respectively. @math{K_h} and @math{K_w}
 * are the spatial height and width of the 2D convolution window respectively, expressed in pixels. Rather than
 * pointing at the start of @tensor{K}, the `k` passed to this function should point to the beginning of the
 * <b>final</b> channel of the output channel group being processed in the call.
 * 
 * For example, if @tensor{K} has shape @tensor_shape{48,5,5,32}, defined as `nn_tensor_t kernel[48][5][5][32]`,
 * then to process the output channel group (output channels `16` through `31`), then the address passed to this
 * call as `k` should be `&kernel[31][0][0][0]`. The `KERNEL_4D_COG_LAST_CHAN_START()` macro can be used to 
 * simplify this. The memory layout of @tensor{K} is the standard memory layout for 4D tensors (see 
 * @ref standard_layout).
 * 
 * `bso` points to the `nn_bso_block_t` describing the biases, shifts and scale for the current group of output 
 * channels. `bso` can be initialized with `nn_standard_BSO_layout()`. See @ref bso_layout for more information.
 * 
 * `K_height` and `K_width` are the height and width respectively of the 2D convolution window.
 * 
 * `K_hori_stride` is the horizontal stride (in pixels) taken by the 2D convolution window between between
 * output pixels.
 * 
 * `C_in` (or @math{X_c}) is the number of channels in the input image @tensor{X}.
 * 
 * `pad_t` and `pad_b` are the number of rows of padding at the top and bottom of the convolution window 
 * respectively. Because this function computes a horizontal strip of output pixels, the padding at the
 * top and bottom of the convolution window does not change during one invocation. These values are non-negative, 
 * and so 'negative' amounts of top or bottom padding should be conveyed as zeros. See example below.
 * 
 * `pad_l_initial` and `pad_r_initial` are the number of columns of padding at the left and right sides of the 
 * convolution window at the begining of the operation. Because this function computes a horizontal strip of output
 * pixels, the amount of left and right padding may change during an invocation. Unlike `pad_t` and `pad_b`, 
 * `pad_l_initial` and `pad_r_initial` indicate the *signed* padding, where gaps between the convolution window
 * and the edge of the input image may be negative, as in the following example.
 * 
 * @inlinecode
 *            pad_r_initial = -3
 *         |-----|  
 *    _____       _
 *   |W W W|      | pad_t = max(1,0) = 1
 *   |W W W|X X X ̅
 *   |W W W|X X X
 *   |W_W_W|X X X _ pad_b = max(-2,0) = 0
 *      X X X X X |
 *      X X X X X |
 *   |-|          ̅
 *     pad_l_initial = 1    
 * @endinlinecode
 * 
 * The diagram above shows a `4x3` convolution window over a `5x5` image. In this case, `pad_t` is `1`, because the
 * top row of the convolution window is outside the input image. `pad_b` is `0`, because the bottom of the convolution
 * window is 2 pixels above the bottom of the image. `pad_l_initial` is `1`, because the left column of the 
 * convolution window is outside the input image. And `pad_r_initial` is `-3`, because if the convolution window
 * moves to the right 3 pixels, the edge of the convolution window will coincide with the edge of the input image.
 * Note also that `pad_r_initial` is *independent of* `K_hori_stride`; `pad_r_initial` is expressed as *input image*
 * pixels, not *output image* pixels.
 * 
 * `x_v_stride` is the memory stride required to move a pointer from (immediately after) the right edge of the 
 * convolution window to (immediately after) the left edge of the convolution window on the following row (note: 
 * "immediately after" because the convolution window's right and left edges are effectively *between* memory 
 * addresses). This is equivalent to @math{\left(X_w-K_w\right)\cdot X_c}.
 * 
 * `k_cout_stride` is the memory stride required to move a pointer from the beginning of one output channel of 
 * @tensor{K} to the output channel with index one *lower*, i.e. stride required to decrement the output channel
 * by one. This is equivalent to @math{-\left(K_h\cdot K_w\cdot X_c\right)}.
 * 
 * `y_h_stride` is the memory stride required to move one pixel to the right in @tensor{Y}. This is equivalent 
 * to @math{Y_c}.
 * 
 * `out_cols` is the number of output pixels for each of which @ttref{VPU_INT8_ACC_PERIOD} channels are to be computed.
 * 
 * `zero_point_vec` is a pointer to an `int8_t` buffer with length @ttref{VPU_INT8_EPV}. This is supplied by
 * the user as a vector to save time (it allows the same vector to be reused, rather than copying a single
 * `int8_t` value several times).
 * 
 * @requires_word_alignment{y,x,k,bso}
 * 
 * @par Requirements and Constraints
 *   - `C_in` must be a multiple of `4`.
 *   - `y_h_stride` must be a multiple of `4`.
 * 
 * @param[out] y                    Pointer to output image @tensor{Y}
 * @param[in]  x                    Pointer to input image @tensor{X}
 * @param[in]  k                    The kernel tensor @tensor{K}
 * @param[in]  bso                  The bias-scale-offset parameters
 * @param[in]  K_height             Kernel height @math{K_h} (in pixels)
 * @param[in]  K_width              Kernel width @math{K_w} (in pixels)
 * @param[in]  K_hori_stride        Horizontal stride of the convolution window (in pixels)
 * @param[in]  C_in                 Number of input channels, @math{X_c}
 * @param[in]  pad_t                (unsigned) Number of rows of padding at the top of the convolution window
 * @param[in]  pad_b                (unsigned) Number or rows of padding at the bottom of the convolution window
 * @param[in]  pad_l_initial        (signed) Number of columns of padding at the left of the convolution window (at 
 *                                  its initial location)
 * @param[in]  pad_r_initial        (signed) Number of columns of padding at the right of the convolution window (at
 *                                  its initial location)
 * @param[in]  x_v_stride           Stride to move from the right edge of a patch to the left-most pixel 
 *                                  of the following row (in bytes)
 * @param[in]  k_cout_stride        
 * @param[in]  y_h_stride           Stride to move the output pointer from one pixel to the next (in bytes)
 * @param[in]  out_cols             Number of output pixels to write.
 * @param[in]  zero_point_vec       Vector containing `VPU_INT8_EPV` copies of the zero point value.
 */
void nn_conv2d_hstrip_deep_padded(
        nn_image_t* Y,
        const nn_image_t* X,
        const nn_tensor_t* K,    
        const nn_bso_block_t* BSO,
        const unsigned K_height,
        const unsigned K_width,
        const unsigned K_hori_stride,
        const channel_count_t C_in,
        const unsigned pad_t,
        const unsigned pad_b,
        const int pad_l_initial,
        const int pad_r_initial,
        const mem_stride_t x_v_stride,
        const mem_stride_t k_cout_stride,
        const mem_stride_t y_h_stride,
        const unsigned out_cols,
        const int8_t* zero_point_vec);


/**
 * @brief Compute a row of output pixels for a deep 2D convolution (Full @ttref{VPU_INT8_ACC_PERIOD} channels) 
 * 
 * A deep 2D convolution is one in which each output channel is a function of every input channel. This function
 * is intended to be invoked multiple times to do a full 2D convolution. Each invocation will compute 
 * @ttref{VPU_INT8_ACC_PERIOD} channels on each of `out_cols` pixels in a single row of @tensor{Y}.
 * 
 * `y` points to output tensor @tensor{Y} with shape @tensor_shape{Y_h, Y_w, Y_c}. @math{Y_w} must
 * be at least `out_cols`, and @math{Y_c} must be at least @ttref{VPU_INT8_ACC_PERIOD}. Rather than pointing
 * at the start of @tensor{Y}, the `y` passed to this function should point to the first channel to be output
 * of the first pixel to be output. The address to which `y` points will be the first memory written.
 * 
 * `x` points to the input tensor @tensor{X} with shape @tensor_shape{X_h, X_w, X_c}. Rather than pointing
 * at the start of @tensor{X}, the `x` passed to this function should point to the beginning of the first 
 * pixel of the convolution window. If the convolution window starts in padding, then this *may point
 * outside of @tensor{X}.
 * 
 * For example, the diagram below shows a `3x3` convolution being applied to a `4x4` image. The input image pixels
 * are marked with `X`'s, and the (implied) padding is marked with `P`'s. In this case, the convolution window 
 * starts in the padding at `(-1, -1, 0)` in @vector{X}'s coordinate space (i.e. @math{X[-1][-1]}) as is indicated 
 * by `P*` in the diagram. If `X_image` points to the start of the input image, then the argument `x` should 
 * be `(nn_image_t*) &X_image[-1][-1][0]`.
 * 
 * @inlinecode
 *  _ _ _      
 * |P*P P|P P P
 * |P X X|X X P
 * |P_X_X|X X P
 *  P X X X X P
 *  P X X X X P
 *  P P P P P P  
 *   
 * @endinlinecode
 * 
 * `k` points to the kernel tensor @tensor{K} with shape @tensor_shape{Y_c, K_h, K_w, X_c}. @math{Y_c} and 
 * @math{X_c} are the channel counts of the output and input images respectively. @math{K_h} and @math{K_w}
 * are the spatial height and width of the 2D convolution window respectively, expressed in pixels. Rather than
 * pointing at the start of @tensor{K}, the `k` passed to this function should point to the beginning of the
 * <b>final</b> channel of the output channel group being processed in the call.
 * 
 * For example, if @tensor{K} has shape @tensor_shape{60,5,5,32}, defined as `nn_tensor_t kernel[60][5][5][32]`,
 * then to process the output channel tail (output channels `48` through `59`), then the address passed to this
 * call as `k` should be `&kernel[59][0][0][0]`. The `KERNEL_4D_COG_LAST_CHAN_START()` macro can be used to 
 * simplify this. The memory layout of @tensor{K} is the standard memory layout for 4D tensors (see 
 * @ref standard_layout).
 * 
 * `bso` points to the `nn_bso_block_t` describing the biases, shifts and scale for the current group of output 
 * channels. `bso` can be initialized with `nn_standard_BSO_layout()`. See @ref bso_layout for more information.
 * 
 * `K_height` and `K_width` are the height and width respectively of the 2D convolution window.
 * 
 * `K_hori_stride` is the horizontal stride (in pixels) taken by the 2D convolution window between between
 * output pixels.
 * 
 * `C_in` (or @math{X_c}) is the number of channels in the input image @tensor{X}.
 * 
 * `pad_t` and `pad_b` are the number of rows of padding at the top and bottom of the convolution window 
 * respectively. Because this function computes a horizontal strip of output pixels, the padding at the
 * top and bottom of the convolution window does not change during one invocation. These values are non-negative, 
 * and so 'negative' amounts of top or bottom padding should be conveyed as zeros. See example below.
 * 
 * `pad_l_initial` and `pad_r_initial` are the number of columns of padding at the left and right sides of the 
 * convolution window at the begining of the operation. Because this function computes a horizontal strip of output
 * pixels, the amount of left and right padding may change during an invocation. Unlike `pad_t` and `pad_b`, 
 * `pad_l_initial` and `pad_r_initial` indicate the *signed* padding, where gaps between the convolution window
 * and the edge of the input image may be negative, as in the following example.
 * 
 * @inlinecode
 *            pad_r_initial = -3
 *         |-----|  
 *    _____       _
 *   |W W W|      | pad_t = max(1,0) = 1
 *   |W W W|X X X ̅
 *   |W W W|X X X
 *   |W_W_W|X X X _ pad_b = max(-2,0) = 0
 *      X X X X X |
 *      X X X X X |
 *   |-|          ̅
 *     pad_l_initial = 1    
 * @endinlinecode
 * 
 * The diagram above shows a `4x3` convolution window over a `5x5` image. In this case, `pad_t` is `1`, because the
 * top row of the convolution window is outside the input image. `pad_b` is `0`, because the bottom of the convolution
 * window is 2 pixels above the bottom of the image. `pad_l_initial` is `1`, because the left column of the 
 * convolution window is outside the input image. And `pad_r_initial` is `-3`, because if the convolution window
 * moves to the right 3 pixels, the edge of the convolution window will coincide with the edge of the input image.
 * Note also that `pad_r_initial` is *independent of* `K_hori_stride`; `pad_r_initial` is expressed as *input image*
 * pixels, not *output image* pixels.
 * 
 * `x_v_stride` is the memory stride required to move a pointer from (immediately after) the right edge of the 
 * convolution window to (immediately after) the left edge of the convolution window on the following row (note: 
 * "immediately after" because the convolution window's right and left edges are effectively *between* memory 
 * addresses). This is equivalent to @math{\left(X_w-K_w\right)\cdot X_c}.
 * 
 * `k_cout_stride` is the memory stride required to move a pointer from the beginning of one output channel of 
 * @tensor{K} to the output channel with index one *lower*, i.e. stride required to decrement the output channel
 * by one. This is equivalent to @math{-\left(K_h\cdot K_w\cdot X_c\right)}.
 * 
 * `y_h_stride` is the memory stride required to move one pixel to the right in @tensor{Y}. This is equivalent 
 * to @math{Y_c}.
 * 
 * `out_cols` is the number of output pixels for each of which @ttref{VPU_INT8_ACC_PERIOD} channels are to be computed.
 * 
 * `zero_point_vec` is a pointer to an `int8_t` buffer with length @ttref{VPU_INT8_EPV}. This is supplied by
 * the user as a vector to save time (it allows the same vector to be reused, rather than copying a single
 * `int8_t` value several times).
 * 
 * `C_out_tail` is the tail of the output channel count. The value given should be @math{left(Y_c mod 16\right}.
 * 
 * @requires_word_alignment{y,x,k,bso}
 * 
 * @par Requirements and Constraints
 *   - `C_in` must be a multiple of `4`.
 *   - `y_h_stride` must be a multiple of `4`.
 *   - `C_out_tail` must be a multiple of `4`.
 * 
 * @param[out] y                    Pointer to output image @tensor{Y}
 * @param[in]  x                    Pointer to input image @tensor{X}
 * @param[in]  k                    The kernel tensor @tensor{K}
 * @param[in]  bso                  The bias-scale-offset parameters
 * @param[in]  K_height             Kernel height @math{K_h} (in pixels)
 * @param[in]  K_width              Kernel width @math{K_w} (in pixels)
 * @param[in]  K_hori_stride        Horizontal stride of the convolution window (in pixels)
 * @param[in]  C_in                 Number of input channels, @math{X_c}
 * @param[in]  pad_t                (unsigned) Number of rows of padding at the top of the convolution window
 * @param[in]  pad_b                (unsigned) Number or rows of padding at the bottom of the convolution window
 * @param[in]  pad_l_initial        (signed) Number of columns of padding at the left of the convolution window (at 
 *                                  its initial location)
 * @param[in]  pad_r_initial        (signed) Number of columns of padding at the right of the convolution window (at
 *                                  its initial location)
 * @param[in]  x_v_stride           Stride to move from the right edge of a patch to the left-most pixel 
 *                                  of the following row (in bytes)
 * @param[in]  k_cout_stride        
 * @param[in]  y_h_stride           Stride to move the output pointer from one pixel to the next (in bytes)
 * @param[in]  out_cols             Number of output pixels to write.
 * @param[in]  zero_point_vec       Vector containing `VPU_INT8_EPV` copies of the zero point value.
 * @param[in]  C_out_tail           The tail of the output channel count.
 */
void nn_conv2d_hstrip_tail_deep_padded(
        nn_image_t* Y,
        const nn_image_t* X,
        const nn_tensor_t* K,    
        const nn_bso_block_t* BSO,
        const unsigned K_height,
        const unsigned K_width,
        const unsigned K_hori_stride,
        const channel_count_t C_in,
        const unsigned pad_t,
        const unsigned pad_b,
        const int pad_l_initial,
        const int pad_r_initial,
        const mem_stride_t x_v_stride,
        const mem_stride_t k_cout_stride,
        const mem_stride_t y_h_stride,
        const unsigned out_cols,
        const int8_t* zero_point_vec,
        const channel_count_t C_out_tail);




void nn_conv2d_hstrip_deep(
        nn_image_t* Y,
        const nn_image_t* X,
        const nn_tensor_t* K,    
        const nn_bso_block_t* BSO,
        const unsigned K_height,
        const unsigned K_width,
        const unsigned K_hori_stride,
        const channel_count_t C_in,
        const mem_stride_t x_v_stride,
        const mem_stride_t k_cout_stride,
        const mem_stride_t y_h_stride,
        const unsigned out_cols);





void nn_conv2d_hstrip_tail_deep(
        nn_image_t* Y,
        const nn_image_t* X,
        const nn_tensor_t* K,    
        const nn_bso_block_t* BSO,
        const unsigned K_height,
        const unsigned K_width,
        const unsigned K_hori_stride,
        const channel_count_t C_in,
        const mem_stride_t x_v_stride,
        const mem_stride_t k_cout_stride,
        const mem_stride_t y_h_stride,
        const unsigned out_cols,
        const channel_count_t C_out_tail);



void nn_conv2d_hstrip_shallowin_padded(
        nn_image_t* Y,
        const nn_image_t* X,
        const nn_tensor_t* K,
        const nn_bso_block_t* BSO,
        const unsigned K_h,
        const unsigned K_h_stride,
        const channel_count_t C_in,
        const unsigned pad_t,
        const unsigned pad_b,
        const int pad_l_initial,
        const int pad_r_initial,
        const mem_stride_t x_v_stride,
        const mem_stride_t y_h_stride,
        const unsigned out_cols,
        const int8_t* zero_point_vec);



void nn_conv2d_hstrip_shallowin(
        nn_image_t* Y,
        const nn_image_t* X,
        const nn_tensor_t* K,
        const nn_bso_block_t* BSO,
        const unsigned K_h,
        const unsigned K_h_stride,
        const channel_count_t C_in,
        const mem_stride_t x_v_stride,
        const mem_stride_t y_h_stride,
        const unsigned out_cols);



void nn_conv2d_hstrip_tail_shallowin_padded(
        nn_image_t* Y,
        const nn_image_t* X,
        const nn_tensor_t* K,
        const nn_bso_block_t* BSO,
        const unsigned K_h,
        const unsigned K_h_stride,
        const channel_count_t C_in,
        const unsigned pad_t,
        const unsigned pad_b,
        const int pad_l_initial,
        const int pad_r_initial,
        const mem_stride_t x_v_stride,
        const mem_stride_t y_h_stride,
        const unsigned out_cols,
        const int8_t* zero_point_vec,
        const channel_count_t C_out_tail);



void nn_conv2d_hstrip_tail_shallowin(
        nn_image_t* Y,
        const nn_image_t* X,
        const nn_tensor_t* K,
        const nn_bso_block_t* BSO,
        const unsigned K_h,
        const unsigned K_h_stride,
        const channel_count_t C_in,
        const mem_stride_t x_v_stride,
        const mem_stride_t y_h_stride,
        const unsigned out_cols,
        const channel_count_t C_out_tail);



void nn_conv2d_hstrip_im2col_padded(
        nn_image_t* Y,
        const nn_image_t* X,
        const nn_tensor_t* K,
        const nn_bso_block_t* BSO,
        const unsigned K_h,
        const unsigned K_h_stride,
        const channel_count_t C_in,
        const unsigned pad_t,
        const unsigned pad_b,
        const int pad_l_initial,
        const int pad_r_initial,
        const mem_stride_t x_v_stride,
        const mem_stride_t y_h_stride,
        const unsigned out_cols,
        const int8_t* zero_point_vec);



void nn_conv2d_hstrip_im2col(
        nn_image_t* Y,
        const nn_image_t* X,
        const nn_tensor_t* K,
        const nn_bso_block_t* BSO,
        const unsigned K_h,
        const unsigned K_h_stride,
        const channel_count_t C_in,
        const mem_stride_t x_v_stride,
        const mem_stride_t y_h_stride,
        const unsigned out_cols);



void nn_conv2d_hstrip_tail_im2col_padded(
        nn_image_t* Y,
        const nn_image_t* X,
        const nn_tensor_t* K,
        const nn_bso_block_t* BSO,
        const unsigned K_h,
        const unsigned K_h_stride,
        const channel_count_t C_in,
        const unsigned pad_t,
        const unsigned pad_b,
        const int pad_l_initial,
        const int pad_r_initial,
        const mem_stride_t x_v_stride,
        const mem_stride_t y_h_stride,
        const unsigned out_cols,
        const int8_t* zero_point_vec,
        const channel_count_t C_out_tail);



void nn_conv2d_hstrip_tail_im2col(
        nn_image_t* Y,
        const nn_image_t* X,
        const nn_tensor_t* K,
        const nn_bso_block_t* BSO,
        const unsigned K_h,
        const unsigned K_h_stride,
        const channel_count_t C_in,
        const mem_stride_t x_v_stride,
        const mem_stride_t y_h_stride,
        const unsigned out_cols,
        const channel_count_t C_out_tail);



#ifdef __XC__
}   //extern "C"
#endif

#endif //NN_OP_ADVANCED_H_
