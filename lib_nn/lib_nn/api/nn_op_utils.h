

#ifndef NN_OP_UTILS_H_
#define NN_OP_UTILS_H_

#include "nn_types.h"
#include "nn_op_structs.h"

#include <stdint.h>

#include "xs3_vpu.h"

#ifdef __XC__
extern "C" {
#endif




/** Helper for computing offsets between pixels in an 8-bit image.
 * 
 * Gives the address delta associated with moving the specified number of
 * rows, columns and channels through an image with the specified parameters.
 * 
 * \param IMG           (nn_image_params_t*) Pointer to image params.
 * \param DELTA_ROWS    (signed int) Number of rows
 * \param DELTA_COLS    (signed int) Number of columns
 * \param DELTA_CHANS   (signed int) Number of channels
 */
#define IMG_ADDRESS_VECT(IMG, DELTA_ROWS, DELTA_COLS, DELTA_CHANS)      (((DELTA_ROWS) * (IMG)->width * (IMG)->channels) + ((DELTA_COLS) * (IMG)->channels) + (DELTA_CHANS))


/** Get the number of output channel groups given the number of output channels.
 * 
 * This macro gives the minimum number of groups required to handle `CHANNELS` channels, which
 * means it is effectively `(int) ceil(CHANNELS / 16.0f)`.
 * 
 * \param CHANNELS  Number of channels
 */
#define OUT_CHANNEL_GROUPS(CHANNELS)    ( ((CHANNELS) + (VPU_INT8_ACC_PERIOD-1)) >> VPU_INT8_ACC_PERIOD_LOG2 )


/** Prepare to execute a 2D deepin-deepout convolution.
 *
 * This function initializes a `nn_conv2d_dido_params_t` struct with
 * the values necessary to perform the specified convolution.
 * 
 * Once initialized, the contents of the `params` struct will not
 * change, so it need only be initialized once for many (identical)
 * convolutions.
 *
 * The convolution itself may require several partial convolutions corresponding
 * to different (non-overlapping) regions of the output image. Each of these 
 * partial convolutions is described by a `nn_conv2d_dido_block_params_t` struct.
 * As the number of these blocks is not known a priori, their memory is
 * allocated from the heap. The `nn_conv2d_dido_params_t.blocks` field of `params` 
 * will point to the (contiguous) array of `nn_conv2d_dido_block_params_t` blocks.
 *
 * The `nn_conv2d_dido_params_t` struct is intended to be opaque, however, because
 * memory is allocated from the heap, if the same params struct is to be 
 * initialized again, or if it is to go out of scope, it should be properly
 * de-initialized using `conv2d_deepin_deepout_deinit()`.
 */
void conv2d_deepin_deepout_init(
    nn_conv2d_dido_params_t* params,
    const nn_conv2d_init_params_t* init_params,
    const nn_conv2d_region_params_t* region_params,
    const int8_t* K,
    const data16_t* B);

/**
 * De-initialize a `nn_conv2d_dido_params_t` struct which
 * has been previously initialized.
 *
 * Because `conv2d_deepin_deepout_init()` uses `malloc()`, these
 * structs should be de-initialized if they are going to be 
 * initialized again or before they are allowed to go out of scope.
 * 
 * This function will free the memory allocated by 
 * `conv2d_deepin_deepout_init()`.
 */
void conv2d_deepin_deepout_deinit(
    nn_conv2d_dido_params_t* params);



/** Prepare to execute a 2D deepin-deepout convolution.
 *
 * This function initializes a `nn_conv2d_dido_params_t` struct with
 * the values necessary to perform the specified convolution.
 * 
 * Once initialized, the contents of the `params` struct will not
 * change, so it need only be initialized once for many (identical)
 * convolutions.
 *
 * The convolution itself may require several partial convolutions corresponding
 * to different (non-overlapping) regions of the output image. Each of these 
 * partial convolutions is described by a `nn_conv2d_dido_block_params_t` struct.
 * As the number of these blocks is not known a priori, their memory is
 * allocated from the heap. The `nn_conv2d_dido_params_t.blocks` field of `params` 
 * will point to the (contiguous) array of `nn_conv2d_dido_block_params_t` blocks.
 *
 * The `nn_conv2d_dido_params_t` struct is intended to be opaque, however, because
 * memory is allocated from the heap, if the same params struct is to be 
 * initialized again, or if it is to go out of scope, it should be properly
 * de-initialized using `conv2d_deepin_deepout_deinit()`.
 */
void conv2d_shallowin_deepout_init(
    nn_conv2d_sido_params_t* params,
    const nn_conv2d_init_params_t* init_params,
    const nn_conv2d_region_params_t* region_params,
    const int8_t* K,
    const data16_t* B);


/**
 * De-initialize a `nn_conv2d_sido_params_t` struct which
 * has been previously initialized.
 *
 * Because `conv2d_shallowin_deepout_init()` uses `malloc()`, these
 * structs should be de-initialized if they are going to be 
 * initialized again or before they are allowed to go out of scope.
 * 
 * This function will free the memory allocated by 
 * `conv2d_shallowin_deepout_init()`.
 */
void conv2d_shallowin_deepout_deinit(
    nn_conv2d_sido_params_t* params);


/**
 *  Rearranges the data in `B` from the standard tensor layout
 * into Bias tensor layout form 1, as required by the 
 * `conv2d_deepin_deepout()` and `conv2d_shallowin_deepout()`
 * functions.
 * 
 * \param B         Bias tensor in standard tensor layout
 * \param C_out     Length of the bias tensor
 * \returns         `B` recast as a `data16_t` pointer.
 */
data16_t* conv2d_boggle_B(
    int32_t* B,
    const unsigned C_out);

/**
 * Rearranges the data in kernel tensor `K`, provided in standard tensor
 * layout ( with shape (C_out, K_h, K_w, C_in) corresponding to the
 * output channel, kernel row, kernel column and input channel
 * respectively) into the layout required by `conv2d_deepin_deepout()`.
 * 
 * \param K         Kernel tensor
 * \param K_h       Kernel height
 * \param K_w       Kernel width
 * \param C_in      Input channel count
 * \param C_out     Output Channel count
 */
void conv2d_dido_boggle_K(
    int8_t* K,
    const unsigned K_h,
    const unsigned K_w,
    const unsigned C_in,
    const unsigned C_out);
    

/**
 * Re-layout the shift-scale tensor to the format expected by the convolution kernels.
 * 
 * The input tensor should contain all of the shifts followed by all of the scales, in
 * channel order. 
 *
 * A scratch buffer parameter may optionally be supplied (same size as `shiftscales`).
 * If `scratch` is `NULL`, a buffer will be `malloc`ed (and `free`ed).
 *
 * \param shiftscales   The shift/scale tensor. Updated in-place
 * \param C_out         The number of output channels
 * \param scratch       Optional scratch buffer.
 */
void conv2d_boggle_shift_scale(
    int16_t* shiftscales,
    const unsigned C_out,
    int16_t* scratch);


/**
 * Rearranges the data in kernel tensor `K`, provided in ..nearly... standard tensor
 * layout ( with shape (C_out, K_h, 32/C_in, C_in) corresponding to the
 * output channel, kernel row, kernel column and input channel
 * respectively) into the layout required by `conv2d_deepin_deepout()`.
 * 
 * \param K         Kernel tensor
 * \param K_h       Kernel height
 * \param K_w       Kernel width
 * \param C_in      Input channel count
 * \param C_out     Output Channel count
 */
void conv2d_sido_boggle_K(
    int8_t* K,
    const unsigned K_h,
    const unsigned K_w,
    const unsigned C_in,
    const unsigned C_out);



/**
 * Lays out the biases, shifts and scales into a format appropriate for the `BSS` argument
 * to the `fully_connected_16()` function.
 * 
 * If `bss_out` and `bias` do not point to the same memory location, it will be assumed that
 * the `bss_out` tensor does not overlap the memory of the `bias`, `shift` and `scale`
 * vectors. In this case, the `scratch` input will be ignored, and no memory will be allocated.
 * 
 * If the `bss_out` and `bias` pointers do point to the same memory location, a temporary scratch 
 * buffer is needed to perform the reformatting. In this case, if the `scratch` parameter is not
 * NULL, the memory to which is points will be used as the scratch buffer. If the `scratch` 
 * parameter is NULL, memory will be `malloc`ed for a scratch buffer. That memory will be `free`ed
 * before returning.
 * 
 * The `bias`, `shift` and `scale` vectors are each `C_out` elements long, and the `i`th
 * of each vector is the value which applies to the `i`th output channel.
 * 
 * The `bss_out` tensor must be large enough to hold `(4 * ((C_out + 15)//16) * 16)` elements
 * of type `data16_t`. `((C_out + 15)//16)*16` is the number of output channels rounded up
 * to the nearest multiple of `16`.
 * 
 * If `scratch` is provided, it must be large enough to store all `C_out` elements of the
 * `bias`, `shift` and `scale` vectors.
 * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * As this function may need to move around a large amount of data, it can be costly to
 * execute it at initialization time. To avoid this cost, you can pre-layout the data in 
 * ROM.
 * 
 * The output `bss_out` tensor will be organized in memory the same as `bss_tensor` in the 
 * following:
 *      
 *      struct {
 *          data16_t biases_high[16];
 *          data16_t biases_low[16];
 *          int16_t shifts[16];
 *          int16_t scales[16];
 *      } bss_tensor[(C_out+15)/16];
 * 
 * Each element of `bss_tensor` contains the biases, shifts and scales for a single output
 * channel group. An output channel group comprises `16` output channels which are computed
 * in parallel. The number of channel output groups is `ceil(C_out/16.0)` == `(C_out+15/16)`, which 
 * is also the length of `bss_tensor`.
 * 
 * The parameters for the first `16` output channels are found in `bss_tensor[0]`. The parameters for 
 * the next `16` output channels are found in `bss_tensor[1]`, and so on.
 * 
 * The `biases_high` field contains the most significant 16 bits of the bias for each of the 16 
 * channels. Likewise, the `biases_low` field contains the 16 least significant bits of the 32-bit 
 * biases. `shifts` and `scales` respectively store the shifts and scales for the output channels. 
 * Within each field, the output channels are in ascending order.
 * 
 * As an example, `bss_tensor[i].scales[k]` stores the scale value for output channel `16*i + k`,
 * taken from `scale[16*i+k]`.
 * 
 * Equivalently, `bss_out` has the shape `((C_out+15)/16, 4, 16)`, where the first index is the
 * channel output group, the second index refers to the parameter (bias high, bias low, shift,
 * scale -- in that order). And the final index corresponds to the offset of the channel within
 * the channel output group (i.e.  `channel_id % 16`).
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * 
 * \param bss_out   The output tensor to be written
 * \param bias      The bias vector
 * \param shift     The shift vector
 * \param scale     The scale vector
 * \param scratch   An optional scratch buffer, or NULL
 * \param C_out     The number of output channels
 */
void fc_boggle_BSS(
    data16_t* bss_out,
    int32_t* bias,
    int16_t* shift,
    int16_t* scale,
    data16_t* scratch,
    const unsigned C_out);



/**
 * Initialize the parameters required by the `avgpool2d()` function.
 * 
 * This function sets the values in `pool` to those needed by the
 * `avgpool2d()` function. This need only be called once during initialization
 * time.
 * 
 * The `x` and `y` inputs describe the input and output images resectively which will be used
 * when `avgpool2d() is called.
 * 
 * The `window` parameter describes the way output pixels are derived from input pixels, including
 * the window size, strides and starting offsets.
 * 
 * This function requies that `x->channels` == `y->channels`. Further, *every pixel* in the input
 * and output images must start at a word-aligned address, which means `x->channels` must be a
 * multiple of 4.
 * 
 * NOTE: If this average pool describes the standard 2x2 average pool with a 2x2 stride across the entire
 *       input image, then the `avgpool2d_2x2()` function should be used instead, which is optimized 
 *       for that common scenario.
 * 
 * NOTE: If this average pool describes a global average pool, then the `avgpool2d_global` function
 *       should be used instead, which is optimized for that scenario.
 * 
 * \param pool      Output. Parameters used by `avgpool2d()`
 * \param x         Parameters of the image to be input to `avgpool2d()`
 * \param y         Parameters of the iamge to be output from `avgpool2d()`
 * \param window    Windowing parameters for the operation.
 */
void avgpool2d_init(
    nn_avgpool_params_t* pool,
    const nn_image_params_t* x,
    const nn_image_params_t* y,
    const nn_window_map_t* window);


/**
 * Obtain the shift and scale parameters required by the `avgpool2d_global()` function.
 * 
 * The `x_height` and `x_width` inputs are the height and width of the image to be input
 * into the `avgpool2d_global()` function.
 * 
 * The `shift` and `scale` values will be set by this function, and should be passed to
 * the `shift` and `scale` inputs to `avgpool2d_global()`.
 * 
 * \param shift     Output. The shift parameter required by `avgpool2d_global()`
 * \param scale     Output. The scale parameter required by `avgpool2d_global()`
 * \param x_height  The height of the image to be input to `avgpool2d_global()`
 * \param x_width   The width of the image to be input to `avgpool2d_global()`
 */
void avgpool2d_global_init(
    uint32_t* shift,
    uint32_t* scale,
    const uint32_t x_height,
    const uint32_t x_width);



#ifdef __XC__
}   //extern "C"
#endif

#endif //NN_OP_UTILS_H_