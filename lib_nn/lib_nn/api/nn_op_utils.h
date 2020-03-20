

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
 * Lays out the biases, shifts and scales into the layout required for the standard bias-
 * shifts-scale tensor used by many of the API functions.
 * 
 * This is a helper function which consumes separate vectors for each component of the BSS
 * tensor and 'boggles' it into the required format. Calls to this function can be avoided
 * if necessary by storing the data in the layout specified in the secion "Bias-Shifts-Scale 
 * Tensor Layout". To understand how these values are used, see the section "Notes on Output
 * Shifts and Scales".
 * 
 * `bias` is a pointer to a 1-D vector with length `C_out` of 32-bit biases. `shift1`, `scale` 
 * and `shift2` are each pointers to 1-D vectors with length `C_out` of 16-bit elements. In
 * each case, the `i`th element corresponds to output channel `i`.
 * 
 * If `bss_out` and `bias` do not point to the same memory location, it will be assumed that
 * the `bss_out` tensor does not overlap the memory of the `bias`, `shift` and `scale`
 * vectors. In this case, the `scratch` input will be ignored, and no memory will be allocated.
 * 
 * If the `bss_out` and `bias` pointers do point to the same memory location, a temporary scratch 
 * buffer is needed to perform the reformatting. In this case, if the `scratch` parameter is not
 * `NULL`, the memory to which it points will be used as the scratch buffer. If the `scratch` 
 * parameter is `NULL`, memory will be `malloc`ed for a scratch buffer. That memory will be `free`ed
 * before returning.
 * 
 * The `bss_out` tensor must be large enough to hold `(5 * ((C_out + 15)//16) * 16)` elements
 * of type `data16_t`. `((C_out + 15)//16)*16` is the number of output channels rounded up
 * to the nearest multiple of `16`.
 * 
 * If `scratch` is provided, it must be large enough to store all `C_out` elements of the
 * `bias`, `shift` and `scale` vectors.
 * 
 * \param bss_out   The output tensor to be written
 * \param bias      The bias vector
 * \param shift1    The shift vector
 * \param scale     The scale vector
 * \param shift2    The shift vector
 * \param scratch   An optional scratch buffer, or NULL
 * \param C_out     The number of output channels
 */
void nn_standard_BSS_layout(
    data16_t* bss_out,
    int32_t* bias,
    int16_t* shift1,
    int16_t* scale,
    int16_t* shift2,
    data16_t* scratch,
    const unsigned C_out);


/**
 * Initialize an instance of `nn_window_op_config_t` for a typical scenario.
 * 
 * This function initializes `config` to specify the following behaviors:
 *      - Window dimensions are `window_height` x `window_width`
 *      - Window strides are `window_v_stride` and `window_h_stride` in the vertical and horizontal dimensions respectively.
 *      - Output begins at the top-left pixel of the output image
 *      - Intitial input window location is at the top-left pixel of the input image
 *      - Output channels equal to input channels
 *      - Input window is dense and rectangle (i.e. no pixels are skipped within a window)
 *      - Output is dense and rectangualr (i.e. fills a rectangular region of output image, without gaps)
 *      - Window is tiled horizontally and vertically as many times as will fit in the input image (window never pads)
 * 
 * \param config              `nn_window_op_config_t` struct to be initialized.
 * \param x                   Input image parameters
 * \param y                   Output image parameters
 * \param window_height       Height of the pooling window (in pixels)
 * \param window_width        Width of the pooling window (in pixels)
 * \param window_v_stride     Vertical stride of the pooling window (in pixels)
 * \param window_h_stride     Horizontal stride of the pooling window (in pixels)
 */
void nn_window_op_config_simple(
    nn_window_op_config_t* config,
    const nn_image_params_t* x,
    const nn_image_params_t* y,
    const unsigned window_height,
    const unsigned window_width,
    const unsigned window_v_stride,
    const unsigned window_h_stride);

void nn_window_op_init(
    nn_window_op_plan_t* plan,
    const nn_image_params_t* x,
    const nn_image_params_t* y,
    const nn_window_op_config_t* config,
    const unsigned channels_per_group);



#ifdef __XC__
}   //extern "C"
#endif

#endif //NN_OP_UTILS_H_
