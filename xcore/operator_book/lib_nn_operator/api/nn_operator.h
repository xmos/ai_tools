

#ifndef NN_OPERATOR_H_
#define NN_OPERATOR_H_

#include "nn_types.h"
#include "nn_op_structs.h"
#include "nn_operator_asm.h"
#include "nn_operator_c.h"
#include "nn_operator_inline.h"

#include <stdint.h>

#include "xs3_vpu.h"

#ifdef __XC__
extern "C" {
#endif



static inline void conv2d_deepin_deepout_block(
    int8_t* Y,
    const nn_conv2d_dido_params_t* params,
    const nn_conv2d_dido_block_params_t* block,
    const int8_t* X,
    const int8_t* K,
    const data16_t* B,
    const int16_t* shifts,
    const int16_t* scales);


static inline void conv2d_shallowin_deepout_block(
    int8_t* Y,
    const nn_conv2d_sido_params_t* params,
    const nn_conv2d_sido_block_params_t* block,
    const int8_t* X,
    const int8_t* K,
    const data16_t* B,
    const int16_t* shifts,
    const int16_t* scales);


/**  2D maxpool for "deep" input and output tensors.
 *
 *  Pool size is 2x2, stride is 2 in both dimensions. Number of input channels
 *  must be divisible by 32.
 *
 *  \param  X       Input tensor of shape (height, width, C_in) using standard
 *                  layout with the last index changing fastest:
 *                  X[h, w, c]  =  X[width * C_in * h  +  C_in * w  +  c]
 *  \param  Y       Output tensor of shape (height//2, width//2, C_in) using
 *                  standard layout with the last index changing fastest:
 *                  Y[h, w, c]  =  Y[(width//2) * C_in * h  +  C_in * w  +  c]
 *  \param  height  Input tensor/image height, must be even.
 *  \param  width   Input tensor/image width, must be even.
 *  \param  C_in    Number of input channels, must be divisible by 32.
 */
static inline void maxpool2d_deep(
    const int8_t* X, 
    int8_t* Y,
    const int32_t height, 
    const int32_t width,
    const int32_t C_in);


/**  2D maxpool for "deep" input and output tensors.
 *
 *  Pool size is 2x2, stride is 2 in both dimensions. Number of input channels
 *  must be divisible by 32.
 *
 *  \param  X       Input tensor of shape (height, width, C_in) using standard
 *                  layout with the last index changing fastest:
 *                  X[h, w, c]  =  X[width * C_in * h  +  C_in * w  +  c]
 *  \param  Y       Output tensor of shape (height//2, width//2, C_in) using
 *                  standard layout with the last index changing fastest:
 *                  Y[h, w, c]  =  Y[(width//2) * C_in * h  +  C_in * w  +  c]
 *  \param  height  Input tensor/image height, must be even.
 *  \param  width   Input tensor/image width, must be even.
 *  \param  C_in    Number of input channels, must be divisible by 32.
 */
static inline void avgpool2d_deep(
    const int8_t* X, 
    int8_t* Y,
    const int32_t height, 
    const int32_t width,
    const int32_t C_in);



/**  Fully connected layer for "deep" input and "shallow" output tensors.
 *
 *  Number of inputs must be divisible by 32. No activation is applied (i.e. linear).
 *
 *  \param  W       Weight tensor of shape (C_out, C_in) using standard layout
 *                  such that:
 *                      W[i, j]  =  K[C_in * i  +  j]
 *  \param  B       Bias tensor of shape (C_out) using a standard layout.
 *  \param  X       Input tensor of shape (C_in) using standard layout.
 *  \param  Y       Output tensor of shape (C_out) using standard layout.
 *  \param  C_out   Number of output channels
 *  \param  C_in    Number of input channels, must be divisible by 32.
 *  \param  shifts  Shift tensor of shape (C_out) using standard layout.
 *                  Defines the shift used in the 32 to 16 bit conversion via
 *                  VLSAT. For c >= C_out, the value shifts[y] is undefined.
 *  \param  scales  Scale tensor of shape (C_out) using standard layout.
 *                  Defines the scale applied after the 32 to 16 bit
 *                  conversion. Optional. Can be assumed to be between 0x4000
 *                  and 0x7FFF. For c >= C_out, the value scales[y] is
 *                  undefined.
 */
static inline void fc_deepin_shallowout_16(
    const int8_t* W, 
    const int32_t* B,
    const int8_t* X, 
    int16_t* Y,
    const int32_t C_out, 
    const int32_t C_in,
    const uint16_t* shifts, 
    const int16_t* scales);



/**  Fully connected layer for "deep" input and "shallow" output tensors.
 *
 *  Number of inputs must be divisible by 32. No activation is applied (i.e. linear).
 *
 *  \param  W       Weight tensor of shape (C_out, C_in) using standard layout
 *                  such that:
 *                      W[i, j]  =  K[C_in * i  +  j]
 *  \param  B       Bias tensor of shape (C_out) using a standard layout.
 *  \param  X       Input tensor of shape (C_in) using standard layout.
 *  \param  Y       Output tensor of shape (C_out) using standard layout.
 *  \param  C_out   Number of output channels.
 *  \param  C_in    Number of input channels, must be divisible by 32.
 *  \param  shifts  Shift tensor of shape (C_out) using standard layout.
 *                  Defines the shift used in the 32 to 8 bit conversion via
 *                  VLSAT. For c >= C_out, the value shifts[y] is undefined.
 *  \param  scales  Scale tensor of shape (C_out) using standard layout.
 *                  Defines the scale applied after the 32 to 8 bit
 *                  conversion. Optional. Can be assumed to be between 0x4000
 *                  and 0x7FFF. For c >= C_out, the value scales[y] is
 *                  undefined.
 */
static inline void fc_deepin_shallowout_8(
    const int8_t* W, 
    const int32_t* B,
    const int8_t* X, 
    int8_t* Y,
    const int32_t C_out, 
    const int32_t C_in,
    const uint16_t* shifts, 
    const int16_t* scales);


/**  Argmax to serve as the final layer of a classifier network.
 *
 *  \param  A       Tensor of shape (N) using a standard layout.
 *  \param  C       Output tensor of shape (1).
 *  \param  N       Number of elements in the input tensor A.
 */
static inline void argmax_16(
    const int16_t* A,
    int32_t* C,
    const int32_t N);


void conv2d_deepin_deepout_init(
    nn_conv2d_dido_params_t* params,
    const uint32_t X_height,
    const uint32_t X_width,
    const uint32_t K_h,
    const uint32_t K_w,
    const uint32_t C_in,
    const uint32_t C_out,
    const padding_mode_t pad_mode,
    const int8_t zero_point,
    const uint32_t region_top,
    const uint32_t region_left,
    const uint32_t region_rows,
    const uint32_t region_cols);


void conv2d_shallowin_deepout_init(
    nn_conv2d_sido_params_t* params,
    const uint32_t X_height,
    const uint32_t X_width,
    const uint32_t K_h,
    const uint32_t K_w,
    const uint32_t C_in,
    const uint32_t C_out,
    const padding_mode_t pad_mode,
    const int8_t zero_point,
    const uint32_t region_top,
    const uint32_t region_left,
    const uint32_t region_rows,
    const uint32_t region_cols);


data16_t* conv2d_boggle_B(
    int32_t* B,
    const unsigned C_out);

void conv2d_dido_boggle_K(
    int8_t* K,
    const unsigned K_h,
    const unsigned K_w,
    const unsigned C_in,
    const unsigned C_out);

void conv2d_sido_boggle_K(
    int8_t* K,
    const unsigned K_h,
    const unsigned K_w,
    const unsigned C_in,
    const unsigned C_out);

    

#if defined(__XS3A__)

/**
 * Copy size bytes from src to dst.
 *   
 * dst and src both must be word-aligned.
 *  
 * \param dst
 * \param src
 * \param size
*/
void vpu_memcpy(
    void* dst,
    void* src,
    unsigned size);

#endif

#ifdef __XC__
} // extern "C"
#endif

#endif //NN_OPERATOR_H_