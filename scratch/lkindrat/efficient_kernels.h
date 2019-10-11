#include <stdint.h>




/**  2D convolution for "deep" input and output tensors.
 *
 *  Stride is 1 in both dimensions. Number of input and output channels must be
 *  divisible by 32 and 16 respectively. Activation is ReLU. Zero padding
 *  should be used on all edges, with size K_h//2 above and below, and K_w//2
 *  on the left and right.
 *
 *  \param  K       Kernel weight tensor of shape
 *                  (C_out//16, K_h, K_w, C_in//32, 16, 32).
 *                  Given semantically layed out kernel tensor with shape
 *                  (C_out, K_h, K_w, C_in), the coefficient corresponding to
 *                  output channel o, row r, column w and input channel i is
 *                  found at the offset (from the beginning of the array) k,
 *                  given by the following:
 *                      q = o // 16   -- (C_out 'group')
 *                      a = o % 16
 *                      w = i // 32   -- (C_in 'group')
 *                      s = i % 32
 *                      k_1 = C_in * 16 * K_h * K_w * q     --   (C_out group offset)
 *                      k_2 = C_in * 16 * K_w * r           --   (row offset within C_out group)
 *                      k_3 = C_in * 16 * c                 --   (column offset within row)
 *                      k_4 = 32 * 16 * w                   --   (C_in group offset within pixel)
 *                      k_5 = 32 * (15-a)                   --   (C_out chunk offset within C_in group)
 *                      k_6 = s                             --   (C_in offset within C_out chunk)
 *                      k = k_1 + k_2 + k_3 + k_4 + k_5 + k_6
 *  \param  B       Bias tensor of shape (2, C_out) using a standard layout
 *                  such that B[i, c]  =  B[C_out * i  +  c]. The value B[0, c]
 *                  encodes the lower 16 bits, while B[1, c] encodes the higher
 *                  16 bits of the 32-bit bias value for output channel c.
 *  \param  X       Input tensor of shape (height, width, C_in) using standard
 *                  layout with the last index changing fastest:
 *                  X[h, w, c]  =  X[width * C_in * h  +  C_in * w  +  c]
 *  \param  Y       Output tensor of shape (height, width, C_out)
 *                  using standard layout with the last index changing fastest:
 *                  Y[h, w, c]  =  Y[width * C_out * h  +  C_out * w  +  c]
 *  \param  height  Input tensor/image height.
 *  \param  width   Input tensor/image width.
 *  \param  K_h     Kernel height, must be odd.
 *  \param  K_w     Kernel width, must be odd.
 *  \param  C_out   Number of output channels, must be divisible by 16.
 *  \param  C_in    Number of input channels, must be divisible by 32.
 *  \param  shifts  Shift tensor of shape (C_out) using standard layout.
 *                  Defines the shift used in the 32 to 16 bit intermediate
 *                  conversion via VLSAT.
 *  \param  scales  Scale tensor of shape (C_out) using standard layout.
 *                  Defines the scale applied after the 32 to 16 bit
 *                  intermediate conversion. Optional. Can be assumed to be
 *                  between 0x4000 and 0x7FFF.
 */
void conv2d_deepin_deepout_relu(
    const int8_t* K, 
    const data16_t* B,
    const int8_t* X, 
    int8_t* Y,
    const int32_t height,
    const int32_t width,
    const int32_t K_h,
    const int32_t K_w,
    const int32_t C_out,
    const int32_t C_in,
    const int16_t* shifts,
    const int16_t* scales);

/**  2D convolution for "shallow" input and "deep" output tensors.
 *
 *  Stride is 1 in both dimensions. Number of output channels must be divisible
 *  by 16. Number of input channels is 4. Activation is ReLU. Zero padding
 *  should be used on all edges, with size K_h//2 above and below, and K_w//2
 *  on the left and right.
 *
 *  \param  K       Kernel weight tensor of shape (C_out, K_h, 8, 4) using
 *                  a non-standard layout such that:
 *                  K[i, j, k, l]  =  K[
 *                    K_h * 8 * 4 * ((i//16 + 1)*16 - i%16 - 1)
 *                    +  8 * 4 * j  +  4 * k  +  l
 *                  ]
 *                  The weights are zero padded in the 3rd dimension, i.e.
 *                  K[i, j, k, l] is zero for K_w <= k < 8. There may or may
 *                  not be zero padding in the 4th dimension.
 *  \param  B       Bias tensor of shape (2, C_out) using a standard layout
 *                  such that B[i, c]  =  B[C_out * i  +  c]. The value B[0, c]
 *                  encodes the lower 16 bits, while B[1, c] encodes the higher
 *                  16 bits of the 32-bit bias value for output channel c.
 *  \param  X       Input tensor of shape (height, width, C_in) using standard
 *                  layout with the last index changing fastest:
 *                  X[h, w, c]  =  X[width * 4 * h  +  4 * w  +  c]
 *  \param  Y       Output tensor of shape (height, width, C_out)
 *                  using standard layout with the last index changing fastest:
 *                  Y[h, w, c]  =  Y[width * C_out * h  +  C_out * w  +  c]
 *  \param  height  Input tensor/image height.
 *  \param  width   Input tensor/image width.
 *  \param  K_h     Kernel height, must be odd.
 *  \param  K_w     Kernel width, must be odd and less than 8.
 *  \param  C_out   Number of output channels, must be divisible by 16.
 *  \param  shifts  Shift tensor of shape (C_out) using standard layout.
 *                  Defines the shift used in the 32 to 16 bit intermediate
 *                  conversion via VLSAT.
 *  \param  scales  Scale tensor of shape (C_out) using standard layout.
 *                  Defines the scale applied after the 32 to 16 bit
 *                  intermediate conversion. Optional. Can be assumed to be
 *                  between 0x4000 and 0x7FFF.
 */
void conv2d_shallowin_deepout_relu(
    const int8_t* K,
    const data16_t* B,
    const int8_t* X,
    int8_t* Y,
    const int32_t height,
    const int32_t width,
    const int32_t K_h,
    const int32_t K_w,
    const int32_t C_out,
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
void maxpool2d_deep(
    const int8_t* X,
    int8_t* Y,
    const int32_t height,
    const int32_t width,
    const int32_t C_in);

/**  Fully connected layer for "deep" input and "shallow" output tensors as the
 *   final (learned) layer of a network.
 *
 *  Number of inputs must be divisible by 32. Number of outputs must be less
 *  than 16. No explicit activation is applied.
 *
 *  \param  W       Weight tensor of shape (C_out, C_in) using standard layout
 *                  such that:
 *                      W[i, j]  =  W[C_in * i  +  j]
 *  \param  B       Bias tensor of shape (C_out) using a standard layout.
 *  \param  X       Input tensor of shape (C_in) using standard layout.
 *  \param  Y       Output tensor of shape (C_out) using standard layout.
 *  \param  C_out   Number of output channels, must be less than 16.
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
void fc_deepin_shallowout_final(
    const int8_t* W,
    const int32_t* B,
    const int8_t* X,
    int16_t* Y,
    const int32_t C_out,
    const int32_t C_in,
    const int16_t* shifts,
    const int16_t* scales);

/**  Argmax to serve as the final layer of a classifier network.
 *
 *  \param  A       Tensor of shape (N) using a standard layout.
 *  \param  C       Output tensor of shape (1).
 *  \param  N       Number of elements in the input tensor A.
 */
void argmax_16(const int16_t* A,
               int32_t* C,
               const int32_t N);
