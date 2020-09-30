#ifndef CONV2D_STRUCTS_H_
#define CONV2D_STRUCTS_H_

#include "nn_image.h"

#define CONV2D_OUTPUT_LENGTH(input_length, filter_size, dilation, stride)     \
  (((input_length - (filter_size + (filter_size - 1) * (dilation - 1)) + 1) + \
    stride - 1) /                                                             \
   stride)

#define CONV2D_INPUT_LENGTH(output_length, filter_size, dilation, stride)  (output_length * stride - (stride - 1) - 1  + (filter_size + (filter_size - 1) * (dilation - 1)))

/**
 * Describes the relationship between the convolution window and the 
 * input image.
 */
typedef struct {

    /** The shape of the convolution window */
    struct {
        /** Height of the convolution window in pixels */
        unsigned height;
        /** Width of the convolution window in pixels */
        unsigned width;
    } shape;

    /** 
     * The initial position of the convolution window, relative to the input image.
     * 
     * The position given by this pair indicates where the top-left pixel of the convolution
     * window begins relative to the top-left pixel of the input image. 
     * 
     * If this pair is, for example, `(0, 0)`, then the convolution window starts at the top 
     * left of the input image and involves no top or left padding.
     */
    struct {
        /** Row offset of convolution window inital position */
        int row;
        /** Column offset of convolution window inital position */
        int column;
    } start;

    /**
     * The strides of the convolution window. These are the number of (input image) pixels that
     * the convolution window moves down and right for each pixel moved down or right in the
     * output image.
     */
    struct {
        /** Vertical stride of the convolution window. */
        int vertical;
        /** Horizontal stride of the convolution window */
        int horizontal;
    } stride;

    /**
     * Note: Only supported where explicitly mentioned.
     */
    struct {
        /** Vertical dilation of the convolution window. */
        int vertical;
        /** Horizontal dilation of the convolution window */
        int horizontal;
    } dilation;

} nn_window_params_t;

#endif //CONV2D_STRUCTS_H_