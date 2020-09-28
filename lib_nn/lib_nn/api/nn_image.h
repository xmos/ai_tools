#ifndef IMAGE_H_
#define IMAGE_H_

#include "nn_types.h"

/**
 * This struct describes the basic parameters for an image tensor
 */
typedef struct {
    /**
     * Height of an image (in pixels)
     */
    uint32_t height;
    /**
     * Width of the image (in pixels)
     */
    uint32_t width;
    /**
     * Number of channels per pixel
     */
    channel_count_t channels;
} nn_image_params_t;

#endif //IMAGE_H_