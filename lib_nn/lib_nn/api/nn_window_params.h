#ifndef STRUCTS_H_
#define STRUCTS_H_

#include <stdint.h>

/**
 * This struct represents an indexing vector for an image.
 */
typedef struct {
    /** Number of image pixel rows */
    int32_t rows;
    /** Number of image pixel columns */
    int32_t cols;
    /** Number of image pixel channels */
    int32_t channels;
} nn_image_vect_t;

/**
 * Some of the functions in this API can have their work split into
 * multiple parts, each called a "job". This is useful, for example, 
 * for parallelizing a computation across multiple cores, or to reduce
 * the delay between which the calling function can service some other
 * resource.
 *  
 * This struct contains the parameters required to specify how work
 * can be split according to the region of the output image in which
 * the job operates.
 * 
 * For an output image `Y` with shape `(Y_height, Y_width, Y_chans)`,
 * the value of this struct indicates that a particular job should
 * compute the output values in the rectangular region
 *   `Y[  start.rows     : (start.rows + size.rows), 
 *        start.cols     : (start.cols + size.cols), 
 *        start.channels : (start.channels + size.channels) ]`
 */
typedef struct {
    /** 
     * Indices in an output image at which to begin producing output.
     * 
     * Typically channels must be a multiple of 4.
     */
    nn_image_vect_t start;

    /**
     * The number of rows, columns and channels of output to produce.
     * 
     * Typically channels must be a multiple of 4.
     */
    nn_image_vect_t size;

} nn_window_op_job_params_t;


#endif //STRUCTS_H_