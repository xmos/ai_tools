#ifndef NN_TYPES_H_
#define NN_TYPES_H_

#include <stdint.h>

typedef uint16_t data16_t;


typedef enum {
    PADDING_VALID = 0,
    PADDING_SAME = 1,
} padding_mode_t;

/**
 * Where functions in the API take an array of values, where the array index corresponds to a padding direction, the indices
 * shall be as indicated in `nn_pad_index_t`, with the order being top padding, left padding, bottom padding, right padding.
 * Rather than hard-coding those values, this enum can be used instead.
 */
enum {
    NN_PAD_TOP = 0,
    NN_PAD_LEFT,
    NN_PAD_BOTTOM,
    NN_PAD_RIGHT,
};

#endif //NN_TYPES_H_