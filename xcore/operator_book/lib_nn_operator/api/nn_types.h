#ifndef NN_TYPES_H_
#define NN_TYPES_H_

#include <stdint.h>

typedef uint16_t data16_t;


typedef enum {
    PADDING_VALID = 0,
    PADDING_SAME = 1,
} padding_mode_t;

#endif //NN_TYPES_H_