#ifndef NN_TYPES_H_
#define NN_TYPES_H_

#include <stdint.h>

typedef uint16_t data16_t;


typedef enum {
    PADDING_VALID,
    PADDING_SAME
} padding_mode_t;

#endif //NN_TYPES_H_