

#ifndef NN_OPERATOR_H_
#define NN_OPERATOR_H_

#include "nn_operator_asm.h"
#include "nn_operator_c.h"
#include "nn_operator_inline.h"

#include <stdint.h>

#include "xs3_vpu.h"

#ifdef __XC__
extern "C" {
#endif


static inline void nn_mat_vec_mul_s8(
    const int8_t* W,
    const int8_t* x,
    const unsigned N_bands,
    const unsigned N_chunks,
    const int16_t* shr,
    int8_t* y);




#ifdef __XC__
} // extern "C"
#endif

#endif //NN_OPERATOR_H_