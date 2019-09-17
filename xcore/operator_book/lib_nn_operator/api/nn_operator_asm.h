

#ifndef NN_OPERATOR_ASM_H_
#define NN_OPERATOR_ASM_H_

#include <stdint.h>

#ifdef __XC__
extern "C" {
#endif

#ifdef __XS3A__


#ifndef USE_ASM_nn_mat_vec_mul_s8
#define USE_ASM_nn_mat_vec_mul_s8   (1)
#endif
void nn_mat_vec_mul_s8_asm(
    const int8_t* W,
    const int8_t* x,
    const unsigned N_bands,
    const unsigned N_chunks,
    const int16_t* shr,
    int8_t* y);


#endif //__XS3A__

#ifdef __XC__
}   //extern "C"
#endif

#endif //NN_OPERATOR_ASM_H_