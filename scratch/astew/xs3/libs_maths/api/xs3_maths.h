
#ifndef XS3_MATHS_H_
#define XS3_MATHS_H_

#ifdef __XS3A__

#include <stdint.h>

#ifdef __XC__
extern "C" {
#endif


void compute_chunk_int8_asm(
    const int8_t* W,
    const int8_t* x);

void compute_tile_int8_asm(
    const int8_t* W,
    const int8_t* x,
    const unsigned N_chunks);

void mat_vec_mul_int8_asm(
    const int8_t* W,
    const int8_t* x,
    const unsigned N_bands,
    const unsigned N_chunks,
    const int16_t* shr,
    int8_t* y);


#ifdef __XC__
} //extern "C"
#endif

#endif //__XS3A__

#endif //XS3_MATHS_H_