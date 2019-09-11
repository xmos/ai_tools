

#ifndef DSP_XS3_VECTOR_H_
#define DSP_XS3_VECTOR_H_

#include <stdint.h>

#ifdef __XC__
extern "C" {
#endif


unsigned dsp_xs3_calc_headroom_s32_c(
    const int32_t* A,
    const unsigned length);

unsigned dsp_xs3_calc_headroom_s32_asm(
    const int32_t* A,
    const unsigned length);

static inline unsigned dsp_xs3_calc_headroom(
    const int32_t* A,
    const unsigned length)
{
#ifdef __XS3A__
    return dsp_xs3_calc_headroom_s32_asm(A, length);
#else
    return dsp_xs3_calc_headroom_s32_c(A, length);
#endif //__XS3A__
}

/**
 * Compute exponents and shifts needed for 
 * dsp_xs3_vec_mul_s32()
 */
void dsp_xs3_vector_mul_s32_prepare(
    int* A_exp,
    int* B_shr,
    int* C_shr,
    const int B_exp,
    const unsigned B_hr,
    const int C_exp,
    const unsigned C_hr);

/**
 *  A[i] = B[i] * C[i]
 * 
 *  A_hr <- New headroom of A[]
 * 
 */
void dsp_xs3_vector_mul_s32_asm(
    int32_t* A,
    unsigned* A_hr,
    const int32_t* B,
    const int32_t* C, 
    const int B_shr,
    const int C_shr,
    const unsigned length);

/**
 *  A[i] = B[i] * C[i]
 * 
 *  A_hr <- New headroom of A[]
 * 
 */
void dsp_xs3_vector_mul_s32_c(
    int32_t* A,
    unsigned* A_hr,
    const int32_t* B,
    const int32_t* C,
    const int B_shr,
    const int C_shr,
    const unsigned length);


#ifdef __XC__
} // extern "C"
#endif

#endif //DSP_XS3_H_