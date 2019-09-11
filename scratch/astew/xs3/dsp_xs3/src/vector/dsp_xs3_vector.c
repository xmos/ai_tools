
#include "dsp_xs3_vector.h"

#include "xs3_vpu.h"

#include <stdint.h>
#include <assert.h>
#include <stdio.h>


static inline int32_t ashr(int32_t num, int32_t shift)
{
    int32_t res;
    asm("ashr %0, %1, %2" : "=r"(res) : "r"(num), "r"(shift) );
    return res;
}
static inline unsigned cls(int32_t num)
{  
    unsigned res;
    asm("cls %0, %1" : "=r"(res) : "r"(num) );

    return res-1;
}



void dsp_xs3_vector_mul_s32_prepare(
    int* A_exp,
    int* B_shr,
    int* C_shr,
    const int B_exp,
    const unsigned B_hr,
    const int C_exp,
    const unsigned C_hr)
{
    const int B_hr_s = B_hr-1;
    const int C_hr_s = C_hr-1;

    *B_shr = -B_hr_s;
    *C_shr = -C_hr_s;

    *A_exp = (B_exp + C_exp) + 30 + *B_shr + *C_shr;
}

void dsp_xs3_vector_mul_s32_c(
    int32_t* A,
    unsigned* A_hr,
    const int32_t* B,
    const int32_t* C,
    const int B_shr,
    const int C_shr,
    const unsigned length)
{
    *A_hr = 31;
    // for(unsigned i = 0; i < 1; i++){
    for(unsigned i = 0; i < length; i++){
        int32_t b = B[i];
        int32_t c = C[i];

        // printf("b: %ld\n", b);
        // printf("c: %ld\n", c);

        b = ashr(b, B_shr);
        c = ashr(c, C_shr);

        // printf("b (shifted): %ld\n", b);
        // printf("c (shifted): %ld\n", c);
        int64_t bc = b*((int64_t)c);
        
        // printf("b*c: %lld\n", bc);

        // const unsigned bit_mask_30   = 0x3FFFFFFF;
        // const unsigned annoying_case = 0x20000000;

        // if(((unsigned)(bc & bit_mask_30)) == annoying_case){
        //     //This should be "round to nearest even" logic. //TODO
        // } else {

        //Unless it's hte annoying case, rounding here is just adding 2^29 and shifting down 30.
        bc = (bc + (1<<29)) >> 30;
        // }

        // printf("bc: %lld\t\t0x%08X\n", bc, (int32_t) bc);

        assert( bc < 0x200000000L && bc > -0x200000000L );

        bc = (bc >  0x7FFFFFFF)?  0x7FFFFFFF : bc;
        bc = (bc < -0x7FFFFFFF)? -0x7FFFFFFF : bc;

        int32_t a = (int32_t) bc;

        A[i] = a;

        unsigned a_hr = cls(a);

        *A_hr = (a_hr < *A_hr)? a_hr : *A_hr;
    }
}

unsigned dsp_xs3_calc_headroom_s32_c(
    const int32_t* A,
    const unsigned length)
{
    unsigned A_hr = 31;
    for(unsigned i = 0; i < length; i++){
        const unsigned rewq = cls(A[i]);
        A_hr = (A_hr <= rewq)? A_hr : rewq;
    }
    return A_hr;
}