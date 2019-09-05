
#ifndef XS3_FPU_H_
#define XS3_FPU_H_

#ifdef __XS3A__

#ifdef __XC__
extern "C" {
#endif


enum {
    FPU_SPEC_POS        = 0,
    FPU_SPEC_POS_ZERO   = 1,
    FPU_SPEC_INF        = 2,
    FPU_SPEC_SIG_NAN    = 3,
    FPU_SPEC_NEG        = 4,
    FPU_SPEC_NEG_ZERO   = 5,
    FPU_SPEC_NINF       = 6,
    FPU_SPEC_QUIET_NAN  = 7,
};


// FADD
float fadd(
    const float a,
    const float b);

// FENAN
void fenan(
    const float a);

// FEQ
unsigned feq(
    const float a,
    const float b);

// FGT
unsigned fgt(
    const float a,
    const float b);

// FLT
unsigned flt(
    const float a,
    const float b);

// FMACC
float fmacc(
    const float acc,
    const float a,
    const float b);

// FMAKE
float fmake(
    const int sign,
    const int exp,
    const unsigned long long mantissa);

// FMANT
int fmant(
    const float a);

// FMUL
float fmul(
    const float a,
    const float b);

// FSEXP
void fsexp(
    int* exp,
    int* sign,
    const float a);

//FSPEC
unsigned fspec(
    const float a);

// FSUB
float fsub(
    const float a,
    const float b);

// FUN
unsigned fun(
    const float a,
    const float b);


#ifdef __XC__
} //extern "C"
#endif

#endif //__XS3A__

#endif //XS3_FPU_H_