
#ifndef XS3_VPU_H_
#define XS3_VPU_H_

#ifdef __XS3A__

#ifdef __XC__
extern "C" {
#endif

#define XS3_VPU_VREG_WIDTH_BITS     (256)
#define XS3_VPU_VREG_WIDTH_BYTES    (XS3_VPU_VREG_WIDTH_BITS  >> 3)
#define XS3_VPU_VREG_WIDTH_WORDS    (XS3_VPU_VREG_WIDTH_BYTES >> 2)

enum {
    VEC_INT_32 = 0,
    VEC_INT_16 = 1,
    VEC_INT_8  = 2,
    VEC_FLT_32 = 4,
    VEC_FLT_16 = 5,
    VEC_FLT_8  = 6,
};

enum {
    VEC_SH0 = 0,
    VEC_SHL = 1,
    VEC_SHR = 2,
};

enum {
    VPU_INT8_MAX =  0x7F,
    VPU_INT8_MIN = -0x7F,

    VPU_INT16_MAX =  0x7FFF,
    VPU_INT16_MIN = -0x7FFF,

    VPU_INT32_MAX =  0x7FFFFFFF,
    VPU_INT32_MIN = -0x7FFFFFFF,
};

enum {
    VPU_INT8_EPV    = 32,
    VPU_INT16_EPV   = 16,
    VPU_INT32_EPV   =  8,
};

enum {
    VPU_INT8_ACC_PERIOD  = 16,
    VPU_INT16_ACC_PERIOD = 16,
    VPU_INT32_ACC_PERIOD =  8,
};


// VSETC
void vsetc(
    const unsigned value);


// VGETC
void vgetc(
    unsigned* value);

// VCLRDR
void vclrdr();

// VLDR
void vldr(
    const void* data);

// VLDD
void vldd(
    const void* data);

// VLDC
void vldc(
    const void* data);

// VSTR
void vstr(
    void* data);

// VSTRPV
void vstrpv(
    void* data,
    const unsigned mask);

// VSTD
void vstd(
    void* data);

// VSTC
void vstc(
    void* data);

// VLADD
void vladd(
    const void* data);

// VLADDD
void vladdd(
    const void* data);

// VLSUB
void vlsub(
    const void* data);

// VLMUL
/*
    8-bit mode:
        vR[i] =  MAX(INT8_MIN+1,MIN(INT8_MAX,((vR[i]*data[i])+0x20)>>6)))
        if vR[i]*data[i] >= 8128  It will saturate to 127   (8128 == (127 << 6))
        rule of thumb: If the absolute value of one or the other multiplicand is less than or equal to 64, the result will be exact.
*/
void vlmul(
    const void* data);



// VDEPTH1
void vdepth1();

// VDEPTH8
void vdepth8();

// VDEPTH16
void vdepth16();

// VSIGN
void vsign();

// VPOS
void vpos();

// VLASHR
void vlashr(
    const void* data, 
    const int shift);

// VEQDR
void veqdr(
    unsigned* equality);

// VEQCR
void veqcr(
    unsigned* equality);

// VLMACC
void vlmacc(
    const void* data);

// VLMACCR
void vlmaccr(
    const void* data);

// VLMACCR1
void vlmaccr1(
    const void* data);

// VLSAT
void vlsat(
    const void* data);

// VCMR
void vcmr();

// VCMI
void vcmi();

// VCMCR
void vcmcr();

// VCMCI
void vcmci();

// VLADSB
void vladsb(
    const void* data);

// VFTFF
void vftff();

// VFTFB
void vftfb();

// VFTTF
void vfttf();

//VFTTB
void vfttb();





/*
    Simplified versions of some of the above
*/
static inline void vsetc_simple(
    const unsigned element_mode, 
    const unsigned shift_mode, 
    const unsigned headroom)
{
    vsetc((element_mode << 8) | (shift_mode << 6) | headroom);
}

static inline void vgetc_simple(
    unsigned* element_mode,
    unsigned* shift_mode,
    unsigned* headroom)
{
    unsigned val;
    vgetc(&val);

    *element_mode = (val & 0b0000111100000000) >> 8;
    *shift_mode   = (val & 0b0000000011000000) >> 6;
    *headroom     = (val & 0b0000000000111111) >> 0;
}

#ifdef __XC__
} //extern "C"
#endif

#endif //__XS3A__

#endif //XS3_VPU_H_