

#ifndef XS3_VPU_H_
#define XS3_VPU_H_

#ifdef __XS3A__


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
    VPU_INT8_EPV_LOG2    = 5,
    VPU_INT16_EPV_LOG2   = 4,
    VPU_INT32_EPV_LOG2   = 3,
};

enum {
    VPU_INT8_ACC_PERIOD  = 16,
    VPU_INT16_ACC_PERIOD = 16,
    VPU_INT32_ACC_PERIOD =  8,
};

enum {
    VPU_INT8_ACC_PERIOD_LOG2  = 4,
    VPU_INT16_ACC_PERIOD_LOG2 = 4,
    VPU_INT32_ACC_PERIOD_LOG2 = 3,
};



#endif //__XS3A__

#endif //XS3_VPU_H_