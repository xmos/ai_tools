

#ifndef XS3_VPU_H_
#define XS3_VPU_H_

#include <xs3a_registers.h>

/* TODO use from xs3a_kernel.h in a future tools release */
#define XS1_VSR_HEADROOM_SHIFT 0x0
#define XS1_VSR_HEADROOM_SIZE 0x5
#define XS1_VSR_HEADROOM_MASK (((1 << XS1_VSR_HEADROOM_SIZE) - 1) << XS1_VSR_HEADROOM_SHIFT)
#define XS1_VSR_HEADROOM(x) (((x) & XS1_VSR_HEADROOM_MASK) >> XS1_VSR_HEADROOM_SHIFT)
#define XS1_VSR_HEADROOM_SET(x, v) (((x) & ~XS1_VSR_HEADROOM_MASK) | (((v) << XS1_VSR_HEADROOM_SHIFT) & XS1_VSR_HEADROOM_MASK))
#define XS1_VSR_SHIFT_SHIFT 0x6
#define XS1_VSR_SHIFT_SIZE 0x2
#define XS1_VSR_SHIFT_MASK (((1 << XS1_VSR_SHIFT_SIZE) - 1) << XS1_VSR_SHIFT_SHIFT)
#define XS1_VSR_SHIFT(x) (((x) & XS1_VSR_SHIFT_MASK) >> XS1_VSR_SHIFT_SHIFT)
#define XS1_VSR_SHIFT_SET(x, v) (((x) & ~XS1_VSR_SHIFT_MASK) | (((v) << XS1_VSR_SHIFT_SHIFT) & XS1_VSR_SHIFT_MASK))
#define XS1_VSR_TYPE_SHIFT 0x8
#define XS1_VSR_TYPE_SIZE 0x4
#define XS1_VSR_TYPE_MASK (((1 << XS1_VSR_TYPE_SIZE) - 1) << XS1_VSR_TYPE_SHIFT)
#define XS1_VSR_TYPE(x) (((x) & XS1_VSR_TYPE_MASK) >> XS1_VSR_TYPE_SHIFT)
#define XS1_VSR_TYPE_SET(x, v) (((x) & ~XS1_VSR_TYPE_MASK) | (((v) << XS1_VSR_TYPE_SHIFT) & XS1_VSR_TYPE_MASK))
#define XS1_VSR_LENGTH_SHIFT 0xc
#define XS1_VSR_LENGTH_SIZE 0x4
#define XS1_VSR_LENGTH_MASK (((1 << XS1_VSR_LENGTH_SIZE) - 1) << XS1_VSR_LENGTH_SHIFT)
#define XS1_VSR_LENGTH(x) (((x) & XS1_VSR_LENGTH_MASK) >> XS1_VSR_LENGTH_SHIFT)
#define XS1_VSR_LENGTH_SET(x, v) (((x) & ~XS1_VSR_LENGTH_MASK) | (((v) << XS1_VSR_LENGTH_SHIFT) & XS1_VSR_LENGTH_MASK))

#define XS1_VSETC_SHIFT_NOSHIFT 0x0
#define XS1_VSETC_SHIFT_SHIFTLEFT 0x1
#define XS1_VSETC_SHIFT_SHIFTRIGHT 0x2
#define XS1_VSETC_TYPE_INT32 0x0
#define XS1_VSETC_TYPE_INT16 0x1
#define XS1_VSETC_TYPE_INT8 0x2

#define XS1_NUM_WORDS_PER_VECTOR 0x8

/* End of xs3a_kernel.h */

#define XS3_VPU_VREG_WIDTH_BITS     (XS1_NUM_WORDS_PER_VECTOR * XS1_ALL_BITS_SIZE)
#define XS3_VPU_VREG_WIDTH_BYTES    (XS3_VPU_VREG_WIDTH_BITS  >> 3)
#define XS3_VPU_VREG_WIDTH_WORDS    (XS3_VPU_VREG_WIDTH_BYTES >> 2)

#ifndef __ASSEMBLER__

enum {
    VEC_INT_32 = 0,   /**< 0 */
    VEC_INT_16 = 1,   /**< 1 */
    VEC_INT_8  = 2,   /**< 2 */
    VEC_FLT_32 = 4,   /**< 4 */
    VEC_FLT_16 = 5,   /**< 5 */
    VEC_FLT_8  = 6,   /**< 6 */
};

enum {
    VEC_SH0 = 0,   /**< 0 */
    VEC_SHL = 1,   /**< 1 */
    VEC_SHR = 2,   /**< 2 */
};

/**
 * The saturation bounds for signed integers in each VPU operating mode.
 */
enum {
    VPU_INT8_MAX =  0x7F,              /**<  0x7F */
    VPU_INT8_MIN = -0x7F,              /**< -0x7F */

    VPU_INT16_MAX =  0x7FFF,           /**<  0x7FFF */
    VPU_INT16_MIN = -0x7FFF,           /**< -0x7FFF */

    VPU_INT32_MAX =  0x7FFFFFFF,       /**<  0x7FFFFFFF */
    VPU_INT32_MIN = -0x7FFFFFFF,       /**< -0x7FFFFFFF */
};

/**
 * Number of accumulator bits in each operating mode.
 * 
 * In each operating mode, the VLMACC, VLMACCR and VLSAT instructions operate on
 * an array of accumulators in the vector registers vR and vD. In each case, the
 * most significant bits are stored in vD, and the least significant bits are stored
 * in vR.
 */
enum {
    VPU_INT8_ACC_SIZE = 32,    /**< 32 */
    VPU_INT16_ACC_SIZE = 32,   /**< 32 */
    VPU_INT32_ACC_SIZE = 40,   /**< 40 */
};

/**
 * When vD and vR contain accumulators, the values in this enum indicate how many least significant 
 * bits are stored in vR, with the remaining bits stored in vD.
 */
enum {
    VPU_INT8_ACC_VR_BITS = 16,     /**< 16 */
    VPU_INT16_ACC_VR_BITS = 16,    /**< 16 */
    VPU_INT32_ACC_VR_BITS = 32,    /**< 32 */
};
/**
 * When vD and vR contain accumulators, the values in this enum can be used to mask off the bits of
 * the accumulator value which correspond to the portion in vR.
 */
enum {
    VPU_INT8_ACC_VR_MASK = 0xFFFF,         /**< 0xFFFF */
    VPU_INT16_ACC_VR_MASK = 0xFFFF,        /**< 0xFFFF */
    VPU_INT32_ACC_VR_MASK = 0xFFFFFFFF,    /**< 0xFFFFFFFF */
};

/**
 * Integer type which fits a single accumulator (32-bits) corresponding to the 8-bit VPU mode.
 */
typedef int32_t vpu_int8_acc_t;

/**
 * Integer type which fits a single accumulator (32-bits) corresponding to the 16-bit VPU mode.
 */
typedef int32_t vpu_int16_acc_t;

/**
 * Integer type which fits a single accumulator (40-bits) corresponding to the 32-bit VPU mode.
 */
typedef int64_t vpu_int32_acc_t;

/**
 * The number of elements which fit into a vector register for each operating mode.
 * 
 * This is also the number of elements which are operated on in the following 
 * instructions: VDEPTH1, VDEPTH16, VDEPTH8, VLADD, VLADDD, VLASHR, VLMACCR, VLMUL, 
 *               VLSUB, VPOS, VSIGN
 *      
 */
enum {
    VPU_INT8_EPV    = 32,   /**< 32 */
    VPU_INT16_EPV   = 16,   /**< 16 */
    VPU_INT32_EPV   =  8,   /**< 8 */
};

/**
 * log-base-2 of the corresponding VPU_INT*_EPV values.
 */
enum {
    VPU_INT8_EPV_LOG2    = 5,   /**< 5 */
    VPU_INT16_EPV_LOG2   = 4,   /**< 4 */
    VPU_INT32_EPV_LOG2   = 3,   /**< 3 */
};

/**
 * The number of accumulators, spread across vR and vD, in each operating mode.
 * 
 * This is also the number of elements consumed (number of multiplies) by the
 * VLMACC instruction.
 */
enum {
    VPU_INT8_ACC_PERIOD  = 16,    /**< 16 */
    VPU_INT16_ACC_PERIOD = 16,    /**< 16 */
    VPU_INT32_ACC_PERIOD =  8,    /**< 8 */
};

/**
 * log-base-2 of the corresponding VPU_INT*_ACC_PERIOD values.
 */
enum {
    VPU_INT8_ACC_PERIOD_LOG2  = 4,   /**< 4 */
    VPU_INT16_ACC_PERIOD_LOG2 = 4,   /**< 4 */
    VPU_INT32_ACC_PERIOD_LOG2 = 3,   /**< 3 */
};

/**
 * The number of elements consumed by a VLMACC instruction in each operating mode.
 * In other words, the number of simultaneous multiply-accumulates performed by the VLMACC
 * instruction.
 */
enum {
    VPU_INT8_VLMACC_ELMS = 16,    /**< 16 */
    VPU_INT16_VLMACC_ELMS = 16,   /**< 16 */
    VPU_INT32_VLMACC_ELMS = 8,    /**< 8 */
};

/**
 * log-base-2 of the corresponding VPU_INT*_VLMACC_ELMS values.
 */
enum {
    VPU_INT8_VLMACC_ELMS_LOG2 = 4,    /**< 4 */
    VPU_INT16_VLMACC_ELMS_LOG2 = 4,   /**< 4 */
    VPU_INT32_VLMACC_ELMS_LOG2 = 3,   /**< 3 */
};

#endif //__ASM__

#endif //XS3_VPU_H_
