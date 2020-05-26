#pragma once

#ifndef __ASSEMBLER__

#include <stdint.h>

#include "xs3_vpu.h"




typedef struct {

    //Word offset = 0
    int16_t vec_0x007F[VPU_INT8_ACC_PERIOD];
    //Word offset = 8
    int8_t vec_0x01[VPU_INT8_ACC_PERIOD];
    //Word offset = 12
    int16_t vec_0x0002[VPU_INT8_ACC_PERIOD];
    //Word offset = 20
    int8_t vec_0x80[VPU_INT8_EPV];
    //Word offset = 28

} vpu_constants_t;

extern const vpu_constants_t vpu_vects;

#endif // __ASSEMBLER__



#define VPU_VEC_0x007F  (0)
#define VPU_VEC_0x01    (8)
#define VPU_VEC_0x0002  (12)
#define VPU_VEC_0x80    (20)