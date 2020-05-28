#pragma once

#include <stdint.h>
#include <assert.h>
#include <string.h>

#include "nn_types.h"
#include "xs3_vpu.h"

typedef union {
    uint8_t  u8[VPU_INT8_EPV];
    int8_t   s8[VPU_INT8_EPV];

    uint16_t u16[VPU_INT16_EPV];
    int16_t  s16[VPU_INT16_EPV];

    uint32_t u32[VPU_INT32_EPV];
    int32_t  s32[VPU_INT32_EPV];
} vpu_vector_t;

typedef enum {
    MODE_S32 = 0x00,
    MODE_S16 = 0x100,
    MODE_S8  = 0x200,
} vector_mode;

typedef struct {
    vector_mode mode;
    vpu_vector_t vR;
    vpu_vector_t vD;
    vpu_vector_t vC;
} xs3_vpu;


void VSETC(
    xs3_vpu* vpu,
    const vector_mode mode);
void VCLRDR(xs3_vpu* vpu);
void VLDR(xs3_vpu* vpu, const void* addr);
void VLDD(xs3_vpu* vpu, const void* addr);
void VLDC(xs3_vpu* vpu, const void* addr);
void VSTR(const xs3_vpu* vpu, void* addr);
void VSTD(const xs3_vpu* vpu, void* addr);
void VSTC(const xs3_vpu* vpu, void* addr);
void VSTRPV(const xs3_vpu* vpu, void* addr, unsigned mask);
void VLMACC(xs3_vpu* vpu, const void* addr);
void VLMACCR(xs3_vpu* vpu, const void* addr);
void VLSAT(xs3_vpu* vpu, const void* addr);
void VLASHR(xs3_vpu* vpu, const void* addr, const int32_t shr);
void VLADD(xs3_vpu* vpu, const void* addr);
void VDEPTH1(xs3_vpu* vpu);
void VDEPTH8(xs3_vpu* vpu);
void VDEPTH16(xs3_vpu* vpu);

void vpu_sim_print(xs3_vpu* vpu);