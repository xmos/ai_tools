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


void VSETC(const vector_mode mode);
void VCLRDR();
void VLDR(const void* addr);
void VLDD(const void* addr);
void VLDC(const void* addr);
void VSTR(void* addr);
void VSTD(void* addr);
void VSTC(void* addr);
void VSTRPV(void* addr, unsigned mask);
void VLMACC(const void* addr);
void VLMACCR(const void* addr);
void VLSAT(const void* addr);


void print_vR(const unsigned hex, const char* extra, const unsigned line);
void print_vD(const unsigned hex, const char* extra, const unsigned line);
void print_vC(const unsigned hex, const char* extra, const unsigned line);
void print_accumulators(const unsigned hex);