
#include "vpu_sim.h"

#include <stdio.h>


/**
 * Saturate to the relevent bounds.
 */
static int64_t saturate(
    const int64_t input,
    const unsigned bits)
{
    const int64_t max_val = (((int64_t)1)<<(bits-1))-1;
    const int64_t min_val = -max_val;
    
    return (input > max_val)?  max_val : (input < min_val)? min_val : input;
}

/**
 * Get the accumulator for the VPU's current mode
 */
static int64_t get_accumulator(
    const xs3_vpu* vpu,
    unsigned index)
{
    if(vpu->mode == MODE_S8 || vpu->mode == MODE_S16){
        union {
            int16_t s16[2];
            int32_t s32;
        } acc;
        acc.s16[1] = vpu->vD.s16[index];
        acc.s16[0] = vpu->vR.s16[index];
        
        return acc.s32;
    } else {
        assert(0); // TODO
    }

}

/** 
 * Set the accumulator for the VPU's current mode
 */
static void set_accumulator(
    xs3_vpu* vpu,
    unsigned index,
    int64_t acc)
{
    if(vpu->mode == MODE_S8 || vpu->mode == MODE_S16){
        
        unsigned mask = (1<<VPU_INT8_ACC_VR_BITS)-1;
        vpu->vR.s16[index] = acc & mask;
        mask = mask << VPU_INT8_ACC_VR_BITS;
        vpu->vD.s16[index] = ((acc & mask) >> VPU_INT8_ACC_VR_BITS);

    } else {
        assert(0); // TODO
    }
}

/**
 * Rotate the accumulators following a VLMACCR
 */
static void rotate_accumulators(
    xs3_vpu* vpu)
{
    if(vpu->mode == MODE_S8 || vpu->mode == MODE_S16){
        data16_t tmpD = vpu->vD.u16[VPU_INT8_ACC_PERIOD-1];
        data16_t tmpR = vpu->vR.u16[VPU_INT8_ACC_PERIOD-1];
        for(int i = VPU_INT8_ACC_PERIOD-1; i > 0; i--){
            vpu->vD.u16[i] = vpu->vD.u16[i-1];
            vpu->vR.u16[i] = vpu->vR.u16[i-1];
        }
        vpu->vD.u16[0] = tmpD;
        vpu->vR.u16[0] = tmpR;
    } else if(vpu->mode == MODE_S32) {
        uint32_t tmpD = vpu->vD.u32[VPU_INT32_ACC_PERIOD-1];
        uint32_t tmpR = vpu->vR.u32[VPU_INT32_ACC_PERIOD-1];
        for(int i = VPU_INT32_ACC_PERIOD-1; i > 0; i--){
            vpu->vD.u32[i] = vpu->vD.u32[i-1];
            vpu->vR.u32[i] = vpu->vR.u32[i-1];
        }
        vpu->vD.u32[0] = tmpD;
        vpu->vR.u32[0] = tmpR;
    } else {
        assert(0); //How'd this happen?
    }
}


void VSETC(
    xs3_vpu* vpu,
    const vector_mode mode){
    vpu->mode = mode;
}

void VCLRDR(xs3_vpu* vpu){
    memset(&vpu->vR.u8[0], 0 ,XS3_VPU_VREG_WIDTH_BYTES);
    memset(&vpu->vD.u8[0], 0 ,XS3_VPU_VREG_WIDTH_BYTES);
}

void VLDR(xs3_vpu* vpu, const void* addr){
    memcpy(&vpu->vR.u8[0], addr, XS3_VPU_VREG_WIDTH_BYTES);
}

void VLDD(xs3_vpu* vpu, const void* addr){
    memcpy(&vpu->vD.u8[0], addr, XS3_VPU_VREG_WIDTH_BYTES);
}

void VLDC(xs3_vpu* vpu, const void* addr){
    memcpy(&vpu->vC.u8[0], addr, XS3_VPU_VREG_WIDTH_BYTES);
}

void VSTR(const xs3_vpu* vpu, void* addr){
    memcpy(addr, &vpu->vR.u8[0], XS3_VPU_VREG_WIDTH_BYTES);
}

void VSTD(const xs3_vpu* vpu, void* addr){
    memcpy(addr, &vpu->vD.u8[0], XS3_VPU_VREG_WIDTH_BYTES);
}

void VSTC(const xs3_vpu* vpu, void* addr){
    memcpy(addr, &vpu->vC.u8[0], XS3_VPU_VREG_WIDTH_BYTES);
}

void VSTRPV(const xs3_vpu* vpu, void* addr, unsigned mask){
    int8_t* addr8 = (int8_t*) addr;

    for(int i = 0; i < 32; i++){
        if(mask & (1 << i)){
            addr8[i] = vpu->vR.s8[i];
        }
    }
}

void VLMACC(
    xs3_vpu* vpu,
    const void* addr)
{
    if(vpu->mode == MODE_S8){
        const int8_t* addr8 = (const int8_t*) addr;

        for(int i = 0; i < VPU_INT8_VLMACC_ELMS; i++){
            int64_t acc = get_accumulator(vpu, i);
            acc = acc + (((int32_t)vpu->vC.s8[i]) * addr8[i]);

            set_accumulator(vpu, i, saturate(acc, 32));
        }
    } else if(vpu->mode == MODE_S16){
        const int16_t* addr16 = (const int16_t*) addr;

        for(int i = 0; i < VPU_INT16_VLMACC_ELMS; i++){
            int64_t acc = get_accumulator(vpu, i);
            acc = acc + (((int32_t)vpu->vC.s16[i]) * addr16[i]);

            set_accumulator(vpu, i, saturate(acc, 32));
        }
    } else if(vpu->mode == MODE_S32){
        const int32_t* addr32 = (const int32_t*) addr;

        for(int i = 0; i < VPU_INT32_VLMACC_ELMS; i++){
            int64_t acc = get_accumulator(vpu, i);
            acc = acc + (((int64_t)vpu->vC.s32[i]) * addr32[i]);

            set_accumulator(vpu, i, saturate(acc, 40));
        }
    } else { 
        assert(0); //How'd this happen?
    }
}

void VLMACCR(
    xs3_vpu* vpu,
    const void* addr)
{
    if(vpu->mode == MODE_S8){
        const int8_t* addr8 = (const int8_t*) addr;
        int64_t acc = get_accumulator(vpu, VPU_INT8_ACC_PERIOD-1);

        for(int i = 0; i < VPU_INT8_EPV; i++)
            acc = acc + (((int32_t)vpu->vC.s8[i]) * addr8[i]);

        acc = saturate(acc, 32);
        rotate_accumulators(vpu);
        set_accumulator(vpu, 0, acc);
    } else if(vpu->mode == MODE_S16){
        const int16_t* addr16 = (const int16_t*) addr;
        int64_t acc = get_accumulator(vpu, VPU_INT16_ACC_PERIOD-1);

        for(int i = 0; i < VPU_INT16_EPV; i++)
            acc = acc + (((int32_t)vpu->vC.s16[i]) * addr16[i]);

        acc = saturate(acc, 32);
        rotate_accumulators(vpu);
        set_accumulator(vpu, 0, acc);
    } else if(vpu->mode == MODE_S32){
        const int32_t* addr32 = (const int32_t*) addr;
        int32_t acc = get_accumulator(vpu, VPU_INT32_ACC_PERIOD-1);

        for(int i = 0; i < VPU_INT32_EPV; i++)
            acc = acc + (((int32_t)vpu->vC.s32[i]) * addr32[i]);

        acc = saturate(acc, 40);
        rotate_accumulators(vpu);
        set_accumulator(vpu, 0, acc);
    } else { 
        assert(0); //How'd this happen?
    }
}

void VLSAT(
    xs3_vpu* vpu,
    const void* addr)
{
    if(vpu->mode == MODE_S8){
        const uint16_t* addr16 = (const uint16_t*) addr;

        for(int i = 0; i < VPU_INT8_ACC_PERIOD; i++){
            int32_t acc = get_accumulator(vpu, i);

            if(addr16[i] != 0)
                acc = acc + (1 << (addr16[i]-1));   //Round
            acc = acc >> addr16[i];             //Shift
            int8_t val = saturate(acc, 8);      //Saturate

            vpu->vR.s8[i] = val;
        }
        memset(&vpu->vD.u8[0], 0, XS3_VPU_VREG_WIDTH_BYTES);
        memset(&vpu->vR.u8[VPU_INT8_ACC_PERIOD], 0, VPU_INT8_ACC_PERIOD);
    } else if(vpu->mode == MODE_S16){
        const uint16_t* addr16 = (const uint16_t*) addr;

        for(int i = 0; i < VPU_INT16_ACC_PERIOD; i++){
            int32_t acc = get_accumulator(vpu, i);
            if(addr16[i] != 0)
                acc = acc + (1 << ((int16_t)(addr16[i]-1)));   //Round

            acc = acc >> addr16[i];             //Shift
            int16_t val = saturate(acc, 16);    //Saturate

            vpu->vR.s16[i] = val;
        }
        memset(&vpu->vD.u8[0], 0, XS3_VPU_VREG_WIDTH_BYTES);

    } else if(vpu->mode == MODE_S32){
        const uint32_t* addr32 = (const uint32_t*) addr;

        for(int i = 0; i < VPU_INT32_ACC_PERIOD; i++){
            int64_t acc = get_accumulator(vpu, i);
            if(addr32[i] != 0)
                acc = acc + (1 << (addr32[i]-1));   //Round
            acc = acc >> addr32[i];             //Shift
            int32_t val = saturate(acc, 32);    //Saturate

            vpu->vR.s32[i] = val;
        }
        memset(&vpu->vD.u8[0], 0, XS3_VPU_VREG_WIDTH_BYTES);
    } else { 
        assert(0); //How'd this happen?
    }
}

void VLASHR(
    xs3_vpu* vpu, 
    const void* addr,
    const int32_t shr)
{
    if(vpu->mode == MODE_S8){
        const int8_t* addr8 = (const int8_t*) addr;

        for(int i = 0; i < VPU_INT8_EPV; i++){
            int32_t val = addr8[i];

            if(shr >= 7)        val = (val < 0)? -1 : 0;
            else if(shr >= 0)   val = val >> shr;
            else                val = val << (-shr);

            vpu->vR.s8[i] = saturate(val, 8);
        }
    } else if(vpu->mode == MODE_S16){
        const int16_t* addr16 = (const int16_t*) addr;

        for(int i = 0; i < VPU_INT16_EPV; i++){
            int32_t val = addr16[i];
            if(shr >= 15)   val = (val < 0)? -1 : 0;
            else if(shr >= 0)   val = val >> shr;
            else                val = val << (-shr);
            vpu->vR.s16[i] = saturate(val, 16);
        }
    } else if(vpu->mode == MODE_S32){
        const int32_t* addr32 = (const int32_t*) addr;

        for(int i = 0; i < VPU_INT32_EPV; i++){
            int64_t val = addr32[i];
            if(shr >= 31)   val = (val < 0)? -1 : 0;
            else if(shr >= 0)   val = val >> shr;
            else                val = val << (-shr);
            vpu->vR.s32[i] = saturate(val, 32);
        }
    } else { 
        assert(0); //How'd this happen?
    }
}

void VLADD(
    xs3_vpu* vpu, 
    const void* addr)
{
    if(vpu->mode == MODE_S8){
        const int8_t* addr8 = (const int8_t*) addr;
        for(int i = 0; i < VPU_INT8_EPV; i++){
            int32_t val = addr8[i];
            vpu->vR.s8[i] = saturate(vpu->vR.s8[i] + val, 8);
        }
    } else if(vpu->mode == MODE_S16){
        const int16_t* addr16 = (const int16_t*) addr;

        for(int i = 0; i < VPU_INT16_EPV; i++){
            int32_t val = addr16[i];
            vpu->vR.s16[i] = saturate(vpu->vR.s16[i] + val, 16);
        }
    } else if(vpu->mode == MODE_S32){
        const int32_t* addr32 = (const int32_t*) addr;

        for(int i = 0; i < VPU_INT32_EPV; i++){
            int64_t val = addr32[i];
            vpu->vR.s32[i] = saturate(vpu->vR.s32[i] + val, 32);
        }
    } else { 
        assert(0); //How'd this happen?
    }
}




void VDEPTH1(xs3_vpu* vpu){

    uint32_t bits = 0;
    
    if(vpu->mode == MODE_S8){
        for(int i = 0; i < VPU_INT8_EPV; i++){
            if(vpu->vR.s8[i] < 0)
                bits |= (1 << i);
        }
    } else if(vpu->mode == MODE_S16){
        for(int i = 0; i < VPU_INT16_EPV; i++){
            if(vpu->vR.s16[i] < 0)
                bits |= (1 << i);
        }
    } else if(vpu->mode == MODE_S32){
        for(int i = 0; i < VPU_INT32_EPV; i++){
            if(vpu->vR.s32[i] < 0)
                bits |= (1 << i);
        }
    } else { 
        assert(0);
    }

    memset(&(vpu->vR), 0, sizeof(vpu_vector_t));
    vpu->vR.s32[0] = bits;
}


void VDEPTH8(xs3_vpu* vpu){

    vpu_vector_t vec_tmp;
    memcpy(&vec_tmp, &(vpu->vR), sizeof(vpu_vector_t));
    memset(&(vpu->vR), 0, sizeof(vpu_vector_t));
    
    if(vpu->mode == MODE_S16){
        for(int i = 0; i < VPU_INT16_EPV; i++){
            int32_t elm = ((int32_t)vec_tmp.s16[i]) + (1 << 7);
            vpu->vR.s8[i] = saturate(elm >> 8, 8);
        }
    } else if(vpu->mode == MODE_S32){
        for(int i = 0; i < VPU_INT32_EPV; i++){
            int64_t elm = ((int64_t)vec_tmp.s32[i]) + (1 << 23);
            vpu->vR.s8[i] = saturate(elm >> 24, 8);
        }
    } else { 
        assert(0);
    }
}


void VDEPTH16(xs3_vpu* vpu){

    
    if(vpu->mode == MODE_S32){
        for(int i = 0; i < VPU_INT32_EPV; i++){
            int64_t elm = ((int64_t)vpu->vR.s32[i]) + (1 << 15);
            vpu->vR.s16[i] = saturate(elm >> 16, 16);
        }

        for(int i = VPU_INT32_EPV; i < VPU_INT16_EPV; i++){
            vpu->vR.s16[i] = 0;
        }
    } else { 
        assert(0);
    }
}


static char signof(int x){return (x>=0? ' ' : '-');}

void vpu_sim_print(xs3_vpu* vpu)
{
    int8_t * vC8 = vpu->vC.s8;
    int8_t * vR8 = vpu->vR.s8;
    int8_t * vD8 = vpu->vD.s8;

    int16_t * vC16 = vpu->vC.s16;
    int16_t * vR16 = vpu->vR.s16;
    int16_t * vD16 = vpu->vD.s16;

    int32_t * vC32 = vpu->vC.s32;
    int32_t * vR32 = vpu->vR.s32;
    int32_t * vD32 = vpu->vD.s32;


    switch (vpu->mode)
    {
    case MODE_S8:
        printf("8-bit:     vC     \t  vR     \t   vD\n");
        for(int i = 0; i< VPU_INT8_EPV; i++){
            printf("%d\t%c0x%0.2X(%d)\t%c0x%0.2X(%d)\t%c0x%0.2X(%d)\n",
            i,signof(vC8[i]),abs(vC8[i]), vC8[i],signof(vR8[i]), abs(vR8[i]), vR8[i], signof(vD8[i]), abs(vD8[i]),vD8[i]);
        }
        break;
    
    case MODE_S16:
        printf("16-bit:  vC     \t    vR      \t    vD\n");
        for(int i = 0; i< VPU_INT16_EPV; i++){
            printf("%d\t%c0x%0.4X(%d)\t%c0x%0.4X(%d)\t%c0x%0.4X(%d)\n",
            i,signof(vC16[i]),abs(vC16[i]), vC16[i],signof(vR16[i]), abs(vR16[i]), vR16[i], signof(vD16[i]), abs(vD16[i]),vD16[i]);
        }
        break;

   case MODE_S32:
        printf("32-bit:  vC     \t\t    vR      \t\t    vD\n");
        for(int i = 0; i< VPU_INT32_EPV; i++){
            printf("%d\t%c0x%0.8X(%d)\t%c0x%0.8X(%d)\t%c0x%0.8X(%d)\n",
            i,signof(vC32[i]),abs(vC32[i]), vC32[i],signof(vR32[i]), abs(vR32[i]), vR32[i], signof(vD32[i]), abs(vD32[i]),vD32[i]);        }
        break;

    default:
        printf("In the future this might print all possible interpretations...");
        break;
    } 

    printf("\n");

}