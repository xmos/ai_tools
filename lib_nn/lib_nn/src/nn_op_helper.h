
#ifndef NN_OP_HELPER_H_
#define NN_OP_HELPER_H_

#define WEAK_FUNC __attribute__((weak))

static inline int8_t sat_s8(
    const int32_t acc32)
{
    if(acc32 >= VPU_INT8_MAX)
        return VPU_INT8_MAX;
    if(acc32 <= VPU_INT8_MIN)
        return VPU_INT8_MIN;
    
    return (int8_t) acc32;
}

static inline int16_t sat_s16(
    const int32_t acc32)
{
    if(acc32 >= VPU_INT16_MAX)
        return VPU_INT16_MAX;
    if(acc32 <= VPU_INT16_MIN)
        return VPU_INT16_MIN;
    
    return (int16_t) acc32;
}

static inline int32_t sat_s32(
    const int64_t acc64)
{
    if(acc64 >= VPU_INT32_MAX)
        return VPU_INT32_MAX;
    if(acc64 <= VPU_INT32_MIN)
        return VPU_INT32_MIN;
    
    return (int32_t) acc64;
}

// static inline void mulsat_s32(int32_t* acc32, const int8_t a, const int8_t b)
// {
//     int64_t acc64 = *acc32 + a*b;
//     *acc32 = sat_s32(acc64);
// }

static inline int8_t vlsat_single_s8(
    int32_t acc, 
    int16_t shr)
{
    shr = (shr <= 0)? 0 : shr;
    int64_t acc64 = acc;
    if(shr > 0) acc64 += 1<<(shr-1);
    return sat_s8(acc64 >> shr);
}

static inline int16_t vlsat_single_s16(
    int32_t acc, 
    int16_t shr)
{
    shr = (shr <= 0)? 0 : shr;
    if(shr > 0) acc += 1<<(shr-1);
    return sat_s16(acc >> shr);
}

static inline int16_t vlmul_single_s16(
    int16_t vR, 
    int16_t mem)
{
    int32_t p = ((int32_t)vR) * mem;
    p = vlsat_single_s16(p, 14);
    return (int16_t)p;
}

static inline int8_t vlmul_single_s8(
    int8_t vR, 
    int8_t mem)
{
    int32_t p = ((int32_t)vR) * mem;
    p = vlsat_single_s8(p, 6);
    return (int8_t)p;
}

static inline int8_t vdepth8_single_s16(
    int16_t vR)
{
    return vlsat_single_s8(vR, 8);
}


static inline unsigned in_bounds(
    const unsigned region_top,
    const unsigned region_left,
    const unsigned region_bottom,
    const unsigned region_right,
    const unsigned bounds_top,
    const unsigned bounds_left,
    const unsigned bounds_bottom,
    const unsigned bounds_right)
{
    return region_top > bounds_bottom
        || region_bottom < bounds_top
        || region_left > bounds_right
        || region_right < bounds_left;
}


static inline unsigned clip(
    const unsigned low, 
    const unsigned val, 
    const unsigned high)
{
    if(val <= low)
        return low;
    else if(val >= high)
        return high;
    else
        return val;
}

static inline unsigned smax(
    const unsigned a,
    const unsigned b)
{
    return (a >= b)? a : b;
}

static inline unsigned smin(
    const unsigned a,
    const unsigned b)
{
    return (a <= b)? a : b;
}


static inline int ceil_log2(
    uint32_t a)
{
    if(a == 0) return -1;
#ifdef  __XS3A__
    unsigned x;
    asm("clz %0, %1" : "=r"(x) : "r"(a));
    unsigned y = 31-x;

    //  clz(1) = 31 -> 31-31 = 0 -> 2^0 = 1
    //  clz(2) = 30 -> 31-30 = 1 -> 2^1 = 2
    //  clz(3) = 30 -> 31-30 = 1 -> 2^1 = 2
    //      2^(y) <= a < 2^(y+1)
    //  check for the lower bound, which yields a different result
    if(a == (1<<y)) return y;
    return y+1;

#else
    for(unsigned i = 0; i < 31; i++){
        if((((unsigned)1)<<i) >= a){
            return i;
        }
    }
#endif
    return -1;
}

#endif //NN_OP_HELPER_H_