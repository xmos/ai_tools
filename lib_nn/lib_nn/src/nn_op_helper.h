
#ifndef NN_OP_HELPER_H_
#define NN_OP_HELPER_H_


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
    int64_t acc64 = acc;
    if(shr > 0) acc64 += 1<<(shr-1);
    return sat_s8(acc64 >> shr);
}

static inline int16_t vlsat_single_s16(
    int32_t acc, 
    int16_t shr)
{
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

#endif //NN_OP_HELPER_H_