#ifndef BSO_H_
#define BSO_H_

#include "nn_types.h"
#include "xs3_vpu.h"

/**
 * Macro returns the number of `nn_bso_block_t`s required for `OUT_CHANS` output channels. This is
 * the same as the number output channel groups, rounded up.
 * 
 * @param[in] OUT_CHANS     Number of output channels
 * 
 * @return  Number of required `nn_bso_block_t`.
 */
#define BSO_BLOCK_COUNT(OUT_CHANS) ((OUT_CHANS+(VPU_INT8_VLMACC_ELMS-1))>>VPU_INT8_VLMACC_ELMS_LOG2)

/**
 * Represents the Bias, shifts and scale for a single output channel group.
 * 
 */
typedef struct {
    /**
     * Contains the upper 16-bits of output channel bias for an operator for (up to) 16 channels.
     * 
     * The full 32-bit bias for an output channel corresponding to index `k` is:
     * 
     * @math{ B_{hi}[k]\cdot 2^{16} + B_{lo}[k] } where @math{ B_{hi}[k] } is `bias_hi[k]` interpreted as a signed 16-bit integer,
     * and @math{B_{lo}[k]} is `bias_lo[k]` interpreted as an unsigned 16-bit integer.
     */
    data16_t bias_hi[VPU_INT8_ACC_PERIOD];

    /**
     * Contains the lower 16-bits of output channel bias for an operator for (up to) 16 channels.
     * 
     * The full bias for an output channel corresponding to index `k` is:
     * 
     * @math{ B_{hi}[k]\cdot 2^{16} + B_{lo}[k] } where @math{ B_{hi}[k] } is `bias_hi[k]` interpreted as a signed 16-bit integer,
     * and @math{B_{lo}[k]} is `bias_lo[k]` interpreted as an unsigned 16-bit integer.
     */
    data16_t bias_lo[VPU_INT8_ACC_PERIOD];

    /**
     * Contains the first shift value for an operator for (up to) 16 channels.
     * 
     * After accumulating all weights and input data, the channel corresponding to index `k` is first divided 
     * by @math{ 2^{s_1[k]} }, where @math{s_1[k]} is `shift1[k]`.
     */
    data16_t shift1[VPU_INT8_ACC_PERIOD];

    /**
     * Contains the scale value for an operator for (up to) 16 channels.
     * 
     * After applying the first shift, the result of that is multiplied by `scale[k]`.
     * 
     */
    data16_t scale[VPU_INT8_ACC_PERIOD];

    /**
     * `offset_scale[k]` and `offset[k]` are multiplied together and added to the result of
     * applying the scale.
     */
    data16_t offset_scale[VPU_INT8_ACC_PERIOD];

    /**
     * `offset_scale[k]` and `offset[k]` are multiplied together and added to the result of
     * applying the scale.
     */
    data16_t offset[VPU_INT8_ACC_PERIOD];

    /**
     * Contains the second shift value for an operator for (up to) 16 channels.
     * 
     * After the offset and offset scale are added, the channel corresponding to index `k` is divided 
     * by @math{ 2^{s_2[k]} }, where @math{s_2[k]} is `shift2[k]`.
     */
    data16_t shift2[VPU_INT8_ACC_PERIOD];

} nn_bso_block_t;



#endif //BSO_H_