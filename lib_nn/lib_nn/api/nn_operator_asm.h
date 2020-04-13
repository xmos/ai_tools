

#ifndef NN_OPERATOR_ASM_H_
#define NN_OPERATOR_ASM_H_

#include <stdint.h>
#include "nn_types.h"

#include "nn_op_structs.h"

#ifdef __XC__
extern "C" {
#endif

#ifdef __XS3A__


 
#ifndef USE_ASM_conv2d_shallowin_deepout_block
#define USE_ASM_conv2d_shallowin_deepout_block     (1)
#endif
void conv2d_shallowin_deepout_block_asm(
    int8_t* Y,
    const nn_conv2d_sido_params_t* params,
    const nn_conv2d_sido_block_params_t* block,
    const int8_t* X,
    const int8_t* K,
    const int16_t* scales);   

#ifndef USE_ASM_fc_deepin_shallowout_16
#define USE_ASM_fc_deepin_shallowout_16    (1)
#endif
void fc_deepin_shallowout_16_asm(
    const int8_t* W, 
    const int32_t* B,
    const int8_t* X, 
    int16_t* Y,
    const int32_t C_out, 
    const int32_t C_in,
    const uint16_t* shifts, 
    const int16_t* scales);
    


#ifndef USE_ASM_fc_deepin_shallowout_8
#define USE_ASM_fc_deepin_shallowout_8    (1)
#endif
void fc_deepin_shallowout_8_asm(
    const int8_t* W, 
    const int32_t* B,
    const int8_t* X, 
    int8_t* Y,
    const int32_t C_out, 
    const int32_t C_in,
    const uint16_t* shifts, 
    const int16_t* scales);


#ifndef USE_ASM_avgpool2d
#define USE_ASM_avgpool2d      (1)
#endif
void avgpool2d_asm(
    int8_t* Y,
    const int8_t* X, 
    const nn_avgpool2d_plan_t* plan);

void avgpool2d_2x2_asm(
    int8_t* Y,
    const int8_t* X, 
    const nn_avgpool2d_plan_t* plan);


#endif //__XS3A__

#ifdef __XC__
}   //extern "C"
#endif

#endif //NN_OPERATOR_ASM_H_