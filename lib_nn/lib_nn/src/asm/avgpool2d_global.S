
#if defined(__XS3A__)

#include "nn_config.h"
#include "asm_constants.h"

/*  
void avgpool2d_global(
    int8_t* Y,
    const int8_t* X, 
    const int32_t bias,
    const int8_t scale,
    const int16_t shift,
    const nn_avgpool2d_global_plan_t* plan,
    const nn_avgpool2d_global_job_t* job);
*/

#define FUNCTION_NAME avgpool2d_global

#define NSTACKWORDS  (5*8 + 12)
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 4
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

#define arg_bias        r2
#define arg_scale       r3


#define PLAN_X_PIXELS   0
#define PLAN_X_CHANS    1

#define JOB_STRIDE_START    0
#define JOB_OUT_CHANNELS    1

#define STACK_SHIFT_VEC     (NSTACKWORDS-8)
#define STACK_BIAS_HI_VEC   (NSTACKWORDS-16)
#define STACK_BIAS_LO_VEC   (NSTACKWORDS-24)
#define STACK_VEC_TMP1      (NSTACKWORDS-32)
#define STACK_VEC_TMP2      (NSTACKWORDS-40)

#define STACK_SHIFT         (NSTACKWORDS+1)
#define STACK_PLAN          (NSTACKWORDS+2)
#define STACK_JOB           (NSTACKWORDS+3)
#define STACK_CP            1

#define Y                   r0
#define X                   r1
#define chans               r2
#define pix_groups          r3
#define chan_grp_left       r4
#define x_chan_incr         r5
#define write_mask          r6
#define chan_tail           r7
#define pix_loop_hop        r8
#define pix_count           r9
#define _16                 r10

#define Q(ASD)          ASD

.align 4
.L_bso_consts:
.word 0x00010001, 0x01010101


.align 16
FUNCTION_NAME:
    dualentsp NSTACKWORDS
    std r4, r5, sp[1]
    std r6, r7, sp[2]
    std r8, r9, sp[3]
    {   ldc _16, 16                             ;   stw r10, sp[8]                          }

    //Change constant pool pointer to refer to the constant VPU vects needed here
    ldaw r11, cp[vpu_vects]
    {   ldaw r11, cp[0]                         ;   set cp, r11                             }
    {   ldap r11, .L_bso_consts                 ;   stw r11, sp[STACK_CP]                   }

    {   ldaw r6, sp[STACK_BIAS_HI_VEC]          ;   ldw r4, r11[0]                          }
    {   ldaw r7, sp[STACK_BIAS_LO_VEC]          ;                                           }
    {   shr r8, arg_bias, 16                    ;   zext arg_bias, 16                       }
        mul arg_bias, r4, arg_bias
        std arg_bias, arg_bias, r7[0]   // Bias low
        std arg_bias, arg_bias, r7[1]
        std arg_bias, arg_bias, r7[2]
        std arg_bias, arg_bias, r7[3]
        mul arg_bias, r4, r8
        std arg_bias, arg_bias, r6[0]   // Bias high
        std arg_bias, arg_bias, r6[1]
        std arg_bias, arg_bias, r6[2]
        std arg_bias, arg_bias, r6[3]

#undef arg_bias
#define job     r2

    {                                           ;   ldw job, sp[STACK_JOB]                  }
    {                                           ;   ldw r11, job[JOB_STRIDE_START]          }
    {                                           ;   ldw r4, sp[STACK_PLAN]                  }
    {   add Y, Y, r11                           ;   ldw pix_count, r4[PLAN_X_PIXELS]        }
    {   add X, X, r11                           ;   shl r11, _16, 5                         }

    {   ldap r11, .L_bso_consts                 ;   vsetc r11                               }

    {   ldaw r6, sp[STACK_SHIFT_VEC]            ;   ldw r4, r11[1]                          }
        mul arg_scale, arg_scale, r4
        std arg_scale, arg_scale, r6[0]
        std arg_scale, arg_scale, r6[1]
        std arg_scale, arg_scale, r6[2]
        std arg_scale, arg_scale, r6[3]

#undef arg_scale

    {                                           ;   vldc r6[0]                              }
    {                                           ;   ldw r4, r11[0]                          }
    {                                           ;   ldw r10, sp[STACK_SHIFT]                }
        mul r10, r4, r10
        std r10, r10, r6[0]
        std r10, r10, r6[1]
        std r10, r10, r6[2]
        std r10, r10, r6[3]
    
    {                                           ;   ldw r4, sp[STACK_PLAN]                  }
    {                                           ;   ldw r3, job[JOB_OUT_CHANNELS]           }
    {                                           ;   ldw chans, r4[PLAN_X_CHANS]             }

#undef job

        mul x_chan_incr, pix_count, chans
    {   ldc _16, 16                             ;   shr chan_grp_left, r3, 4                }
    {   sub x_chan_incr, x_chan_incr, _16       ;   mkmsk r11, 4                            }

    {   and chan_tail, r3, r11                  ;   and pix_loop_hop, pix_count, r11        }
    
    {                                           ;   sub pix_loop_hop, _16, pix_loop_hop     }
    {   mkmsk write_mask, 16                    ;   bt chan_grp_left, .L_chan_loop          }
    {   mkmsk write_mask, chan_tail             ;   ldc chan_grp_left, 1                    }
     
    .align 4
    .L_chan_loop:
        {   shl r11, _16, 5                         ;   sub chan_grp_left, chan_grp_left, 1     }
        {   ldaw r11, sp[STACK_BIAS_HI_VEC]         ;   vsetc r11                               }
        {   ldaw r11, sp[STACK_BIAS_LO_VEC]         ;   vldd r11[0]                             }
        {   shr pix_groups, pix_count, 4            ;   vldr r11[0]                             }
#if CONFIG_SYMMETRIC_SATURATION_avgpool2d_global
        {   ldaw r11, sp[STACK_SHIFT_VEC]           ;   bru pix_loop_hop                        } // Do pix tail first
#else //CONFIG_SYMMETRIC_SATURATION_avgpool2d_global
        {   ldaw r11, sp[STACK_VEC_TMP1]            ;   bru pix_loop_hop                        } // Do pix tail first
#endif //CONFIG_SYMMETRIC_SATURATION_avgpool2d_global
        .L_pix_loop:
            {   add X, X, chans                         ;   vlmacc X[0]                             }   // 0  -- should be a no-op
            {   add X, X, chans                         ;   vlmacc X[0]                             }
            {   add X, X, chans                         ;   vlmacc X[0]                             }
            {   add X, X, chans                         ;   vlmacc X[0]                             }
            {   add X, X, chans                         ;   vlmacc X[0]                             }
            {   add X, X, chans                         ;   vlmacc X[0]                             }
            {   add X, X, chans                         ;   vlmacc X[0]                             }
            {   add X, X, chans                         ;   vlmacc X[0]                             }
            {   add X, X, chans                         ;   vlmacc X[0]                             }
            {   add X, X, chans                         ;   vlmacc X[0]                             }
            {   add X, X, chans                         ;   vlmacc X[0]                             }   // 10
            {   add X, X, chans                         ;   vlmacc X[0]                             }
            {   add X, X, chans                         ;   vlmacc X[0]                             }
            {   add X, X, chans                         ;   vlmacc X[0]                             }
            {   add X, X, chans                         ;   vlmacc X[0]                             }
            {   add X, X, chans                         ;   vlmacc X[0]                             }   // 15
            {   sub pix_groups, pix_groups, 1           ;   bt pix_groups, .L_pix_loop              }   // 16
        .L_pix_loop_end:        

#if CONFIG_SYMMETRIC_SATURATION_avgpool2d_global
        {   sub X, X, x_chan_incr                   ;   vlsat r11[0]                            }
        vstrpv Y[0], write_mask
        {   add Y, Y, _16                           ;   bt chan_grp_left, .L_chan_loop          }
#else //CONFIG_SYMMETRIC_SATURATION_avgpool2d_global

        {   ldaw r11, sp[STACK_VEC_TMP2]            ;   vstd r11[0]                             }
        {   ldaw r11, sp[STACK_SHIFT_VEC]           ;   vstr r11[0]                             }
        {   shl r11, _16, 4                         ;   vlsat r11[0]                            }
        {   ldaw r11, sp[STACK_VEC_TMP1]            ;   vsetc r11       /* 16-bit mode */       }
        vstrpv Y[0], write_mask
        {   ldaw r11, sp[STACK_VEC_TMP2]            ;   vldd r11[0]                             }
        {   ldaw r11, sp[STACK_SHIFT_VEC]           ;   vldr r11[0]                             }
        {   ldaw r11, cp[VPU_VEC_0x007F]            ;   vlsat r11[0]                            }
        {   sub X, X, x_chan_incr                   ;   vladd r11[0]                            }   //only 0xFF80 will remain negative (0xFFFF)
        {   ldaw r11, sp[STACK_VEC_TMP1]            ;   vdepth1                                 }   //values that should result in -0x80 will get 1 here
        {                                           ;   vstr r11[0]                             }
        {   ldaw r11, cp[VPU_VEC_0x80]              ;   ldw Q(pix_groups), sp[STACK_VEC_TMP1]   }
        {   and r11, Q(pix_groups), write_mask      ;   vldr r11[0]                             }
        vstrpv Y[0], r11
        {   add Y, Y, _16                           ;   bt chan_grp_left, .L_chan_loop          }
#endif //CONFIG_SYMMETRIC_SATURATION_avgpool2d_global

    .L_chan_loop_end:
    
    {   mkmsk r11, 16                           ;   ldc chan_grp_left, 1                }
    {   eq r11, write_mask, r11                 ;                                       }
    {   mkmsk write_mask, chan_tail             ;   bf r11, .L_img_end                  }
    {                                           ;   bt write_mask, .L_chan_loop         }


.L_img_end:

.Lfunc_end:
    //Restore the original constant pool pointer
    {                                           ;   ldw r11, sp[STACK_CP]                   }
    {                                           ;   set cp, r11                             }

    {                                           ;   ldw r10, sp[8]                      }
    ldd r8, r9, sp[3]
    ldd r6, r7, sp[2]
    ldd r4, r5, sp[1]
    retsp NSTACKWORDS


    .cc_bottom FUNCTION_NAME.function
    .set FUNCTION_NAME.nstackwords,NSTACKWORDS
    .globl FUNCTION_NAME.nstackwords
    .set FUNCTION_NAME.maxcores,1
    .globl FUNCTION_NAME.maxcores
    .set FUNCTION_NAME.maxtimers,0
    .globl FUNCTION_NAME.maxtimers
    .set FUNCTION_NAME.maxchanends,0
    .globl FUNCTION_NAME.maxchanends
.Ltmp0:
    .size FUNCTION_NAME, .Ltmp0-FUNCTION_NAME
    .issue_mode  single

#endif



