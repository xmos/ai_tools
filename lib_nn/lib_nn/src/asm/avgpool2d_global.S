
#if defined(__XS3A__)

#include "nn_config.h"

/*  
void avgpool2d_global_asm(
    int8_t* Y,
    const int8_t* X, 
    const uint32_t x_height, 
    const uint32_t x_width,
    const uint32_t x_chans,
    const int32_t  bias,
    const uint32_t shift,
    const uint32_t scale);
*/

#define FUNCTION_NAME avgpool2d_global

#define NSTACKWORDS  (4*8 + 12)
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 4
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

#define arg_Y           r0
#define arg_X           r1
#define arg_x_height    r2
#define arg_x_width     r3

#define STACK_X_CHANS       (NSTACKWORDS+1)
#define STACK_BIAS          (NSTACKWORDS+2)
#define STACK_SHIFT         (NSTACKWORDS+3)
#define STACK_SCALE         (NSTACKWORDS+4)
#define STACK_SHIFT_VEC     (NSTACKWORDS-8)
#define STACK_BIAS_HI_VEC   (NSTACKWORDS-16)
#define STACK_BIAS_LO_VEC   (NSTACKWORDS-24)
#define STACK_VEC_TMP       (NSTACKWORDS-32)

#define Y               r0
#define X               r1
#define pix_groups      r2
#define chans           r3
#define chan_grp_left   r4
#define x_chan_incr     r5
#define write_mask      r6
#define chan_tail       r7
#define pix_loop_hop    r8
#define pix_count       r9
#define _16             r10

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
    mul pix_count, arg_x_height, arg_x_width
    {   shl r11, _16, 5                         ;   ldw chans, sp[STACK_X_CHANS]            }
    {   ldap r11, .L_bso_consts                 ;   vsetc r11                               }

    {                                           ;   ldw r10, sp[STACK_SCALE]                }
    {   ldaw r6, sp[STACK_SHIFT_VEC]            ;   ldw r4, r11[1]                          }
    mul r10, r10, r4
    std r10, r10, r6[0]
    std r10, r10, r6[1]
    std r10, r10, r6[2]
    std r10, r10, r6[3]
    {                                           ;   vldc r6[0]                              }
    {                                           ;   ldw r4, r11[0]                          }
    {                                           ;   ldw r10, sp[STACK_SHIFT]                }
    mul r10, r4, r10
    std r10, r10, r6[0]
    std r10, r10, r6[1]
    std r10, r10, r6[2]
    std r10, r10, r6[3]
    {   ldaw r6, sp[STACK_BIAS_HI_VEC]          ;   ldw r4, r11[0]                          }
    {   ldaw r7, sp[STACK_BIAS_LO_VEC]          ;   ldw r10, sp[STACK_BIAS]                 }
    {   shr r8, r10, 16                         ;   zext r10, 16                            }
    mul r10, r4, r10
    std r10, r10, r7[0]
    std r10, r10, r7[1]
    std r10, r10, r7[2]
    std r10, r10, r7[3]
    mul r10, r4, r8
    std r10, r10, r6[0]
    std r10, r10, r6[1]
    std r10, r10, r6[2]
    std r10, r10, r6[3]
    

    mul x_chan_incr, pix_count, chans
    {   ldc _16, 16                             ;   shr chan_grp_left, chans, 4             }
    {   sub x_chan_incr, x_chan_incr, _16       ;   mkmsk r11, 4                            }

    {   and chan_tail, chans, r11               ;   and pix_loop_hop, pix_count, r11        }
    
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
        {   shl r11, _16, 4                         ;   bru pix_loop_hop                        } // Do pix tail first
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
        {   ldaw r11, sp[STACK_SHIFT_VEC]           ;   vsetc r11       /* 16-bit mode */       }
        {   ldaw r11, sp[STACK_VEC_TMP]             ;   vlsat r11[0]                            }
        {   sub X, X, x_chan_incr                   ;   vstr r11[0]                             }
        {                                           ;   vdepth8                                 }
        vstrpv Y[0], write_mask
        {   ldaw r11, cp[vec_0x007F]                ;   vldr r11[0]                             }
        {                                           ;   vladd r11[0]                            }   //only 0xFF80 will remain negative (0xFFFF)
        {   ldaw r11, sp[STACK_VEC_TMP]             ;   vdepth1                                 }   //values that should result in -0x80 will get 1 here
        {                                           ;   vstr r11[0]                             }
        {   ldaw r11, cp[vec_0x80]                  ;   ldw Q(pix_groups), sp[STACK_VEC_TMP]    }
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



