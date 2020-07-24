
#if defined(__XS3A__)
#include "nn_config.h"
#include "xs3_vpu.h"

/*
void bsign_8_(
    uint32_t* Y,
    const int8_t* X,
    const nn_bsign_8_job_t* job);
*/

#define FUNCTION_NAME bsign_8_

#define NSTACKWORDS     (2)
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 16
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

#define arg_Y           r0
#define arg_X           r1
#define arg_zero_pv     r2
#define arg_job         r3

#define Y               arg_Y
#define X               arg_X
#define length          r2
#define tail_mask       r3
#define tmp             r4
#define _32             r5
#define zero_pv         r11

#define JOB_START       0
#define JOB_LENGTH      1

#define PLAN_ZERO_POINT 0

FUNCTION_NAME:
    dualentsp NSTACKWORDS
    std r4, r5, sp[0]

    {   ldw tmp, arg_job[JOB_START]             ;   ldc _32, 32                             }
    {   add X, X, tmp                           ;   shl r11, _32, 4                         }
    {   shr tmp, tmp, 3                         ;   vsetc r11                               }    
    {   ldw length, arg_job[JOB_LENGTH]         ;   mov zero_pv, arg_zero_pv                }
    {   mov tail_mask, length                   ;   add Y, Y, tmp                           }
    {   zext tail_mask, 5                       ;   shr length, length, 5                   }
    {   shr tail_mask, tail_mask, 3             ;   mkmsk tmp, 4                            }
    {   mkmsk tail_mask, tail_mask              ;   bf length, .L_loop_end                  }

.L_loop_top:
    {   sub length, length, 1                   ;   vldr zero_pv[0]                         }
    {   add X, X, _32                           ;   vlsub X[0]                              }
    {                                           ;   vdepth1                                 }
    {   vstrpv Y[0], tmp                        }
    {   add Y, Y, 4                             ;   bt length, .L_loop_top                  }
.L_loop_end:
    {                                           ;   bf tail_mask, .Lfunc_end                }
    {                                           ;   vldr zero_pv[0]                         }
    {                                           ;   vlsub X[0]                              }
    {                                           ;   vdepth1                                 }
    {                                           ;   stw length, Y[0]                        } // Store 0 to upper bytes of tail
    {   vstrpv Y[0], tail_mask                  }

.Lfunc_end:
    ldd r4, r5, sp[0]
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

