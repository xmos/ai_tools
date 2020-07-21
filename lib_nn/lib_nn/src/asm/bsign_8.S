
#if defined(__XS3A__)
#include "nn_config.h"

/*
void bsign_8(
    uint32_t* Y,
    const int8_t* X,
    const nn_bsign_8_job_t* job);
*/

#define FUNCTION_NAME bsign_8

#define NSTACKWORDS  (2)
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 16
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

#define Y               r0
#define X               r1
#define length          r2
#define tail_mask       r3
#define tmp             r4
#define _32             r5
#define X_              r11

#define JOB_START   0
#define JOB_LENGTH  1

FUNCTION_NAME:
    dualentsp NSTACKWORDS
    std r4, r5, sp[0]
    
    
#define job     r2
    {   ldw tmp, job[JOB_START]                 ;   ldc _32, 32                             }
    {   add X, X, tmp                           ;   shl r11, _32, 4                         }
    {   shr tmp, tmp, 3                         ;   vsetc r11                               }    
    {   ldw length, job[JOB_LENGTH]             ;   mov X_, X                               }
    {   mov tail_mask, length                   ;   add Y, Y, tmp                           }
    {   zext tail_mask, 5                       ;   shr length, length, 5                   }
    {   shr tail_mask, tail_mask, 3             ;   mkmsk tmp, 4                            }
    {   mkmsk tail_mask, tail_mask              ;   bf length, .L_loop_end                  }

.L_loop_top:
    {   add X_, X_, _32                         ;   vldr X_[0]                              }
    {   sub length, length, 1                   ;   vdepth1                                 }
    {                                           ;   vstrpv Y[0], tmp                        }
    {   add Y, Y, 4                             ;   bt length, .L_loop_top                  }
.L_loop_end:
    {                                           ;   bf tail_mask, .Lfunc_end                }
    {                                           ;   vldr X_[0]                              }
    {                                           ;   vdepth1                                 }
    {                                           ;   vstrpv Y[0], tail_mask                  }

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



