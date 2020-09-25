
#if defined(__XS3A__)

#include "nn_config.h"

/*
void add_elementwise(
    int8_t Y[],
    const int8_t X1[],
    const int8_t X2[],
    const add_params_t* params, //per-channel? If so, need to add C_in and make this an array.
    const unsigned output_start,
    const unsigned output_count);
*/

#define FUNCTION_NAME add_elementwise

#define NSTACKVECS      11
#define NSTACKWORDS     ((NSTACKVECS)*8 + 10)
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 16
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

       
#define PARAMS_LEFT_SHIFT   0
#define PARAMS_IN1_OFFSET   1
#define PARAMS_IN2_OFFSET   2
#define PARAMS_IN1_MULT     3
#define PARAMS_IN2_MULT     4
#define PARAMS_IN1_SHIFT    5
#define PARAMS_IN2_SHIFT    6
#define PARAMS_OUT_MULT     7
#define PARAMS_OUT_SHIFT    8
#define PARAMS_OUT_OFFSET   9

#define STACK_VECS_START        (NSTACKWORDS - (NSTACKVECS * 8))
#define STACK_VEC_IN1           ((STACK_VECS_START)+0)
#define STACK_VEC_IN2           ((STACK_VECS_START)+8)
#define STACK_VEC_IN1_OFFSET    ((STACK_VECS_START)+16)
#define STACK_VEC_IN2_OFFSET    ((STACK_VECS_START)+24)
#define STACK_VECS_DP_START     ((STACK_VECS_START)+32)

#define DP_VEC_IN1_MULT      0
#define DP_VEC_IN2_MULT      8
#define DP_VEC_IN1_SHIFT     16
#define DP_VEC_IN2_SHIFT     24
#define DP_VEC_OUT_MULT      32
#define DP_VEC_OUT_SHIFT     40
#define DP_VEC_OUT_OFFSET    48

#define STACK_OUT_START     NSTACKWORDS+1
#define STACK_OUT_COUNT     NSTACKWORDS+2

#define STACK_ORIG_DP       8

#define Y               r0
#define X1              r1
#define X2              r2
#define N               r3
#define left_shift      r4
#define ones            r5
#define _32             r6
#define next_Y          r7
#define proc_chans      r8
#define tmp             r10


// Simple non-ABI-compliant function that significantly reduces the code memory footprint of add_elementwise()
// (unfortunately this function uses a lot of stack space)
.L_std_func:
    std tmp, tmp, r11[0]
    std tmp, tmp, r11[1]
    std tmp, tmp, r11[2]
    std tmp, tmp, r11[3]
    retsp 0

FUNCTION_NAME:
    dualentsp NSTACKWORDS
    std r4, r5, sp[1]
    std r6, r7, sp[2]
    std r8, r9, sp[3]

#define params      r3

{   ldc _32, 32                             ;   stw r10, sp[1]                          }
{   ldaw dp, sp[STACK_VECS_DP_START]        ;   stw dp, sp[STACK_ORIG_DP]               }
    ldw tmp, sp[STACK_OUT_START]
{   add X1, X1, tmp                         ;   add X2, X2, tmp                         }
{   add Y, Y, tmp                           ;   ldw left_shift, params[PARAMS_LEFT_SHIFT]}

// Create vectors on the stack for each of these...

{   ldaw r11, dp[DP_VEC_IN1_SHIFT]          ;   ldw tmp, params[PARAMS_IN1_SHIFT]       }
{   add tmp, tmp, 1                         ;   bl .L_std_func                          }
{   ldaw r11, dp[DP_VEC_IN2_SHIFT]          ;   ldw tmp, params[PARAMS_IN2_SHIFT]       }
{   add tmp, tmp, 1                         ;   bl .L_std_func                          }
{   ldaw r11, dp[DP_VEC_OUT_MULT]           ;   ldw tmp, params[PARAMS_OUT_MULT]        }
{                                           ;   bl .L_std_func                          }
{   ldaw r11, dp[DP_VEC_OUT_SHIFT]          ;   ldw tmp, params[PARAMS_OUT_SHIFT]       }
{   add tmp, tmp, 1                         ;   bl .L_std_func                          }
{   ldaw r11, dp[DP_VEC_OUT_OFFSET]         ;   ldw tmp, params[PARAMS_OUT_OFFSET]      }
{                                           ;   bl .L_std_func                          }

//If we didn't have to add 1 to the shifts, we could do the 5 vector things above in this loop, too
{   ldaw r11, sp[STACK_VEC_IN1_OFFSET]      ;   ldc next_Y, 4                           }
.L_load_loop:
    {   add params, params, 4                   ;   ldw tmp, params[0]                      }
    {   sub next_Y, next_Y, 1                   ;   bl .L_std_func                          }
    {   add r11, r11, _32                       ;   bt next_Y, .L_load_loop                 }


#undef params

    
    ldw N, sp[STACK_OUT_COUNT]
    ldap r11, vpu_vects_vec_0x01
{   mov next_Y, Y                           ;   mov ones, r11                           }
{   shl r11, _32, 4                         ;                                           }

.L_loop_top:
    {   ldc tmp, 0                              ;   vsetc r11                               }
    {   mov Y, next_Y                           ;   vldc ones[0]                            }
    {   shr proc_chans, N, 3                    ;   vclrdr                                  }
    {   ldaw r11, sp[STACK_VEC_IN1]             ;   vlmacc X1[0]                            }
    {   lsu tmp, tmp, proc_chans                ;   vstr r11[0]                             }
    {   shl proc_chans, tmp, 3                  ;   vclrdr                                  }
    {   mov tmp, N                              ;   vlmacc r11[0]                           }
    {   zext tmp, 3                             ;   vstr r11[0]                             }
        vlashr r11[0], left_shift
    {   shr tmp, tmp, proc_chans                ;   vstr r11[0]                             }
    {   ldaw r11, sp[STACK_VEC_IN2]             ;   vclrdr                                  }
    {   add proc_chans, proc_chans, tmp         ;   vlmacc X2[0]                            }
    {   sub N, N, proc_chans                    ;   vstr r11[0]                             }
    {   add X1, X1, proc_chans                  ;   vclrdr                                  }
    {   add X2, X2, proc_chans                  ;   vlmacc r11[0]                           }
    {   add next_Y, Y, proc_chans               ;   vstr r11[0]                             }
        vlashr r11[0], left_shift
    {   ldc r11, 0                              ;   vstr r11[0]                             }
    {                                           ;   vsetc r11                               }
    {   ldaw r11, dp[DP_VEC_IN1_MULT]           ;   vclrdr                                  }
    {   ldaw r11, sp[STACK_VEC_IN1]             ;   vldc r11[0]                             }
    {   ldaw r11, sp[STACK_VEC_IN1_OFFSET]      ;   vlmacc r11[0]                           }
    {   ldaw r11, dp[DP_VEC_IN1_SHIFT]          ;   vlmacc r11[0]                           }
    {   ldaw r11, sp[STACK_VEC_IN1]             ;   vlsat r11[0]                            }
    {                                           ;   vstr r11[0]                             }
    {   ldaw r11, dp[DP_VEC_IN2_MULT]           ;   vclrdr                                  }
    {   ldaw r11, sp[STACK_VEC_IN2]             ;   vldc r11[0]                             }
    {   ldaw r11, sp[STACK_VEC_IN2_OFFSET]      ;   vlmacc r11[0]                           }
    {   ldaw r11, dp[DP_VEC_IN2_SHIFT]          ;   vlmacc r11[0]                           }
    {   ldaw r11, sp[STACK_VEC_IN2]             ;   vlsat r11[0]                            }
    {   ldc tmp, 8                              ;   vstr r11[0]                             }
    {   ldaw r11, dp[DP_VEC_OUT_MULT]           ;   vclrdr                                  }
    {   ldaw r11, sp[STACK_VEC_IN1]             ;   vldc r11[0]                             }
    {   ldaw r11, sp[STACK_VEC_IN2]             ;   vlmacc r11[0]                           }
    {   ldaw r11, dp[DP_VEC_OUT_SHIFT]          ;   vlmacc r11[0]                           }
    {   ldaw r11, dp[DP_VEC_OUT_OFFSET]         ;   vlsat r11[0]                            }
    {   ldaw r11, sp[STACK_VEC_IN1]             ;   vladd r11[0]                            }
    {   sub tmp, tmp, _32                       ;   vstr r11[0]                             }
        vlashr r11[0], tmp
    {   mkmsk proc_chans, proc_chans            ;   vdepth8                                 }
        vstrpv Y[0], proc_chans
    {   shl r11, _32, 4                         ;   bt N, .L_loop_top                       }

.L_loop_end:


    
.Lfunc_end:
{                                           ;   ldw dp, sp[STACK_ORIG_DP]               }
{                                           ;   ldw r10, sp[1]                          }
    ldd r4, r5, sp[1]
    ldd r6, r7, sp[2]
    ldd r8, r9, sp[3]
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



