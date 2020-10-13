
#if defined(__XS3A__)

#include "nn_config.h"

/*
void add_elementwise(
    int8_t Y[],
    const int8_t X1[],
    const int8_t X2[],
    const add_params3_t* params, //per-channel? If so, need to add C_in and make this an array.
    const unsigned output_start,
    const unsigned output_count);
*/

#define FUNCTION_NAME add_elementwise3

#define NSTACKVECS      8
#define NSTACKWORDS     ((NSTACKVECS)*8 + 10)
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 16
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

#define PARAMS_SHR1         0
#define PARAMS_SHR2         1
#define PARAMS_IN1_OFFSET   2
#define PARAMS_IN2_OFFSET   3
#define PARAMS_IN1_MULT     4
#define PARAMS_IN2_MULT     5
#define PARAMS_OUT_MULT     7
#define PARAMS_OUT_OFFSET   8

#define PARAMS_IN12_SHR     0
#define PARAMS_IN12_OFFSET  1
#define PARAMS_IN12_MULT    2
#define PARAMS_OUT_MOFF     3

#define STACK_VECS_START        (NSTACKWORDS - (NSTACKVECS * 8))
#define STACK_VEC_IN1           ((STACK_VECS_START)+0)
#define STACK_VEC_IN2           ((STACK_VECS_START)+8)
#define STACK_VEC_IN1_OFFSET    ((STACK_VECS_START)+16)
#define STACK_VEC_IN2_OFFSET    ((STACK_VECS_START)+24)
#define STACK_VECS_DP_START     ((STACK_VECS_START)+32)

#define DP_VEC_IN1_MULT      0
#define DP_VEC_IN2_MULT      8
#define DP_VEC_OUT_MULT      16
#define DP_VEC_OUT_OFFSET    24

#define STACK_OUT_START     NSTACKWORDS+1
#define STACK_OUT_COUNT     NSTACKWORDS+2

#define STACK_ORIG_DP       8


#define Y           r0
#define X1          r1
#define X2          r2
#define N           r3
#define _32         r4
#define X1_shr      r5
#define X2_shr      r6
#define chans       r7
#define vec_tmp1    r8
#define vec_tmp2    r9
#define tmp         r10


FUNCTION_NAME:
    dualentsp NSTACKWORDS
    std r4, r5, sp[1]
    std r6, r7, sp[2]
    std r8, r9, sp[3]
{   ldc _32, 32                             ;   stw r10, sp[1]                          }
{   ldaw dp, sp[STACK_VECS_DP_START]        ;   stw dp, sp[STACK_ORIG_DP]               }

#define params      r3

    ldw tmp, sp[STACK_OUT_START]
{   add X1, X1, tmp                         ;   add X2, X2, tmp                         }
{   add Y, Y, tmp                           ;   ldw X1_shr, params[PARAMS_IN12_SHR]     }
{   sext X1_shr, 16                         ;   shr X2_shr, X1_shr, 16                  }

{   ldaw r11, sp[STACK_VEC_IN1_OFFSET]      ;   ldw tmp, params[PARAMS_IN12_OFFSET]     }
{   zext tmp, 16                            ;   shr vec_tmp1, tmp, 16                   }
{   shl chans, tmp, 16                      ;   sext X2_shr, 16                         }
{   or tmp, tmp, chans                      ;   shl vec_tmp2, vec_tmp1, 16              }
{                                           ;   bl .L_std_func                          }

{   ldaw r11, sp[STACK_VEC_IN2_OFFSET]      ;   or tmp, vec_tmp1, vec_tmp2              }
{                                           ;   bl .L_std_func                          }

{   ldaw r11, dp[DP_VEC_IN1_MULT]           ;   ldw tmp, params[PARAMS_IN12_MULT]       }
{   zext tmp, 16                            ;   shr vec_tmp1, tmp, 16                   }
{   shl chans, tmp, 16                      ;   shl vec_tmp2, vec_tmp1, 16              }
{   or tmp, tmp, chans                      ;   bl .L_std_func                          }

{   ldaw r11, dp[DP_VEC_IN2_MULT]           ;   or tmp, vec_tmp1, vec_tmp2              }
{                                           ;   bl .L_std_func                          }

{   ldaw r11, dp[DP_VEC_OUT_MULT]           ;   ldw tmp, params[PARAMS_OUT_MOFF]        }
{   zext tmp, 16                            ;   shr vec_tmp1, tmp, 16                   }
{   shl chans, tmp, 16                      ;   shl vec_tmp2, vec_tmp1, 16              }
{   or tmp, tmp, chans                      ;   bl .L_std_func                          }

{   ldaw r11, dp[DP_VEC_OUT_OFFSET]         ;   or tmp, vec_tmp1, vec_tmp2              }
{                                           ;   bl .L_std_func                          }
{                                           ;   bu .L_std_func_end                      }

.L_std_func:
    std tmp, tmp, r11[0]
    std tmp, tmp, r11[1]
    std tmp, tmp, r11[2]
    std tmp, tmp, r11[3]
    retsp 0
.L_std_func_end:


#undef params

    
    ldw N, sp[STACK_OUT_COUNT]
    ldap r11, vpu_vects_vec_0x01
{   ldaw vec_tmp1, sp[STACK_VEC_IN1]        ;                                           }
{   ldaw vec_tmp2, sp[STACK_VEC_IN2]        ;   vldc r11[0]                             }
{   shl r11, _32, 4                         ;   bf N, .L_loop_end                       }

.L_loop_top:

    {   ldc tmp, 15                             ;   vsetc r11                               }
    {   lsu chans, tmp, N                       ;   vclrdr                                  }
    {   shl chans, chans, 4                     ;   vlmacc X2[0]                            }
    {   mov tmp, N                              ;   vstr vec_tmp2[0]                        }
    {   zext tmp, 4                             ;   vclrdr                                  }
    {   shr tmp, tmp, chans                     ;   vlmacc X1[0]                            }
    {   shl r11, _32, 3                         ;   vstr vec_tmp1[0]                        }

    {   ldaw r11, sp[STACK_VEC_IN1_OFFSET]      ;   vsetc r11                               }
        vlashr vec_tmp1[0], X1_shr
    {   ldaw r11, dp[DP_VEC_IN1_MULT]           ;   vladd r11[0]                            }
    {   ldaw r11, sp[STACK_VEC_IN2_OFFSET]      ;   vlmul r11[0]                            }
    {   add chans, tmp, chans                   ;   vstr vec_tmp1[0]                        }
        vlashr vec_tmp2[0], X2_shr
    {   ldaw r11, dp[DP_VEC_IN2_MULT]           ;   vladd r11[0]                            }
    {   add X1, X1, chans                       ;   vlmul r11[0]                            }
    {   ldaw r11, dp[DP_VEC_OUT_MULT]           ;   vladd vec_tmp1[0]                       }

    {   ldaw r11, dp[DP_VEC_OUT_OFFSET]         ;   vlmul r11[0]                            }
    {   add X2, X2, chans                       ;   vladd r11[0]                            }
    {   shl r11, _32, 4                         ;   vdepth8                                 }
    {   mkmsk tmp, chans                        ;   sub N, N, chans                         }
        vstrpv Y[0], tmp
    {   add Y, Y, chans                         ;   bt N, .L_loop_top                       }

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



