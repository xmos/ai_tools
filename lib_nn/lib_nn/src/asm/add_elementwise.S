
#if defined(__XS3A__)

#include "nn_config.h"

/*


typedef struct {
    struct {
        int16_t shr;
        int16_t multiplier;
    } input[2];
    struct {
        int32_t bias;
        int8_t shr;
    } output;
} nn_add_params_t;


void add_elementwise(
    int8_t Y[],
    const int8_t X1[],
    const int8_t X2[],
    const nn_add_params_t* params,
    const unsigned output_start,
    const unsigned output_count);
*/

#define FUNCTION_NAME add_elementwise

#define NSTACKVECS      8
#define NSTACKWORDS     ((NSTACKVECS)*8 + 10)
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 16
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

#define PARAMS_SHR1         0
#define PARAMS_IN1_MULT     1
#define PARAMS_SHR2         2
#define PARAMS_IN2_MULT     3
#define PARAMS_OUT_BIAS_LO  4
#define PARAMS_OUT_BIAS_HI  5
#define PARAMS_OUT_SHR_BYTE 12


#define STACK_VECS_START        (NSTACKWORDS - (NSTACKVECS * 8))
#define STACK_VEC_IN1           ((STACK_VECS_START)+0)
#define STACK_VEC_IN2           ((STACK_VECS_START)+8)
#define STACK_VECS_DP_START     ((STACK_VECS_START)+16)

#define DP_VEC_IN1_MULT     0
#define DP_VEC_IN2_MULT     8
#define DP_VEC_BIAS_HI      16
#define DP_VEC_BIAS_LO      24
#define DP_VEC_OUT_SHR      32

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
{   ldc r4, PARAMS_SHR1                     ;                                           }
{   add X1, X1, tmp                         ;   add X2, X2, tmp                         }
{   ldc r4, PARAMS_SHR2                     ;   ld16s X1_shr, params[r4]                }
{   add Y, Y, tmp                           ;   ld16s X2_shr, params[r4]                }

{   ldc r4, PARAMS_IN1_MULT                 ;                                           }
{   ldaw r11, dp[DP_VEC_IN1_MULT]           ;   ld16s tmp, params[r4]                   }
{   zext tmp, 16                            ;   shl chans, tmp, 16                      }
{   or tmp, tmp, chans                      ;   bl .L_std_func                          }

{   ldc r4, PARAMS_IN2_MULT                 ;                                           }
{   ldaw r11, dp[DP_VEC_IN2_MULT]           ;   ld16s tmp, params[r4]                   }
{   zext tmp, 16                            ;   shl chans, tmp, 16                      }
{   or tmp, tmp, chans                      ;   bl .L_std_func                          }

{   ldc r4, PARAMS_OUT_BIAS_LO              ;                                           }
{   ldaw r11, dp[DP_VEC_BIAS_LO]            ;   ld16s tmp, params[r4]                   }
{   zext tmp, 16                            ;   shl chans, tmp, 16                      }
{   or tmp, tmp, chans                      ;   bl .L_std_func                          }

{   ldc r4, PARAMS_OUT_BIAS_HI              ;                                           }
{   ldaw r11, dp[DP_VEC_BIAS_HI]            ;   ld16s tmp, params[r4]                   }
{   zext tmp, 16                            ;   shl chans, tmp, 16                      }
{   or tmp, tmp, chans                      ;   bl .L_std_func                          }

{   ldc r4, PARAMS_OUT_SHR_BYTE             ;                                           }
{   ldaw r11, dp[DP_VEC_OUT_SHR]            ;   ld8u tmp, params[r4]                    }
{   zext tmp, 16                            ;   shl chans, tmp, 16                      }
{   or tmp, tmp, chans                      ;   bl .L_std_func                          }

{                                           ;   bu .L_std_func_end                      }
#undef params

.L_std_func:
    std tmp, tmp, r11[0]
    std tmp, tmp, r11[1]
    std tmp, tmp, r11[2]
    std tmp, tmp, r11[3]
    retsp 0
.L_std_func_end:

    ldw N, sp[STACK_OUT_COUNT]
{   ldaw vec_tmp1, sp[STACK_VEC_IN1]        ;   ldc _32, 32                             }
{   ldaw vec_tmp2, sp[STACK_VEC_IN2]        ;                                           }
{   shl r11, _32, 4                         ;                                           }
{                                           ;   vsetc r11                               }
{                                           ;   bf N, .L_loop_end                       }
.L_loop_top:
    {   ldc tmp, 15                             ;   vclrdr                                  }
        ldap r11, vpu_vects_vec_0x01
    {   lsu chans, tmp, N                       ;   vldc r11[0]                             }
    {   shl chans, chans, 4                     ;   vlmacc X1[0]                            }
    {   mov tmp, N                              ;   vstr vec_tmp1[0]                        }
    {   zext tmp, 4                             ;   vclrdr                                  }
    {   shr tmp, tmp, chans                     ;   vlmacc X2[0]                            }
    {   shl r11, _32, 3                         ;   vstr vec_tmp2[0]                        }
    {   add chans, tmp, chans                   ;   vsetc r11                               }
        vlashr vec_tmp1[0], X1_shr
    {   add X1, X1, chans                       ;   vstr vec_tmp1[0]                        }
        vlashr vec_tmp2[0], X2_shr
    {   ldaw r11, dp[DP_VEC_BIAS_HI]            ;   vstr vec_tmp2[0]                        }

#if !(CONFIG_SYMMETRIC_SATURATION_add_elementwise)
        ldaw r11, cp[vpu_vects_vec_0x80]
    {   mkmsk tmp, chans                        ;   vldr r11[0]                             }
        vstrpv Y[0], tmp
    {   ldaw r11, dp[DP_VEC_BIAS_HI]            ;                                           }
#endif // CONFIG_SYMMETRIC_SATURATION_add_elementwise


    {   ldaw r11, dp[DP_VEC_BIAS_LO]            ;   vldd r11[0]                             }
    {   ldaw r11, dp[DP_VEC_IN1_MULT]           ;   vldr r11[0]                             }
    {   add X2, X2, chans                       ;   vldc r11[0]                             }
    {   ldaw r11, dp[DP_VEC_IN2_MULT]           ;   vlmacc vec_tmp1[0]                      }
    {   sub N, N, chans                         ;   vldc r11[0]                             }
    {   shl r11, _32, 4                         ;   vlmacc vec_tmp2[0]                      }

#if CONFIG_SYMMETRIC_SATURATION_add_elementwise

    {   ldaw r11, dp[DP_VEC_OUT_SHR]            ;   vsetc r11                               }
    {   mkmsk tmp, chans                        ;   vlsat r11[0]                            }
        vstrpv Y[0], tmp

#else 

    {   ldaw r11, dp[DP_VEC_OUT_SHR]            ;   vstd vec_tmp2[0]                        }
    {                                           ;   vstr vec_tmp1[0]                        }
    {                                           ;   vlsat r11[0]                            }
    {                                           ;   vldd vec_tmp2[0]                        }
        ldaw r11, cp[vpu_vects_vec_0x007F]
    {                                           ;   vladd r11[0]                            }
    {   mkmsk tmp, chans                        ;   vdepth1                                 }
    {   mov r11, vec_tmp1                       ;   vstr vec_tmp2[0]                        }

    {                                           ;   ldw vec_tmp2, sp[STACK_VEC_IN2]         }
    {   andnot tmp, vec_tmp2                    ;   vldr r11[0]                             }
    {   shl r11, _32, 4                         ;                                           }
    {   ldaw r11, dp[DP_VEC_OUT_SHR]            ;   vsetc r11                               }
    {   ldaw vec_tmp2, sp[STACK_VEC_IN2]        ;   vlsat r11[0]                            }
        vstrpv Y[0], tmp

#endif // CONFIG_SYMMETRIC_SATURATION_add_elementwise

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



