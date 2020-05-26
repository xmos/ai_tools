
#if defined(__XS3A__)

#include "nn_config.h"
#include "window_op_plan.h"
#include "asm_constants.h"

/*  
void conv2d_1x1(
    int8_t* Y,
    const int8_t* X, 
    const int8_t* K,
    const data16_t* BSO,
    const nn_conv2d_1x1_plan_t* plan);


Optimizing assumptions:
    - input window is 1x1
    - input window stride is 1x1 (So, input image is just a long row of pixels)
    - output stride is 1x1 (So, output image is also just a long row of pixels)

*/

#define FUNCTION_NAME conv2d_1x1

#define NSTACKWORDS  36
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 4
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME


#define VLMACCR_GROUP(OUT_CHAN_STRIDE, CIG_STRIDE)                                                              \
                    {   sub K, K, OUT_CHAN_STRIDE           ;   vlmaccr K[0]                            };      \
                    {   sub K, K, OUT_CHAN_STRIDE           ;   vlmaccr K[0]                            };      \
                    {   sub K, K, OUT_CHAN_STRIDE           ;   vlmaccr K[0]                            };      \
                    {   sub K, K, OUT_CHAN_STRIDE           ;   vlmaccr K[0]                            };      \
                    {   sub K, K, OUT_CHAN_STRIDE           ;   vlmaccr K[0]                            };      \
                    {   sub K, K, OUT_CHAN_STRIDE           ;   vlmaccr K[0]                            };      \
                    {   sub K, K, OUT_CHAN_STRIDE           ;   vlmaccr K[0]                            };      \
                    {   sub K, K, OUT_CHAN_STRIDE           ;   vlmaccr K[0]                            };      \
                    {   sub K, K, OUT_CHAN_STRIDE           ;   vlmaccr K[0]                            };      \
                    {   sub K, K, OUT_CHAN_STRIDE           ;   vlmaccr K[0]                            };      \
                    {   sub K, K, OUT_CHAN_STRIDE           ;   vlmaccr K[0]                            };      \
                    {   sub K, K, OUT_CHAN_STRIDE           ;   vlmaccr K[0]                            };      \
                    {   sub K, K, OUT_CHAN_STRIDE           ;   vlmaccr K[0]                            };      \
                    {   sub K, K, OUT_CHAN_STRIDE           ;   vlmaccr K[0]                            };      \
                    {   sub K, K, OUT_CHAN_STRIDE           ;   vlmaccr K[0]                            };      \
                    {   add K, K, CIG_STRIDE                ;   vlmaccr K[0]                            }    

#define VLMACCR_GROUP_TAIL(OUT_CHAN_STRIDE, CIG_STRIDE)                                                         \
                    {   sub K, K, OUT_CHAN_STRIDE           ;   vlmaccr K[0]                            };      \
                    {   sub K, K, OUT_CHAN_STRIDE           ;   vlmaccr K[0]                            };      \
                    {   sub K, K, OUT_CHAN_STRIDE           ;   vlmaccr K[0]                            };      \
                    {   sub K, K, OUT_CHAN_STRIDE           ;   vlmaccr K[0]                            };      \
                    {   sub K, K, OUT_CHAN_STRIDE           ;   vlmaccr K[0]                            };      \
                    {   sub K, K, OUT_CHAN_STRIDE           ;   vlmaccr K[0]                            };      \
                    {   sub K, K, OUT_CHAN_STRIDE           ;   vlmaccr K[0]                            };      \
                    {   sub K, K, OUT_CHAN_STRIDE           ;   vlmaccr K[0]                            };      \
                    {   sub K, K, OUT_CHAN_STRIDE           ;   vlmaccr K[0]                            };      \
                    {   sub K, K, OUT_CHAN_STRIDE           ;   vlmaccr K[0]                            };      \
                    {   sub K, K, OUT_CHAN_STRIDE           ;   vlmaccr K[0]                            };      \
                    {   add K, K, CIG_STRIDE                ;   vlmaccr K[0]                            }    


#define PLAN_START_STRIDE_X     0
#define PLAN_START_STRIDE_Y     1
#define PLAN_START_STRIDE_K     2
#define PLAN_COG_STRIDE_Y       3
#define PLAN_COG_STRIDE_K       4
#define PLAN_BODY_KSTRIDE       5
#define PLAN_TAIL_KSTRIDE       6
#define PLAN_PIX_COUNT          7
#define PLAN_CIN                8
#define PLAN_COUT               9

#define STACK_VEC_TMP       (NSTACKWORDS-8)
#define STACK_VEC_TMP2      (NSTACKWORDS-16)

#define STACK_PLAN          (NSTACKWORDS+1)

#define STACK_CP            1
#define STACK_X             9
#define STACK_PIX_COUNT     10
#define STACK_COG_STRIDE_Y  11
#define STACK_COG_STRIDE_K  12
#define STACK_TAIL_MASK     13
#define STACK_C_OUT         14
#define STACK_BSO           15
#define STACK_TMP           16


#define Y               r0
#define X               r1
#define K               r2
#define BSO             r3
#define C_in            r4
#define cin_tail        r5
#define _32             r6
#define cog_left        r7
#define cig_left        r8
  #define plan            cig_left
#define pix_left        r9
#define K_cig_stride    r10

#define Q(R)    R

.align 16


FUNCTION_NAME:
    dualentsp NSTACKWORDS
    std r4, r5, sp[1]
    std r6, r7, sp[2]
    std r8, r9, sp[3]
    {   ldc _32, 32                             ;   stw r10, sp[8]                          }

    //Change constant pool pointer to refer to the constant VPU vects needed here
    ldaw r11, cp[vpu_vects]
    {   ldaw r11, cp[0]                         ;   set cp, r11                             }
    {                                           ;   stw r11, sp[STACK_CP]                   }


    {   shl r11, _32, 4                         ;   stw BSO, sp[STACK_BSO]                  }
    {                                           ;   vsetc r11                               }
    {                                           ;   ldw plan, sp[STACK_PLAN]                }

    {                                           ;   ldw C_in, plan[PLAN_CIN]                }
    {                                           ;   ldw K_cig_stride, plan[PLAN_BODY_KSTRIDE]}

    {   ldaw r11, sp[STACK_VEC_TMP2]            ;   vclrdr                                  }
    {   ldaw r11, sp[STACK_VEC_TMP]             ;   vstr r11[0]                             }
    {                                           ;   vstr r11[0]                             }
    {                                           ;   ldw r11, plan[PLAN_START_STRIDE_X]      }
    {   add X, X, r11                           ;   ldw r11, plan[PLAN_START_STRIDE_Y]      }
    {   add Y, Y, r11                           ;   ldw r11, plan[PLAN_START_STRIDE_K]      }
    {   add K, K, r11                           ;   stw X, sp[STACK_X]                      }

    {   mov cin_tail, C_in                      ;   ldw r11, plan[PLAN_PIX_COUNT]           }
    {   zext cin_tail, 5                        ;   stw r11, sp[STACK_PIX_COUNT]            }
    {                                           ;   ldw r11, plan[PLAN_COG_STRIDE_Y]        }
    {                                           ;   stw r11, sp[STACK_COG_STRIDE_Y]         }
    {                                           ;   ldw r11, plan[PLAN_COG_STRIDE_K]        }
    {                                           ;   stw r11, sp[STACK_COG_STRIDE_K]         }
    {                                           ;   ldw r11, plan[PLAN_COUT]                }
    {   mov r8, r11                             ;   stw r11, sp[STACK_C_OUT]                }
    {   zext r8, 4                              ;                                           }
    {   mkmsk r8, r8                            ;                                           }
    {   shr cog_left, r11, 4                    ;   stw r8, sp[STACK_TAIL_MASK]             }

    {                                           ;   bf cog_left, .L_cog_tail                }
    .L_cog_loop:
        {   add K, K, K_cig_stride              ;   ldw pix_left, sp[STACK_PIX_COUNT]       }
        {   sub K, K, _32                       ;   sub cog_left, cog_left, 1               }
        .L_pix_loop:    
#if CONFIG_SYMMETRIC_SATURATION_conv2d_1x1
            {                                           ;   ldw BSO, sp[STACK_BSO]                  }
#else
            {   ldaw r11, cp[VPU_VEC_0x80]              ;   ldw BSO, sp[STACK_BSO]                  }
            {   mkmsk r11, 16                           ;   vldr r11[0]                             }
            vstrpv Y[0], r11
#endif
            {   add r11, BSO, _32                       ;   vldd BSO[0]                             }
            {   shr cig_left, C_in, 5                   ;   vldr r11[0]                             }
            {   add BSO, r11, _32                       ;   bf cig_left, .L_cig_tail                }
            .L_cig_loop:
                {   sub cig_left, cig_left, 1               ;   vldc X[0]                               }
                VLMACCR_GROUP(C_in, K_cig_stride)
                {   add X, X, _32                           ;   bt cig_left, .L_cig_loop                }
            .L_cig_tail:    
            {   add K, K, cin_tail                      ;   bf cin_tail, .L_get_res                 }
            {   sub K, K, _32                           ;   sub cin_tail, _32, cin_tail             }
                {   ldaw r11, sp[STACK_VEC_TMP]             ;   vldc X[0]                               }
                {   sub r11, r11, cin_tail                  ;   vstc r11[0]                             }
                {   sub cin_tail, _32, cin_tail             ;   vldc r11[0]                             }
                VLMACCR_GROUP(C_in, K_cig_stride)
            .L_get_res:
            
#if CONFIG_SYMMETRIC_SATURATION_conv2d_1x1
            {   shl r11, _32, 3                         ;   sub pix_left, pix_left, 1               }
            {   ldaw r11, sp[STACK_VEC_TMP]             ;   vsetc r11                               }
            {   add BSO, BSO, _32                       ;   vlsat BSO[0]                            }
            {   add X, X, cin_tail                      ;   vstr r11[0]                             }   
            {   add BSO, BSO, _32                       ;   vldc BSO[0]                             }
            {                                           ;   vclrdr                                  }
            {   shl r11, _32, 4                         ;   vlmacc r11[0]                           }
            {   add BSO, BSO, _32                       ;   vldc BSO[0]                             }
            {   add BSO, BSO, _32                       ;   vlmacc BSO[0]                           }
            {   mkmsk r11, 16                           ;   vsetc r11                               }
            {   add BSO, BSO, _32                       ;   vlsat BSO[0]                            }
            vstrpv Y[0], r11
            {   sub K, K, C_in                          ;   ldw Q(cig_left), sp[STACK_C_OUT]        }
            {   add Y, Y, Q(cig_left)                   ;   bt pix_left, .L_pix_loop                }



            // Move 32-bit accumulators into memory:

            // {   shl r11, _32, 3                         ;   ldw r10, sp[STACK_0x11111111]           }
            // {   ldaw r8, sp[STACK_VEC_A]                ;   vsetc r11                               }
            // {   ldaw r9, sp[STACK_VEC_B]                ;   vstr r8[0]                              }
            // vlashr r8[0], tmp
            // {   ldap r11, .L_vec_zero                   ;   vstd r9[0]                              }
            // vstrpv r9[0], r10
            // {   ldap r11, .L_vec_0x0008                 ;   vldr r11[0]                             }
            // {   shl r10, r10, 1                         ;   vlsat r11[0]                            }
            // vstrpv r8[0], r10
            // {   shl r11, _32, 4                         ;   vclrdr                                  }
            // {   ldap r11, .L_vec_0x01                   ;   vsetc r11                               }
            // {   ldaw r11, sp[STACK_VEC_32BIT]           ;   vldc r11[0]                             }
            // {                                           ;   vlmacc r8[0]                            }
            // {   ldaw r8, sp[STACK_VEC_A + 4]            ;   vstr r11[0]                             }
            // {                                           ;   vclrdr                                  }
            // {   ldaw r11, sp[STACK_VEC_32BIT + 8]       ;   vlmacc r8[0]                            }
            // {   shl r8, r10, 2                          ;   vstr r11[0]                             }
            // {   ldap r11, .L_vec_0x7F                   ;   vclrdr                                  }
            // {   add r10, r8, r10                        ;   vldc r9[0]                              }
            // {   ldaw r9, sp[STACK_VEC_B + 4]            ;   vlmacc r11[0]                           }
            // {   ldap r11, .L_vec_0x02                   ;   vlmacc r11[0]                           }
            // {   ldaw r11, sp[STACK_VEC_32BIT]           ;   vlmacc r11[0]                           }
            // vstrpv r11[0], r10
            // {                                           ;   vclrdr                                  }
            // {   ldap r11, .L_vec_0x7F                   ;   vldc r9[0]                              }
            // {                                           ;   vlmacc r11[0]                           }
            // {   ldap r11, .L_vec_0x02                   ;   vlmacc r11[0]                           }
            // {   ldaw r11, sp[STACK_VEC_32BIT+8]         ;   vlmacc r11[0]                           }
            // vstrpv r11[0], r10





#else
            {   shl r11, _32, 3                         ;   sub pix_left, pix_left, 1               }
            {   ldaw r11, sp[STACK_VEC_TMP]             ;   vsetc r11    /* 16-bit mode */          }
            {   add BSO, BSO, _32                       ;   vlsat BSO[0]                            }
            {   add X, X, cin_tail                      ;   vstr r11[0]                             }   
            {   add BSO, BSO, _32                       ;   vldc BSO[0]                             }
            {                                           ;   vclrdr                                  }
            {                                           ;   vlmacc r11[0]                           }
            {   add BSO, BSO, _32                       ;   vldc BSO[0]                             }
            {   add BSO, BSO, _32                       ;   vlmacc BSO[0]                           }

            {   add BSO, BSO, _32                       ;   vlsat BSO[0]                            }
            {   ldaw r11, cp[VPU_VEC_0x007F]            ;   vstr r11[0]                             }
            {   ldaw r11, sp[STACK_TMP]                 ;   vladd r11[0]                            }
            {   mkmsk Q(cig_left), 4                    ;   vdepth1                                 }
            vstrpv r11[0], Q(cig_left)
            {   ldc Q(cig_left), 0                      ;                                           }
            {   ldaw r11, sp[STACK_VEC_TMP]             ;   sub Q(cig_left), Q(cig_left), 8         }
            vlashr r11[0], Q(cig_left)
            {   mkmsk r11, 16                           ;   ldw Q(cig_left), sp[STACK_TMP]          }
            {   andnot r11, Q(cig_left)                 ;   vdepth8                                 }
            vstrpv Y[0], r11
            {   shl r11, _32, 4                         ;   ldw Q(cig_left), sp[STACK_C_OUT]        }
            {   sub K, K, C_in                          ;   vsetc r11   /* 8-bit mode */            }
            {   add Y, Y, Q(cig_left)                   ;                                           }
            bt pix_left, .L_pix_loop

#endif
        {   add K, K, C_in                          ;   ldw r11, sp[STACK_COG_STRIDE_Y]         }
        {   ldc cig_left, 40                        ;   ldw X, sp[STACK_X]                      }
        {                                           ;   stw BSO, sp[STACK_BSO]                  }
        {   add Y, Y, r11                           ;                                           }
        bt cog_left, .L_cog_loop

#define macc_hop        cog_left
#define reset_offset    pix_left

    .L_cog_tail:
        {   sub K, K, _32                           ;   ldw plan, sp[STACK_PLAN]                }
        {   ldc pix_left, 12                        ;   ldw r11, plan[PLAN_COUT]                }
        {   zext r11, 4                             ;                                           }
        {   sub macc_hop, pix_left, r11             ;                                           }
        {   sub reset_offset, _32, r11              ;                                           }
        bf r11, .Lfunc_end
        {   sub reset_offset, reset_offset, r11     ;   ldw K_cig_stride, plan[PLAN_TAIL_KSTRIDE]}
        {   add K, K, K_cig_stride                  ;   ldw r8, sp[STACK_PIX_COUNT]             }
        .L_pix_loop2:
#if CONFIG_SYMMETRIC_SATURATION_conv2d_1x1
            {                                           ;   ldw BSO, sp[STACK_BSO]                  }
#else
            {   ldaw r11, cp[VPU_VEC_0x80]              ;   ldw BSO, sp[STACK_BSO]                  }
            {                                           ;   vldr r11[0]                             }
            {                                           ;   ldw r11, sp[STACK_TAIL_MASK]            }
            vstrpv Y[0], r11
#endif
            {   add r11, BSO, _32                       ;   vldd BSO[0]                             }
            {   sub r8, r8, 1                           ;   vldr r11[0]                             }
            {   shr cig_left, C_in, 5                   ;   stw r8, sp[STACK_PIX_COUNT]             }
            {   add BSO, r11, _32                       ;   bf cig_left, .L_cig_tail2               }
            .L_cig_loop2:
                {   ldaw r11, sp[STACK_VEC_TMP]             ;   vldc X[0]                               }
                {   sub r11, r11, reset_offset              ;   vstr r11[0]                             }
                {   ldaw r11, sp[STACK_VEC_TMP]             ;   vldr r11[0]                             }
                {   sub r11, r11, reset_offset              ;   vstd r11[0]                             }
                {   shl r11, _32, 3                         ;   vldd r11[0]                             }
                {   sub cig_left, cig_left, 1               ;   bru macc_hop                            }
                VLMACCR_GROUP_TAIL(C_in, K_cig_stride)
                {   add X, X, _32                           ;   bt cig_left, .L_cig_loop2               }
            .L_cig_tail2:
            {   add K, K, cin_tail                      ;   bf cin_tail, .L_get_res2                }
            {   sub K, K, _32                           ;   sub cin_tail, _32, cin_tail             }
                {   ldaw r11, sp[STACK_VEC_TMP]             ;   vldc X[0]                               }
                {   sub r11, r11, cin_tail                  ;   vstc r11[0]                             }
                {   ldaw r11, sp[STACK_VEC_TMP]             ;   vldc r11[0]                             }
                {   sub r11, r11, reset_offset              ;   vstr r11[0]                             }
                {   ldaw r11, sp[STACK_VEC_TMP]             ;   vldr r11[0]                             }
                {   sub r11, r11, reset_offset              ;   vstd r11[0]                             }
                {   sub cin_tail, _32, cin_tail             ;   vldd r11[0]                             }
                {   sub cig_left, cig_left, 1               ;   bru macc_hop                            }
                VLMACCR_GROUP_TAIL(C_in, K_cig_stride)
            .L_get_res2:
#if CONFIG_SYMMETRIC_SATURATION_conv2d_1x1
            {   shl r11, _32, 3                         ;                                           }     
            {                                           ;   vsetc r11                               }
            {   ldaw r11, sp[STACK_VEC_TMP]             ;   vlsat BSO[0]                            }
            {   add BSO, BSO, _32                       ;   vstr r11[0]                             }   
            {   add BSO, BSO, _32                       ;   vldc BSO[0]                             }
            {   add X, X, cin_tail                      ;   vclrdr                                  }
            {   shl r11, _32, 4                         ;   vlmacc r11[0]                           }
            {   add BSO, BSO, _32                       ;   vldc BSO[0]                             }
            {   add BSO, BSO, _32                       ;   vlmacc BSO[0]                           }
            {                                           ;   vsetc r11                               }
            {                                           ;   vlsat BSO[0]                            }
            {                                           ;   ldw r11, sp[STACK_TAIL_MASK]            }
            vstrpv Y[0], r11
            {                                           ;   ldw r8, sp[STACK_C_OUT]                 }
            {   add Y, Y, r8                            ;   ldw r8, sp[STACK_PIX_COUNT]             }
            {   sub K, K, C_in                          ;   bt r8, .L_pix_loop2                     }
#else
            {   shl r11, _32, 3                         ;                                           }     
            {                                           ;   vsetc r11                               }
            {   ldaw r11, sp[STACK_VEC_TMP]             ;   vlsat BSO[0]                            }
            {   add BSO, BSO, _32                       ;   vstr r11[0]                             }   
            {   add BSO, BSO, _32                       ;   vldc BSO[0]                             }
            {   add X, X, cin_tail                      ;   vclrdr                                  }
            {                                           ;   vlmacc r11[0]                           }
            {   add BSO, BSO, _32                       ;   vldc BSO[0]                             }
            {   add BSO, BSO, _32                       ;   vlmacc BSO[0]                           }

            {   add BSO, BSO, _32                       ;   vlsat BSO[0]                            }
            {   ldaw r11, cp[VPU_VEC_0x007F]            ;   vstr r11[0]                             }
            {   ldaw r11, sp[STACK_TMP]                 ;   vladd r11[0]                            }
            {   mkmsk Q(cig_left), 4                    ;   vdepth1                                 }
            vstrpv r11[0], Q(cig_left)
            {   ldc Q(cig_left), 0                      ;                                           }
            {   ldaw r11, sp[STACK_VEC_TMP]             ;   sub Q(cig_left), Q(cig_left), 8         }
            vlashr r11[0], Q(cig_left)
            {                                           ;   ldw Q(cig_left), sp[STACK_TMP]          }
            {                                           ;   ldw r11, sp[STACK_TAIL_MASK]            }
            {   andnot r11, Q(cig_left)                 ;   vdepth8                                 }
            vstrpv Y[0], r11
            {   shl r11, _32, 4                         ;   ldw Q(cig_left), sp[STACK_C_OUT]        }
            {   sub K, K, C_in                          ;   vsetc r11   /* 8-bit mode */            }
            {   add Y, Y, Q(cig_left)                   ;   ldw Q(cig_left), sp[STACK_PIX_COUNT]    }
            bt Q(cig_left), .L_pix_loop2

#endif


.L_img_end:

.Lfunc_end:
    //Restore the original constant pool pointer
    {                                           ;   ldw r11, sp[STACK_CP]                   }
    {                                           ;   set cp, r11                             }

    {                                           ;   ldw r10, sp[8]                          }
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



