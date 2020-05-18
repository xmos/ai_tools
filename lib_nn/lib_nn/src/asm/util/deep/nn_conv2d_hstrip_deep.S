
#if defined(__XS3A__)

#include "nn_config.h"

/**


void nn_conv2d_hstrip_deep(
        nn_image_t* Y,
        const nn_image_t* X,
        const nn_tensor_t* K,
        const nn_bso_block_t* BSO,
        const unsigned K_h,
        const unsigned K_w,
        const unsigned K_h_stride,
        const channel_count_t C_in,
        const mem_stride_t x_v_stride,
        const mem_stride_t k_cout_stride,
        const mem_stride_t y_h_stride,
        const unsigned out_cols)
*/

#define FUNCTION_NAME nn_conv2d_hstrip_deep


#define NSTACKVECS  (2)

#define NSTACKWORDS  ((NSTACKVECS*8)+14)

.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 16
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME


#define STACK_K_H               (NSTACKWORDS+1)
#define STACK_K_W               (NSTACKWORDS+2)
#define STACK_K_h_stride        (NSTACKWORDS+3)
#define STACK_C_IN              (NSTACKWORDS+4)
#define STACK_X_V_STRIDE        (NSTACKWORDS+5)
#define STACK_K_COUT_STRIDE     (NSTACKWORDS+6)
#define STACK_Y_H_STRIDE        (NSTACKWORDS+7)
#define STACK_OUT_COLS          (NSTACKWORDS+8)

#define STACK_VEC_TMP2          (NSTACKWORDS-8)
#define STACK_VEC_TMP1          (NSTACKWORDS-16)

#define STACK_C_IN_GROUPS       7
#define STACK_C_IN_TAIL         8
#define STACK_Y                 9
#define STACK_X                 10
#define STACK_K                 11
#define STACK_BSO               12
#define STACK_TMP               13
#define STACK_H_STRIDES         STACK_K_h_stride


#define Q(R)      R


FUNCTION_NAME:
    dualentsp NSTACKWORDS
    std r4, r5, sp[0]
    std r6, r7, sp[1]
    std r8, r9, sp[2]

    {                                           ;   stw r10, sp[6]                          }
    {                                           ;   stw r0, sp[STACK_Y]                     }
    {                                           ;   stw r1, sp[STACK_X]                     }
    {   ldc r0, 32                              ;   stw r2, sp[STACK_K]                     }

    {   shl r11, r0, 4                          ;                                           }
    {   mov r11, r1                             ;   vsetc r11                               }
    {   ldaw r1, sp[STACK_VEC_TMP1]             ;   vclrdr                                  }
    {                                           ;   vstr r1[0]                              }
    {                                           ;   stw r3, sp[STACK_BSO]                   }
    
    {                                           ;   ldw r1, sp[STACK_C_IN]                  }
    {                                           ;   ldw r0, sp[STACK_K_h_stride]            }
    mul r0, r0, r1
    {   shl r0, r0, 16                          ;   ldw r2, sp[STACK_Y_H_STRIDE]            }
    {   zext r2, 16                             ;                                           }
    {   or r0, r2, r0                           ;                                           }
    {                                           ;   stw r0, sp[STACK_H_STRIDES]             }
    {   shr r2, r1, 5                           ;                                           }
    {   zext r1, 5                              ;   stw r2, sp[STACK_C_IN_GROUPS]           }
    {                                           ;   stw r1, sp[STACK_C_IN_TAIL]             }
    {                                           ;   nop                                     }

#define X               r0
#define K               r1
#define BSO             r2
#define rows_left       r3
#define cols_left       r4
#define cig_left        r5
#define c_in_tail       r6
#define k_cout_stride   r7
#define _32             r8
#define tmp             r9

.align 16
    {   ldc _32, 32                             ;   ldw tmp, sp[STACK_OUT_COLS]             }
    {                                           ;   ldw c_in_tail, sp[STACK_C_IN_TAIL]      }
    {                                           ;   ldw X, sp[STACK_X]                      }
    {                                           ;   ldw k_cout_stride, sp[STACK_K_COUT_STRIDE]}


    .L_out_col_start:
#if !(CONFIG_SYMMETRIC_SATURATION_conv2d_deep)
        {   ldaw r11, cp[vec_0x80]                  ;   ldw Q(cols_left), sp[STACK_Y]           }
        {   mkmsk Q(rows_left), 16                  ;   vldr r11[0]                             }
        vstrpv Q(cols_left)[0], Q(rows_left)
#endif
        {                                           ;   ldw BSO, sp[STACK_BSO]                  }
        {   sub tmp, tmp, 1                         ;   ldw rows_left, sp[STACK_K_H]            }
        {   add r11, BSO, _32                       ;   vldd BSO[0]                             }
        {   add BSO, r11, _32                       ;   vldr r11[0]                             }

        {                                           ;   stw tmp, sp[STACK_OUT_COLS]             }
        {                                           ;   ldw K, sp[STACK_K]                      }

        .L_patch_row_start:
            {   sub rows_left, rows_left, 1             ;   ldw cols_left, sp[STACK_K_W]            }
            .L_center_start:
                {   sub cols_left, cols_left, 1             ;   ldw cig_left, sp[STACK_C_IN_GROUPS]     }

                {                                           ;   bf cig_left, .L_center_cig_end          }
                .L_center_cig_start:
                    {   add X, X, _32                           ;   vldc X[0]                               }
                    {   add tmp, K, k_cout_stride               ;   vlmaccr K[0]                            }
                    {   add tmp, tmp, k_cout_stride             ;   vlmaccr tmp[0]                          }

                    {   add tmp, tmp, k_cout_stride             ;   vlmaccr tmp[0]                          }
                    {   add tmp, tmp, k_cout_stride             ;   vlmaccr tmp[0]                          }
                    {   add tmp, tmp, k_cout_stride             ;   vlmaccr tmp[0]                          }
                    {   add tmp, tmp, k_cout_stride             ;   vlmaccr tmp[0]                          }

                    {   add tmp, tmp, k_cout_stride             ;   vlmaccr tmp[0]                          }
                    {   add tmp, tmp, k_cout_stride             ;   vlmaccr tmp[0]                          }
                    {   add tmp, tmp, k_cout_stride             ;   vlmaccr tmp[0]                          }
                    {   add tmp, tmp, k_cout_stride             ;   vlmaccr tmp[0]                          }

                    {   add tmp, tmp, k_cout_stride             ;   vlmaccr tmp[0]                          }
                    {   add tmp, tmp, k_cout_stride             ;   vlmaccr tmp[0]                          }
                    {   add tmp, tmp, k_cout_stride             ;   vlmaccr tmp[0]                          }
                    {   add tmp, tmp, k_cout_stride             ;   vlmaccr tmp[0]                          }

                    {   add tmp, tmp, k_cout_stride             ;   vlmaccr tmp[0]                          }
                    {   sub cig_left, cig_left, 1               ;   vlmaccr tmp[0]                          }
                    {   add K, K, _32                           ;   bt cig_left, .L_center_cig_start        }
                .L_center_cig_end:
                {   ldaw r11, sp[STACK_VEC_TMP2]            ;   bf c_in_tail, .L_center_tail_end            }

                .L_center_tail_start:
                    {   sub tmp, c_in_tail, _32                 ;   vldc X[0]                               }
                    {   add r11, r11, tmp                       ;   vstc r11[0]                             }
                    {   add X, X, c_in_tail                     ;   vldc r11[0]                             }
                    {   add tmp, K, tmp                         ;                                           }
                    {   add tmp, tmp, k_cout_stride             ;   vlmaccr tmp[0]                          }
                    {   add tmp, tmp, k_cout_stride             ;   vlmaccr tmp[0]                          }
                    {   add tmp, tmp, k_cout_stride             ;   vlmaccr tmp[0]                          }
                    {   add tmp, tmp, k_cout_stride             ;   vlmaccr tmp[0]                          }

                    {   add tmp, tmp, k_cout_stride             ;   vlmaccr tmp[0]                          }
                    {   add tmp, tmp, k_cout_stride             ;   vlmaccr tmp[0]                          }
                    {   add tmp, tmp, k_cout_stride             ;   vlmaccr tmp[0]                          }
                    {   add tmp, tmp, k_cout_stride             ;   vlmaccr tmp[0]                          }

                    {   add tmp, tmp, k_cout_stride             ;   vlmaccr tmp[0]                          }
                    {   add tmp, tmp, k_cout_stride             ;   vlmaccr tmp[0]                          }
                    {   add tmp, tmp, k_cout_stride             ;   vlmaccr tmp[0]                          }
                    {   add tmp, tmp, k_cout_stride             ;   vlmaccr tmp[0]                          }

                    {   add tmp, tmp, k_cout_stride             ;   vlmaccr tmp[0]                          }
                    {   add tmp, tmp, k_cout_stride             ;   vlmaccr tmp[0]                          }
                    {   add tmp, tmp, k_cout_stride             ;   vlmaccr tmp[0]                          }
                    {   add K, K, c_in_tail                     ;   vlmaccr tmp[0]                          }

                .L_center_tail_end:
                {                                           ;   bt cols_left, .L_center_start           }
            .L_center_end:
            {                                           ;   ldw tmp, sp[STACK_X_V_STRIDE]           }
            {   add X, X, tmp                           ;                                           }
            bt rows_left, .L_patch_row_start
        .L_patch_row_end:
#if CONFIG_SYMMETRIC_SATURATION_conv2d_deep

        {   shl r11, _32, 3                         ;                                           }
        {   ldaw r11, sp[STACK_VEC_TMP2]            ;   vsetc r11                               }
        {   add BSO, BSO, _32                       ;   vlsat BSO[0]                            }
        {                                           ;   vstr r11[0]                             }
        {   add BSO, BSO, _32                       ;   vldc BSO[0]                             }
        {                                           ;   vclrdr                                  }
        {   shl r11, _32, 4                         ;   vlmacc r11[0]                           }
        {   add BSO, BSO, _32                       ;   vldc BSO[0]                             }
        {   add BSO, BSO, _32                       ;   vlmacc BSO[0]                           }


        {                                           ;   vsetc r11                               }
        {                                           ;   vlsat BSO[0]                            }
        {   mkmsk tmp, 16                           ;   ldw Q(rows_left), sp[STACK_Y]           }
        vstrpv Q(rows_left)[0], tmp
        {                                           ;   ldw tmp, sp[STACK_H_STRIDES]            }
        {   shr Q(cols_left), tmp, 16               ;   zext tmp, 16                            }
        {   add Q(rows_left), Q(rows_left), tmp     ;   ldw X, sp[STACK_X]                      }
        {   add X, X, Q(cols_left)                  ;   stw Q(rows_left), sp[STACK_Y]           }
        {                                           ;   stw X, sp[STACK_X]                      }
        {                                           ;   ldw tmp, sp[STACK_OUT_COLS]             }
        {                                           ;   bt tmp, .L_out_col_start                }

#else

        {   shl r11, _32, 3                         ;                                           }
        {   ldaw r11, sp[STACK_VEC_TMP2]            ;   vsetc r11                               }
        {   add BSO, BSO, _32                       ;   vlsat BSO[0]                            }
        {                                           ;   vstr r11[0]                             }
        {   add BSO, BSO, _32                       ;   vldc BSO[0]                             }
        {                                           ;   vclrdr                                  }
        {                                           ;   vlmacc r11[0]                           }
        {   add BSO, BSO, _32                       ;   vldc BSO[0]                             }
        {   add BSO, BSO, _32                       ;   vlmacc BSO[0]                           }

        {                                           ;   vlsat BSO[0]                            }
        
        {   ldaw r11, cp[vec_0x007F]                ;   vstr r11[0]                             }
        {   ldaw r11, sp[STACK_TMP]                 ;   vladd r11[0]                            }
        {   mkmsk Q(rows_left), 4                   ;   vdepth1                                 }
        vstrpv r11[0], Q(rows_left)
        {   ldc Q(rows_left), 0                     ;   ldw Q(cig_left), sp[STACK_Y]            }
        {   ldaw r11, sp[STACK_VEC_TMP2]            ;   sub Q(rows_left), Q(rows_left), 8       }
        vlashr r11[0], Q(rows_left)
        {   mkmsk r11, 16                           ;   ldw Q(rows_left), sp[STACK_TMP]         }
        {   andnot r11, Q(rows_left)                ;   vdepth8                                 }
        vstrpv Q(cig_left)[0], r11

        {   shl r11, _32, 4                         ;   ldw tmp, sp[STACK_H_STRIDES]            }
        {   shr Q(cols_left), tmp, 16               ;   zext tmp, 16                            }
        {   add Q(cig_left), Q(cig_left), tmp       ;   vsetc r11                               }
        {                                           ;   ldw X, sp[STACK_X]                      }
        {   add X, X, Q(cols_left)                  ;   stw Q(cig_left), sp[STACK_Y]            }
        {                                           ;   stw X, sp[STACK_X]                      }
        {                                           ;   ldw tmp, sp[STACK_OUT_COLS]             }
        {                                           ;   bt tmp, .L_out_col_start                }

#endif

    .L_out_col_end:


.Lfunc_end:
    {                                           ;   ldw r10, sp[6]                          }
    ldd r8, r9, sp[2]
    ldd r6, r7, sp[1]
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
