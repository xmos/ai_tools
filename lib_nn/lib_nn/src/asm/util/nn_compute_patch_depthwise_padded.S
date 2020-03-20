
#if defined(__XS3A__)

/**


void nn_compute_patch_depthwise_padded_asm(
    int8_t* Y,
    const int8_t* X, 
    const int8_t* K,
    const nn_bss_block_t* BSS,
    const unsigned K_h,
    const unsigned K_w,
    const unsigned pad_t,
    const unsigned pad_l,
    const unsigned pad_b,
    const unsigned pad_r,
    const int32_t xk_col_stride,
    const int32_t x_row_stride,
    const unsigned chans_to_write,
    const int8_t* zero_point_vec);
*/

#define FUNCTION_NAME nn_compute_patch_depthwise_padded_asm

#define NSTACKWORDS  48
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 4
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME


#define STACK_K_H               (NSTACKWORDS+1)
#define STACK_K_W               (NSTACKWORDS+2)
#define STACK_PAD_T             (NSTACKWORDS+3)
#define STACK_PAD_L             (NSTACKWORDS+4)
#define STACK_PAD_B             (NSTACKWORDS+5)
#define STACK_PAD_R             (NSTACKWORDS+6)
#define STACK_XK_COL_STRIDE     (NSTACKWORDS+7)
#define STACK_X_ROW_STRIDE      (NSTACKWORDS+8)
#define STACK_CHAN2WRITE        (NSTACKWORDS+9)
#define STACK_ZERO_VEC_PTR      (NSTACKWORDS+10)

#define STACK_VEC_TMP           (NSTACKWORDS-8)
#define STACK_VEC_ZERO          (NSTACKWORDS-32)


#define STACK_MID_ROWS      (9)
#define STACK_CTR_COLS      (10)


#define Y               r0
#define X               r1
#define K               r2
#define BSS             r3


.align 16
FUNCTION_NAME:
/**/dualentsp NSTACKWORDS 
    {   ldc r11, 32                             ;   stw r10, sp[8]                          }
    std r8, r9, sp[3]   
    {   shl r11, r11, 4                         ;                                           }
/**/std r4, r5, sp[1]
    std r6, r7, sp[2]
    {   ldc r10, 32                             ;   vsetc r11                               }

    {   add r11, BSS, r10                       ;   vldd BSS[0]                             }
/**/{   add BSS, r11, r10                       ;   vldr r11[0]                             }

    {    ldc r10, 0                             ;   ldw r7, sp[STACK_K_H]                   }
    {                                           ;   ldw r11, sp[STACK_PAD_T]                }
    {   sub r7, r7, r11                         ;   ldw r11, sp[STACK_PAD_B]                }
/**/{   sub r7, r7, r11                         ;   ldw r11, sp[STACK_PAD_L]                }
    {   lss r9, r10, r11                        ;   ldw r5, sp[STACK_XK_COL_STRIDE]         }
    mul r6, r5, r11
    {   neg r9, r9                              ;   stw r6, sp[STACK_PAD_L]                 }
/**/{                                           ;   ldw r4, sp[STACK_K_W]                   }
    maccs r10, r4, r9, r11
    {   ldaw r8, sp[STACK_ZERO_VEC_PTR]         ;   ldw r11, sp[STACK_PAD_R]                }
    {   lss r9, r10, r11                        ;   ldw r8, r8[0]                           }
/**/mul r6, r5, r11
    {   neg r9, r9                              ;   stw r6, sp[STACK_PAD_R]                 }
    maccs r10, r4, r9, r11
    {                                           ;   stw r7, sp[STACK_MID_ROWS]              }
/**/{   mov r11, r8                             ;   stw r4, sp[STACK_CTR_COLS]              }
    {   ldaw r11, sp[STACK_VEC_ZERO]            ;   vldc r11[0]                             }
    {                                           ;   bu .L_vbias_adj                         }
    
// Use of this macro is a hint to you that the register isn't being used for it's nominal purpose
#define Q(RSOMETHING)    RSOMETHING

#define xk_col_stride   r4
#define x_row_stride    r5
#define _32             r6
#define rows_left       r7
#define cols_left       r8
#define pad_t           r9
#define K_w             r10

    .align 16
    .L_vbias_adj:
/**/{   mov xk_col_stride, r5                   ;   vstc r11[0]                             }
    {   ldc Q(cols_left), 0                     ;   ldw x_row_stride, sp[STACK_X_ROW_STRIDE]}
    {   ldc _32, 32                             ;   ldw K_w, sp[STACK_K_W]                  }
    mul r11, K_w, xk_col_stride
/**/{   add r11, r11, x_row_stride              ;   ldw pad_t, sp[STACK_PAD_T]              }
    {                                           ;   bf pad_t, .L_pix_start                  }
    maccu Q(cols_left), X, pad_t, r11
    mul r11, pad_t, K_w
/**/{   zext r11, 1                             ;   shr cols_left, r11, 1                   }
    {                                           ;   bf r11, .L_tpad_col                     }
    {   add K, K, xk_col_stride                 ;   vlmacc K[0]                             }
    {                                           ;   bf cols_left, .L_pix_start              }
    .L_tpad_col:
/**/    {   add K, K, xk_col_stride             ;   vlmacc K[0]                             }
        {   sub cols_left, cols_left, 1         ;   vlmacc K[0]                             }
        {   add K, K, xk_col_stride             ;   bt cols_left, .L_tpad_col               }

    {                                           ;   bu .L_pix_start                         }


// #undef xk_col_stride   r4
// #undef x_row_stride    r5
// #undef _32             r6
// #undef rows_left       r7
// #undef cols_left       r8
#undef pad_t           
#undef K_w 


    .align 16
    .L_pix_start:
/**/    {                                           ;   ldw rows_left, sp[STACK_MID_ROWS]       }
        .L_patch_row:
            {   ldaw r11, sp[STACK_VEC_ZERO]            ;   ldw cols_left, sp[STACK_PAD_L]          }
            {                                           ;   bf cols_left, .L_patch_left_end         }
            {   add X, X, cols_left                     ;   vldc r11[0]                             }
            .L_patch_left:
/**/            {   sub cols_left, cols_left, xk_col_stride ;   vlmacc K[0]                             }
                {   add K, K, xk_col_stride                 ;   bt cols_left, .L_patch_left             }
            .L_patch_left_end:
            {   sub rows_left, rows_left, 1             ;   ldw cols_left, sp[STACK_CTR_COLS]       }
            {                                           ;   bu .L_patch_ctr                         }

            .align 16
            .L_patch_ctr:
/**/            {   sub cols_left, cols_left, 1             ;   vldc X[0]                               }
                {   add X, X, xk_col_stride                 ;   vlmacc K[0]                             }
                {   add K, K, xk_col_stride                 ;   bt cols_left, .L_patch_ctr              }
            {   ldaw r11, sp[STACK_VEC_ZERO]            ;   ldw cols_left, sp[STACK_PAD_R]          }
/**/        {                                           ;   bf cols_left, .L_patch_right_end        }
            {   add X, X, cols_left                     ;   vldc r11[0]                             }
            .L_patch_right:
                {   sub cols_left, cols_left, xk_col_stride ;   vlmacc K[0]                             }
                {   add K, K, xk_col_stride                 ;   bt cols_left, .L_patch_right            }
            .L_patch_right_end:
/**/        {   add X, X, x_row_stride                  ;   bt rows_left, .L_patch_row              }    

    {                                           ;   ldw r11, sp[STACK_PAD_B]                }
    {   ldaw Q(cols_left), sp[STACK_VEC_ZERO]   ;   bf r11, .L_pix_finish                   }
    {                                           ;   vldc Q(cols_left)[0]                    }
/**/{                                           ;   ldw Q(cols_left), sp[STACK_K_W]         }
    mul r11, Q(cols_left), r11
    {   zext r11, 1                             ;   shr cols_left, r11, 1                   }
    {                                           ;   bf r11, .L_bpad_col                     }
/**/{   add K, K, xk_col_stride                 ;   vlmacc K[0]                             }
    {                                           ;   bt cols_left, .L_bpad_col               }
    {                                           ;   bu .L_pix_finish                        }
    {   nop                                     ;                                           }
    
    .align 16
    .L_bpad_col:
/**/    {   add K, K, xk_col_stride             ;   vlmacc K[0]                             }
        {   sub cols_left, cols_left, 1         ;   vlmacc K[0]                             }
        {   add K, K, xk_col_stride             ;   bt cols_left, .L_bpad_col               }
    {                                           ;   bu .L_pix_finish                        }


    .align 16
    .L_pix_finish:
        {   shl r11, _32, 3                         ;   ldw Q(cols_left), sp[STACK_CHAN2WRITE]  }
        {   ldaw r11, sp[STACK_VEC_TMP]             ;   vsetc r11                               }
        {   add BSS, BSS, _32                       ;   vlsat BSS[0]                            }
        {                                           ;   vstr r11[0]                             }   
        {   add BSS, BSS, _32                       ;   vldc BSS[0]                             }
        {                                           ;   vclrdr                                  }
        {   shl r11, _32, 4                         ;   vlmacc r11[0]                           }
        {   mkmsk Q(cols_left), Q(cols_left)        ;   vsetc r11                               }
        {                                           ;   vlsat BSS[0]                            }
        vstrpv Y[0], Q(cols_left)

.Lfunc_end:
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



