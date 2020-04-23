
#if defined(__XS3A__)

/**


void nn_conv2d_hstrip_depthwise_padded(
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
    const int32_t window_hstride,
    const int32_t y_col_stride,
    const unsigned out_cols,
    const unsigned chans_to_write,
    const int8_t* zero_point_vec);
*/

#define FUNCTION_NAME nn_conv2d_hstrip_depthwise_padded

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
#define STACK_WINDOW_HSTRIDE    (NSTACKWORDS+9)
#define STACK_Y_COL_STRIDE      (NSTACKWORDS+10)
#define STACK_OUT_COLS          (NSTACKWORDS+11)
#define STACK_CHAN2WRITE        (NSTACKWORDS+12)
#define STACK_ZERO_VEC_PTR      (NSTACKWORDS+13)


#define STACK_VEC_VD            (NSTACKWORDS-8)
#define STACK_VEC_VR            (NSTACKWORDS-16)
#define STACK_VEC_TMP           (NSTACKWORDS-24)
#define STACK_VEC_ZERO          (NSTACKWORDS-32)


#define STACK_MID_ROWS      (9)
#define STACK_CTR_COLS      (10)
#define STACK_X_PSTART      (11)
#define STACK_K_PSTART      (12)
#define STACK_BSS_PSTART    (13)


#define Y               r0
#define X               r1
#define K               r2


.align 16
FUNCTION_NAME:
/**/dualentsp NSTACKWORDS 
    {   ldc r11, 32                             ;   stw r10, sp[8]                          }
    std r8, r9, sp[3]   
    {   shl r11, r11, 4                         ;                                           }
/**/std r4, r5, sp[1]
    std r6, r7, sp[2]
    {   ldc r10, 32                             ;   vsetc r11                               }

    {   add r11, r3, r10                        ;   vldd r3[0]                              }
/**/{   add r11, r11, r10                       ;   vldr r11[0]                             }
    {                                           ;   stw r11, sp[STACK_BSS_PSTART]           }

    {    ldc r10, 0                             ;   ldw r3, sp[STACK_K_H]                   }
    {                                           ;   ldw r11, sp[STACK_PAD_T]                }
/**/{   sub r3, r3, r11                         ;   ldw r11, sp[STACK_PAD_B]                }
    {   sub r3, r3, r11                         ;   ldw r11, sp[STACK_PAD_L]                }
    {   lss r9, r10, r11                        ;   ldw r5, sp[STACK_XK_COL_STRIDE]         }
    mul r6, r5, r11
/**/{   neg r9, r9                              ;   stw r6, sp[STACK_PAD_L]                 }
    {                                           ;   ldw r4, sp[STACK_K_W]                   }
    maccs r10, r4, r9, r11
    {   ldaw r8, sp[STACK_ZERO_VEC_PTR]         ;   ldw r11, sp[STACK_PAD_R]                }
/**/{   lss r9, r10, r11                        ;   ldw r8, r8[0]                           }
    mul r6, r5, r11
    {   neg r9, r9                              ;   stw r6, sp[STACK_PAD_R]                 }
    maccs r10, r4, r9, r11
/**/{                                           ;   stw r3, sp[STACK_MID_ROWS]              }
    mul r4, r4, r5
    {   mov r11, r8                             ;   stw r4, sp[STACK_CTR_COLS]              }
    {   ldaw r11, sp[STACK_VEC_ZERO]            ;   vldc r11[0]                             }
/**/{                                           ;   bu .L_vbias_adj                         }
    

#define Q(RSOMETHING)    RSOMETHING

#define xk_col_stride   r3
#define x_row_stride    r4
#define _32             r5
#define rows_left       r6
#define cols_left       r7
#define pad_t           r8
#define pad_b           r9
#define K_w             r10

// FUNCTION_NAME + 0x80
    .align 16
    .L_vbias_adj:
/**/{   mov xk_col_stride, r5                   ;   vstc r11[0]                             }
    {   ldc Q(cols_left), 0                     ;   ldw x_row_stride, sp[STACK_X_ROW_STRIDE]}
    {   ldc _32, 32                             ;   ldw K_w, sp[STACK_K_W]                  }
    mul r11, K_w, xk_col_stride
/**/{   add r11, r11, x_row_stride              ;   ldw pad_t, sp[STACK_PAD_T]              }
    {                                           ;   ldw pad_b, sp[STACK_PAD_B]              }
    {                                           ;   bf pad_t, .L_vbias_adj_bot              }
    maccu Q(cols_left), X, pad_t, r11
/**/mul cols_left, pad_t, K_w
    .L_tpad_col:
        {   sub cols_left, cols_left, 1             ;   vlmacc K[0]                             }
        {   add K, K, xk_col_stride                 ;   bt cols_left, .L_tpad_col               }

    .L_vbias_adj_bot:
    mul r11, xk_col_stride, K_w
/**/{                                           ;   ldw rows_left, sp[STACK_MID_ROWS]       }
    {   ldc Q(cols_left), 0                     ;   stw K, sp[STACK_K_PSTART]               }
    maccu Q(cols_left), K, r11, rows_left
    {   ldaw r11, sp[STACK_VEC_VD]              ;   bf pad_b, .L_vbias_adj_save             }
/**/mul cols_left, K_w, pad_b
    .L_bpad_col:
        {   sub cols_left, cols_left, 1             ;   vlmacc K[0]                             }
        {   add K, K, xk_col_stride                 ;   bt cols_left, .L_bpad_col               }

    .L_vbias_adj_save:
    {   ldaw r11, sp[STACK_VEC_VR]              ;   vstd r11[0]                             }

////
////
///////////////////////////////////////////////

// #undef xk_col_stride   r3
// #undef x_row_stride    r4
// #undef _32             r5
// #undef rows_left       r6
// #undef cols_left       r7
#undef pad_t           
#undef pad_b           
#undef K_w             

#define window_hstride  r8
#define pix_left        r9
#define X_start         r10

// FUNCTION_NAME + d0
/**/{                                           ;   vstr r11[0]                             }
    {   mov X_start, X                          ;   ldw window_hstride, sp[STACK_WINDOW_HSTRIDE]}
    {                                           ;   ldw pix_left, sp[STACK_OUT_COLS]        }
    {                                           ;   bu .L_pix_start                         }

    .align 16
    .L_pix_start:
/**/    {   sub pix_left, pix_left, 1               ;   mov X, X_start                          }
        {   ldaw r11, sp[STACK_VEC_VD]              ;   ldw K, sp[STACK_K_PSTART]               }
        {   ldaw r11, sp[STACK_VEC_VR]              ;   vldd r11[0]                             }
        {   add X_start, X_start, window_hstride    ;   vldr r11[0]                             }
/**/    {   ldc Q(_32), 0                           ;   ldw rows_left, sp[STACK_MID_ROWS]       }
        .L_patch_row:
            {   ldaw r11, sp[STACK_VEC_ZERO]            ;   ldw cols_left, sp[STACK_PAD_L]          }
            {   lss r11, Q(_32), cols_left              ;   vldc r11[0]                             }
            mul cols_left, cols_left, r11
/**/        {   add X, X, cols_left                     ;   bf cols_left, .L_patch_left_end     }
            .L_patch_left:
                {   sub cols_left, cols_left, xk_col_stride ;   vlmacc K[0]                             }
                {   add K, K, xk_col_stride                 ;   bt cols_left, .L_patch_left             }
            .L_patch_left_end:
            {   sub rows_left, rows_left, 1             ;   ldw cols_left, sp[STACK_CTR_COLS]       }

            .L_patch_ctr:
/**/            {   sub cols_left, cols_left, xk_col_stride ;   vldc X[0]                               }
                {   add X, X, xk_col_stride                 ;   vlmacc K[0]                             }
                {   add K, K, xk_col_stride                 ;   bt cols_left, .L_patch_ctr              }

            {   ldaw r11, sp[STACK_VEC_ZERO]            ;   ldw cols_left, sp[STACK_PAD_R]          }
/**/        {   lss r11, Q(_32), cols_left              ;   vldc r11[0]                             } // + 0x120
            mul cols_left, cols_left, r11
            {   add X, X, cols_left                     ;   bf cols_left, .L_patch_right_end        }
            {   nop                                     ;   nop                                     }
            .L_patch_right:
/**/            {   sub cols_left, cols_left, xk_col_stride ;   vlmacc K[0]                             }
                {   add K, K, xk_col_stride                 ;   bt cols_left, .L_patch_right            }
            .L_patch_right_end:
            {   add X, X, x_row_stride                  ;   bt rows_left, .L_patch_row              }
        {   ldc _32, 32                             ;   bu .L_pix_finish                        }

    .align 16
    .L_pix_finish:
/**/    {   shl r11, _32, 3                         ;   ldw Q(cols_left), sp[STACK_CHAN2WRITE]  }
        {   ldaw r11, sp[STACK_VEC_TMP]             ;   vsetc r11                               }
        {                                           ;   ldw Q(rows_left), sp[STACK_BSS_PSTART]  }
        {   add Q(rows_left), Q(rows_left), _32     ;   vlsat Q(rows_left)[0]                   }
/**/    {                                           ;   vstr r11[0]                             }
        {   add Q(rows_left), Q(rows_left), _32     ;   vldc Q(rows_left)[0]                    }
        {                                           ;   vclrdr                                  }
        {   shl r11, _32, 4                         ;   vlmacc r11[0]                           }
        {   add Q(rows_left), Q(rows_left), _32     ;   vldc Q(rows_left)[0]                    }
        {   add Q(rows_left), Q(rows_left), _32     ;   vlmacc Q(rows_left)[0]                  }
/**/    {   mkmsk Q(cols_left), Q(cols_left)        ;   vsetc r11                               }
        {                                           ;   vlsat Q(rows_left)[0]                   }
        vstrpv Y[0], Q(cols_left)
        {   ldc r11, 0                              ;   ldw Q(cols_left), sp[STACK_Y_COL_STRIDE]}
/**/    {   add Y, Y, Q(cols_left)                  ;   ldw Q(rows_left), sp[STACK_PAD_L]       }

        //  (NOTE: PAD_L, PAD_R, CTR_COLS and window_hstride have been multiplied by xk_col_stride)
        //  (NOTE: The left padding can only decrease over time, right padding can only increase)

        //  We're shifting the patch to the right, so SUBTRACT from left padding. But if it was
        //  positive (>0) BEFORE, then CTR_COLS has to INCREASE commensurately, because we have 
        //  less left-padding

        {   lss Q(cols_left), r11, Q(rows_left)             ;   ldw Q(_32), sp[STACK_CTR_COLS]          }
        {   lss Q(cols_left), Q(rows_left), window_hstride  ;   bf Q(cols_left), .L_werwer              }
        {   add Q(cols_left), Q(cols_left), 1               ;   bru Q(cols_left)                        }
/**/    {   add Q(_32), Q(_32), window_hstride              ;   bru Q(cols_left)                        }
        {   add Q(_32), Q(_32), Q(rows_left)                ;                                           }
    .L_werwer:
        {   sub Q(rows_left), Q(rows_left), window_hstride  ;   ldw Q(cols_left), sp[STACK_PAD_R]       }
        //  We're shifting the patch to the right, so ADD to right padding. But if it is
        //  positive (>0) AFTER, then CTR_COLS has to DECREASE commensurately, because we have 
        //  more right-padding
        {   add Q(cols_left), Q(cols_left), window_hstride  ;   stw Q(rows_left), sp[STACK_PAD_L]       }
/**/    {   ldc r11, 0                                      ;                                           }
        {   lss Q(rows_left), r11, Q(cols_left)             ;   stw Q(cols_left), sp[STACK_PAD_R]       } 
        {   lss Q(rows_left), Q(cols_left), window_hstride  ;   bf Q(rows_left), .L_rewrew              }
        {   add Q(rows_left), Q(rows_left), 1               ;   bru Q(rows_left)                        }
/**/    {   sub Q(_32), Q(_32), window_hstride              ;   bru Q(rows_left)                        }
        {   sub Q(_32), Q(_32), Q(cols_left)                ;                                           }
    .L_rewrew:
        {                                                   ;   stw Q(_32), sp[STACK_CTR_COLS]          }
        {   ldc _32, 32                                     ;   bt pix_left, .L_pix_start               }

#undef pix_left
#undef _32

.Lfunc_end:
/**/{                                           ;   ldw r10, sp[8]                          }
    ldd r8, r9, sp[3]
    ldd r6, r7, sp[2]
    ldd r4, r5, sp[1]
/**/retsp NSTACKWORDS


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



