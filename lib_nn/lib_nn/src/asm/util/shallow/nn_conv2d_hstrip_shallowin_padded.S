
#if defined(__XS3A__)

/**
void nn_conv2d_hstrip_shallowin_padded_asm(
        nn_image_t* Y,
        const nn_image_t* X,
        const nn_tensor_t* K,
        const nn_bss_block_t* BSS,
        const unsigned K_h,
        const unsigned K_h_stride,
        const channel_count_t C_in,
        const unsigned pad_t,
        const unsigned pad_b,
        const int pad_l_initial,
        const int pad_r_initial,
        const mem_stride_t x_v_stride,
        const mem_stride_t y_h_stride,
        const unsigned out_cols,
        const int8_t* zero_point_vec)
*/

#define FUNCTION_NAME nn_conv2d_hstrip_shallowin_padded_asm


#define NSTACKVECS  (4)

#define NSTACKWORDS  ((NSTACKVECS*8)+10)
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 16
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME


#define STACK_K_H               (NSTACKWORDS+1)
#define STACK_K_h_stride        (NSTACKWORDS+2)
#define STACK_C_IN              (NSTACKWORDS+3)
#define STACK_PAD_T             (NSTACKWORDS+4)
#define STACK_PAD_B             (NSTACKWORDS+5)
#define STACK_PAD_L             (NSTACKWORDS+6)
#define STACK_PAD_R             (NSTACKWORDS+7)
#define STACK_X_V_STRIDE        (NSTACKWORDS+8)
#define STACK_Y_H_STRIDE        (NSTACKWORDS+9)
#define STACK_OUT_COLS          (NSTACKWORDS+10)
#define STACK_VEC_ZERO_POINT    (NSTACKWORDS+11)


#define STACK_VEC_ADJ_B_HI      (NSTACKWORDS-8)
#define STACK_VEC_ADJ_B_LO      (NSTACKWORDS-16)
#define STACK_VEC_TMP           (NSTACKWORDS-24)
#define STACK_VEC_VR            (NSTACKWORDS-32)

#define STACK_PATCH_ROWS        STACK_K_H
#define STACK_K                 7
#define STACK_BSS               8
#define STACK_WIN_H_STRIDE      STACK_K_h_stride



#define Q(R)      R

.align 16
FUNCTION_NAME:
    dualentsp NSTACKWORDS
    std r4, r5, sp[0]
    std r6, r7, sp[1]
    std r8, r9, sp[2]
    {                                           ;   stw r10, sp[6]                          }

#define Y               r0
#define X               r1
#define K               r2
#define BSS             r3
#define x_v_stride      r4
#define k_cout_str      r5
#define rows_left       r6
#define _32             r7
#define tmp             r8
#define vec_zp          r9
#define X_patch         r10


    {   ldc _32, 32                             ;                                           }
    {   shl r11, _32, 4                         ;                                           }
    {                                           ;   vsetc r11                               }
    {   add r11, BSS, _32                       ;   vldd BSS[0]                             }
    {   add r11, r11, _32                       ;   vldr r11[0]                             }
    {                                           ;   stw r11, sp[STACK_BSS]                  }
#undef BSS
    {                                           ;   ldw tmp, sp[STACK_K_H]                  }
    {   shl k_cout_str, tmp, 5                  ;   ldw rows_left, sp[STACK_PAD_T]          }
    {   sub tmp, tmp, rows_left                 ;   ldw rows_left, sp[STACK_PAD_B]          }
    {   sub tmp, tmp, rows_left                 ;                                           }
    {                                           ;   stw tmp, sp[STACK_PATCH_ROWS]           }

    {                                           ;   ldw tmp, sp[STACK_K_h_stride]           }
    {                                           ;   ldw r11, sp[STACK_C_IN]                 }
    mul tmp, tmp, r11
    {                                           ;   stw tmp, sp[STACK_WIN_H_STRIDE]         }

    {                                           ;   ldw vec_zp, sp[STACK_VEC_ZERO_POINT]    }
    {                                           ;   vldc vec_zp[0]                          }
    {   mov tmp, K                              ;   ldw x_v_stride, sp[STACK_X_V_STRIDE]    }

    //Check if any top padding
    {   ldc r11, 0                              ;   ldw rows_left, sp[STACK_PAD_T]          }
    maccu r11, X, x_v_stride, rows_left
    maccu r11, K, _32, rows_left
    {   mov K, tmp                              ;   stw K, sp[STACK_K]                      }

    {                                           ;   bf rows_left, .L_pad_tb_end             }
    .L_pad_tb_row_start:
        {   sub tmp, K, k_cout_str                  ;   vlmaccr K[0]                            }
        {   sub tmp, tmp, k_cout_str                ;   vlmaccr tmp[0]                          }
        {   sub tmp, tmp, k_cout_str                ;   vlmaccr tmp[0]                          }
        {   sub tmp, tmp, k_cout_str                ;   vlmaccr tmp[0]                          }
        {   sub tmp, tmp, k_cout_str                ;   vlmaccr tmp[0]                          }
        {   sub tmp, tmp, k_cout_str                ;   vlmaccr tmp[0]                          }
        {   sub tmp, tmp, k_cout_str                ;   vlmaccr tmp[0]                          }
        {   sub tmp, tmp, k_cout_str                ;   vlmaccr tmp[0]                          }
        {   sub tmp, tmp, k_cout_str                ;   vlmaccr tmp[0]                          }
        {   sub tmp, tmp, k_cout_str                ;   vlmaccr tmp[0]                          }
        {   sub tmp, tmp, k_cout_str                ;   vlmaccr tmp[0]                          }
        {   sub tmp, tmp, k_cout_str                ;   vlmaccr tmp[0]                          }
        {   sub tmp, tmp, k_cout_str                ;   vlmaccr tmp[0]                          }
        {   sub tmp, tmp, k_cout_str                ;   vlmaccr tmp[0]                          }
        {   sub tmp, tmp, k_cout_str                ;   vlmaccr tmp[0]                          }
        {   sub rows_left, rows_left, 1             ;   vlmaccr tmp[0]                          }
        {   add K, K, _32                           ;   bt rows_left, .L_pad_tb_row_start       }
    .L_pad_tb_end:
    {   ldaw Q(vec_zp), sp[STACK_VEC_ADJ_B_HI]  ;   bt r11, .L_pad_tb_done                  }
    {                                           ;   ldw rows_left, sp[STACK_PAD_B]          }
    {                                           ;   ldw tmp, sp[STACK_PATCH_ROWS]           }
    maccu r11, K, _32, tmp  //  move K past non-padding rows
    {   ldc r11, 1                              ;   bt rows_left, .L_pad_tb_row_start       }

    .L_pad_tb_done:
    //Save adjusted biases on the stack.
    {   ldaw Q(vec_zp), sp[STACK_VEC_ADJ_B_LO]  ;   vstd Q(vec_zp)[0]                       }
    {                                           ;   vstr Q(vec_zp)[0]                       }


#define pad_mask        r3

    {                                           ;   ldw vec_zp, sp[STACK_VEC_ZERO_POINT]    }
    {   mkmsk pad_mask, 32                      ;   ldw Q(rows_left), sp[STACK_C_IN]        }
    
    // Figure out the initial padding mask. From here on the left/right padding will be
    //  stored as bytes, rather than pixels
    {   ldc tmp, 1                              ;   ldw r11, sp[STACK_PAD_L]                }
    mul r11, r11, Q(rows_left)
    {   lss tmp, r11, tmp                       ;   stw r11, sp[STACK_PAD_L]                }
    {   mkmsk r11, r11                          ;   bru tmp                                 }
    {   andnot pad_mask, r11                    ;                                           }
    {   ldc tmp, 1                              ;   ldw r11, sp[STACK_PAD_R]                }
    mul r11, r11, Q(rows_left)
    {   lss tmp, r11, tmp                       ;   stw r11, sp[STACK_PAD_R]                }
    {   sub r11, _32, r11                       ;   bru tmp                                 }
    {   zext pad_mask, r11                      ;   ldw tmp, sp[STACK_OUT_COLS]             }

    .L_out_col_start:
        // load number of patch rows, initialize accumulators, decrement out cols
        {   ldaw r11, sp[STACK_VEC_ADJ_B_HI]        ;   ldw rows_left, sp[STACK_PATCH_ROWS]     }
        {   ldaw r11, sp[STACK_VEC_ADJ_B_LO]        ;   vldd r11[0]                             }
        {   sub tmp, tmp, 1                         ;   vldr r11[0]                             }
        {   mov X_patch, X                          ;   stw tmp, sp[STACK_OUT_COLS]             }
        
        // K resets for every output pixel
        {   ldc _32, 32                             ;   ldw K, sp[STACK_K]                      }

        // Throw the zero-point vector into the temp vector on the stack. We'll use VSTRPV to
        // only overwrite the non-padding bytes, which won't change until we're done accumulating
        // for this patch.
        {   ldaw r11, sp[STACK_VEC_TMP]             ;   vldc vec_zp[0]                          }
        {   ldaw r11, sp[STACK_VEC_VR]              ;   vstc r11[0]                             }

        .L_patch_row_start:
            // Stash vR (low acc), use VSTRPV to mask input into VEC_TMP, load masked input into vC
            //  unstash vR
            {   mov r11, X_patch                        ;   vstr r11[0]                             }
            {   ldaw tmp, sp[STACK_VEC_TMP]             ;   vldr r11[0]                             }
            vstrpv tmp[0], pad_mask
            {   ldaw r11, sp[STACK_VEC_VR]              ;   vldc tmp[0]                             }
            {   add X_patch, X_patch, x_v_stride        ;   vldr r11[0]                             }

            // Do VLMACCRs
            {   sub tmp, K, k_cout_str                  ;   vlmaccr K[0]                            }
            {   sub tmp, tmp, k_cout_str                ;   vlmaccr tmp[0]                          }
            {   sub tmp, tmp, k_cout_str                ;   vlmaccr tmp[0]                          }
            {   sub tmp, tmp, k_cout_str                ;   vlmaccr tmp[0]                          }
            {   sub tmp, tmp, k_cout_str                ;   vlmaccr tmp[0]                          }
            {   sub tmp, tmp, k_cout_str                ;   vlmaccr tmp[0]                          }
            {   sub tmp, tmp, k_cout_str                ;   vlmaccr tmp[0]                          }
            {   sub tmp, tmp, k_cout_str                ;   vlmaccr tmp[0]                          }
            {   sub tmp, tmp, k_cout_str                ;   vlmaccr tmp[0]                          }
            {   sub tmp, tmp, k_cout_str                ;   vlmaccr tmp[0]                          }
            {   sub tmp, tmp, k_cout_str                ;   vlmaccr tmp[0]                          }
            {   sub tmp, tmp, k_cout_str                ;   vlmaccr tmp[0]                          }
            {   sub tmp, tmp, k_cout_str                ;   vlmaccr tmp[0]                          }
            {   sub tmp, tmp, k_cout_str                ;   vlmaccr tmp[0]                          }
            {   sub tmp, tmp, k_cout_str                ;   vlmaccr tmp[0]                          }
            {   sub rows_left, rows_left, 1             ;   vlmaccr tmp[0]                          }

            // Each row of K (second dimension) is exactly 32 bytes. Iterate if rows remain.
            {   add K, K, _32                           ;   bt rows_left, .L_patch_row_start        }

        .L_patch_row_end:

        {   shl r11, _32, 3                         ;   ldw tmp, sp[STACK_BSS]      /*load BSS*/}
        {   ldaw r11, sp[STACK_VEC_TMP]             ;   vsetc r11            /*set 16-bit mode*/}
        {   add tmp, tmp, _32                       ;   vlsat tmp[0]       /*apply first shift*/}
        {                                           ;   vstr r11[0] /*save 16-bit intermediate*/}
        {   add tmp, tmp, _32                       ;   vldc tmp[0]       /*load scale into vC*/}
        {                                           ;   vclrdr   /*clear accumulate for VLMACC*/}
        {   shl r11, _32, 4                         ;   vlmacc r11[0]              /*do VLMACC*/}
        {                                           ;   vsetc r11             /*set 8-bit mode*/}
        {   mkmsk tmp, 16        /*16 out channels*/;   vlsat tmp[0]       /*apply final shift*/}
        vstrpv Y[0], tmp         /* Store output */
        {   ldc pad_mask, 32                        ;   ldw tmp, sp[STACK_Y_H_STRIDE]           }
        {   add Y, Y, tmp                           ;   ldw tmp, sp[STACK_WIN_H_STRIDE]         }
        {   add X, X, tmp                           ;                                           }

#define l_padding   Q(rows_left)
#define r_padding   Q(_32)    

        {   ldc r11, 0                              ;   ldw r_padding, sp[STACK_PAD_R]          }
        {   add r_padding, r_padding, tmp           ;   ldw l_padding, sp[STACK_PAD_L]          }
        {   sub l_padding, l_padding, tmp           ;                                           }
        {   lss tmp, r11, r_padding                 ;   stw r_padding, sp[STACK_PAD_R]          }
        mul r_padding, r_padding, tmp
        {   lss tmp, r11, l_padding                 ;   sub pad_mask, pad_mask, r_padding       }
        mul l_padding, l_padding, tmp
        {   sub pad_mask, pad_mask, l_padding       ;   stw l_padding, sp[STACK_PAD_L]          }
        {   mkmsk pad_mask, pad_mask                ;   ldw tmp, sp[STACK_OUT_COLS]             }
        {   shl pad_mask, pad_mask, l_padding       ;   bt tmp, .L_out_col_start                }
    .L_out_col_end:

#undef l_padding
#undef r_padding
#undef center_cols
#undef k_h_stride


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
