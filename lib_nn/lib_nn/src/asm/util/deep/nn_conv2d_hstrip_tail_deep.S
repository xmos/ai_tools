
#if defined(__XS3A__)

/**


void nn_conv2d_hstrip_tail_deep(
        nn_image_t* Y,
        const nn_image_t* X,
        const nn_tensor_t* K,
        const nn_bss_block_t* BSS,
        const unsigned K_h,
        const unsigned K_w,
        const unsigned K_h_stride,
        const channel_count_t C_in,
        const mem_stride_t x_v_stride,
        const mem_stride_t k_cout_stride,
        const mem_stride_t y_h_stride,
        const unsigned out_cols,
        const channel_count_t C_out_tail)
*/

#define FUNCTION_NAME nn_conv2d_hstrip_tail_deep


#define NSTACKVECS  (2)

#define NSTACKWORDS  ((NSTACKVECS*8)+12)
    
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
#define STACK_C_OUT_TAIL        (NSTACKWORDS+9)

#define STACK_VEC_TMP2          (NSTACKWORDS-8)
#define STACK_VEC_TMP1          (NSTACKWORDS-16)

#define STACK_C_IN_GROUPS       STACK_C_IN
#define STACK_C_IN_TAIL         7
#define STACK_Y                 8
#define STACK_X                 9
#define STACK_K                 10
#define STACK_BSS               11
#define STACK_H_STRIDES         STACK_K_h_stride
#define STACK_WRITE_MASK        12


#define Q(R)      R

.align 16
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
    {                                           ;   stw r3, sp[STACK_BSS]                   }
    
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

    //  To move N accumulators from the beginning of vD:vR to the end of vD:vR (which is where
    //  they need to be at the beginning of a VLMACCR sequence), we can store them to address
    //  V, then load them back from address  V-2*(16-N). So store C_OUT_TAIL = 2*(16-N)
    //  If we have a sequence of 12 VLMACCRs (max C_out_tail size) preceded by a BRU 
    //  instruction, and we have N C_out tail channels, we should use a BRU operand value
    //  of  12-N.
    //  And 12-N = ((2*(16-N))/2 - 4), so we can use one register for both of these things if
    //  we can squeeze in 2 instructions between the spots where they're needed (shr one bit
    //  and sub 4; then the reverse)
    {   ldc r11, 16                             ;   ldw r0, sp[STACK_C_OUT_TAIL]            }
    {   sub r0, r11, r0                         ;   mkmsk r2, r0                            }
    {   shl r0, r0, 1                           ;   stw r2, sp[STACK_WRITE_MASK]            }
    {                                           ;   stw r0, sp[STACK_C_OUT_TAIL]            }




#define X               r0
#define K               r1
#define rows_left       r2
#define cols_left       r3
#define cig_left        r4
#define c_in_tail       r5
#define BSS             r6
#define k_cout_stride   r7
#define _32             r8
#define tmp             r9
#define c_out_tail      r10

    {   ldc _32, 32                             ;   ldw tmp, sp[STACK_OUT_COLS]             }
    {                                           ;   ldw c_in_tail, sp[STACK_C_IN_TAIL]      }
    {                                           ;   ldw X, sp[STACK_X]                      }
    {                                           ;   ldw k_cout_stride, sp[STACK_K_COUT_STRIDE]}
    {                                           ;   ldw c_out_tail, sp[STACK_C_OUT_TAIL]    }

    .L_out_col_start:
        {                                           ;   ldw BSS, sp[STACK_BSS]                  }
        {   sub tmp, tmp, 1                         ;   ldw rows_left, sp[STACK_K_H]            }
        {   add r11, BSS, _32                       ;   vldd BSS[0]                             }
        {   add BSS, r11, _32                       ;   vldr r11[0]                             }
        {                                           ;   stw tmp, sp[STACK_OUT_COLS]             }
        {   ldc tmp, 0                              ;   ldw K, sp[STACK_K]                      }

        .L_patch_row_start: //tmp is 0 on first pass, x_v_stride on subsequent iterations
            {   add X, X, tmp                           ;   ldw cols_left, sp[STACK_K_W]            }
            {   ldaw tmp, sp[STACK_VEC_TMP2]            ;   bf cols_left, .L_center_end             }
            .L_center_start:
                {   sub cols_left, cols_left, 1             ;   ldw cig_left, sp[STACK_C_IN_GROUPS]     }
                {   sub r11, tmp, c_out_tail                ;   bf cig_left, .L_center_cig_end          }
                .L_center_cig_start:

                // load the next set of inputs into vC, to be multiplied by each output channel's coefficients
                    {   add X, X, _32                           ;   vldc X[0]                               }
                    
                //reset accumulator positions
                    {   shr c_out_tail, c_out_tail, 1           ;   vstr tmp[0]                             }
                    {   sub c_out_tail, c_out_tail, 4           ;   vldr r11[0]                             }
                    {   sub cig_left, cig_left, 1               ;   vstd tmp[0]                             }
                    {   mov tmp, K                              ;   vldd r11[0]                             }

                //do vlmaccrs
                    {   add c_out_tail, c_out_tail, 4           ;   bru c_out_tail                          }

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
                    {   add K, K, _32                           ;   vlmaccr tmp[0]                          }

                    {   shl c_out_tail, c_out_tail, 1           ;                                           }
                    {   ldaw tmp, sp[STACK_VEC_TMP2]            ;   bt cig_left, .L_center_cig_start        }

                .L_center_cig_end:
                {                                           ;   bf c_in_tail, .L_center_tail_end            }
                .L_center_tail_start:

                //reset accumulator positions
                    {   shr c_out_tail, c_out_tail, 1           ;   vstr tmp[0]                             }
                    {   sub c_out_tail, c_out_tail, 4           ;   vldr r11[0]                             }
                    {   sub tmp, c_in_tail, _32                 ;   vstd tmp[0]                             }
                    {   ldaw r11, sp[STACK_VEC_TMP2]            ;   vldd r11[0]                             }

                //mask C_in_tail
                    {   add X, X, c_in_tail                     ;   vldc X[0]                               }
                    {   add r11, r11, tmp                       ;   vstc r11[0]                             }
                    {   add tmp, K, tmp                         ;   vldc r11[0]                             }

                //do vlmaccrs
                    {   add c_out_tail, c_out_tail, 4           ;   bru c_out_tail                          }

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
                {   shl c_out_tail, c_out_tail, 1           ;                                           }
                {   ldaw tmp, sp[STACK_VEC_TMP2]            ;   bt cols_left, .L_center_start           }
            .L_center_end:
            //                                          // Added to X back up at the start of the row
            {   sub rows_left, rows_left, 1             ;   ldw tmp, sp[STACK_X_V_STRIDE]           }
            bt rows_left, .L_patch_row_start
        .L_patch_row_end:

        {   shl r11, _32, 3                         ;                                           }
        {   ldaw r11, sp[STACK_VEC_TMP2]            ;   vsetc r11                               }
        {   add BSS, BSS, _32                       ;   vlsat BSS[0]                            }
        {                                           ;   vstr r11[0]                             }
        {   add BSS, BSS, _32                       ;   vldc BSS[0]                             }
        {                                           ;   vclrdr                                  }
        {   shl r11, _32, 4                         ;   vlmacc r11[0]                           }
        {   add BSS, BSS, _32                       ;   vldc BSS[0]                             }
        {   add BSS, BSS, _32                       ;   vlmacc BSS[0]                           }
        {                                           ;   vsetc r11                               }
        {                                           ;   vlsat BSS[0]                            }
        {   mkmsk tmp, 16                           ;   ldw Q(rows_left), sp[STACK_Y]           }
        {                                           ;   ldw tmp, sp[STACK_WRITE_MASK]           }
        vstrpv Q(rows_left)[0], tmp
        {                                           ;   ldw tmp, sp[STACK_H_STRIDES]            }
        {   shr Q(cols_left), tmp, 16               ;   zext tmp, 16                            }

        {   add Q(rows_left), Q(rows_left), tmp     ;   ldw X, sp[STACK_X]                      }
        {   add X, X, Q(cols_left)                  ;   stw Q(rows_left), sp[STACK_Y]           }
        {                                           ;   stw X, sp[STACK_X]                      }
        {                                           ;   ldw tmp, sp[STACK_OUT_COLS]             }
        {                                           ;   bt tmp, .L_out_col_start                }
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
