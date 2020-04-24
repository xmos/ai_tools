
#if defined(__XS3A__)

/**


void nn_conv2d_hstrip_tail_deep_padded(
        nn_image_t* Y,
        const nn_image_t* X,
        const nn_tensor_t* K,
        const nn_bso_block_t* BSO,
        const unsigned K_h,
        const unsigned K_w,
        const unsigned K_h_stride,
        const channel_count_t C_in,
        const unsigned pad_t,
        const unsigned pad_b,
        const int pad_l_initial,
        const int pad_r_initial,
        const mem_stride_t x_v_stride,
        const mem_stride_t k_cout_stride,
        const mem_stride_t y_h_stride,
        const unsigned out_cols,
        const int8_t* zero_point_vec,
        const channel_count_t C_out_tail)
*/

#define FUNCTION_NAME nn_conv2d_hstrip_tail_deep_padded


#define NSTACKVECS  (4)

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
#define STACK_PAD_T             (NSTACKWORDS+5)
#define STACK_PAD_B             (NSTACKWORDS+6)
#define STACK_PAD_L             (NSTACKWORDS+7)
#define STACK_PAD_R             (NSTACKWORDS+8)
#define STACK_X_V_STRIDE        (NSTACKWORDS+9)
#define STACK_K_COUT_STRIDE     (NSTACKWORDS+10)
#define STACK_Y_H_STRIDE        (NSTACKWORDS+11)
#define STACK_OUT_COLS          (NSTACKWORDS+12)
#define STACK_VEC_ZERO_POINT    (NSTACKWORDS+13)
#define STACK_C_OUT_TAIL        (NSTACKWORDS+14)


#define STACK_VEC_ADJ_B_HI      (NSTACKWORDS-8)
#define STACK_VEC_ADJ_B_LO      (NSTACKWORDS-16)
#define STACK_VEC_TMP2          (NSTACKWORDS-24)
#define STACK_VEC_TMP1          (NSTACKWORDS-32)

#define STACK_C_IN_GROUPS       STACK_C_IN
#define STACK_C_IN_TAIL         7
#define STACK_PATCH_ROWS        8
#define STACK_Y                 9
#define STACK_X                 10
#define STACK_K                 11
#define STACK_BSO               12
#define STACK_CENTER_COLS       STACK_K_W
#define STACK_WIN_H_STRIDE      STACK_K_H
#define STACK_WRITE_MASK        13


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
    {   add r11, r3, r0                         ;   vldd r3[0]                              }
    {   add r11, r11, r0                        ;   vldr r11[0]                             }
    {                                           ;   stw r11, sp[STACK_BSO]                  }
    
    {                                           ;   ldw r0, sp[STACK_K_H]                   }
    {                                           ;   ldw r1, sp[STACK_PAD_T]                 }
    {   sub r0, r0, r1                          ;   ldw r1, sp[STACK_PAD_B]                 }
    {   sub r0, r0, r1                          ;                                           }
    {                                           ;   stw r0, sp[STACK_PATCH_ROWS]            }
    {                                           ;   ldw r0, sp[STACK_K_h_stride]            }
    {                                           ;   ldw r1, sp[STACK_C_IN]                  }
    mul r0, r0, r1
    {   shr r0, r1, 5                           ;   stw r0, sp[STACK_WIN_H_STRIDE]          }
    {   zext r1, 5                              ;   stw r0, sp[STACK_C_IN_GROUPS]           }
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

#define rows_left       r0
#define cols_left       r1
#define cig_left        r2
#define K               r3
#define X               r4
#define tmp             r5
#define c_in_tail       r6
#define c_out_tail      r7
#define _32             r8
#define k_cout_stride   r9
#define vec_zp          r10



    {                                           ;   ldw X, sp[STACK_X]                      }
    {                                           ;   ldw K, sp[STACK_K]                      }
    {   ldc _32, 32                             ;   ldw k_cout_stride, sp[STACK_K_COUT_STRIDE]}
    {                                           ;   ldw c_in_tail, sp[STACK_C_IN_TAIL]      }
    {                                           ;   ldw c_out_tail, sp[STACK_C_OUT_TAIL]    }

    //Check if any top padding
    {                                           ;   ldw rows_left, sp[STACK_PAD_T]          }
    {                                           ;   bf rows_left, .L_pad_t_end              }
    .L_pad_t_start:
        {                                           ;   ldw vec_zp, sp[STACK_VEC_ZERO_POINT]    }

        .L_pad_t_row_start:
            {   sub rows_left, rows_left, 1             ;   ldw cols_left, sp[STACK_K_W]             }
            .L_pad_t_col_start:
                {   sub cols_left, cols_left, 1             ;   vldc vec_zp[0]                          }
                {   ldaw tmp, sp[STACK_VEC_TMP2]            ;   ldw cig_left, sp[STACK_C_IN_GROUPS]     }
                {   sub r11, tmp, c_out_tail                ;   bf cig_left, .L_pad_t_cig_end           }
                .L_pad_t_cig_start:
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

                    {   shl c_out_tail, c_out_tail, 1           ;   add X, X, _32                           }
                    {   ldaw tmp, sp[STACK_VEC_TMP2]            ;   bt cig_left, .L_pad_t_cig_start         }
                .L_pad_t_cig_end:
                {                                           ;   bf c_in_tail, .L_pad_t_tail_end         }
                .L_pad_t_tail_start:
            
                //reset accumulator positions
                    {   shr c_out_tail, c_out_tail, 1           ;   vstr tmp[0]                             }
                    {   sub c_out_tail, c_out_tail, 4           ;   vldr r11[0]                             }
                    {   sub tmp, c_in_tail, _32                 ;   vstd tmp[0]                             }
                    {   ldaw r11, sp[STACK_VEC_TMP2]            ;   vldd r11[0]                             }

                //mask C_in_tail
                    {   add r11, r11, tmp                       ;   vstc r11[0]                             }
                    {   add X, X, c_in_tail                     ;   vldc r11[0]                             }
                    {   add tmp, K, tmp                         ;                                           }

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

                .L_pad_t_tail_end:
                {   shl c_out_tail, c_out_tail, 1           ;   bt cols_left, .L_pad_t_col_start        }    
            .L_pad_t_col_end:
            {                                           ;   ldw tmp, sp[STACK_X_V_STRIDE]           }
            {   add X, X, tmp                           ;   bt rows_left, .L_pad_t_row_start        }
        .L_pad_t_row_end:
        {                                           ;   stw X, sp[STACK_X]                      }
        {                                           ;   stw K, sp[STACK_K]                      }
    .L_pad_t_end:


    // Check if any bottom padding

    {                                           ;   ldw rows_left, sp[STACK_PAD_B]          }
    {   ldaw Q(X), sp[STACK_VEC_ADJ_B_HI]       ;   bf rows_left, .L_pad_b_end              }
    
    //Move K past the non-padding rows
    {                                           ;   ldw Q(X), sp[STACK_C_IN_GROUPS]         }
    { shl Q(X), Q(X), 5                         ;   ldw tmp, sp[STACK_K_W]                  }
    { add Q(X), Q(X), c_in_tail                 ;                                           }
    mul Q(X), Q(X), tmp
    {   ldc Q(_32), 0                           ;   ldw tmp, sp[STACK_PATCH_ROWS]           }
    maccu Q(_32), K, Q(X), tmp

    .L_pad_b_start:
        {   ldc _32, 32                             ;   ldw vec_zp, sp[STACK_VEC_ZERO_POINT]    }

        .L_pad_b_row_start:
            {   sub rows_left, rows_left, 1             ;   ldw cols_left, sp[STACK_K_W]             }
            .L_pad_b_col_start:
                {   sub cols_left, cols_left, 1             ;   vldc vec_zp[0]                          }
                {   ldaw tmp, sp[STACK_VEC_TMP2]            ;   ldw cig_left, sp[STACK_C_IN_GROUPS]     }
                {   sub r11, tmp, c_out_tail                ;   bf cig_left, .L_pad_b_cig_end           }
                .L_pad_b_cig_start:
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
                    {   ldaw tmp, sp[STACK_VEC_TMP2]            ;   bt cig_left, .L_pad_b_cig_start         }
                .L_pad_b_cig_end:
                {                                           ;   bf c_in_tail, .L_pad_b_tail_end         }
                .L_pad_b_tail_start:
                //reset accumulator positions
                    {   shr c_out_tail, c_out_tail, 1           ;   vstr tmp[0]                             }
                    {   sub c_out_tail, c_out_tail, 4           ;   vldr r11[0]                             }
                    {   sub tmp, c_in_tail, _32                 ;   vstd tmp[0]                             }
                    {   ldaw r11, sp[STACK_VEC_TMP2]            ;   vldd r11[0]                             }

                //mask C_in_tail
                    {   add r11, r11, tmp                       ;   vstc r11[0]                             }
                    {   add X, X, c_in_tail                     ;   vldc r11[0]                             }
                    {   add tmp, K, tmp                         ;                                           }

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

                .L_pad_b_tail_end:
                {   shl c_out_tail, c_out_tail, 1           ;   bt cols_left, .L_pad_b_col_start        }    
            .L_pad_b_col_end:
            {   ldaw Q(X), sp[STACK_VEC_ADJ_B_HI]       ;   bt rows_left, .L_pad_b_row_start        }
        .L_pad_b_row_end:
    .L_pad_b_end:

    //Save adjusted biases on the stack.
    {   ldaw Q(X), sp[STACK_VEC_ADJ_B_LO]       ;   vstd Q(X)[0]                            }
    {                                           ;   vstr Q(X)[0]                            }




#undef rows_left  
#undef cols_left  
#undef cig_left   
#undef K          
#undef X          
#undef tmp      
#undef c_in_tail  
#undef c_out_tail 
#undef vec_z_tail 
#undef k_cout_stride 
#undef vec_zp     
#undef _32        

    // Calculate the number of center (not left or right padding) columns
    {   ldc r9, 0                               ;   ldw r11, sp[STACK_PAD_L]                }
    {   lss r1, r11, r9                         ;   ldw r10, sp[STACK_PAD_R]                }
    {   lss r2, r10, r9                         ;   ldw r0, sp[STACK_K_W]                   } 
    {                                           ;   bt r1, .L_neg_pad_l                     }
    {   sub r0, r0, r11                         ;   bt r2, .L_neg_pad_r                     }
    .L_neg_pad_l:
    {                                           ;   bt r2, .L_neg_pad_r                     }
    {   sub r0, r0, r10                         ;                                           }
    .L_neg_pad_r:
    {                                           ;   stw r0, sp[STACK_CENTER_COLS]           }



#define X               r0
#define K               r1
#define rows_left       r2
#define cols_left       r3
#define cig_left        r4
#define c_in_tail       r5
#define vec_zp          r6
#define k_cout_stride   r7
#define _32             r8
#define tmp             r9
#define c_out_tail      r10

    {                                           ;   ldw tmp, sp[STACK_OUT_COLS]             }
    {                                           ;   ldw c_in_tail, sp[STACK_C_IN_TAIL]      }
    {                                           ;   ldw X, sp[STACK_X]                      }
    {                                           ;   ldw k_cout_stride, sp[STACK_K_COUT_STRIDE]}
    {                                           ;   ldw vec_zp, sp[STACK_VEC_ZERO_POINT]    }
    {                                           ;   ldw c_out_tail, sp[STACK_C_OUT_TAIL]    }

    .L_out_col_start:
        {   ldaw r11, sp[STACK_VEC_ADJ_B_HI]        ;   ldw rows_left, sp[STACK_PATCH_ROWS]     }
        {   ldaw r11, sp[STACK_VEC_ADJ_B_LO]        ;   vldd r11[0]                             }
        {   sub tmp, tmp, 1                         ;   vldr r11[0]                             }
        {                                           ;   stw tmp, sp[STACK_OUT_COLS]             }
        {   ldc tmp, 0                              ;   ldw K, sp[STACK_K]                      }

        .L_patch_row_start: //tmp is 0 on first pass, x_v_stride on subsequent iterations
            {   add X, X, tmp                           ;   ldw cols_left, sp[STACK_PAD_L]          }
            {   shr Q(cig_left), cols_left, 24          ;   bf cols_left, .L_pad_l_end              }   //If cols_left == 0, skip pad_l
            {                                           ;   bt Q(cig_left), .L_pad_l_end            }   //if cols_left < 0, skip  pad_;
            .L_pad_l_start:
                {   sub cols_left, cols_left, 1             ;   vldc vec_zp[0]                          }
                {   ldaw tmp, sp[STACK_VEC_TMP2]            ;   ldw cig_left, sp[STACK_C_IN_GROUPS]     }
                {   sub r11, tmp, c_out_tail                ;   bf cig_left, .L_pad_l_cig_end           }
                .L_pad_l_cig_start:
                    
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

                    {   shl c_out_tail, c_out_tail, 1           ;   add X, X, _32                           }
                    {   ldaw tmp, sp[STACK_VEC_TMP2]            ;   bt cig_left, .L_pad_l_cig_start         }

                .L_pad_l_cig_end:
                {                                           ;   bf c_in_tail, .L_pad_l_tail_end         }
                .L_pad_l_tail_start:
            
                //reset accumulator positions
                    {   shr c_out_tail, c_out_tail, 1           ;   vstr tmp[0]                             }
                    {   sub c_out_tail, c_out_tail, 4           ;   vldr r11[0]                             }
                    {   sub tmp, c_in_tail, _32                 ;   vstd tmp[0]                             }
                    {   ldaw r11, sp[STACK_VEC_TMP2]            ;   vldd r11[0]                             }

                //mask C_in_tail
                    {   add r11, r11, tmp                       ;   vstc r11[0]                             }
                    {   add X, X, c_in_tail                     ;   vldc r11[0]                             }
                    {   add tmp, K, tmp                         ;                                           }

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

                .L_pad_l_tail_end:
                {   shl c_out_tail, c_out_tail, 1           ;   bt cols_left, .L_pad_l_start            }
            .L_pad_l_end:
            {                                           ;   ldw cols_left, sp[STACK_CENTER_COLS]    }
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
            {                                           ;   ldw cols_left, sp[STACK_PAD_R]          }
            {   shr tmp, cols_left, 24                  ;   bf cols_left, .L_pad_r_end              }
            {                                           ;   bt tmp, .L_pad_r_end                    }
            .L_pad_r_start:
                {   sub cols_left, cols_left, 1             ;   vldc vec_zp[0]                          }
                {   ldaw tmp, sp[STACK_VEC_TMP2]            ;   ldw cig_left, sp[STACK_C_IN_GROUPS]     }
                {   sub r11, tmp, c_out_tail                ;   bf cig_left, .L_pad_r_cig_end           }
                .L_pad_r_cig_start:
                    
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

                    {   shl c_out_tail, c_out_tail, 1           ;   add X, X, _32                           }
                    {   ldaw tmp, sp[STACK_VEC_TMP2]            ;   bt cig_left, .L_pad_r_cig_start         }

                .L_pad_r_cig_end:
                {                                           ;   bf c_in_tail, .L_pad_r_tail_end         }
                .L_pad_r_tail_start:
            
                //reset accumulator positions
                    {   shr c_out_tail, c_out_tail, 1           ;   vstr tmp[0]                             }
                    {   sub c_out_tail, c_out_tail, 4           ;   vldr r11[0]                             }
                    {   sub tmp, c_in_tail, _32                 ;   vstd tmp[0]                             }
                    {   ldaw r11, sp[STACK_VEC_TMP2]            ;   vldd r11[0]                             }

                //mask C_in_tail
                    {   add r11, r11, tmp                       ;   vstc r11[0]                             }
                    {   add X, X, c_in_tail                     ;   vldc r11[0]                             }
                    {   add tmp, K, tmp                         ;                                           }

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

                .L_pad_r_tail_end:
                {   shl c_out_tail, c_out_tail, 1           ;   bt cols_left, .L_pad_r_start            }
            .L_pad_r_end:
            //                                          // Added to X back up at the start of the row
            {   sub rows_left, rows_left, 1             ;   ldw tmp, sp[STACK_X_V_STRIDE]           }
            bt rows_left, .L_patch_row_start
        .L_patch_row_end:

        {   shl r11, _32, 3                         ;   ldw tmp, sp[STACK_BSO]                  }
        {   ldaw r11, sp[STACK_VEC_TMP2]            ;   vsetc r11                               }
        {   add tmp, tmp, _32                       ;   vlsat tmp[0]                            }
        {                                           ;   vstr r11[0]                             }
        {   add tmp, tmp, _32                       ;   vldc tmp[0]                             }
        {                                           ;   vclrdr                                  }
        {   shl r11, _32, 4                         ;   vlmacc r11[0]                           }
        {   add tmp, tmp, _32                       ;   vldc tmp[0]                             }
        {   add tmp, tmp, _32                       ;   vlmacc tmp[0]                           }
        {                                           ;   vsetc r11                               }
        {                                           ;   vlsat tmp[0]                            }
        {   mkmsk tmp, 16                           ;   ldw Q(rows_left), sp[STACK_Y]           }
        {                                           ;   ldw tmp, sp[STACK_WRITE_MASK]           }
        vstrpv Q(rows_left)[0], tmp
        {                                           ;   ldw tmp, sp[STACK_Y_H_STRIDE]           }
        {   add Q(rows_left), Q(rows_left), tmp     ;   ldw tmp, sp[STACK_WIN_H_STRIDE]         }
        {                                           ;   ldw X, sp[STACK_X]                      }
        {   add X, X, tmp                           ;   stw Q(rows_left), sp[STACK_Y]           }
        {                                           ;   stw X, sp[STACK_X]                      }

#define l_padding       cols_left
#define r_padding       cig_left
#define center_cols     _32
#define k_h_stride      rows_left

        {   ldc r11, 0                              ;   ldw l_padding, sp[STACK_PAD_L]          }
        {                                           ;   ldw center_cols, sp[STACK_CENTER_COLS]  }
        {   lss tmp, r11, l_padding                 ;   ldw k_h_stride, sp[STACK_K_h_stride]    }
        {   lss tmp, l_padding, k_h_stride          ;   bf tmp, .L_werwer                       }
        {   add tmp, tmp, 1                         ;   bru tmp                                 }
        {   add center_cols, center_cols, k_h_stride;   bru tmp                                 }
        {   add center_cols, center_cols, l_padding ;                                           }
        .L_werwer:
        {   sub l_padding, l_padding, k_h_stride    ;   ldw r_padding, sp[STACK_PAD_R]          }
        {   add r_padding, r_padding, k_h_stride    ;   stw l_padding, sp[STACK_PAD_L]          }
        {   lss tmp, r11, r_padding                 ;   stw r_padding, sp[STACK_PAD_R]          }
        {   lss tmp, r_padding, k_h_stride          ;   bf tmp, .L_rewrew                       }
        {   add tmp, tmp, 1                         ;   bru tmp                                 }
        {   sub center_cols, center_cols, k_h_stride;   bru tmp                                 }
        {   sub center_cols, center_cols, r_padding ;                                           }
        .L_rewrew:
        {                                           ;   stw center_cols, sp[STACK_CENTER_COLS]  }
        {   ldc _32, 32                             ;   ldw tmp, sp[STACK_OUT_COLS]             }
        bt tmp, .L_out_col_start
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
