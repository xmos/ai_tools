
#if defined(__XS3A__)






/*  
void conv2d_shallowin_deepout_relu_asm(
    const int8_t* K, 
    const data16_t* B,
    const int8_t* X, 
    int8_t* Y,
    const int32_t height, 
    const int32_t width,
    const int32_t K_h, 
    const int32_t K_w,
    const int32_t C_out,
    const int16_t* shifts, 
    const int16_t* scales);
*/

#define FUNCTION_NAME conv2d_shallowin_deepout_relu_asm

#define NSTACKWORDS  36

    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 4
.type FUNCTION_NAME,@function
.call FUNCTION_NAME, conv2d_deepin_deepout_relu_asm_patch
.cc_top FUNCTION_NAME.function,FUNCTION_NAME


#define C_IN                        (4)
#define VPU_INT8_ACC_PERIOD_LOG2    (4)
#define K_W_DIM                     (8)

#define NOOOOP          nop
    
#define arg_K               r0
#define arg_B               r1
#define arg_X               r2
#define arg_Y               r3

#define patch_arg_y               r0
#define patch_arg_k               r1
#define patch_arg_cout            r2
#define patch_arg_pad_mask        r3


#define STACK_HEIGHT        (NSTACKWORDS+1)
#define STACK_WIDTH         (NSTACKWORDS+2)
#define STACK_K_H           (NSTACKWORDS+3)
#define STACK_K_W           (NSTACKWORDS+4)
#define STACK_C_OUT         (NSTACKWORDS+5)
#define STACK_SHIFTS        (NSTACKWORDS+6)
#define STACK_SCALES        (NSTACKWORDS+7)

#define STACK_K             (NSTACKWORDS-1 )
#define STACK_X             (NSTACKWORDS-2 )
#define STACK_BIAS_LO       (NSTACKWORDS-3 )
#define STACK_BIAS_HI       (NSTACKWORDS-4 )
#define STACK_K_H_HALF      (NSTACKWORDS-5 )
#define STACK_K_W_HALF      (NSTACKWORDS-6 )
#define STACK_X_ROW_BYTES   (NSTACKWORDS-7 )
#define STACK_K_ROW_BYTES   (NSTACKWORDS-8 )
#define STACK_ROWS_LEFT     (NSTACKWORDS-9 )
#define STACK_COLS_LEFT     (NSTACKWORDS-10)
#define STACK_CUR_ROW       (NSTACKWORDS-11)
#define STACK_CUR_COL       (NSTACKWORDS-12)
#define STACK_Y             (NSTACKWORDS-13)
#define STACK_ROW_PAD_T     (NSTACKWORDS-14)
#define STACK_ROW_PAD_B     (NSTACKWORDS-15)
#define STACK_X_ROW_OFFSET  (NSTACKWORDS-16)
#define STACK_K_START       (NSTACKWORDS-17)
#define STACK_INIT_PADR     (NSTACKWORDS-18)
#define STACK_START_ROW     (NSTACKWORDS-19)
#define STACK_BIAS_OFFSET   (NSTACKWORDS-20)


#define PATCH_X_INITIAL         (1) //0
#define PATCH_ROWS              (2) //1
#define PATCH_BIAS_LO           (3) //1     /////
#define PATCH_BIAS_HI           (4) //2     /////
#define PATCH_SHIFTS            (5) //2     /////
#define PATCH_SCALES            (6) //3     /////
#define PATCH_X_ROW_INCR        (7) //3
#define PATCH_KERNEL_ADVANCE    (8) //4


#define SAVED_REGS_DBL              (5)


FUNCTION_NAME:
    dualentsp NSTACKWORDS
// we need all the registers
    std r4, r5, sp[SAVED_REGS_DBL + 0]
    std r6, r7, sp[SAVED_REGS_DBL + 1]
    std r8, r9, sp[SAVED_REGS_DBL + 2]
    {   NOOOOP                                  ;   stw r10, sp[2*SAVED_REGS_DBL + 6]   }

.L_cdd_init:
    {   NOOOOP                                  ;   stw arg_K, sp[STACK_K]              }
    {   NOOOOP                                  ;   stw arg_X, sp[STACK_X]              }
    {   NOOOOP                                  ;   stw arg_Y, sp[STACK_Y]              }
    {   NOOOOP                                  ;   stw arg_B, sp[STACK_BIAS_LO]        }
    {   NOOOOP                                  ;   ldw r10, sp[STACK_C_OUT]            }
    lda16 arg_B, arg_B[r10]
    {   NOOOOP                                  ;   stw arg_B, sp[STACK_BIAS_HI]        }
    {   ldc r6, K_W_DIM                         ;   ldw r0, sp[STACK_K_H]               }
    {   shr r4, r0, 1                           ;   ldw r1, sp[STACK_K_W]               }
    {   shr r5, r1, 1                           ;   stw r4, sp[STACK_K_H_HALF]          }
    {   sub r6, r6, r1                          ;   stw r5, sp[STACK_K_W_HALF]          }
    {   NOOOOP                                  ;   stw r6, sp[STACK_INIT_PADR]         }
    {   ldc r4, 4                               ;   ldw r5, sp[STACK_WIDTH]             }
    mul r6, r4, r5  //r6 = width * C_in
    {   NOOOOP                                  ;   stw r6, sp[STACK_X_ROW_BYTES]       }
    {   ldc r6, (K_W_DIM * C_IN)                ;   stw r6, sp[PATCH_X_ROW_INCR]        }
    {   shr r10, r10, VPU_INT8_ACC_PERIOD_LOG2  ;   shl r6, r6, VPU_INT8_ACC_PERIOD_LOG2}
    {   shr r4, r4, 5                           ;   stw r6, sp[STACK_K_ROW_BYTES]       }
    {   NOOOOP                                  ;   stw r10, sp[STACK_C_OUT]            }

//Some of the arguments passed to the patch function are passed on the stack, but will be the same
//  with every call. Those arguments can be placed on the stack here.
    {   NOOOOP                                  ;   ldw r10, sp[STACK_SHIFTS]           }
    {   NOOOOP                                  ;   stw r10, sp[PATCH_SHIFTS]           }
    {   NOOOOP                                  ;   ldw r10, sp[STACK_SCALES]           }
    {   NOOOOP                                  ;   stw r10, sp[PATCH_SCALES]           }

//Initialize the row and column counters
    {   NOOOOP                                  ;   ldw r4, sp[STACK_HEIGHT]            }
    {   NOOOOP                                  ;   ldw r10, sp[STACK_WIDTH]            }
    {   NOOOOP                                  ;   stw r4, sp[STACK_ROWS_LEFT]         }
    {   ldc r10, 0                              ;   stw r10, sp[STACK_COLS_LEFT]        }
    {   NOOOOP                                  ;   stw r10, sp[STACK_CUR_ROW]          }
    {   NOOOOP                                  ;   stw r10, sp[STACK_CUR_COL]          }

.L_cdd_row_loop_top:
#define r_start_row         r11
#define r_pad_t             r10
#define r_pad_b             r9
#define r_end_row           r8
    {   ldc r5, 0                               ;   ldw r7, sp[STACK_CUR_ROW]           }
    {   NOOOOP                                  ;   ldw r3, sp[STACK_K_H_HALF]          }
    {   add r_end_row, r7, r3                   ;   ldw r6, sp[STACK_HEIGHT]            }
    {   sub r_start_row, r7, r3                 ;   sub r6, r6, 1                       }
    {   lss r4, r_start_row, r5                 ;   lss r0, r6, r_end_row               }
    {   ldc r_pad_t, 0                          ;   bf r4, .L_lbl_A                     }
        {   ldc r_start_row, 0                      ;   neg r_pad_t, r_start_row            }
    .L_lbl_A:
    {   ldc r_pad_b, 0                          ;   bf r0, .L_lbl_B                     }
        {   mov r_end_row, r6                       ;   sub r_pad_b, r_end_row, r6          }
    .L_lbl_B:
    {   sub r5, r_end_row, r_start_row          ;   ldw r0, sp[STACK_WIDTH]             }   
    {   add r5, r5, 1                           ;   stw r0, sp[STACK_COLS_LEFT]         }   
    {   ldc r5, 0                               ;   stw r5, sp[PATCH_ROWS]              } // patch_rows = (end_row-start_row)+1
    {   ldc r0, (C_IN * K_W_DIM)                ;   stw r5, sp[STACK_CUR_COL]           }

    {   shl r0, r0, VPU_INT8_ACC_PERIOD_LOG2    ;   ldw r7, sp[STACK_X_ROW_BYTES]       }   
    mul r2, r_start_row, r7
    {   add r7, r_pad_t, r_pad_b                ;   stw r2, sp[STACK_X_ROW_OFFSET]      }
    mul r6, r7, r0

    {   NOOOOP                                  ;   stw r6, sp[PATCH_KERNEL_ADVANCE]    }   //NOTE: doesn't change for this whole image row
    {   NOOOOP                                  ;   ldw r6, sp[STACK_K]                 }
    mul r2, r0, r_pad_t
    {   add r2, r6, r2                          ;   stw r11, sp[STACK_START_ROW]        }
    {   NOOOOP                                  ;   stw r2, sp[STACK_K_START]           }  //TODO: can probably just go on the stack.... wait.. why did I say that. It is going on the stack.. maybe I meant into a register? It ultimately needs to find its way into r1

// Calculate row contribution to bias offset
// B_offset_row = K_w * (K_h//2 + pad_b-pad_t) )
    {   sub r2, r_pad_b, r_pad_t                ;   ldw r1, sp[STACK_K_H_HALF]          }
    {   add r2, r2, r1                          ;   ldw r1, sp[STACK_K_W]               }
    mul r2, r2, r1
    {   NOOOOP                                  ;   stw r2, sp[STACK_BIAS_OFFSET]       }

#undef r_start_row      //r11
#undef r_end_row        //r8
#undef r_pad_t          //r10
#undef r_pad_b          //r9

#define r_start_col         r8
#define r_end_col           r7
#define r_pad_l             r6
#define r_pad_r             r5
.L_cdd_col_loop_top:
    {   NOOOOP                                  ;   ldw r2, sp[STACK_CUR_COL]           }
    {   ldc r1, 0                               ;   ldw r4, sp[STACK_K_W_HALF]          }
    {   sub r_start_col, r2, r4                 ;   NOOOOP                              }
    {   lss r0, r_start_col, r1                 ;   ldw r_pad_r, sp[STACK_INIT_PADR]    }
    {   ldc r_pad_l, 0                          ;   bf r0, .L_lbl_C                     }
        {   NOOOOP                                  ;   neg r_pad_l, r_start_col            }
    .L_lbl_C:
    {   add r_end_col, r2, r4                   ;   ldw r4, sp[STACK_WIDTH]             }
    {   sub r4, r4, 1                           ;   NOOOOP                              }
    {   lss r0, r4, r_end_col                   ;   NOOOOP                              }
    {   sub r4, r_end_col, r4                   ;   bf r0, .L_lbl_D                     }
        {   NOOOOP                                  ;   add r_pad_r, r_pad_r, r4            }
    .L_lbl_D:

#define r_pad_mask      r4
    {   ldc r3, 4                               ;   ldc r4, 32                          }
    // {   shl r5, r_pad_r, 2                      ;   shl r_pad_l, r_pad_l, r3            }


//Set the pointers to the bias subtensor before we lose r_pad_r and r_pad_l
//  &B[2 * C_out * (sp[STACK_BIAS_OFFSET]    + (K_w//2 + pad_r-pad_l))]
    {   sub r0, r_pad_r, r_pad_l                ;   ldw r2, sp[STACK_K_W]               }
    {   sub r2, r2, 8                           ;   NOOOOP                              }
    {   add r0, r0, r2                          ;   ldw r2, sp[STACK_K_W_HALF]          }
    {   add r0, r0, r2                          ;   ldw r2, sp[STACK_BIAS_OFFSET]       }
    {   add r0, r0, r2                          ;   ldw r2, sp[STACK_C_OUT]             }
    {   shl r2, r2, (VPU_INT8_ACC_PERIOD_LOG2+2);   ldw r1, sp[STACK_BIAS_LO]           }
    mul r0, r0, r2
    {   add r2, r0, r1                          ;   ldw r1, sp[STACK_BIAS_HI]           }
    {   add r1, r0, r1                          ;   stw r2, sp[PATCH_BIAS_LO]           }
    {   NOOOOP                                  ;   stw r1, sp[PATCH_BIAS_HI]           }
    // {   NOOOOP                                  ;   NOOOOP                              }
    // {   NOOOOP                                  ;   NOOOOP                              }

    mul r5, r_pad_r, r3
    mul r_pad_l, r_pad_l, r3
    {   sub r4, r4, r5                          ;   mkmsk r6, r_pad_l                   }
    {   mkmsk r_pad_mask, r4                    ;   NOOOOP                              }
    xor r_pad_mask, r_pad_mask, r6

    // mul r8, r_start_col, r3
    {   shl r8, r_start_col, 2                  ;   ldw r0, sp[STACK_X_ROW_OFFSET]      }
    {   add r8, r8, r0                          ;   ldw r1, sp[STACK_X]                 }
    {   add r1, r1, r8                          ;   ldw r2, sp[STACK_START_ROW]         }
    {   NOOOOP                                  ;   stw r1, sp[PATCH_X_INITIAL]         }

//Now set r0-r3 with Y, K, cout_groups and pad_mask
    {   NOOOOP                                  ;   ldw patch_arg_y, sp[STACK_Y]        }
    {   NOOOOP                                  ;   ldw patch_arg_k, sp[STACK_K_START]  }
    {   mov patch_arg_pad_mask, r_pad_mask      ;   ldw patch_arg_cout, sp[STACK_C_OUT] }
    
    ldap r11, conv2d_shallowin_deepout_relu_asm_patch
    bla r11

    {   NOOOOP                                  ;   stw r0, sp[STACK_Y]                 }

    {   NOOOOP                                  ;   ldw r11, sp[STACK_CUR_COL]          }
    {   add r11, r11, 1                         ;   ldw r10, sp[STACK_COLS_LEFT]        }
    {   sub r10, r10, 1                         ;   stw r11, sp[STACK_CUR_COL]          }
    {   NOOOOP                                  ;   stw r10, sp[STACK_COLS_LEFT]        }
    bt r10, .L_cdd_col_loop_top
.L_cdd_col_loop_end:
    {   NOOOOP                                  ;   ldw r11, sp[STACK_CUR_ROW]          }
    {   add r11, r11, 1                         ;   ldw r10, sp[STACK_ROWS_LEFT]        }
    {   sub r10, r10, 1                         ;   stw r11, sp[STACK_CUR_ROW]          }
    {   NOOOOP                                  ;   stw r10, sp[STACK_ROWS_LEFT]        }
    bt r10, .L_cdd_row_loop_top
.L_cdd_row_loop_end:

.L_cdd_func_end:
    {   NOOOOP                                  ;   ldw r10, sp[2*SAVED_REGS_DBL + 6]   }
    ldd r8, r9, sp[SAVED_REGS_DBL + 2]
    ldd r6, r7, sp[SAVED_REGS_DBL + 1]
    ldd r4, r5, sp[SAVED_REGS_DBL + 0]
    retsp NSTACKWORDS

    .cc_bottom FUNCTION_NAME.function
    .linkset FUNCTION_NAME.nstackwords, NSTACKWORDS + conv2d_deepin_deepout_relu_asm_patch.nstackwords
    .globl FUNCTION_NAME.nstackwords
    .set FUNCTION_NAME.maxcores,1
    .globl FUNCTION_NAME.maxcores
    .set FUNCTION_NAME.maxtimers,0
    .globl FUNCTION_NAME.maxtimers
    .set FUNCTION_NAME.maxchanends,0
    .globl FUNCTION_NAME.maxchanends
.L_cdd_tmp0:
    .size FUNCTION_NAME, .L_cdd_tmp0-FUNCTION_NAME
    .issue_mode  single

#endif



