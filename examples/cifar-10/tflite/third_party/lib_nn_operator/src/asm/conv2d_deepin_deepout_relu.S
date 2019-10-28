
#if defined(__XS3A__)






/*  
void conv2d_deepin_deepout_relu_asm(
    const int8_t* K, 
    const data16_t* B,
    const int8_t* X, 
    int8_t* Y,
    const int32_t height, 
    const int32_t width,
    const int32_t K_h, 
    const int32_t K_w,
    const int32_t C_out, 
    const int32_t C_in,
    const int16_t* shifts, 
    const int16_t* scales);
*/

#define FUNCTION_NAME conv2d_deepin_deepout_relu_asm

#define NSTACKWORDS  40

    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 4
.type FUNCTION_NAME,@function
.call FUNCTION_NAME, conv2d_deepin_deepout_relu_asm_patch
.cc_top FUNCTION_NAME.function,FUNCTION_NAME


#define NOOOOP          nop
    
#define arg_K               r0
#define arg_B               r1
#define arg_X               r2
#define arg_Y               r3


#define STACK_HEIGHT        (NSTACKWORDS+1)
#define STACK_WIDTH         (NSTACKWORDS+2)
#define STACK_K_H           (NSTACKWORDS+3)
#define STACK_K_W           (NSTACKWORDS+4)
#define STACK_C_OUT         (NSTACKWORDS+5)
#define STACK_C_IN          (NSTACKWORDS+6)
#define STACK_SHIFTS        (NSTACKWORDS+7)
#define STACK_SCALES        (NSTACKWORDS+8)

#define STACK_K             (NSTACKWORDS-1)
#define STACK_X             (NSTACKWORDS-2)
#define STACK_BIASES_LO     (NSTACKWORDS-3)
#define STACK_BIASES_HI     (NSTACKWORDS-4)
#define STACK_K_H_HALF      (NSTACKWORDS-5)
#define STACK_K_W_HALF      (NSTACKWORDS-6)
#define STACK_X_PXL_BYTES   (NSTACKWORDS-7)
#define STACK_X_ROW_BYTES   (NSTACKWORDS-8)
#define STACK_K_PXL_BYTES   (NSTACKWORDS-9)
#define STACK_K_ROW_BYTES   (NSTACKWORDS-10)
#define STACK_ROWS_LEFT     (NSTACKWORDS-11)
#define STACK_COLS_LEFT     (NSTACKWORDS-12)
#define STACK_CUR_ROW       (NSTACKWORDS-13)
#define STACK_CUR_COL       (NSTACKWORDS-14)
#define STACK_Y             (NSTACKWORDS-15)
#define STACK_ROW_PAD_T     (NSTACKWORDS-16)
#define STACK_ROW_PAD_B     (NSTACKWORDS-17)
#define STACK_ROW_START     (NSTACKWORDS-18)
#define STACK_BIAS_OFFSET   (NSTACKWORDS-19)


#define STACK_DONT_USE              (0)
#define PATCH_PATCH_ROW_INCR        (1)

#define PATCH_KERNEL_ROW_INCR       (2)
#define PATCH_X_INITIAL             (3)

#define PATCH_ROWS                  (4)
#define PATCH_ROW_MACCS             (5)

#define PATCH_CHAN_OUTS             (6)
#define PATCH_KERNEL_ADVANCE        (7)

#define PATCH_SHIFTS                (8)
#define PATCH_SCALES                (9)

#define SAVED_REGS_DBL              (5)

#define VPU_INT8_ACC_PERIOD_LOG2    (4)

FUNCTION_NAME:
    dualentsp NSTACKWORDS
// we need all the registers
    std r4, r5, sp[SAVED_REGS_DBL + 0]
    std r6, r7, sp[SAVED_REGS_DBL + 1]
    std r8, r9, sp[SAVED_REGS_DBL + 2]
    {   NOOOOP                                  ;   stw r10, sp[2*SAVED_REGS_DBL + 6]   }

.L_cdd_init:
//Store initial K, X and Y on the stack
    {   NOOOOP                                  ;   stw arg_K, sp[STACK_K]              }
    {   NOOOOP                                  ;   stw arg_X, sp[STACK_X]              }
    {   NOOOOP                                  ;   stw arg_Y, sp[STACK_Y]              }
//Store B_hi and B_lo on the stack
    {   NOOOOP                                  ;   stw arg_B, sp[STACK_BIASES_LO]      }
    {   NOOOOP                                  ;   ldw r10, sp[STACK_C_OUT]            }
    lda16 arg_B, arg_B[r10]
    {   NOOOOP                                  ;   stw arg_B, sp[STACK_BIASES_HI]      }
//Store half-kernel vals on the stack
    {   NOOOOP                                  ;   ldw r0, sp[STACK_K_H]               }
    {   shr r4, r0, 1                           ;   ldw r1, sp[STACK_K_W]               }
    {   shr r5, r1, 1                           ;   stw r4, sp[STACK_K_H_HALF]          }
    {   NOOOOP                                  ;   stw r5, sp[STACK_K_W_HALF]          }
//Store byte counts on stack
    {   NOOOOP                                  ;   ldw r4, sp[STACK_C_IN]              }
    {   NOOOOP                                  ;   ldw r5, sp[STACK_WIDTH]             }
    {   NOOOOP                                  ;   stw r4, sp[STACK_X_PXL_BYTES]       }
    mul r6, r4, r5  //r6 = width * C_in
    {   shl r6, r4, VPU_INT8_ACC_PERIOD_LOG2    ;   stw r6, sp[STACK_X_ROW_BYTES]       }
    {   shr r10, r10, 4                         ;   stw r6, sp[STACK_K_PXL_BYTES]       }
    mul r6, r6, r1  //r6 = K_pxl_bytes * K_w
    {   shr r4, r4, 5                           ;   stw r6, sp[STACK_K_ROW_BYTES]       }
//Store the channel-in and channel-out group counts
    {   NOOOOP                                  ;   stw r4, sp[STACK_C_IN]              }
    {   NOOOOP                                  ;   stw r10, sp[STACK_C_OUT]            }
//Initialize the row and column counters
    {   NOOOOP                                  ;   ldw r4, sp[STACK_HEIGHT]            }
    {   NOOOOP                                  ;   ldw r10, sp[STACK_WIDTH]            }
    {   NOOOOP                                  ;   stw r4, sp[STACK_ROWS_LEFT]         }
    {   ldc r10, 0                              ;   stw r10, sp[STACK_COLS_LEFT]        }
    {   NOOOOP                                  ;   stw r10, sp[STACK_CUR_ROW]          }
    {   NOOOOP                                  ;   stw r10, sp[STACK_CUR_COL]          }

//Some of the arguments passed to the patch function are passed on the stack, but will be the same
//  with every call. Those arguments can be placed on the stack here.
    {   NOOOOP                                  ;   ldw r10, sp[STACK_SHIFTS]           }
    {   NOOOOP                                  ;   stw r10, sp[PATCH_SHIFTS]           }
    {   NOOOOP                                  ;   ldw r10, sp[STACK_SCALES]           }
    {   NOOOOP                                  ;   stw r10, sp[PATCH_SCALES]           }
    {   NOOOOP                                  ;   ldw r10, sp[STACK_C_OUT]            }
    {   NOOOOP                                  ;   stw r10, sp[PATCH_CHAN_OUTS]        }


.L_cdd_row_loop_top:
#define r_start_row         r11
#define r_pad_t             r10
#define r_pad_b             r9
#define r_end_row           r8
    {   ldc r5, 0                               ;   ldw r7, sp[STACK_CUR_ROW]           }
    {   NOOOOP                                  ;   ldw r3, sp[STACK_K_H_HALF]          }
    {   add r_end_row, r7, r3                   ;   ldw r6, sp[STACK_HEIGHT]            }
    {   sub r_start_row, r7, r3                 ;   NOOOOP                              }
    {   sub r6, r6, 1                           ;   NOOOOP                              }
    {   lss r4, r_start_row, r5                 ;   NOOOOP                              }
    {   ldc r_pad_t, 0                          ;   bf r4, .L_lbl_A                     }
        {   ldc r_start_row, 0                      ;   neg r_pad_t, r_start_row            }
    .L_lbl_A:
    {   lss r4, r6, r_end_row                   ;   stw r_start_row, sp[STACK_ROW_START]}
    {   ldc r_pad_b, 0                          ;   bf r4, .L_lbl_B                     }
        {   mov r_end_row, r6                       ;   sub r_pad_b, r_end_row, r6          }
    .L_lbl_B:
    {   sub r5, r_end_row, r_start_row          ;   stw r_pad_t, sp[STACK_ROW_PAD_T]    }
    {   add r5, r5, 1                           ;   stw r_pad_b, sp[STACK_ROW_PAD_B]    }
    {   NOOOOP                                  ;   stw r5, sp[PATCH_ROWS]              }

    {   ldc r5, 0                               ;   ldw r0, sp[STACK_WIDTH]             }
    {   NOOOOP                                  ;   stw r5, sp[STACK_CUR_COL]           }
    {   NOOOOP                                  ;   stw r0, sp[STACK_COLS_LEFT]         }
#undef  r_end_row         //r8
#define r_start_col         r8
#define r_end_col           r7
#define r_pad_l             r6
#define r_pad_r             r5
.L_cdd_col_loop_top:
    {   NOOOOP                                  ;   ldw r2, sp[STACK_CUR_COL]           }
    {   ldc r1, 0                               ;   ldw r4, sp[STACK_K_W_HALF]          }
    {   sub r_start_col, r2, r4                 ;   ldw r_start_row,sp[STACK_ROW_START] }
    {   lss r0, r_start_col, r1                 ;   ldw r_pad_t, sp[STACK_ROW_PAD_T]    }
    {   ldc r_pad_l, 0                          ;   bf r0, .L_lbl_C                     }
        {   ldc r_start_col, 0                      ;   neg r_pad_l, r_start_col            }
    .L_lbl_C:
    {   add r_end_col, r2, r4                   ;   ldw r4, sp[STACK_WIDTH]             }
    {   sub r4, r4, 1                           ;   ldw r_pad_b, sp[STACK_ROW_PAD_B]    }
    {   lss r0, r4, r_end_col                   ;   NOOOOP                              }
    {   ldc r_pad_r, 0                          ;   bf r0, .L_lbl_D                     }
        {   mov r_end_col, r4                       ;   sub r_pad_r, r_end_col, r4          }
    .L_lbl_D:

//We now have the start/end row/col and all the pads

//compute bias offset  
// &B[2 * C_out * (K_w * (K_h//2 + pad_b-pad_t) + (K_w//2 + pad_r-pad_l))]  
    {   sub r0, r9, r10                         ;   ldw r4, sp[STACK_K_H_HALF]          }
    {   add r0, r0, r4                          ;   ldw r4, sp[STACK_K_W]               }
    mul r0, r0, r4
    {   sub r1, r5, r6                          ;   ldw r4, sp[STACK_K_W_HALF]          }
    {   add r1, r1, r4                          ;   ldw r4, sp[STACK_C_OUT]             }
    {   add r0, r0, r1                          ;   shl r4, r4, 6                       } //shl 6 because B is int16_t*, not int8_t*
    mul r0, r0, r4
    {   NOOOOP                                  ;   stw r0, sp[STACK_BIAS_OFFSET]       }
    // {   NOOOOP                                  ;   NOOOOP                              }
    // {   NOOOOP                                  ;   NOOOOP                              }
    // {   NOOOOP                                  ;   NOOOOP                              }

// const int8_t* patch_x = X + (patch_start_row * X_row_bytes + patch_start_col * X_pxl_bytes);
    
#define patch_x         r11
    {   NOOOOP                                  ;   ldw r4, sp[STACK_X_ROW_BYTES]       }
    mul patch_x, r_start_row, r4
#undef  r_start_row   //r11
    {   NOOOOP                                  ;   ldw r4, sp[STACK_X]                 }
    {   add patch_x, patch_x, r4                ;   ldw r4, sp[STACK_X_PXL_BYTES]       }
    mul r3, r4, r_start_col
    {   add patch_x, patch_x, r3                ;   NOOOOP                              }
    {   NOOOOP                                  ;   stw patch_x, sp[PATCH_X_INITIAL]    }
#undef patch_x

//  const unsigned patch_cols = patch_end_col - patch_start_col + 1;

#define patch_cols      r11
    {   sub patch_cols, r_end_col, r_start_col  ;   NOOOOP                              }
    {   add patch_cols, patch_cols, 1           ;   NOOOOP                              }

#undef  r_start_col   //r8
#undef  r_end_col     //r7

// const unsigned patch_row_incr =  X_pxl_bytes * (width - patch_cols);
// const unsigned kernel_row_incr = K_pxl_bytes * (K_w - patch_cols);
// const unsigned patch_row_maccs = (C_in >> 5) * patch_cols;
    {   NOOOOP                                  ;   ldw r7, sp[STACK_WIDTH]             }
    {   sub r7, r7, patch_cols                  ;   ldw r4, sp[STACK_K_W]               }
    {   sub r4, r4, patch_cols                  ;   ldw r8, sp[STACK_X_PXL_BYTES]       }
    mul r7, r7, r8
    {   NOOOOP                                  ;   stw r7, sp[PATCH_PATCH_ROW_INCR]    } 
    {   NOOOOP                                  ;   ldw r8, sp[STACK_K_PXL_BYTES]       }
    mul r4, r4, r8
    {   NOOOOP                                  ;   stw r4, sp[PATCH_KERNEL_ROW_INCR]   }
    {   NOOOOP                                  ;   ldw r7, sp[STACK_C_IN]              }
    mul r7, r7, patch_cols
    {   NOOOOP                                  ;   stw r7, sp[PATCH_ROW_MACCS]         }
#undef  patch_cols   //r11

// const unsigned K_offset = K_row_bytes * pad_t + K_pxl_bytes * pad_l;
    {   NOOOOP                                  ;   ldw r11, sp[STACK_K_ROW_BYTES]      }
    {   NOOOOP                                  ;   ldw r7, sp[STACK_K_PXL_BYTES]       }
    {   NOOOOP                                  ;   ldw r4, sp[PATCH_KERNEL_ROW_INCR]   }
    mul r10, r_pad_t, r11
#undef  pad_t     //r10
    mul r6, r_pad_l, r7
#undef  pad_l     //r6
    {   add r0, r10, r6                         ;   ldw r1, sp[STACK_K]                 }

// const unsigned kernel_advance = K_row_bytes * pad_b + K_pxl_bytes * pad_r
//                               + K_offset
//                               - kernel_row_incr;
    mul r9, r_pad_b, r11
#undef  pad_b     //r9
    mul r5, r_pad_r, r7 
#undef  pad_r     //r5
    {   add r2, r9, r5                          ;   NOOOOP                              }
    {   add r2, r2, r0                          ;   ldw r9, sp[PATCH_KERNEL_ROW_INCR]   }
    {   sub r2, r2, r9                          ;   NOOOOP                              }
    {   NOOOOP                                  ;   stw r2, sp[PATCH_KERNEL_ADVANCE]    }

// const int8_t* patch_k = K + K_offset;
    {   add r1, r1, r0                          ;   NOOOOP                              }

//Now set r0-r3 with Y, K, BIAS_LO and BIAS_HI
    {   NOOOOP                                  ;   ldw r0, sp[STACK_Y]                 }
    {   NOOOOP                                  ;   ldw r2, sp[STACK_BIASES_LO]         }
    {   NOOOOP                                  ;   ldw r3, sp[STACK_BIASES_HI]         }
    
    {   NOOOOP                                  ;   ldw r11, sp[STACK_BIAS_OFFSET]      }
    {   add r2, r2, r11                         ;   NOOOOP                              }
    {   add r3, r3, r11                         ;   NOOOOP                              }

//And call conv2d_deepin_deepout_relu_asm_patch()


    ldap r11, conv2d_deepin_deepout_relu_asm_patch
    bla r11

    {   NOOOOP                                  ;   stw r0, sp[STACK_Y]                 }
    {   NOOOOP                                  ;   NOOOOP                              }
    {   NOOOOP                                  ;   NOOOOP                              }





    {   NOOOOP                                  ;   ldw r11, sp[STACK_CUR_COL]          }
    {   add r11, r11, 1                         ;   ldw r10, sp[STACK_COLS_LEFT]        }
    {   sub r10, r10, 1                         ;   stw r11, sp[STACK_CUR_COL]          }
    {   NOOOOP                                  ;   stw r10, sp[STACK_COLS_LEFT]        }
    bt r10, .L_cdd_col_loop_top
    {   NOOOOP                                  ;   NOOOOP                              }
    {   NOOOOP                                  ;   NOOOOP                              }
    {   NOOOOP                                  ;   NOOOOP                              }
.L_cdd_col_loop_end:
    {   NOOOOP                                  ;   ldw r11, sp[STACK_CUR_ROW]          }
    {   add r11, r11, 1                         ;   ldw r10, sp[STACK_ROWS_LEFT]        }
    {   sub r10, r10, 1                         ;   stw r11, sp[STACK_CUR_ROW]          }
    {   NOOOOP                                  ;   stw r10, sp[STACK_ROWS_LEFT]        }
    bt r10, .L_cdd_row_loop_top

    {   NOOOOP                                  ;   NOOOOP                              }
    {   NOOOOP                                  ;   NOOOOP                              }
    {   NOOOOP                                  ;   NOOOOP                              }
    {   NOOOOP                                  ;   NOOOOP                              }
.L_cdd_row_loop_end:
    {   NOOOOP                                  ;   NOOOOP                              }
    {   NOOOOP                                  ;   NOOOOP                              }

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



