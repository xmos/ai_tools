
#if defined(__XS3A__)


/*  
void avgpool2d_asm(
    int8_t* Y,
    const int8_t* X, 
    const nn_pool_params_t* params);
*/

#define FUNCTION_NAME avgpool2d_asm

#define NSTACKWORDS  26
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 4
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

#define NOOOOP          nop

#define arg_Y           r0
#define arg_X           r1
#define arg_params      r2
    

#define Y                       r0
#define X                       r1
#define params                  r2
#define win_col_incr_x          r3
#define win_row_incr_x          r4
#define chan_incr_x             r5

#define STACK_WCOL_TAIL         (9)
#define STACK_WCOL_BODY         (10)
#define STACK_SPEC_2X2          (11)
#define STACK_ROWS_LEFT         (12)
#define STACK_SHIFT_VEC         (NSTACKWORDS-8)     //Each only needs 16 bytes
#define STACK_SCALE_VEC         (NSTACKWORDS-12)     

#define PARAMS_OUT_ROWS         (0)
#define PARAMS_OUT_COLS         (1)
#define PARAMS_OUT_CHANS        (2)
#define PARAMS_W_H              (3)
#define PARAMS_W_W              (4)
#define PARAMS_HSTRIDE_INCR_X   (5)
#define PARAMS_VSTRIDE_INCR_X   (6)
#define PARAMS_VSTRIDE_INCR_Y   (7)
#define PARAMS_VSTRIDE_DBL      (3)
#define PARAMS_SHIFT            (8)
#define PARAMS_SCALE            (9)
#define PARAMS_CHAN_INCR_X      (10)
#define PARAMS_WIN_COL_INCR_X   (11)
#define PARAMS_WIN_ROW_INCR_X   (12)
#define PARAMS_START_INCR_X     (13)
#define PARAMS_START_INCR_Y     (14)


.align 4
.L_macc_const: //multiplies input in VLMACC instruction
.byte 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0


.align 4
FUNCTION_NAME:
    dualentsp NSTACKWORDS
    std r4, r5, sp[1]
    std r6, r7, sp[2]
    std r8, r9, sp[3]
    {   ldc r7, 32                              ;   stw r10, sp[8]                          }
    {   shl r11, r7, 4                          ;                                           }
    {                                           ;   vsetc r11                               }

    ldaw r11, params[PARAMS_SHIFT]
    {   mov r10, r11                            ;   ldw win_col_incr_x, r11[PARAMS_WIN_COL_INCR_X - PARAMS_SHIFT]   }
    {                                           ;   ldw win_row_incr_x, r10[PARAMS_WIN_ROW_INCR_X - PARAMS_SHIFT]   }
    {   ldc r8, 16                              ;   ldw chan_incr_x, r10[PARAMS_CHAN_INCR_X - PARAMS_SHIFT]         }
    {                                           ;   ldw r11, r10[PARAMS_START_INCR_X - PARAMS_SHIFT]                }
    {   add X, X, r11                           ;   ldw r11, r10[PARAMS_START_INCR_Y - PARAMS_SHIFT]                }
    {   add Y, Y, r11                           ;                                                                   }

    {   ldaw r11, sp[STACK_SHIFT_VEC]           ;   ldw r9, r10[PARAMS_SHIFT - PARAMS_SHIFT]    }
    std r9, r9, r11[0]
    std r9, r9, r11[1]
    std r9, r9, r11[2]
    std r9, r9, r11[3] //16  16-bit shifts

    {   ldaw r11, sp[STACK_SCALE_VEC]           ;   ldw r9, r10[PARAMS_SCALE - PARAMS_SHIFT]    }
    std r9, r9, r11[0]
    std r9, r9, r11[1] //only need 16 elements in vR
    {                                           ;   vldc r11[0]                         }
    
    {   mkmsk r9, 4                             ;   ldw r11, params[PARAMS_W_W]         }
    {   and r9, r9, r11                         ;   ldc r10, 16                         }
    {   sub r10, r10, r9                        ;   shr r11, r11, 4                     }
    {                                           ;   stw r10, sp[STACK_WCOL_TAIL]        }
    {                                           ;   stw r11, sp[STACK_WCOL_BODY]        }
    {                                           ;   ldw r11, params[PARAMS_OUT_ROWS]    }
    {                                           ;   stw r11, sp[STACK_ROWS_LEFT]        }
    .align 4
    .L_vstride_loop:

#define out_col     r6
#define chans_left  r7
#define win_row     r8
        {                                           ;   ldw out_col, params[PARAMS_OUT_COLS]}
        .L_hstride_loop:
            {    mkmsk r9, 4                            ;   ldw chans_left, params[PARAMS_OUT_CHANS]}
            .L_chan_loop:
                {                                               ;   vclrdr                          }                
                {                                               ;   ldw win_row, params[PARAMS_W_H]     }
#define win_tail    r9
#define win_body    r10
                {   ldc r11, 16                                 ;   ldw win_tail, sp[STACK_WCOL_TAIL]   }
                {   lsu r11, r11, chans_left                    ;   ldw win_body, sp[STACK_WCOL_BODY]   }
                .L_win_row_loop:
                    {                                           ;   bf win_body, .L_maccs_tail          }
                    .L_maccs_loop:
                        {   add X, X, win_col_incr_x                ;   vlmacc X[0]                     } // 1
                        {   add X, X, win_col_incr_x                ;   vlmacc X[0]                     }
                        {   add X, X, win_col_incr_x                ;   vlmacc X[0]                     }
                        {   add X, X, win_col_incr_x                ;   vlmacc X[0]                     }
                        {   add X, X, win_col_incr_x                ;   vlmacc X[0]                     }
                        {   add X, X, win_col_incr_x                ;   vlmacc X[0]                     }
                        {   add X, X, win_col_incr_x                ;   vlmacc X[0]                     }
                        {   add X, X, win_col_incr_x                ;   vlmacc X[0]                     }
                        {   add X, X, win_col_incr_x                ;   vlmacc X[0]                     }
                        {   add X, X, win_col_incr_x                ;   vlmacc X[0]                     }
                        {   add X, X, win_col_incr_x                ;   vlmacc X[0]                     }
                        {   add X, X, win_col_incr_x                ;   vlmacc X[0]                     }
                        {   add X, X, win_col_incr_x                ;   vlmacc X[0]                     }
                        {   add X, X, win_col_incr_x                ;   vlmacc X[0]                     }
                        {   add X, X, win_col_incr_x                ;   vlmacc X[0]                     }
                        {   sub win_body, win_body, 1               ;   vlmacc X[0]                     } // 16
                        {   add X, X, win_col_incr_x                ;   bt win_body, .L_maccs_loop      }
                .L_maccs_tail:    
                    {                                           ;   bru win_tail                        }
                    {   add X, X, win_col_incr_x                ;   vlmacc X[0]                         } // r10 == 0
                    {   add X, X, win_col_incr_x                ;   vlmacc X[0]                         }
                    {   add X, X, win_col_incr_x                ;   vlmacc X[0]                         }
                    {   add X, X, win_col_incr_x                ;   vlmacc X[0]                         }
                    {   add X, X, win_col_incr_x                ;   vlmacc X[0]                         }
                    {   add X, X, win_col_incr_x                ;   vlmacc X[0]                         }
                    {   add X, X, win_col_incr_x                ;   vlmacc X[0]                         }
                    {   add X, X, win_col_incr_x                ;   vlmacc X[0]                         }
                    {   add X, X, win_col_incr_x                ;   vlmacc X[0]                         }
                    {   add X, X, win_col_incr_x                ;   vlmacc X[0]                         }
                    {   add X, X, win_col_incr_x                ;   vlmacc X[0]                         }
                    {   add X, X, win_col_incr_x                ;   vlmacc X[0]                         }
                    {   add X, X, win_col_incr_x                ;   vlmacc X[0]                         }
                    {   add X, X, win_col_incr_x                ;   vlmacc X[0]                         }
                    {   add X, X, win_col_incr_x                ;   vlmacc X[0]                         }
                    {   add X, X, win_col_incr_x                ;   vlmacc X[0]                         } // r10 == 15
                    {   sub win_row, win_row, 1                 ;   ldw win_body, sp[STACK_WCOL_BODY]   } // r10 == 16
                    {   add X, X, win_row_incr_x                ;   bt win_row, .L_win_row_loop         }
                .L_win_row_loop_end:
#undef win_row
#undef win_body
#undef win_tail
                {   ldaw r10, sp[STACK_SHIFT_VEC]           ;   ldc r9, 16                          }   
                {   mkmsk r10, 16                           ;   vlsat r10[0]                        }   
                {   mkmsk r11, chans_left                   ;   bf r11, .L_chan_tail                }
                vstrpv Y[0], r10
                {   add Y, Y, r9                            ;   sub chans_left, chans_left, r9      }
                {   add X, X, chan_incr_x                   ;   bu .L_chan_loop                     }

            .L_chan_tail:

            vstrpv Y[0], r11
            {   add Y, Y, chans_left                    ;   add X, X, chan_incr_x                   }
            {   sub out_col, out_col, 1                 ;   ldw r11, params[PARAMS_HSTRIDE_INCR_X]  }
            {   add X, X, r11                           ;   bt out_col, .L_hstride_loop             }
#undef chans_left
#undef out_col

        .L_hstride_loop_end:

        {                                           ;   ldw r9, sp[STACK_ROWS_LEFT]             }
        {   sub r9, r9, 1                           ;   ldw r10, params[PARAMS_VSTRIDE_INCR_X]  }
        {   add X, X, r10                           ;   ldw r11, params[PARAMS_VSTRIDE_INCR_Y]  }
        {   add Y, Y, r11                           ;   stw r9, sp[STACK_ROWS_LEFT]             }
        {                                           ;   bt r9, .L_vstride_loop                  }
        
    .L_vstride_loop_end:



.L_img_end:

.Lfunc_end:
    {                                           ;   ldw r10, sp[8]                      }
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



