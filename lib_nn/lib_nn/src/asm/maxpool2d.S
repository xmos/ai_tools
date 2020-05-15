
#if defined(__XS3A__)


/*  
void maxpool2d(
    int8_t* Y,
    const int8_t* X, 
    const nn_window_op_plan_t* params);
*/

#include "window_op_plan.h"

#define FUNCTION_NAME maxpool2d

#define NSTACKWORDS     (40)
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 16
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME


#define STACK_CURMAX_VEC    (NSTACKWORDS-8)
#define STACK_TMP_VEC       (NSTACKWORDS-16)     


#define STACK_OUT_ROWS          9
#define STACK_OUT_COLS          10
#define STACK_POOL_ROWS         11
#define STACK_POOL_COLS         12
#define STACK_CHAN_TAIL         13
#define STACK_CHAN_GRP_LEFT     14
#define STACK_OUT_ROWS_LEFT     15
#define STACK_OUT_COLS_LEFT     16
#define STACK_X_CHAN_GRP_INCR   17
#define STACK_Y_CHAN_GRP_INCR   18
#define STACK_Y_VSTRIDE         19
#define STACK_X_VSTRIDE         20

#define arg_Y           r0
#define arg_X           r1
#define arg_params      r2

#define Y                   r0
#define X                   r1
#define chan_mask           r2
#define x_pool_col_incr     r3
#define x_pool_row_incr     r4
#define pool_rows_left      r5
#define pool_cols_left      r6
#define x_hstride_incr      r7
#define y_hstride_incr      r8


FUNCTION_NAME:
    dualentsp NSTACKWORDS
    std r4, r5, sp[1]
    std r6, r7, sp[2]
    std r8, r9, sp[3]
    ldaw r11, cp[0]
    {                                           ;   stw r10, sp[8]                          }
    {                                           ;   ldw r10, arg_params[WOP_OUTPUT_ROWS]    }
    {                                           ;   stw r10, sp[STACK_OUT_ROWS]             }
    {                                           ;   ldw r10, arg_params[WOP_OUTPUT_COLS]    }
    {                                           ;   stw r10, sp[STACK_OUT_COLS]             }
    {                                           ;   ldw r10, arg_params[WOP_WINDOW_ROWS]    }
    {                                           ;   stw r10, sp[STACK_POOL_ROWS]            }
    {                                           ;   ldw r10, arg_params[WOP_WINDOW_COLS]    }
    {                                           ;   stw r10, sp[STACK_POOL_COLS]            }
    {                                           ;   ldw r10, arg_params[WOP_OUTER_STRIDE_VERT_X]   }
    {                                           ;   stw r10, sp[STACK_X_VSTRIDE]            }
    {                                           ;   ldw r10, arg_params[WOP_OUTER_STRIDE_VERT_Y]   }
    {   ldc r5, WOP_CHAN_STRIDE_X               ;   stw r10, sp[STACK_Y_VSTRIDE]            }
    {                                           ;   ldw r10, arg_params[r5]                 }
    {   ldc r5, WOP_CHAN_STRIDE_Y               ;   stw r10, sp[STACK_X_CHAN_GRP_INCR]      }
    {                                           ;   ldw r10, arg_params[r5]                 }
    {   ldc r10, WOP_START_STRIDE_X             ;   stw r10, sp[STACK_Y_CHAN_GRP_INCR]      }
    {   ldc r5, WOP_START_STRIDE_Y              ;   ldw r10, arg_params[r10]                }
    {   add X, X, r10                           ;   ldw r10, arg_params[r5]                 }
    {   add Y, Y, r10                           ;   ldw x_pool_col_incr, arg_params[WOP_INNER_STRIDE_HORI]  }
    {   ldc r10, WOP_OUTER_STRIDE_HORI_Y        ;   ldw x_pool_row_incr, arg_params[WOP_INNER_STRIDE_VERT]  }
    {   ldc r11, 32                             ;   ldw x_hstride_incr, arg_params[WOP_OUTER_STRIDE_HORI_X]    }
    {   shl r11, r11, 4                         ;   ldw y_hstride_incr, arg_params[r10]     }
    {   mkmsk chan_mask, 32                     ;   ldw r10, arg_params[WOP_OUTPUT_CHANS]   }
    {   mov r5, r10                             ;   vsetc r11                               }
    {   zext r5, 5                              ;                                           }
    {   shr r11, r10, 5                         ;   stw r5, sp[STACK_CHAN_TAIL]             }
    {                                           ;   bt r11, .L_chan_grp_loop                }
    {   mkmsk chan_mask, r5                     ;   ldc r11, 1                              }

    .L_chan_grp_loop:
        {   sub r11, r11, 1                         ;   ldw r10, sp[STACK_OUT_ROWS]             }
        {   mov r11, r10                            ;   stw r11, sp[STACK_CHAN_GRP_LEFT]        }
        .L_out_rows_loop:
            {   sub r11, r11, 1                         ;   ldw r10, sp[STACK_OUT_COLS]             }
            {                                           ;   stw r11, sp[STACK_OUT_ROWS_LEFT]        }
            .L_out_cols_loop:
                {   sub r11, r10, 1                         ;   ldw pool_rows_left, sp[STACK_POOL_ROWS] }
                {   ldaw r11, cp[vec_0x80]                  ;   stw r11, sp[STACK_OUT_COLS_LEFT]        }
                {   ldaw r11, sp[STACK_CURMAX_VEC]          ;   vldr r11[0]                             }
                {   mov r11, X                              ;   vstr r11[0]                             }
                .L_pool_row_loop:
                    {   sub pool_rows_left, pool_rows_left, 1   ;   ldw pool_cols_left, sp[STACK_POOL_COLS] }
                    .L_pool_col_loop:
                        {   ldaw r10, sp[STACK_CURMAX_VEC]          ;   vldr r11[0]                             }   // vR[i] = CURMAX[i]
                        {   ldaw r10, sp[STACK_TMP_VEC]             ;   vlsub r10[0]                            }   // vR[i] = X[i] - CURMAX[i]
                        {   add X, r11, x_pool_col_incr             ;   vdepth1                                 }   //vR = bitmask of CURMAX[i] < X[i]
                        {   sub pool_cols_left, pool_cols_left, 1   ;   vstr r10[0]                             }   // write bitmask to it can be loaded into a register
                        {                                           ;   ldw r10, r10[0]                         }   // load mask into r10
                        {   ldaw r11, sp[STACK_CURMAX_VEC]          ;   vldr r11[0]                             }   // vR = X[i]
                        vstrpv r11[0], r10                                                                          // CURMAX[i] = X[i] where X[i] > CURMAX[i]
                        {   mov r11, X                              ;   bt pool_cols_left, .L_pool_col_loop     }

                    {   add r11, X, x_pool_row_incr             ;   bt  pool_rows_left, .L_pool_row_loop    }

                {   add X, r11, x_hstride_incr              ;                                           }
                {   ldaw r11, sp[STACK_CURMAX_VEC]          ;   ldw r10, sp[STACK_OUT_COLS_LEFT]        }
                {   mov r11, X                              ;   vldr r11[0]                             }
                vstrpv Y[0], chan_mask
                {   add Y, Y, y_hstride_incr                ;   bt r10, .L_out_cols_loop                }
            
            {                                           ;   ldw r5, sp[STACK_Y_VSTRIDE]             }
            {   add Y, Y, r5                            ;   ldw r6, sp[STACK_X_VSTRIDE]             }
            {   add X, X, r6                            ;   ldw r11, sp[STACK_OUT_ROWS_LEFT]        }
            {                                           ;   bt r11, .L_out_rows_loop                }
    
        {   mkmsk r10, 32                           ;   ldw r5, sp[STACK_X_CHAN_GRP_INCR]       }
        {   eq r10, r10, chan_mask                  ;   ldw r6, sp[STACK_Y_CHAN_GRP_INCR]       }
        {                                           ;   bf r10, .L_img_end                      }
        {   add Y, Y, r6                            ;   ldw r11, sp[STACK_CHAN_GRP_LEFT]        }
        {   add X, X, r5                            ;   bt r11, .L_chan_grp_loop                }
        {   ldc r11, 1                              ;   ldw r10, sp[STACK_CHAN_TAIL]            }
        {   mkmsk chan_mask, r10                    ;   bu .L_chan_grp_loop                     }

    .L_img_end:        

.L_func_end:
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



