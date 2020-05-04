
#if defined(__XS3A__)

#include "nn_config.h"
#include "window_op_plan.h"

/*  
void avgpool2d_2x2(
    int8_t* Y,
    const int8_t* X, 
    const nn_avgpool2d_plan_t* params);
*/

#define FUNCTION_NAME avgpool2d_2x2

#define NSTACKWORDS  (20+8*1)
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 4
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

#define arg_Y           r0
#define arg_X           r1
#define arg_params      r2


#define STACK_VEC_TMP           (NSTACKWORDS-8)

#define STACK_X_CHAN_GRP_INCR   9
#define STACK_Y_CHAN_GRP_INCR   10
#define STACK_TAIL_LEN          11
#define STACK_OUT_ROWS          12
#define STACK_OUT_COLS          13
#define STACK_COG_LEFT          14
#define STACK_TMP               15


#define Y                   r0
#define X                   r1
#define pool_col_x          r2
#define pool_row_x          r3
#define cols_left           r4
#define write_mask          r5
#define rows_left           r6
#define hstride_y           r7
#define vstride_x           r8
#define vstride_y           r9
#define tmp                 r10

.align 16
.L_macc_const: //multiplies input in VLMACC instruction
.byte 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1

.L_sat_vec:  //used in VLSAT instruction
.short 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2

.L_vec_0x007F:
.short 0x007F,0x007F,0x007F,0x007F,0x007F,0x007F,0x007F,0x007F,0x007F,0x007F,0x007F,0x007F,0x007F,0x007F,0x007F,0x007F

.L_vec_0x80:
.byte 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80


FUNCTION_NAME:
    dualentsp NSTACKWORDS
    std r4, r5, sp[1]
    std r6, r7, sp[2]
    std r8, r9, sp[3]
    {   ldc r10, 32                             ;   stw r10, sp[8]                          }
    {   shl r11, r10, 4                         ;                                           }
    {   mov r11, arg_params                     ;   vsetc r11                               }

    {                                           ;   ldw r10, r11[WOP_OUTPUT_ROWS]           }
    {                                           ;   stw r10, sp[STACK_OUT_ROWS]             }
    {                                           ;   ldw r10, r11[WOP_OUTPUT_COLS]           }
    {                                           ;   stw r10, sp[STACK_OUT_COLS]             }
    {   ldc r6, WOP_OUTER_STRIDE_HORI_Y         ;   ldw r4, r11[WOP_OUTPUT_CHANS]           }
    {   mov r5, r4                              ;   ldw hstride_y, r11[r6]                  }
    {   zext r5, 4                              ;   ldw vstride_x, r11[WOP_OUTER_STRIDE_VERT_X]}
    {   ldc r6, WOP_CHAN_STRIDE_Y               ;   ldw vstride_y, r11[WOP_OUTER_STRIDE_VERT_Y]}
    {                                           ;   ldw r10, r11[r6]                        }
    {   ldc r6, WOP_CHAN_STRIDE_X               ;   stw r10, sp[STACK_Y_CHAN_GRP_INCR]      }
    {                                           ;   ldw r10, r11[r6]                        }
    {   ldc r6, WOP_INNER_STRIDE_HORI           ;   stw r10, sp[STACK_X_CHAN_GRP_INCR]      }
    {   ldc r6, WOP_INNER_STRIDE_VERT           ;   ldw pool_col_x, r11[r6]                 }
    {   ldc r6, WOP_START_STRIDE_X              ;   ldw pool_row_x, r11[r6]                 }
    {   add pool_row_x, pool_row_x, pool_col_x  ;                                           }
    {   ldc r6, WOP_START_STRIDE_Y              ;   ldw r10, r11[r6]                        }
    {   add X, X, r10                           ;   ldw r11, r11[r6]                        }
    {   add Y, Y, r11                           ;   stw r5, sp[STACK_TAIL_LEN]              }
    {   ldap r11, .L_macc_const                 ;   mov r6, r5                              }
    {   shr tmp, r4, 4                          ;   vldc r11[0]                             }

    {   mkmsk write_mask, 16                    ;   bt tmp, .L_chan_loop                    }
    {   mkmsk write_mask, r6                    ;   ldc tmp, 1                              }
    
    .L_chan_loop:
        {   sub tmp, tmp, 1                         ;   ldw rows_left, sp[STACK_OUT_ROWS]       }
        {                                           ;   stw tmp, sp[STACK_COG_LEFT]             }
        .L_row_loop:
            {   sub rows_left, rows_left, 1             ;   ldw cols_left, sp[STACK_OUT_COLS]       }
#if CONFIG_SYMMETRIC_SATURATION_avgpool2d
            .L_col_loop:
                {   ldap r11, .L_sat_vec                    ;   vclrdr                                  }
                {   add X, X, pool_col_x                    ;   vlmacc X[0]                             }
                {   add X, X, pool_row_x                    ;   vlmacc X[0]                             }
                {   add X, X, pool_col_x                    ;   vlmacc X[0]                             }
                {   sub X, X, pool_row_x                    ;   vlmacc X[0]                             }
                {   sub cols_left, cols_left, 1             ;   vlsat r11[0]                            }
                vstrpv Y[0], write_mask
                {   add Y, Y, hstride_y                     ;   bt cols_left, .L_col_loop               }
#else
            .L_col_loop:
                {   ldap r11, .L_vec_0x80                   ;   ldc tmp, 32                             }
                {   shl r11, tmp, 3                         ;   vldr r11[0]                             }
                vstrpv Y[0], write_mask
                {   sub cols_left, cols_left, 1             ;   vclrdr                                  }

                {   add X, X, pool_col_x                    ;   vlmacc X[0]                             }
                {   add X, X, pool_row_x                    ;   vlmacc X[0]                             }
                {   add X, X, pool_col_x                    ;   vlmacc X[0]                             }
                {   sub X, X, pool_row_x                    ;   vlmacc X[0]                             }

                {   ldap r11, .L_sat_vec                    ;   vsetc r11       /* 16-bit mode */       }
                {   ldaw r11, sp[STACK_VEC_TMP]             ;   vlsat r11[0]                            }
                {   ldap r11, .L_vec_0x007F                 ;   vstr r11[0]                             }
                {   ldaw r11, sp[STACK_TMP]                 ;   vladd r11[0]                            }
                {   mkmsk tmp, 4                            ;   vdepth1                                 }
                vstrpv r11[0], write_mask
                {   ldc tmp, 0                              ;                                           }
                {   ldaw r11, sp[STACK_VEC_TMP]             ;   sub tmp, tmp, 8                         }
                vlashr r11[0], tmp
                {   mov r11, write_mask                     ;   ldw tmp, sp[STACK_TMP]                  }
                {   andnot r11, tmp                         ;   vdepth8                                 }
                vstrpv Y[0], r11
                {   ldc tmp, 32                             ;                                           }
                {   shl r11, tmp, 4                         ;                                           }
                {                                           ; vsetc r11                                 }
                {   add Y, Y, hstride_y                     ;   bt cols_left, .L_col_loop               }
#endif
            {   add Y, Y, vstride_y                     ;   ldw r11, sp[STACK_X_CHAN_GRP_INCR]      }
            {   add X, X, vstride_x                     ;   bt rows_left, .L_row_loop               }
        {   add X, X, r11                           ;   ldw r11, sp[STACK_Y_CHAN_GRP_INCR]      }
        {                                           ;   ldw tmp, sp[STACK_COG_LEFT]             }
        {   add Y, Y, r11                           ;   bt tmp, .L_chan_loop                    }
    {   mkmsk r4, 16                            ;   ldw r11, sp[STACK_TAIL_LEN]         }
    {   eq r4, write_mask, r4                   ;   mkmsk write_mask, r11               }
    {   ldc tmp, 1                              ;   bf r4, .L_img_end                   }
    {                                           ;   bt tmp, .L_chan_loop                }

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



