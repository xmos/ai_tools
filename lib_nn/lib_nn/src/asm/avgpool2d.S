
#if defined(__XS3A__)

#include "nn_config.h"
#include "window_op_plan.h"

/*  
void avgpool2d_gen(
    int8_t* Y,
    const int8_t* X, 
    const nn_avgpool2d_plan_t* plan);
*/

#define FUNCTION_NAME avgpool2d_gen

#define NSTACKWORDS  38
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 4
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

#define NOOOOP          nop

#define arg_Y           r0
#define arg_X           r1
#define arg_plan        r2

#define STACK_OUT_ROWS          (10)
#define STACK_OUT_COLS          (11)
#define STACK_OUTER_STRIDE_VX   (12)
#define STACK_OUTER_STRIDE_VY   (13)
#define STACK_CHAN_STRIDE_X     (14)
#define STACK_CHAN_STRIDE_Y     (15)
#define STACK_WIN_ROWS          (16)
#define STACK_OUTER_STRIDE_HY   (17)
#define STACK_CHAN_TAIL         (18)
#define STACK_WIN_COL_LOOPS     (19)
#define STACK_TMP               (20)


#define STACK_SHIFT_VEC         (NSTACKWORDS-8)
#define STACK_VEC_TMP           (NSTACKWORDS-16)

#define APPLAN_SHIFT            (WOP_WORDS)
#define APPLAN_SCALE            (WOP_WORDS+1)
    

#define Y                       r0
#define X                       r1
#define chan_mask               r2
#define inner_stride_h          r3
#define inner_stride_v          r4
#define win_col_jump            r5
#define outer_stride_hx         r6
#define chan_grp_left           r7
#define out_row_left            r8
#define out_col_left            r9
#define tmp                     r10   

#define Q(DSA)  DSA


.align 16

.L_vec_0x0000:
.short 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0


FUNCTION_NAME:
    dualentsp NSTACKWORDS
    std r4, r5, sp[1]
    std r6, r7, sp[2]
    std r8, r9, sp[3]
    ldc r11, 0x200
    {                                           ;   vsetc r11                               }
    {   ldc r11, APPLAN_SCALE                   ;   stw r10, sp[8]                          }
    {   ldaw r11, sp[STACK_SHIFT_VEC]           ;   ldw r10, arg_plan[r11]                  }
    std r10, r10, r11[0]
    std r10, r10, r11[1]
    {   ldc r10, APPLAN_SHIFT                   ;   vldc r11[0]                             }
    {                                           ;   ldw r10, arg_plan[r10]                  }
    std r10, r10, r11[0]
    std r10, r10, r11[1]
    std r10, r10, r11[2]
    std r10, r10, r11[3]

    {                                           ;   ldw r10, arg_plan[WOP_START_STRIDE_X]   }
    {   add X, X, r10                           ;   ldw r10, arg_plan[WOP_START_STRIDE_Y]   }
    {   add Y, Y, r10                           ;   ldw r10, arg_plan[WOP_OUTPUT_ROWS]      }
    {                                           ;   ldw r11, arg_plan[WOP_OUTPUT_COLS]      }
    std r11, r10, sp[STACK_OUT_ROWS/2]
    {                                           ;   ldw r10, arg_plan[WOP_OUTER_STRIDE_VERT_X]}
    {   ldc r9, WOP_CHAN_STRIDE_X               ;   ldw r11, arg_plan[WOP_OUTER_STRIDE_VERT_Y]}
    std r11, r10, sp[STACK_OUTER_STRIDE_VX/2]
    {   ldc r9, WOP_CHAN_STRIDE_Y               ;   ldw r10, arg_plan[r9]                   }
    {                                           ;   ldw r11, arg_plan[r9]                   }
    std r11, r10, sp[STACK_CHAN_STRIDE_X/2]  
    {   ldc r9, WOP_OUTER_STRIDE_HORI_Y         ;   ldw r10, arg_plan[WOP_WINDOW_ROWS]      }
    {   ldc r7, 0                               ;   ldw r11, arg_plan[r9]                   }
    std r11, r10, sp[STACK_WIN_ROWS/2]

    {                                           ;   ldw r10, arg_plan[WOP_OUTPUT_CHANS]     }
    {   zext r10, 4                             ;                                           }
    {                                           ;   stw r10, sp[STACK_CHAN_TAIL]            }


    {   ldc r8, 16                              ;   ldw r11, arg_plan[WOP_WINDOW_COLS]      }
    {   shr r10, r11, 4                         ;   zext r11, 4                             }
    {   lsu r7, r7, r11                         ;   sub r11, r8, r11                        }
    mul win_col_jump, r7, r11
    {   add r10, r10, r7                        ;                                           }
    {                                           ;   stw r10, sp[STACK_WIN_COL_LOOPS]        }

    {                                           ;   ldw inner_stride_h, arg_plan[WOP_INNER_STRIDE_HORI]}
    {                                           ;   ldw inner_stride_v, arg_plan[WOP_INNER_STRIDE_VERT]}
    {                                           ;   ldw outer_stride_hx, arg_plan[WOP_OUTER_STRIDE_HORI_X]}



    {                                           ;   ldw r11, arg_plan[WOP_OUTPUT_CHANS]     }
    {   shr chan_grp_left, r11, 4               ;   zext r11, 4                             }
    {   mkmsk chan_mask, 16                     ;   bt chan_grp_left, .L_chan_grp_loop      }
    {   mkmsk chan_mask, r11                    ;                                           }    
    .L_chan_grp_loop:
        {   sub chan_grp_left, chan_grp_left, 1     ;   ldw out_row_left, sp[STACK_OUT_ROWS]    }
        .L_out_rows_loop:
            {   sub out_row_left, out_row_left, 1       ;   ldw out_col_left, sp[STACK_OUT_COLS]    }
            .L_out_cols_loop:
#if CONFIG_SYMMETRIC_SATURATION_avgpool2d
                {   sub out_col_left, out_col_left, 1       ;   ldw tmp, sp[STACK_WIN_ROWS]             }
                {                                           ;   vclrdr                                  }
#else
                {   ldaw r11, cp[vec_0x80]                  ;   ldw tmp, sp[STACK_WIN_ROWS]             }
                {                                           ;   vldr r11[0]                             }
                vstrpv Y[0], chan_mask
                {   sub out_col_left, out_col_left, 1       ;   vclrdr                                  }
#endif
                .L_pool_row_loop:
                    {   sub tmp, tmp, 1                         ;   ldw r11, sp[STACK_WIN_COL_LOOPS]        }
                    {                                           ;   bru win_col_jump                        }
                    .L_pool_col_loop:
                        {   add X, X, inner_stride_h                ;   vlmacc X[0]                             }   // 0
                        {   add X, X, inner_stride_h                ;   vlmacc X[0]                             }   
                        {   add X, X, inner_stride_h                ;   vlmacc X[0]                             }   
                        {   add X, X, inner_stride_h                ;   vlmacc X[0]                             }   
                        {   add X, X, inner_stride_h                ;   vlmacc X[0]                             }
                        {   add X, X, inner_stride_h                ;   vlmacc X[0]                             }
                        {   add X, X, inner_stride_h                ;   vlmacc X[0]                             }
                        {   add X, X, inner_stride_h                ;   vlmacc X[0]                             }
                        {   add X, X, inner_stride_h                ;   vlmacc X[0]                             }   //8
                        {   add X, X, inner_stride_h                ;   vlmacc X[0]                             }
                        {   add X, X, inner_stride_h                ;   vlmacc X[0]                             }
                        {   add X, X, inner_stride_h                ;   vlmacc X[0]                             }
                        {   add X, X, inner_stride_h                ;   vlmacc X[0]                             }
                        {   add X, X, inner_stride_h                ;   vlmacc X[0]                             }
                        {   add X, X, inner_stride_h                ;   vlmacc X[0]                             }
                        {   sub r11, r11, 1                         ;   vlmacc X[0]                             }   //15
                        {   add X, X, inner_stride_h                ;   bt r11, .L_pool_col_loop                }
                    {   add X, X, inner_stride_v                ;   bt  tmp, .L_pool_row_loop               }
                    
                {  ldaw tmp, sp[STACK_SHIFT_VEC]            ;   ldw r11, sp[STACK_OUTER_STRIDE_HY]      }   
#if CONFIG_SYMMETRIC_SATURATION_avgpool2d
                {   add X, X, outer_stride_hx               ;   vlsat tmp[0]                }   
                vstrpv Y[0], chan_mask
#else
                {   ldc tmp, 32                             ;                                           }
                {   shl r11, tmp, 3                         ;                                           }
                {   ldaw r11, sp[STACK_SHIFT_VEC]           ;   vsetc r11       /* 16-bit mode */       }
                {   ldaw r11, sp[STACK_VEC_TMP]             ;   vlsat r11[0]                            }
                {   ldaw r11, cp[vec_0x007F]                ;   vstr r11[0]                             }
                {   ldaw r11, sp[STACK_TMP]                 ;   vladd r11[0]                            }
                {   mkmsk tmp, 4                            ;   vdepth1                                 }
                vstrpv r11[0], tmp
                {   ldc tmp, 0                              ;                                           }
                {   ldaw r11, sp[STACK_VEC_TMP]             ;   sub tmp, tmp, 8                         }
                vlashr r11[0], tmp
                {   mov r11, chan_mask                      ;   ldw tmp, sp[STACK_TMP]                  }
                {   andnot r11, tmp                         ;   vdepth8                                 }
                vstrpv Y[0], r11
                {   ldc tmp, 32                             ;                                           }
                {   shl r11, tmp, 4                         ;                                           }
                {                                           ;   vsetc r11                               }
                {   add X, X, outer_stride_hx               ;   ldw r11, sp[STACK_OUTER_STRIDE_HY]      }


#endif
                {   add Y, Y, r11                           ;   bt out_col_left, .L_out_cols_loop       }
            {                                           ;   ldw r11, sp[STACK_OUTER_STRIDE_VX]      }
            {   add X, X, r11                           ;   ldw r11, sp[STACK_OUTER_STRIDE_VY]      }
            {   add Y, Y, r11                           ;   bt out_row_left, .L_out_rows_loop       }
        {   mkmsk r10, 16                           ;   ldw r9, sp[STACK_CHAN_STRIDE_X]         }
        {   eq r10, r10, chan_mask                  ;   ldw r11, sp[STACK_CHAN_STRIDE_Y]        }
        {   add X, X, r9                            ;   bf r10, .L_img_end                      }
        {   add Y, Y, r11                           ;   bt chan_grp_left, .L_chan_grp_loop      }
        {                                           ;   ldw r10, sp[STACK_CHAN_TAIL]            }
        {   mkmsk chan_mask, r10                    ;   bu .L_chan_grp_loop                     }


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



