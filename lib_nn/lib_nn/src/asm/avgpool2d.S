
#if defined(__XS3A__)

#include "nn_config.h"
#include "asm_constants.h"

/*  
void avgpool2d_gen(
    int8_t* Y,
    const int8_t* X, 
    const nn_avgpool2d_plan_t* plan,
    const nn_pool2d_job_t* job);
*/

#define FUNCTION_NAME avgpool2d_gen

#define NSTACKWORDS  38
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 4
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME


#define PLAN_WINDOW_ROWS        0
#define PLAN_WINDOW_COLS        1
#define PLAN_CHANNELS           2
#define PLAN_SHIFT              3
#define PLAN_SCALE              4


#define JOB_OUT_ROWS            0
#define JOB_OUT_COLS            1
#define JOB_OUT_CHANNELS        2
#define JOB_STRIDE_X_START      3
#define JOB_STRIDE_X_ROW        4
#define JOB_STRIDE_X_COG        5
#define JOB_STRIDE_WIN_ROW      6
#define JOB_STRIDE_WIN_COL      7
#define JOB_STRIDE_Y_START      8
#define JOB_STRIDE_Y_ROW        9
#define JOB_STRIDE_Y_COG        10


#define STACK_CP                (1)
#define STACK_OUT_ROWS          (10)
#define STACK_OUT_COLS          (11)
#define STACK_OUTER_STRIDE_VX   (12)
#define STACK_OUTER_STRIDE_VY   (13)
#define STACK_STRIDE_X_COG      (14)
#define STACK_STRIDE_Y_COG      (15)
#define STACK_WIN_ROWS          (16)
#define STACK_CHAN_TAIL         (17)
#define STACK_WIN_COL_LOOPS     (18)
#define STACK_TMP               (19)


#define STACK_SHIFT_VEC         (NSTACKWORDS-8)
#define STACK_VEC_TMP           (NSTACKWORDS-16)

#define APPLAN_SHIFT            (WOP_WORDS)
#define APPLAN_SCALE            (WOP_WORDS+1)
    

#define Y                       r0
#define X                       r1
#define XY_chans                r2
#define chan_mask               r3
#define inner_stride_v          r4
#define win_col_jump            r5
#define outer_stride_hx         r6
#define cog_left                r7
#define out_row_left            r8
#define out_col_left            r9
#define tmp                     r10   


.align 16

.L_vec_0x0000:
.short 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0

#define arg_plan        r2
#define arg_job         r3

#define Q(R) R

FUNCTION_NAME:
    dualentsp NSTACKWORDS
    std r4, r5, sp[1]
    std r6, r7, sp[2]
    std r8, r9, sp[3]
    {                                           ;   stw r10, sp[8]                          }


    //Change constant pool pointer to refer to the constant VPU vects needed here
    ldaw r11, cp[vpu_vects]
    {   ldaw r11, cp[0]                         ;   set cp, r11                             }
    {                                           ;   stw r11, sp[STACK_CP]                   }

    ldc r11, VPU_MODE_8BIT
    {                                           ;   vsetc r11                               }
    

    {   ldaw r11, sp[STACK_SHIFT_VEC]           ;   ldw r10, arg_plan[PLAN_SCALE]           }
    std r10, r10, r11[0]
    std r10, r10, r11[1]
    {                                           ;   vldc r11[0]                             }
    {                                           ;   ldw r10, arg_plan[PLAN_SHIFT]           }
    std r10, r10, r11[0]
    std r10, r10, r11[1]
    std r10, r10, r11[2]
    std r10, r10, r11[3]


    {   ldc r7, 0                               ;   ldw r11, arg_plan[PLAN_WINDOW_ROWS]     }
    {                                           ;   stw r11, sp[STACK_WIN_ROWS]             }
    
    {   ldc r8, 16                              ;   ldw r11, arg_plan[PLAN_WINDOW_COLS]     }
    {   shr r10, r11, 4                         ;   zext r11, 4                             }
    {   lsu r7, r7, r11                         ;   sub r11, r8, r11                        }
    mul win_col_jump, r7, r11
    {   add r10, r10, r7                        ;                                           }
    {                                           ;   stw r10, sp[STACK_WIN_COL_LOOPS]        }

    {                                           ;   ldw XY_chans, arg_plan[PLAN_CHANNELS]   }

#undef arg_plan

    {                                           ;   ldw r11, arg_job[JOB_STRIDE_X_START]    }
    {   add X, X, r11                           ;   ldw r11, arg_job[JOB_STRIDE_Y_START]    }
    {   add Y, Y, r11                           ;                                           }

    {                                           ;   ldw r11, arg_job[JOB_OUT_ROWS]          }
    {                                           ;   stw r11, sp[STACK_OUT_ROWS]             }
    {                                           ;   ldw r11, arg_job[JOB_OUT_COLS]          }
    {                                           ;   stw r11, sp[STACK_OUT_COLS]             }

    {                                           ;   ldw inner_stride_v, arg_job[JOB_STRIDE_X_ROW]   }
    {                                           ;   ldw outer_stride_hx, arg_job[JOB_STRIDE_WIN_COL]}

    {                                           ;   ldw r11, arg_job[JOB_STRIDE_WIN_ROW]    }
    {                                           ;   stw r11, sp[STACK_OUTER_STRIDE_VX]      }
    {                                           ;   ldw r11, arg_job[JOB_STRIDE_Y_ROW]      }
    {                                           ;   stw r11, sp[STACK_OUTER_STRIDE_VY]      }
    {                                           ;   ldw r11, arg_job[JOB_STRIDE_X_COG]      }
    {                                           ;   stw r11, sp[STACK_STRIDE_X_COG]         }
    {                                           ;   ldw r11, arg_job[JOB_STRIDE_Y_COG]      }
    {                                           ;   stw r11, sp[STACK_STRIDE_Y_COG]         }
    {                                           ;   ldw r11, arg_job[JOB_OUT_CHANNELS]      }

#undef arg_job

    {   shr cog_left, r11, 4                    ;   zext r11, 4                             }
    {                                           ;   stw r11, sp[STACK_CHAN_TAIL]            }
    {   mkmsk chan_mask, 16                     ;   bt cog_left, .L_chan_grp_loop           }
    {   mkmsk chan_mask, r11                    ;                                           }    

    .L_chan_grp_loop:
        {   sub cog_left, cog_left, 1               ;   ldw out_row_left, sp[STACK_OUT_ROWS]    }
        .L_out_rows_loop:
            {   sub out_row_left, out_row_left, 1       ;   ldw out_col_left, sp[STACK_OUT_COLS]    }
            .L_out_cols_loop:
#if CONFIG_SYMMETRIC_SATURATION_avgpool2d
                {   sub out_col_left, out_col_left, 1       ;   ldw tmp, sp[STACK_WIN_ROWS]             }
                {                                           ;   vclrdr                                  }
#else
                {   ldaw r11, cp[VPU_VEC_0x80]              ;   ldw tmp, sp[STACK_WIN_ROWS]             }
                {                                           ;   vldr r11[0]                             }
                vstrpv Y[0], chan_mask
                {   sub out_col_left, out_col_left, 1       ;   vclrdr                                  }
#endif
                .L_pool_row_loop:
                    {   sub tmp, tmp, 1                         ;   ldw r11, sp[STACK_WIN_COL_LOOPS]        }
                    {                                           ;   bru win_col_jump                        }
                    .L_pool_col_loop:
                        {   add X, X, XY_chans                      ;   vlmacc X[0]                             }   // 0
                        {   add X, X, XY_chans                      ;   vlmacc X[0]                             }   
                        {   add X, X, XY_chans                      ;   vlmacc X[0]                             }   
                        {   add X, X, XY_chans                      ;   vlmacc X[0]                             }   
                        {   add X, X, XY_chans                      ;   vlmacc X[0]                             }
                        {   add X, X, XY_chans                      ;   vlmacc X[0]                             }
                        {   add X, X, XY_chans                      ;   vlmacc X[0]                             }
                        {   add X, X, XY_chans                      ;   vlmacc X[0]                             }
                        {   add X, X, XY_chans                      ;   vlmacc X[0]                             }   //8
                        {   add X, X, XY_chans                      ;   vlmacc X[0]                             }
                        {   add X, X, XY_chans                      ;   vlmacc X[0]                             }
                        {   add X, X, XY_chans                      ;   vlmacc X[0]                             }
                        {   add X, X, XY_chans                      ;   vlmacc X[0]                             }
                        {   add X, X, XY_chans                      ;   vlmacc X[0]                             }
                        {   add X, X, XY_chans                      ;   vlmacc X[0]                             }
                        {   sub r11, r11, 1                         ;   vlmacc X[0]                             }   //15
                        {   add X, X, XY_chans                      ;   bt r11, .L_pool_col_loop                }
                    {   add X, X, inner_stride_v                ;   bt  tmp, .L_pool_row_loop               }
                    
#if CONFIG_SYMMETRIC_SATURATION_avgpool2d

                {  ldaw tmp, sp[STACK_SHIFT_VEC]            ;                                           }   
                {   add X, X, outer_stride_hx               ;   vlsat tmp[0]                            }   
                vstrpv Y[0], chan_mask

#else

                ldc r11, VPU_MODE_16BIT
                {   ldaw r11, sp[STACK_SHIFT_VEC]           ;   vsetc r11       /* 16-bit mode */       }
                {   ldaw r11, sp[STACK_VEC_TMP]             ;   vlsat r11[0]                            }
                {   ldaw r11, cp[VPU_VEC_0x007F]            ;   vstr r11[0]                             }
                {   ldaw r11, sp[STACK_TMP]                 ;   vladd r11[0]                            }
                {   mkmsk tmp, 4                            ;   vdepth1                                 }
                vstrpv r11[0], tmp
                {   ldc tmp, 0                              ;                                           }
                {   ldaw r11, sp[STACK_VEC_TMP]             ;   sub tmp, tmp, 8                         }
                vlashr r11[0], tmp
                {   mov r11, chan_mask                      ;   ldw tmp, sp[STACK_TMP]                  }
                {   andnot r11, tmp                         ;   vdepth8                                 }
                vstrpv Y[0], r11
                ldc r11, VPU_MODE_8BIT
                {   add X, X, outer_stride_hx               ;   vsetc r11                               }

#endif

                {   add Y, Y, XY_chans                       ;   bt out_col_left, .L_out_cols_loop       }
            {                                           ;   ldw r11, sp[STACK_OUTER_STRIDE_VX]      }
            {   add X, X, r11                           ;   ldw r11, sp[STACK_OUTER_STRIDE_VY]      }
            {   add Y, Y, r11                           ;   bt out_row_left, .L_out_rows_loop       }
        {   mkmsk r10, 16                           ;   ldw r9, sp[STACK_STRIDE_X_COG]          }
        {   eq r10, r10, chan_mask                  ;   ldw r11, sp[STACK_STRIDE_Y_COG]         }
        {   add X, X, r9                            ;   bf r10, .L_img_end                      }
        {   add Y, Y, r11                           ;   bt cog_left, .L_chan_grp_loop           }
        {                                           ;   ldw r10, sp[STACK_CHAN_TAIL]            }
        {   mkmsk chan_mask, r10                    ;   bu .L_chan_grp_loop                     }


    .L_img_end:        

.Lfunc_end:
    //Restore the original constant pool pointer
    {                                           ;   ldw r11, sp[STACK_CP]                   }
    {                                           ;   set cp, r11                             }
    
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



