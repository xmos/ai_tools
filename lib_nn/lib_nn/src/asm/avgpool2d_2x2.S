
#if defined(__XS3A__)

#include "nn_config.h"
#include "asm_constants.h"

/*  
void avgpool2d_2x2(
    int8_t* Y,
    const int8_t* X, 
    const nn_avgpool2d_plan_t* plan,
    const nn_pool2d_job_t* job);
*/

#define FUNCTION_NAME avgpool2d_2x2

#define NSTACKWORDS  (16+8*1)
    
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


#define STACK_VEC_TMP           (NSTACKWORDS-8)

#define STACK_CP                1
#define STACK_STRIDE_X_COG      9
#define STACK_STRIDE_Y_COG      10
#define STACK_TAIL_LEN          11
#define STACK_OUT_ROWS          12
#define STACK_OUT_COLS          13
#define STACK_COG_LEFT          14
#define STACK_TMP               15


#define Y                   r0
#define X                   r1
// #define TODO: NOT USED   r2
#define pool_row_x          r3
#define cols_left           r4
#define write_mask          r5
#define rows_left           r6
#define XY_chans            r7
#define vstride_x           r8
#define vstride_y           r9
#define tmp                 r10

#define Q(R) R


.align 16

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

//Function args
#define plan        r2
#define job         r3
    
    {                                           ;   ldw XY_chans, plan[PLAN_CHANNELS]       }

#undef plan

    {                                           ;   ldw r11, job[JOB_STRIDE_X_START]        }
    {   add X, X, r11                           ;   ldw r11, job[JOB_STRIDE_Y_START]        }
    {   add Y, Y, r11                           ;   ldw r11, job[JOB_OUT_ROWS]              }
    {                                           ;   stw r11, sp[STACK_OUT_ROWS]             }
    {                                           ;   ldw r11, job[JOB_OUT_COLS]              }
    {                                           ;   stw r11, sp[STACK_OUT_COLS]             }
    {                                           ;   ldw Q(write_mask), job[JOB_OUT_CHANNELS]}
    {   shr Q(cols_left), Q(write_mask), 4      ;   ldw vstride_x, job[JOB_STRIDE_WIN_ROW]  }
    {   zext Q(write_mask), 4                   ;   ldw vstride_y, job[JOB_STRIDE_Y_ROW]    }
    {                                           ;   ldw r11, job[JOB_STRIDE_X_COG]          }
    {                                           ;   stw r11, sp[STACK_STRIDE_X_COG]         }
    {                                           ;   ldw r11, job[JOB_STRIDE_Y_COG]          }
    {                                           ;   stw r11, sp[STACK_STRIDE_Y_COG]         }
    {                                           ;   ldw Q(pool_row_x), job[JOB_STRIDE_X_ROW]}
    {   add pool_row_x, pool_row_x, XY_chans    ;   stw Q(write_mask), sp[STACK_TAIL_LEN]   }
    {   ldaw r11, cp[VPU_VEC_0x01]              ;   mov Q(rows_left), Q(write_mask)         }
    {   mov tmp, Q(cols_left)                   ;   vldc r11[0]                             }

    {   mkmsk write_mask, 16                    ;   bt tmp, .L_chan_loop                    }
    {   mkmsk write_mask, Q(rows_left)          ;   ldc tmp, 1                              }
    
    .L_chan_loop:
        {   sub tmp, tmp, 1                         ;   ldw rows_left, sp[STACK_OUT_ROWS]       }
        {                                           ;   stw tmp, sp[STACK_COG_LEFT]             }
        .L_row_loop:
            {   sub rows_left, rows_left, 1             ;   ldw cols_left, sp[STACK_OUT_COLS]       }

#if CONFIG_SYMMETRIC_SATURATION_avgpool2d

            .L_col_loop:
                {   ldaw r11, cp[VPU_VEC_0x0002]            ;   vclrdr                                  }
                {   add X, X, XY_chans                      ;   vlmacc X[0]                             }
                {   add X, X, pool_row_x                    ;   vlmacc X[0]                             }
                {   add X, X, XY_chans                      ;   vlmacc X[0]                             }
                {   sub X, X, pool_row_x                    ;   vlmacc X[0]                             }
                {   sub cols_left, cols_left, 1             ;   vlsat r11[0]                            }
                vstrpv Y[0], write_mask
                {   add Y, Y, XY_chans                      ;   bt cols_left, .L_col_loop               }

#else

            .L_col_loop:
                {   ldaw r11, cp[VPU_VEC_0x80]              ;   ldc tmp, 32                             }
                {   shl r11, tmp, 3                         ;   vldr r11[0]                             }
                vstrpv Y[0], write_mask
                {   sub cols_left, cols_left, 1             ;   vclrdr                                  }

                {   add X, X, XY_chans                      ;   vlmacc X[0]                             }
                {   add X, X, pool_row_x                    ;   vlmacc X[0]                             }
                {   add X, X, XY_chans                      ;   vlmacc X[0]                             }
                {   sub X, X, pool_row_x                    ;   vlmacc X[0]                             }

                {   ldaw r11, cp[VPU_VEC_0x0002]            ;   vsetc r11       /* 16-bit mode */       }
                {   ldaw r11, sp[STACK_VEC_TMP]             ;   vlsat r11[0]                            }
                {   ldaw r11, cp[VPU_VEC_0x007F]            ;   vstr r11[0]                             }
                {   ldaw r11, sp[STACK_TMP]                 ;   vladd r11[0]                            }
                {   mkmsk tmp, 4                            ;   vdepth1                                 }
                vstrpv r11[0], write_mask
                {   ldc tmp, 0                              ;                                           }
                {   ldaw r11, sp[STACK_VEC_TMP]             ;   sub tmp, tmp, 8                         }
                vlashr r11[0], tmp
                {   mov r11, write_mask                     ;   ldw tmp, sp[STACK_TMP]                  }
                {   andnot r11, tmp                         ;   vdepth8                                 }
                vstrpv Y[0], r11
                ldc tmp, VPU_MODE_8BIT
                {                                           ;   vsetc r11                               }
                {   add Y, Y, XY_chans                      ;   bt cols_left, .L_col_loop               }

#endif

            {   add Y, Y, vstride_y                     ;   ldw r11, sp[STACK_STRIDE_X_COG]         }
            {   add X, X, vstride_x                     ;   bt rows_left, .L_row_loop               }

        {   add X, X, r11                           ;   ldw r11, sp[STACK_STRIDE_Y_COG]         }
        {                                           ;   ldw tmp, sp[STACK_COG_LEFT]             }
        {   add Y, Y, r11                           ;   bt tmp, .L_chan_loop                    }

    {   mkmsk Q(cols_left), 16                  ;   ldw r11, sp[STACK_TAIL_LEN]             }
    {  eq Q(cols_left), write_mask, Q(cols_left);   mkmsk write_mask, r11                   }
    {   ldc tmp, 1                              ;   bf Q(cols_left), .L_img_end             }
    {                                           ;   bt tmp, .L_chan_loop                    }

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



