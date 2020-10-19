
#if defined(__XS3A__)

#include "nn_config.h"
#include "asm_constants.h"

/*  
void avgpool2d_2x2(
    int8_t* Y,
    const int8_t* X, 
    const channel_count_t XY_chans,
    const nn_window_params_t* pooling_window,
    const nn_window_op_job_params_t* job_params,
    const nn_avgpool2d_flags_e flags,
    const nn_avgpool2d_job_t* job);
*/

#define FUNCTION_NAME avgpool2d_2x2

#define NSTACKWORDS  (16+8*1)
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 4
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME


#define WINDOW_SHAPE_ROWS   (0)
#define WINDOW_SHAPE_COLS   (1)
#define WINDOW_START_ROW    (2)
#define WINDOW_START_COL    (3)
#define WINDOW_STRIDE_VERT  (4)
#define WINDOW_STRIDE_HORI  (5)

#define JOB_START_ROW   (0)
#define JOB_START_COL   (1)
#define JOB_START_CHAN  (2)
#define JOB_SIZE_ROWS   (3)
#define JOB_SIZE_COLS   (4)
#define JOB_SIZE_CHANS  (5)

#define JOB_STRIDE_X_ROW        (0)
#define JOB_STRIDE_X_COG        (1)
#define JOB_STRIDE_WIN_ROW      (2)
#define JOB_STRIDE_WIN_COL      (3)
#define JOB_STRIDE_Y_ROW        (4)
#define JOB_STRIDE_Y_COG        (5)
#define JOB_SCALE               (6)
#define JOB_SHIFT               (7)

#define STACK_JOB_PARAMS        (NSTACKWORDS+1)
#define STACK_FLAGS             (NSTACKWORDS+2)
#define STACK_JOB               (NSTACKWORDS+3)


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
#define XY_chans            r2
#define pool_row_x          r3
#define cols_left           r4
#define write_mask          r5
#define rows_left           r6

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

#define job         r6
#define job_params  r7

    {                                           ;   ldw job, sp[STACK_JOB]                  }
    {                                           ;   ldw job_params, sp[STACK_JOB_PARAMS]    }

    {                                           ;   ldw r11, job_params[JOB_SIZE_ROWS]      }
    {                                           ;   stw r11, sp[STACK_OUT_ROWS]             }
    {                                           ;   ldw r11, job_params[JOB_SIZE_COLS]      }
    {                                           ;   stw r11, sp[STACK_OUT_COLS]             }
    {                                           ;   ldw Q(write_mask), job_params[JOB_SIZE_CHANS]  }

#undef job_params

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

#undef job

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
                    vstrpv r11[0], tmp
                {   ldc tmp, 0                              ;                                           }
                {   ldaw r11, sp[STACK_VEC_TMP]             ;   sub tmp, tmp, 8                         }
                    vlashr r11[0], tmp
                {   mov r11, write_mask                     ;   ldw tmp, sp[STACK_TMP]                  }
                {   andnot r11, tmp                         ;   vdepth8                                 }
                    vstrpv Y[0], r11
                    ldc r11, VPU_MODE_8BIT
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



