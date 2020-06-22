
#if defined(__XS3A__)


/*  
void maxpool2d(
    int8_t* Y,
    const int8_t* X, 
    const nn_maxpool_plan_t* plan,
    const nn_maxpool_job_t* job);
*/

#include "asm_constants.h"

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

#define STACK_CP                1
#define STACK_OUT_ROWS          9
#define STACK_OUT_COLS          10
#define STACK_WIN_ROWS          11
#define STACK_WIN_COLS          12
#define STACK_CHAN_TAIL         13
#define STACK_COG_LEFT          14
#define STACK_OUT_ROWS_LEFT     15
#define STACK_OUT_COLS_LEFT     16
#define STACK_X_COG_STRIDE      17
#define STACK_Y_COG_STRIDE      18
#define STACK_Y_ROW_STRIDE      19
#define STACK_WIN_ROW_STRIDE    20

#define PLAN_WIN_ROWS           0
#define PLAN_WIN_COLS           1
#define PLAN_C_IN               2
#define PLAN_C_OUT              3

#define JOB_ROWS                0
#define JOB_COLS                1
#define JOB_CHANNELS            2
#define JOB_X_START_STRIDE      3
#define JOB_X_ROW_STRIDE        4
#define JOB_X_COG_STRIDE        5
#define JOB_WIN_ROW_STRIDE      6
#define JOB_WIN_COL_STRIDE      7
#define JOB_Y_START_STRIDE      8
#define JOB_Y_ROW_STRIDE        9  
#define JOB_Y_COG_STRIDE        10



#define Y                   r0
#define X                   r1
#define chan_mask           r2
#define C_in                r3
#define x_vstride           r4
#define rows_left           r5
#define cols_left           r6
#define win_hstride         r7
#define C_out               r8
#define tmp_A               r9
#define tmp_B               r10

#define Q(R)    R


#define plan            r2
#define job             r3

FUNCTION_NAME:
    dualentsp NSTACKWORDS
    std r4, r5, sp[1]
    std r6, r7, sp[2]
    std r8, r9, sp[3]
    {                                           ;   stw r10, sp[8]                          }

    // Change constant pool pointer to refer to the constant VPU vects needed here
    ldaw r11, cp[vpu_vects]
    {   ldaw r11, cp[0]                         ;   set cp, r11                             }
    {                                           ;   stw r11, sp[STACK_CP]                   }

    // Extract everything we need from the job struct
    {                                           ;   ldw tmp_B, job[JOB_CHANNELS]            }
    {   shr Q(C_out), tmp_B, 5                  ;   ldw tmp_A, job[JOB_ROWS]                }
    {   zext tmp_B, 5                           ;   stw tmp_A, sp[STACK_OUT_ROWS]           }
    {                                           ;   stw Q(C_out), sp[STACK_COG_LEFT]        }
    {                                           ;   stw tmp_B, sp[STACK_CHAN_TAIL]          }
    {                                           ;   ldw tmp_A, job[JOB_COLS]                }
    {                                           ;   stw tmp_A, sp[STACK_OUT_COLS]           }
    {                                           ;   ldw tmp_A, job[JOB_X_START_STRIDE]      }
    {   add X, X, tmp_A                         ;   ldw x_vstride, job[JOB_X_ROW_STRIDE]    }
    {                                           ;   ldw tmp_A, job[JOB_X_COG_STRIDE]        }
    {                                           ;   stw tmp_A, sp[STACK_X_COG_STRIDE]       }
    {                                           ;   ldw win_hstride, job[JOB_WIN_COL_STRIDE]}
    {                                           ;   ldw tmp_A, job[JOB_WIN_ROW_STRIDE]      }
    {                                           ;   stw tmp_A, sp[STACK_WIN_ROW_STRIDE]     }
    {                                           ;   ldw tmp_A, job[JOB_Y_START_STRIDE]      }
    {   add Y, Y, tmp_A                         ;   ldw tmp_A, job[JOB_Y_ROW_STRIDE]        }
    {                                           ;   stw tmp_A, sp[STACK_Y_ROW_STRIDE]       }
    {                                           ;   ldw tmp_A, job[JOB_Y_COG_STRIDE]        }
    {                                           ;   stw tmp_A, sp[STACK_Y_COG_STRIDE]       }

#undef job

    // Extract everything else we need from the plan struct
    {                                           ;   ldw C_out, plan[PLAN_C_OUT]             }
    {                                           ;   ldw C_in, plan[PLAN_C_IN]               }
    {                                           ;   ldw Q(cols_left), plan[PLAN_WIN_COLS]   }
    {                                           ;   ldw Q(rows_left), plan[PLAN_WIN_ROWS]   }
    {                                           ;   stw Q(cols_left), sp[STACK_WIN_COLS]    }
    {   ldc tmp_A, 32                           ;   stw Q(rows_left), sp[STACK_WIN_ROWS]    }
    {   shl r11, tmp_A, 4                       ;   ldw tmp_B, sp[STACK_OUT_COLS]           }

#undef plan

    // Set the VPU mode to 8-bit
    {                                           ;   vsetc r11                               }
    
    // If there are any full channel groups to process, hop to the cog loop.
    {   mkmsk chan_mask, 32                      ;   ldw r11, sp[STACK_COG_LEFT]             }
    {                                           ;   bt r11, .L_cog_loop                     }

    //Otherwise, process the tail
    {                                           ;   ldw chan_mask, sp[STACK_CHAN_TAIL]      }
    {   mkmsk chan_mask, chan_mask              ;   ldc r11, 1                              }

    // COG loop: expects r11 to contain remaining channel groups (1 if on tail)
    .L_cog_loop:
        {   sub r11, r11, 1                         ;   ldw tmp_B, sp[STACK_OUT_ROWS]           }
        {   mov r11, tmp_B                          ;   stw r11, sp[STACK_COG_LEFT]             }

        // Outer row loop: expects r11 to contain remaining output rows
        .L_out_rows_loop:
            {   sub r11, r11, 1                         ;   ldw tmp_B, sp[STACK_OUT_COLS]           }
            {                                           ;   stw r11, sp[STACK_OUT_ROWS_LEFT]        }

            // Outer col loop: 1 output pixel is computed per iteration (for the current channel group)
            //      Expects tmp_B to contain remaining output columns
            .L_out_cols_loop:
                {   sub r11, tmp_B, 1                       ;   ldw rows_left, sp[STACK_WIN_ROWS]       }
                {   ldaw r11, cp[VPU_VEC_0x80]              ;   stw r11, sp[STACK_OUT_COLS_LEFT]        }
                {   ldaw r11, sp[STACK_CURMAX_VEC]          ;   vldr r11[0]                             } 
                {   mov r11, X                              ;   vstr r11[0]                             } //Set CURMAX to all -128's

                // Inner row loop
                .L_pool_row_loop:
                    {   sub rows_left, rows_left, 1             ;   ldw cols_left, sp[STACK_WIN_COLS]       }

                    //Inner col loop
                    .L_pool_col_loop:
                        {   ldaw tmp_B, sp[STACK_CURMAX_VEC]        ;   vldr r11[0]                             }   // vR[i] = CURMAX[i]
                        {   ldaw tmp_B, sp[STACK_TMP_VEC]           ;   vlsub tmp_B[0]                          }   // vR[i] = X[i] - CURMAX[i]
                        {   add X, r11, C_in                        ;   vdepth1                                 }   //vR = bitmask of CURMAX[i] < X[i]
                        {   sub cols_left, cols_left, 1             ;   vstr tmp_B[0]                           }   // write bitmask to it can be loaded into a register
                        {                                           ;   ldw tmp_B, tmp_B[0]                     }   // load mask into r10
                        {   ldaw r11, sp[STACK_CURMAX_VEC]          ;   vldr r11[0]                             }   // vR = X[i]
                        vstrpv r11[0], tmp_B                                                                        // CURMAX[i] = X[i] where X[i] > CURMAX[i]
                        {   mov r11, X                              ;   bt cols_left, .L_pool_col_loop          }

                    {   add r11, X, x_vstride               ;   bt  rows_left, .L_pool_row_loop         }

                {   add X, r11, win_hstride                 ;                                           }
                {   ldaw r11, sp[STACK_CURMAX_VEC]          ;   ldw tmp_B, sp[STACK_OUT_COLS_LEFT]      }
                {   mov r11, X                              ;   vldr r11[0]                             }
                vstrpv Y[0], chan_mask
                {   add Y, Y, C_out                         ;   bt tmp_B, .L_out_cols_loop              }
            
            {                                           ;   ldw Q(rows_left), sp[STACK_Y_ROW_STRIDE]}
            {   add Y, Y, Q(rows_left)                  ;   ldw Q(cols_left), sp[STACK_WIN_ROW_STRIDE]}
            {   add X, X, Q(cols_left)                  ;   ldw r11, sp[STACK_OUT_ROWS_LEFT]        }
            {                                           ;   bt r11, .L_out_rows_loop                }
    
        {   mkmsk tmp_B, 32                         ;   ldw Q(rows_left), sp[STACK_X_COG_STRIDE]}
        {   eq tmp_B, tmp_B, chan_mask              ;   ldw Q(cols_left), sp[STACK_Y_COG_STRIDE]}
        {                                           ;   bf tmp_B, .L_img_end                    }
        {   add Y, Y, Q(cols_left)                  ;   ldw r11, sp[STACK_COG_LEFT]             }
        {   add X, X, Q(rows_left)                  ;   bt r11, .L_cog_loop                     }
        {   ldc r11, 1                              ;   ldw tmp_B, sp[STACK_CHAN_TAIL]          }
        {   mkmsk chan_mask, r10                    ;   bu .L_cog_loop                          }

    .L_img_end:        

.L_func_end:
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



