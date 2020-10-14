
#if defined(__XS3A__)


/*  
void maxpool2d_ext(
    nn_image_t* Y,
    const nn_image_t* X, 
    const nn_image_params_t* x_params,
    const nn_image_params_t* y_params,
    const nn_window_params_t* pooling_window,
    const nn_window_op_job_params_t* job_params,
    const nn_maxpool2d_flags_e flags);
*/

#include "asm_constants.h"

#define FUNCTION_NAME maxpool2d_adv

#define NSTACKWORDS     (40)
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 16
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME


#define IMG_PARAMS_ROWS     (0)
#define IMG_PARAMS_COLS     (1)
#define IMG_PARAMS_CHANS    (2)

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

#define STACK_POOLING_WINDOW    (NSTACKWORDS+1)
#define STACK_JOB_PARAMS        (NSTACKWORDS+2)
#define STACK_FLAGS             (NSTACKWORDS+3)

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


#define Y                   r0
#define X                   r1
#define C_in                r2
#define C_out               r3
#define x_vstride           r4
#define rows_left           r5
#define cols_left           r6
#define win_hstride         r7
#define chan_mask           r8
#define tmp_A               r9
#define tmp_B               r10

#define Q(R)    R


#define x_params            r2
#define y_params            r3

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

#define job_params  r10
#define window      r11
    {                                           ;   ldw job_params, sp[STACK_JOB_PARAMS]    }
    {                                           ;   ldw window, sp[STACK_POOLING_WINDOW]    }

    // Move Y to its start position
    {                                           ;   ldw r4, y_params[IMG_PARAMS_COLS]       }
    {                                           ;   ldw r5, y_params[IMG_PARAMS_CHANS]      }

        mul r4, r4, r5  // r4 <-- y_row_bytes = y.chans * y.cols
    {                                           ;   ldw r6, job_params[JOB_START_ROW]       }
        mul r4, r4, r6  // r4 <-- y_row_bytes * job.start.row

    {   add Y, Y, r4                            ;   ldw r4, job_params[JOB_START_COL]       }
        mul r4, r4, r5  // r4 <-- y.chans * job.start.cols
    {   add Y, Y, r4                            ;   ldw r4, job_params[JOB_START_CHAN]      }
    {   add Y, Y, r4                            ;   ldw r4, window[WINDOW_STRIDE_VERT]      }


    // Move X to its start position
    {                                           ;   ldw r5, job_params[JOB_START_ROW]       }
        mul r4, r4, r5
    {                                           ;   ldw r5, window[WINDOW_START_ROW]        }
    
    
    {   add r4, r4, r5                          ;   ldw r5, x_params[IMG_PARAMS_CHANS]      } // r4 <-- job_start_row_X
    {                                           ;   ldw r6, x_params[IMG_PARAMS_COLS]       }
        mul r5, r5, r6  // r5 <-- x_row_bytes
        mul r4, r4, r5  // r4 <-- x_row_bytes * job_start_row_x
    {   add X, X, r4                            ;   ldw r4, window[WINDOW_STRIDE_HORI]      }
    {                                           ;   ldw r5, job_params[JOB_START_COL]       }
        mul r4, r4, r5  // r4 <-- job.start.cols * window.stride.hori
    {                                           ;   ldw r5, window[WINDOW_START_COL]        }
    {   add r4, r4, r5                          ;   ldw r5, x_params[IMG_PARAMS_CHANS]      }
        mul r4, r4, r5
    {   add X, X, r4                            ;   ldw r4, job_params[JOB_START_CHAN]      }
    {   add X, X, r4                            ;   ldw r4, window[WINDOW_SHAPE_COLS]       }

    // Save window shape to stack
    {                                           ;   stw r4, sp[STACK_WIN_COLS]              }
    {                                           ;   ldw r4, window[WINDOW_SHAPE_ROWS]       }
    {                                           ;   stw r4, sp[STACK_WIN_ROWS]              }

    // 
    {                                           ;   ldw r4, job_params[JOB_SIZE_CHANS]      }
    {   shr r5, r4, 5                           ;   ldw r6, job_params[JOB_SIZE_ROWS]       }
    {   zext r4, 5                              ;   stw r6, sp[STACK_OUT_ROWS]              }
    {                                           ;   stw r5, sp[STACK_COG_LEFT]              }
    {                                           ;   stw r4, sp[STACK_CHAN_TAIL]             }
    {                                           ;   ldw r4, job_params[JOB_SIZE_COLS]       }
    {                                           ;   stw r4, sp[STACK_OUT_COLS]              }

    // Calculate the Y strides we'll need
    {   ldc r8, 32                              ;   ldw r4, y_params[IMG_PARAMS_COLS]       }
    {                                           ;   ldw r5, y_params[IMG_PARAMS_CHANS]      }
        mul r4, r4, r5  // r4 <-- y_row_bites
    {                                           ;   ldw r6, job_params[JOB_SIZE_ROWS]       }
        mul r6, r6, r4  // r6 <-- y_row_bites * job.size.rows
    {   sub r8, r8, r6                          ;   ldw r6, job_params[JOB_SIZE_COLS]       }
        mul r6, r6, r5  // r6 <-- y.chans * job.size.cols
    {   sub r4, r4, r6                          ;   stw r8, sp[STACK_Y_COG_STRIDE]          }
    {                                           ;   stw r4, sp[STACK_Y_ROW_STRIDE]          }

    // Calculate the X strides we'll need
    {   ldc r8, 32                              ;   ldw r4, x_params[IMG_PARAMS_COLS]       }
    {                                           ;   ldw r5, x_params[IMG_PARAMS_CHANS]      }
        mul r4, r4, r5  // r4 <-- x_row_bytes
    {                                           ;   ldw r6, window[WINDOW_STRIDE_VERT]      }
        mul r6, r6, r4  // r6 <-- win_vstride_from_prev_start
    {                                           ;   ldw r7, job_params[JOB_SIZE_ROWS]       }
        mul r7, r7, r6  // r7 <-- win_vstride_from_prev_start * job.size.rows
    {   sub r8, r8, r7                          ;   ldw r7, window[WINDOW_STRIDE_HORI]      }
    {                                           ;   stw r8, sp[STACK_X_COG_STRIDE]          }
        mul r7, r7, r5  // r7 <-- win_hstride_from_prev_start
    {                                           ;   ldw r8, job_params[JOB_SIZE_COLS]       }
        mul r8, r7, r8  // r8 <-- win_hstride_from_prev_start * job.size.cols
    {   sub r8, r6, r8                          ;   ldw r6, window[WINDOW_SHAPE_ROWS]       }
        mul r6, r6, r4  // r6 <-- window.shape.height * x_row_bytes
    {   sub win_hstride, r7, r6                 ;   stw r8, sp[STACK_WIN_ROW_STRIDE]        }
    {                                           ;   ldw r6, window[WINDOW_SHAPE_COLS]       }
        mul r6, r6, r5  // r6 <-- window.shape.width * x.chans

    // Keep the channel counts handy
    {   sub x_vstride, r4, r6                   ;   ldw C_out, y_params[IMG_PARAMS_CHANS]   }
    {   ldc tmp_A, 32                           ;   ldw C_in, x_params[IMG_PARAMS_CHANS]    }
    {   shl r11, tmp_A, 4                       ;   ldw tmp_B, sp[STACK_OUT_COLS]           }

#undef x_params
#undef y_params
#undef job_params
#undef window


    // Set the VPU mode to 8-bit
    {                                           ;   vsetc r11                               }
    
    // If there are any full channel groups to process, hop to the cog loop.
    {   mkmsk chan_mask, 32                     ;   ldw r11, sp[STACK_COG_LEFT]             }
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
                        {   ldaw tmp_B, sp[STACK_CURMAX_VEC]        ;   vldr r11[0]                             }   // vR[i] = X[i]
                        {   ldaw tmp_B, sp[STACK_TMP_VEC]           ;   vlsub tmp_B[0]                          }   // vR[i] = CURMAX[i] - X[i]
                        {   add X, r11, C_in                        ;   vdepth1                                 }   // vR = bitmask[i] = CURMAX[i] < X[i]   
                        {   sub cols_left, cols_left, 1             ;   vstr tmp_B[0]                           }   // write bitmask to it can be loaded into a register
                        {                                           ;   ldw tmp_B, tmp_B[0]                     }   // load mask into r10
                        {   ldaw r11, sp[STACK_CURMAX_VEC]          ;   vldr r11[0]                             }   // vR = X[i]
                            vstrpv r11[0], tmp_B                                                                    // CURMAX[i] = X[i] where X[i] > CURMAX[i]
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



