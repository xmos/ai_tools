
#if defined(__XS3A__)


/*  
void conv2d_deepin_deepout_block_asm(
    int8_t* Y,
    const nn_conv2d_dido_params_t* params,
    const nn_conv2d_dido_block_params_t* block,
    const int8_t* X,
    const int8_t* K,
    const int16_t* shifts,
    const int16_t* scales);
*/


#define FUNCTION_NAME conv2d_deepin_deepout_block_asm

#define NSTACKVECTORS   (0)
#define NSTACKWORDS     (NSTACKVECTORS*8 + 30)
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 4
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

#define NOOOOP          nop

#define Y                       r0
#define params                  r1
#define block                   r2
#define arg_X                   r3

#define STACK_K_START           (NSTACKWORDS+1)
#define STACK_SHIFTS_START      (NSTACKWORDS+2)
#define STACK_SCALES_START      (NSTACKWORDS+3)

#define STACK_R4_R5_DBL         (1)
#define STACK_R6_R7_DBL         (2)
#define STACK_R8_R9_DBL         (3)
#define STACK_R10               (1)               

#define STACK_X_START           (8)

#define STACK_B_NEXT            (9)
#define STACK_SHIFTS_NEXT       (10)
#define STACK_SCALES_NEXT       (11)

#define STACK_OUT_COLS          (12)
#define STACK_COUT_GRPS         (13)
#define STACK_OUT_X_PXL_INCR    (14)
#define STACK_OUT_X_ROW_INCR    (15)
#define STACK_OUT_Y_ROW_INCR    (16)
#define STACK_PATCH_ROWS        (17)
#define STACK_PATCH_MACCS       (18)
#define STACK_PATCH_X_ROW_INCR  (19)
#define STACK_PATCH_K_ROW_INCR  (20)
#define STACK_COUT_GRP_INCR_K   (21)


#define STACK_OUT_ROWS_LEFT     (22)
#define STACK_OUT_COLS_LEFT     (23)
#define STACK_COUT_GRPS_LEFT    (24)

#define STACK_B_START           (25)


// #define STACK_PADC_VEC          (NSTACKWORDS-8)
// #define STACK_ZEROP_VEC         (NSTACKWORDS-16)

.align 4

FUNCTION_NAME:
    {                                           ;   dualentsp NSTACKWORDS               }
    std r4, r5, sp[STACK_R4_R5_DBL]
    std r6, r7, sp[STACK_R6_R7_DBL]
    std r8, r9, sp[STACK_R8_R9_DBL]
    {                                           ;   stw r10, sp[STACK_R10]              }

//Do some initialization stuff

    //X = X + block->init.start_offset.X
    {                                                   ;   ldw r11, block[0]                               }
    {   add arg_X, arg_X, r11                           ;                                                   }
    {                                                   ;   stw arg_X, sp[STACK_X_START]                    }

    //Y = Y + block->init.start_offset.Y
    {                                                   ;   ldw r11, block[1]                               }
    {   add Y, Y, r11                                   ;                                                   }

    //K = K + block->init.start_offset.K
    {                                                   ;   ldw r11, block[2]                               }
    {                                                   ;   ldw r10, sp[STACK_K_START]                      }
    {   add r10, r10, r11                               ;                                                   }
    {                                                   ;   stw r10, sp[STACK_K_START]                      }

    //Load our adjusted bias vector (which is where padding_count used to be!)
    //  and store it on the stack
    {                                                   ;   ldw r4, block[3]                                }
    {                                                   ;   stw r4, sp[STACK_B_START]                       }


// Other stuff that should be on the stack
//  TODO: some of this could be expedited with ldd and std if the params are arranged nicely on the stack

    {                                                   ;   ldw r4, params[4]                               }
    {                                                   ;   stw r4, sp[STACK_COUT_GRPS]                     }
    {                                                   ;   ldw r4, params[1]                               }
    {                                                   ;   stw r4, sp[STACK_OUT_X_PXL_INCR]                }
    {                                                   ;   ldw r4, block[4]                                }
    {                                                   ;   stw r4, sp[STACK_COUT_GRP_INCR_K]               }
    {                                                   ;   ldw r4, block[5]                                }
    {                                                   ;   bf r4, .Lfunc_end                               }
    {                                                   ;   stw r4, sp[STACK_OUT_ROWS_LEFT]                 }
    {                                                   ;   ldw r4, block[6]                                }
    {                                                   ;   stw r4, sp[STACK_OUT_COLS]                      }
    {                                                   ;   ldw r4, block[7]                                }
    {                                                   ;   stw r4, sp[STACK_OUT_X_ROW_INCR]                }
    {                                                   ;   ldw r4, block[8]                                }
    {                                                   ;   stw r4, sp[STACK_OUT_Y_ROW_INCR]                }
    {                                                   ;   ldw r4, block[9]                                }
    {                                                   ;   stw r4, sp[STACK_PATCH_MACCS]                   }
    {                                                   ;   ldw r4, block[10]                               }
    {                                                   ;   stw r4, sp[STACK_PATCH_ROWS]                    }
    {                                                   ;   ldw r4, block[11]                               }
    {   ldc r7, 12                                      ;   stw r4, sp[STACK_PATCH_X_ROW_INCR]              }
    {                                                   ;   ldw r4, block[r7]                               }
    {                                                   ;   stw r4, sp[STACK_PATCH_K_ROW_INCR]              }

#undef params
#undef block

//Outermost loop -- output rows

.L_out_rows_loop:

    {                                                   ;   ldw r11, sp[STACK_OUT_COLS]                     }
    {                                                   ;   stw r11, sp[STACK_OUT_COLS_LEFT]                }

    .L_out_cols_loop:

        //We'll keep K in a register for the duration of the out col loop
        #define K   r1

        //reset K to start of K tensor (plus block offset)
        {                                                   ;   ldw K, sp[STACK_K_START]                        }

        //Reset number of output channel groups
        {                                                   ;   ldw r11, sp[STACK_COUT_GRPS]                    }
        {                                                   ;   stw r11, sp[STACK_COUT_GRPS_LEFT]               }

        //Reset biases, shifts and scales
        {                                                   ;   ldw r2, sp[STACK_B_START]                       }
        {                                                   ;   ldw r3, sp[STACK_SHIFTS_START]                  }
        {                                                   ;   ldw r4, sp[STACK_SCALES_START]                  }
        
        {                                                   ;   stw r2, sp[STACK_B_NEXT]                        }
        {                                                   ;   stw r3, sp[STACK_SHIFTS_NEXT]                   }
        {                                                   ;   stw r4, sp[STACK_SCALES_NEXT]                   }

        .L_cout_grp_loop:

            //We'll keep X in a register for the duration of the out channel group
            #define X   r2

            //Reset X to start of patch
            {   ldc r10, 32                                     ;   ldw X, sp[STACK_X_START]                        }

            //Set VPU mode to 8-bit
            {   shl r11, r10, 4                                 ;                                                   }
            {                                                   ;   vsetc r11                                       }

            //initialize biases
            {                                                   ;   ldw r3, sp[STACK_B_NEXT]                        }
            {   add r11, r3, r10                                ;   vldd r3[0]                                      }
            {   add r3, r11, r10                                ;   vldr r11[0]                                     }
            {                                                   ;   stw r3, sp[STACK_B_NEXT]                        }

            //Keep patch_rows in a register until we're done with the patch
            #define patch_rows  r3

            //begin the patch maccs
            {                                                   ;   ldw patch_rows, sp[STACK_PATCH_ROWS]            }
            

            //Put some constants we need in registers
            #define patch_x_row_incr    r5
            #define patch_k_row_incr    r6

            {                                                       ;   ldw patch_k_row_incr, sp[STACK_PATCH_K_ROW_INCR]    }
            {                                                       ;   ldw patch_x_row_incr, sp[STACK_PATCH_X_ROW_INCR]    }

            .L_patch_row_loop:

                //Keep patch_maccs in a register until we're done with the row
                #define patch_maccs r4

                //reset the patch row macc count
                {                                           ;   ldw patch_maccs, sp[STACK_PATCH_MACCS]      }
                
                //A single VLMACCR group
                .L_patch_row_macc_loop:

                    {   add X, X, r10                           ;   vldc X[0]                                   }

                    {   add K, K, r10                           ;   vlmaccr K[0]                                }
                    {   add K, K, r10                           ;   vlmaccr K[0]                                }
                    {   add K, K, r10                           ;   vlmaccr K[0]                                }
                    {   add K, K, r10                           ;   vlmaccr K[0]                                }
                    {   add K, K, r10                           ;   vlmaccr K[0]                                }
                    {   add K, K, r10                           ;   vlmaccr K[0]                                }
                    {   add K, K, r10                           ;   vlmaccr K[0]                                }
                    {   add K, K, r10                           ;   vlmaccr K[0]                                }
                    {   add K, K, r10                           ;   vlmaccr K[0]                                }
                    {   add K, K, r10                           ;   vlmaccr K[0]                                }
                    {   add K, K, r10                           ;   vlmaccr K[0]                                }
                    {   add K, K, r10                           ;   vlmaccr K[0]                                }
                    {   add K, K, r10                           ;   vlmaccr K[0]                                }
                    {   add K, K, r10                           ;   vlmaccr K[0]                                }
                    {   add K, K, r10                           ;   vlmaccr K[0]                                }
                    {   sub patch_maccs, patch_maccs, 1         ;   vlmaccr K[0]                                }
                    
                    {   add K, K, r10                           ;   bt patch_maccs, .L_patch_row_macc_loop      }
                    //End of patch row macc loop
                
                {   add X, X, patch_x_row_incr              ;   sub patch_rows, patch_rows, 1               }
                {   add K, K, patch_k_row_incr              ;   bt patch_rows, .L_patch_row_loop            }
                //End of patch_row_loop
                #undef patch_maccs

            #undef patch_k_row_incr
            #undef patch_x_row_incr

            //Set accumulator mode to 16-bit and saturate accumulators to 16 bits
            {   shl r11, r10, 3                                 ;   ldw r9, sp[STACK_SHIFTS_NEXT]                   }
            {                                                   ;   vsetc r11                                       }
            {   add r9, r9, r10                                 ;   vlsat r9[0]                                     }
            {                                                   ;   stw r9, sp[STACK_SHIFTS_NEXT]                   }

            //Apply scales
            {   mkmsk r9, 16                                    ;   ldw r8, sp[STACK_SCALES_NEXT]                   }
            {   add r8, r8, r10                                 ;   vlmul r8[0]                                     }
            {   ldc r10, 16                                     ;   stw r8, sp[STACK_SCALES_NEXT]                   }

            //Change depth to 8-bit
            {                                                   ;   vdepth8                                         }

            //Store results in Y and advance K to next channel output group, and
            //Decrement channel output groups left and branch if there are more
            vstrpv Y[0], r9
            {   add Y, Y, r10                                   ;   ldw r11, sp[STACK_COUT_GRPS_LEFT]               }
            {   sub r11, r11, 1                                 ;   ldw r10, sp[STACK_COUT_GRP_INCR_K]              }
            {                                                   ;   stw r11, sp[STACK_COUT_GRPS_LEFT]               }
            {   add K, K, r10                                   ;   bt r11, .L_cout_grp_loop                        }

            //End of output channel group loop
            #undef patch_rows
            #undef X

        //Advance our patch start pointer by one pixel
        {                                                   ;   ldw r11, sp[STACK_OUT_X_PXL_INCR]               }
        {                                                   ;   ldw r2, sp[STACK_X_START]                       }
        {   add r2, r2, r11                                 ;                                                   }
        {                                                   ;   stw r2, sp[STACK_X_START]                       }

        //Decrement output cols and loop if more
        {                                                   ;   ldw r11, sp[STACK_OUT_COLS_LEFT]                }
        {   sub r11, r11, 1                                 ;                                                   }
        {                                                   ;   stw r11, sp[STACK_OUT_COLS_LEFT]                }
        {                                                   ;   bt r11, .L_out_cols_loop                        }

        //End output columns loop
        #undef K

    //Advance the output rows of X and Y
    //  (r2 still contains X pointer)
    {                                                   ;   ldw r11, sp[STACK_OUT_X_ROW_INCR]               }
    {                                                   ;   ldw r10, sp[STACK_OUT_Y_ROW_INCR]               }
    
    {   add r2, r2, r11                                 ;                                                   }
    {   add Y, Y, r10                                   ;                                                   } 

    {                                                   ;   stw r2, sp[STACK_X_START]                       }
    
    {                                                   ;   ldw r11, sp[STACK_OUT_ROWS_LEFT]                }
    {   sub r11, r11, 1                                 ;                                                   }
    {                                                   ;   stw r11, sp[STACK_OUT_ROWS_LEFT]                }
    {                                                   ;   bt r11, .L_out_rows_loop                        }

    //End output rows loop



.Lfunc_end:
    {                                           ;   ldw r10, sp[STACK_R10]              }
    ldd r8, r9, sp[STACK_R8_R9_DBL]
    ldd r6, r7, sp[STACK_R6_R7_DBL]
    ldd r4, r5, sp[STACK_R4_R5_DBL]
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



