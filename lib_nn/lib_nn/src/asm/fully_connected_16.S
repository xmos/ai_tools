
#if defined(__XS3A__)


/*  
void fully_connected_16_asm(
    int16_t* Y,
    const int8_t* W, 
    const int8_t* X, 
    const data16_t* BSS,
    const nn_fully_connected_16_plan_t* plan);
*/

#define FUNCTION_NAME fully_connected_16_asm

#define NSTACKWORDS  30
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 4
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

#define PLAN_C_IN               0
#define PLAN_C_OUT              1
#define PLAN_CIG_END            2
#define PLAN_TAIL_STRAT         3

#define STACK_ARG_PLAN  (NSTACKWORDS+1)
#define STACK_X         8
#define STACK_CIG       9
#define STACK_CIN_TAIL  10
#define STACK_C_OUT     11

#define STACK_X_TAIL    (NSTACKWORDS-8)
#define STACK_TMP_VEC   (NSTACKWORDS-16)

#define Y               r0
#define W               r1
#define X               r2
  #define plan            X
#define BSS             r3
#define C_in            r4
#define C_out           r5
#define _32             r6
#define cig_end         r7
#define cig_left        r8
#define x_tail_adj      r9

.align 4

FUNCTION_NAME:
    dualentsp NSTACKWORDS
    std r4, r5, sp[0]
    std r6, r7, sp[1]
    std r8, r9, sp[2]
    {   ldc _32, 32                             ;   stw r10, sp[6]                      }
    {   mov r8, X                               ;   stw X, sp[STACK_X]                  }
    {                                           ;   ldw plan, sp[STACK_ARG_PLAN]        }
    {   ldaw r10, sp[STACK_X_TAIL]              ;   ldw C_in, plan[PLAN_C_IN]           }
    {   shr r11, C_in, 5                        ;   ldw C_out, plan[PLAN_C_OUT]         }
    {   shl r11, r11, 5                         ;   ldw cig_end, plan[PLAN_CIG_END]     }

    // Save the tail of X to STACK_TMP_X
    {   add r11, r8, r11                        ;   vclrdr                              }
    {   mov r8, C_in                            ;   vstr r10[0]                         }
    {   zext r8, 5                              ;   vldr r11[0]                         }
    {   mkmsk r8, r8                            ;                                       }
    vstrpv r10[0], r8

    {   neg x_tail_adj, C_in                    ;   mov r11, C_in                       }
    {   zext x_tail_adj, 5                      ;   shr r10, C_in, 5                    }
    {   zext r11, 5                             ;   stw r10, sp[STACK_CIG]              }
    {   shr C_out, C_out, 4                     ;   stw r11, sp[STACK_CIN_TAIL]         }

    {                                           ;   bf C_out, .L_cog_loop_end           }

    .align 4
    .L_cog_loop:
        {   add r11, BSS, _32                       ;   vldd BSS[0]                         }   //load high biases
        {   add BSS, r11, _32                       ;   vldr r11[0]                         }   //load low biases
        {   shl r11, _32, 4                         ;                                       }
        {   add W, W, cig_end                       ;   vsetc r11                           }   //8 bit mode
        {   shr cig_left, C_in, 5                   ;   ldw X, sp[STACK_X]                  }
        {   sub W, W, _32                           ;   bf cig_left, .L_cig_loop_tail       }

        .align 4
        .L_cig_loop:
            {   sub cig_left, cig_left, 1           ;   vldc X[0]                           }
        
#define MACCR_BLOCK(INCRR)                                                                              \
            {   sub W, W, C_in                          ;   vlmaccr W[0]                        };      \
            {   sub W, W, C_in                          ;   vlmaccr W[0]                        };      \
            {   sub W, W, C_in                          ;   vlmaccr W[0]                        };      \
            {   sub W, W, C_in                          ;   vlmaccr W[0]                        };      \
            {   sub W, W, C_in                          ;   vlmaccr W[0]                        };      \
            {   sub W, W, C_in                          ;   vlmaccr W[0]                        };      \
            {   sub W, W, C_in                          ;   vlmaccr W[0]                        };      \
            {   sub W, W, C_in                          ;   vlmaccr W[0]                        };      \
            {   sub W, W, C_in                          ;   vlmaccr W[0]                        };      \
            {   sub W, W, C_in                          ;   vlmaccr W[0]                        };      \
            {   sub W, W, C_in                          ;   vlmaccr W[0]                        };      \
            {   sub W, W, C_in                          ;   vlmaccr W[0]                        };      \
            {   sub W, W, C_in                          ;   vlmaccr W[0]                        };      \
            {   sub W, W, C_in                          ;   vlmaccr W[0]                        };      \
            {   sub W, W, C_in                          ;   vlmaccr W[0]                        };      \
            {   add W, W, INCRR                         ;   vlmaccr W[0]                        }

            MACCR_BLOCK(cig_end);

            {   add X, X, _32                           ;   bt cig_left, .L_cig_loop            }

            .L_cig_loop_tail:
            {   shl r11, _32, 3                         ;   bf x_tail_adj, .L_cig_loop_end      }

            {   ldaw r8, sp[STACK_X_TAIL]               ;                                       }
            {                                           ;   vldc r8[0]                          }

            // now just do the VLMACCRs normally
            MACCR_BLOCK(cig_end);

        .L_cig_loop_end:
        
        {   sub W, W, x_tail_adj                    ;   vsetc r11                           }   //16 bit mode
        {   add BSS, BSS, _32                       ;   vlsat BSS[0]                        }
        {   ldaw r11, sp[STACK_TMP_VEC]             ;   vldc BSS[0]                         }
        {   add BSS, BSS, _32                       ;   vstr r11[0]                         }
        {                                           ;   vclrdr                              }
        {                                           ;   vlmacc r11[0]                       }
        {   add BSS, BSS, _32                       ;   vlsat BSS[0]                        }
        {   sub C_out, C_out, 1                     ;   vstr Y[0]                           }
        {   add Y, Y, _32                           ;   bt C_out, .L_cog_loop               }

    .L_cog_loop_end:



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////// C_out tail starts here
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



        {                                           ;   ldw plan, sp[STACK_ARG_PLAN]        }
        {                                           ;   ldw C_out, plan[PLAN_C_OUT]         }
        {   zext C_out, 4                           ;   ldw r11, plan[PLAN_TAIL_STRAT]      }
        bf C_out, .Lfunc_end
        {                                           ;   bru r11                             }
        {                                           ;   bu .L_tail_strat_default            }

/////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////// TAIL STRATEGY DEFAULT

#define needs_maccr     r5
#define maccr_jump      r9
#define acc_offset      r10
    .L_tail_strat_default:
        {   mov r11, BSS                            ;                                       }
        {   add r11, r11, _32                       ;   vldd r11[0]                         }   //load high biases
        {   add BSS, r11, _32                       ;   vldr r11[0]                         }   //load low biases
        {   shl r11, _32, 4                         ;   ldw X, sp[STACK_X]                  }   // Reset X
        {   sub r11, C_out, 1                       ;   vsetc r11                           }   //8 bit mode
        mul r11, r11, C_in
        {   add W, W, r11                           ;                                       }   // W = W + (C_out-1) * C_in
        {   add cig_end, r11, _32                   ;   ldc r11, 16                         }   //cig_end = (C_out-1) * C_in + 32
        {   sub maccr_jump, r11, C_out              ;   ldc r11, 1                          }   // r9 = 16 - C_out
        {   shl acc_offset, maccr_jump, 1           ;                                       }   // r6 = 2 * (16 - C_out)
        {   and needs_maccr, needs_maccr, r11       ;                                       }
        {   sub acc_offset, acc_offset, needs_maccr ;   shr cig_left, C_in, 5               }
        {   sub acc_offset, acc_offset, needs_maccr ;   eq needs_maccr, needs_maccr, 0      }
        
        {                                           ;   bf cig_left, .L_tail_loop_cig_tail  }

        .align 4
        .L_tail_loop:
            {   ldaw r11, sp[STACK_TMP_VEC]             ;   vldc X[0]                           }

            {   sub r11, r11, acc_offset                ;   vstr r11[0]                         }
            {   ldaw r11, sp[STACK_TMP_VEC]             ;   vldr r11[0]                         }
            {   sub r11, r11, acc_offset                ;   vstd r11[0]                         }
            {                                           ;   vldd r11[0]                         }
            
            {                                           ;   bru needs_maccr                     } // r5 should be (C_out % 2)
            {                                           ;   vlmaccr W[0]                        }
            {   sub cig_left, cig_left, 1               ;   bru maccr_jump                      }

            MACCR_BLOCK(cig_end);

            {   add X, X, _32                           ;   bt cig_left, .L_tail_loop           }

            .L_tail_loop_cig_tail:

            {   ldaw r8, sp[STACK_X_TAIL]               ;   bf x_tail_adj, .L_tail_loop_end     }
            
            {   ldaw r11, sp[STACK_TMP_VEC]             ;   vldc r8[0]                          }
        
            // Re-align accumulators at end of vector registers
            {   sub r11, r11, acc_offset                ;   vstr r11[0]                         }
            {   ldaw r11, sp[STACK_TMP_VEC]             ;   vldr r11[0]                         }
            {   sub r11, r11, acc_offset                ;   vstd r11[0]                         }
            {                                           ;   vldd r11[0]                         }


            // now just do the remaining VLMACCRs
            {                                           ;   bru needs_maccr                     } // r5 should be (C_out % 2)
            {                                           ;   vlmaccr W[0]                        }
            {                                           ;   bru maccr_jump                      }

            MACCR_BLOCK(cig_end);
            
        .L_tail_loop_end:
        {   shl r11, _32, 3                         ;   ldw plan, sp[STACK_ARG_PLAN]        }
        {   add r8, BSS, _32                        ;   ldw r5, plan[PLAN_C_OUT]            }
        {   zext r5, 4                              ;   vsetc r11                           }
        {   add BSS, BSS, _32                       ;   vlsat BSS[0]                        }
        {   ldaw r11, sp[STACK_TMP_VEC]             ;   vldc BSS[0]                         }
        {   add BSS, BSS, _32                       ;   vstr r11[0]                         }
        {                                           ;   vclrdr                              }
        {   shl r5, r5, 1                           ;   vlmacc r11[0]                       }
        {   mkmsk r5, r5                            ;   vlsat BSS[0]                        }

        vstrpv Y[0], r5
        bu .Lfunc_end

#undef needs_maccr     
#undef maccr_jump      
#undef acc_offset   

.Lfunc_end:
    {                                            ;   ldw r10, sp[6]                      }
    ldd r8, r9, sp[2]
    ldd r6, r7, sp[1]
    ldd r4, r5, sp[0]
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


