
#if defined(__XS3A__)


/*  
void fully_connected_16_asm(
    int16_t* Y,
    const int8_t* W, 
    const int8_t* X, 
    const data16_t* BSS,
    const unsigned C_in, 
    const unsigned C_out);
*/

#define FUNCTION_NAME fully_connected_16_asm

#define NSTACKWORDS  30
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 4
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME



#define STACK_CIN       (NSTACKWORDS+1)
#define STACK_COUT      (NSTACKWORDS+2)
#define STACK_X         8
#define STACK_CIG       9
#define STACK_CIN_TAIL  10
#define STACK_COG_START 11

#define STACK_TMP_X     (NSTACKWORDS-8)
#define STACK_TMP_VR    (NSTACKWORDS-16)

#define Y               r0
#define W               r1
#define X               r2
#define BSS             r3
#define C_in            r4
#define C_out           r5

#define _32             r6
#define cig_end         r7
#define cig             r8
#define x_tail_incr     r9             
#define x_tail_mask     r10

.align 4

FUNCTION_NAME:
    dualentsp NSTACKWORDS
    std r4, r5, sp[0]
    std r6, r7, sp[1]
    std r8, r9, sp[2]
    {   ldc r8, 32                              ;   stw r10, sp[6]                      }

    {   ldc r11, 32                             ;   ldw C_in, sp[STACK_CIN]             }
    {   shl r6, C_in, 4                         ;   mkmsk r11, 5                        }
    {   and r11, C_in, r11                      ;   stw X, sp[STACK_X]                  }
    {   sub r6, r6, C_in                        ;   ldw C_out, sp[STACK_COUT]           }  // cog_start = 16*C_in - C_in = 15 * C_in
    {   add x_tail_incr, r6, r11                ;   stw r11, sp[STACK_CIN_TAIL]         }  // x_tail_incr = (15 * C_in) + (C_in % 32)
    {   add cig_end, r6, r8                     ;   shr C_out, C_out, 4                 }  // cig_end = (15 * C_in) + 32
    {   mkmsk x_tail_mask, r11                  ;                                       }  
    {   ldc _32, 32                             ;   stw r6, sp[STACK_COG_START]         }

    {   ldaw r8, sp[STACK_TMP_X]                ;   mov r11, C_in                       }
    {   add X, X, r11                           ;   zext r11, 5                         }
    {   sub r11, X, r11                         ;   vclrdr                              }
    {                                           ;   vstr r8[0]                          }
    {                                           ;   vldr r11[0]                         }
    vstrpv r8[0], x_tail_mask

    {                                           ;   bf C_out, .L_cog_loop_end           }

    .align 4
    .L_cog_loop:
        {   add r11, BSS, _32                       ;   vldd BSS[0]                         }   //load high biases
        {   add BSS, r11, _32                       ;   vldr r11[0]                         }   //load low biases
        {   shl r11, _32, 4                         ;   ldw r8, sp[STACK_COG_START]         }
        {   add W, W, r8                            ;   vsetc r11                           }   //8 bit mode
        {   shr cig, C_in, 5                        ;   ldw X, sp[STACK_X]                  }
        {                                           ;   bf cig, .L_cig_loop_tail            }

        .align 4
        .L_cig_loop:
            {   sub cig, cig, 1                         ;   vldc X[0]                           }
        
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

            {   add X, X, _32                           ;   bt cig, .L_cig_loop                 }

            .L_cig_loop_tail:
            {   shl r11, _32, 3                         ;   bf x_tail_mask, .L_cig_loop_end     }

            // load the tail of X into vC, masking it using x_tail_mask
            {   ldaw r8, sp[STACK_TMP_X]                ;                                       }
            {                                           ;   vldc r8[0]                          }

            // now just do the VLMACCRs normally
            MACCR_BLOCK(x_tail_incr);

        .L_cig_loop_end:
        
        {                                           ;   vsetc r11                           }   //16 bit mode
        {   add BSS, BSS, _32                       ;   vlsat BSS[0]                        }
        {   add BSS, BSS, _32                       ;   vlmul BSS[0]                        }
        {   sub C_out, C_out, 1                     ;   vstr Y[0]                           }
        {   add Y, Y, _32                           ;   bt C_out, .L_cog_loop               }

    .L_cog_loop_end:



        {   mkmsk r11, 4                            ;   ldw C_out, sp[STACK_COUT]           }
        {   and C_out, C_out, r11                   ;                                       }
        {                                           ;   bf C_out, .Lfunc_end                }

        //Load biases
        {                                           ;   mov r11, BSS                        }
        {   add r11, r11, _32                       ;   vldd r11[0]                         }   //load high biases
        {                                           ;   vldr r11[0]                         }   //load low biases
        {   shl r11, _32, 4                         ;   add BSS, r11, _32                   }
        {                                           ;   vsetc r11                           }   //8 bit mode
        {   sub r11, C_out, 1                       ;   ldw X, sp[STACK_X]                  }   // Reset X
        mul r11, r11, C_in
        {   add W, W, r11                           ;                                       }   // W = W + (C_out-1) * C_in
        {   add cig_end, r11, _32                   ;   ldc r11, 16                         }   //cig_end = (C_out-1) * C_in + 32
        {   sub r9, r11, C_out                      ;   ldc r11, 1                          }   // r9 = 16 - C_out
        {   shl r6, r9, 1                           ;   stw C_out, sp[STACK_COUT]           }   // r6 = 2 * (16 - C_out)
        {   and r5, r5, r11                         ;                                       }
        {   sub r6, r6, r5                          ;                                       }
        {   sub r6, r6, r5                          ;   eq r5, r5, 0                        }
        
        {   shr cig, C_in, 5                        ;   ldc r11, 32                         }
        {                                           ;   bf cig, .L_tail_loop_cig_tail       }


        .align 4
        .L_tail_loop:
            {   ldaw r11, sp[STACK_TMP_VR]          ;   vldc X[0]                           }

            // Reposition accumulators at end of vector registers
            {   sub r11, r11, r6                        ;   vstr r11[0]                         }
            {   ldaw r11, sp[STACK_TMP_VR]              ;   vldr r11[0]                         }
            {   sub r11, r11, r6                        ;   vstd r11[0]                         }
            {   ldc r11, 32                             ;   vldd r11[0]                         }
            
            {                                           ;   bru r5                              } // r5 should be (C_out % 2)
            {                                           ;   vlmaccr W[0]                        }
            {   sub cig, cig, 1                         ;   bru r9                              }

            MACCR_BLOCK(cig_end);

            {   add X, X, r11                           ;   bt cig, .L_tail_loop                }

            .L_tail_loop_cig_tail:

            {   ldaw r8, sp[STACK_TMP_X]                ;   bf x_tail_mask, .L_tail_loop_end    }

            // load the tail of X into vC, masking it using x_tail_mask
            
            {   ldaw r11, sp[STACK_TMP_VR]              ;   vldc r8[0]                          }
        
            // Re-align accumulators at end of vector registers
            {   sub r11, r11, r6                        ;   vstr r11[0]                         }
            {   ldaw r11, sp[STACK_TMP_VR]              ;   vldr r11[0]                         }
            {   sub r11, r11, r6                        ;   vstd r11[0]                         }
            {   ldc r11, 32                             ;   vldd r11[0]                         }


            // now just do the remaining VLMACCRs
            {                                           ;   bru r5                              } // r5 should be (C_out % 2)
            {                                           ;   vlmaccr W[0]                        }
            {                                           ;   bru r9                              }

            MACCR_BLOCK(x_tail_incr);
            
        .L_tail_loop_end:
        {   shl r11, r11, 3                         ;   ldw r5, sp[STACK_COUT]              }
        {   shl r5, r5, 1                           ;   vsetc r11                           }
        {   ldc r8, 32                              ;   mkmsk r5, r5                        }
        {   add BSS, BSS, r8                        ;   vlsat BSS[0]                        }
        {                                           ;   vlmul BSS[0]                        }

        vstrpv Y[0], r5


    


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



