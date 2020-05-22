
#if defined(__XS3A__)

#include "nn_config.h"
#include "window_op_plan.h"
#include "asm_constants.h"

/*  
void conv2d_1x1(
    nn_image_t* Y,
    const nn_image_t* X,
    const nn_tensor_t* K,
    const nn_bso_block_t* BSO,
    const nn_conv2d_1x1_plan_t* plan,
    const nn_conv2d_1x1_job_t* job);


Optimizing assumptions:
    - input window is 1x1
    - input window stride is 1x1 (So, input image is just a long row of pixels)
    - output stride is 1x1 (So, output image is also just a long row of pixels)

*/

#define FUNCTION_NAME conv2d_1x1

#define NSTACKWORDS  36
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 4
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME



#define PLAN_CIN                0
#define PLAN_COUT               1

#define JOB_START_STRIDE_X     0
#define JOB_START_STRIDE_Y     1
#define JOB_START_STRIDE_K     2
#define JOB_START_STRIDE_BSO   3
#define JOB_PIXELS             4
#define JOB_CHANNELS           5

#define STACK_VEC_TMP       (NSTACKWORDS-8)
#define STACK_VEC_TMP2      (NSTACKWORDS-16)

#define STACK_PLAN          (NSTACKWORDS+1)
#define STACK_JOB           (NSTACKWORDS+2)

#define STACK_CP            1
#define STACK_X             9
#define STACK_Y_NEXT_COG    10
#define STACK_BSO           11
#define STACK_PIX_COUNT     12
#define STACK_JOB_OUT_TAIL  13
#define STACK_TMP           14
#define STACK_C_OUT         15


#define Y               r0
#define X               r1
#define K               r2
#define BSO             r3
#define C_in            r4
#define cin_tail        r5
#define _32             r6
#define cog_left        r7
#define cig_left        r8
#define pix_left        r9
#define K_cig_stride    r10

// This macro is just a hint to you that the register R is being used for an off-label purpose
// (e.g.   Q(cig_left) means that you, the reader, should NOT interpret its value to be the number
//  of channel input groups left.)
#define Q(R)    R

.align 16


FUNCTION_NAME:
    dualentsp NSTACKWORDS
    std r4, r5, sp[1]
    std r6, r7, sp[2]
    std r8, r9, sp[3]
    {   ldc _32, 32                             ;   stw r10, sp[8]                          }

    //Change constant pool pointer to refer to the constant VPU vects needed here
    ldaw r11, cp[vpu_vects]
    {   ldaw r11, cp[0]                         ;   set cp, r11                             }
    {                                           ;   stw r11, sp[STACK_CP]                   }

#define plan            cig_left
#define job             pix_left

    {   shl r11, _32, 4                         ;   ldw plan, sp[STACK_PLAN]                }
    {                                           ;   vsetc r11                               }
    {                                           ;   ldw job, sp[STACK_JOB]                  }
    {                                           ;   ldw C_in, plan[PLAN_CIN]                }
    {   mov cin_tail, C_in                      ;   ldw r11, plan[PLAN_COUT]                }
    {   zext cin_tail, 5                        ;   stw r11, sp[STACK_C_OUT]                }
    {                                           ;   ldw r11, job[JOB_START_STRIDE_X]        }
    {   add X, X, r11                           ;   ldw r11, job[JOB_START_STRIDE_Y]        }
    {   add Y, Y, r11                           ;   ldw r11, job[JOB_START_STRIDE_K]        }
    {   add K, K, r11                           ;   ldw r11, job[JOB_START_STRIDE_BSO]      }
    {   add BSO, BSO, r11                       ;   stw X, sp[STACK_X]                      }
    {                                           ;   stw Y, sp[STACK_Y_NEXT_COG]             }
    {                                           ;   stw BSO, sp[STACK_BSO]                  }
    {                                           ;   ldw r11, job[JOB_PIXELS]                }
    {                                           ;   stw r11, sp[STACK_PIX_COUNT]            }

    {   ldaw r11, sp[STACK_VEC_TMP2]            ;   ldw r1, job[JOB_CHANNELS]               }
    {   shr cog_left, r1, 4                     ;   vclrdr                                  }
    {   zext r1, 4                              ;   vstr r11[0]                             }
    {   ldaw r11, sp[STACK_VEC_TMP]             ;   stw r1, sp[STACK_JOB_OUT_TAIL]          }
    {                                           ;   vstr r11[0]                             }

#undef plan
#undef job

    //If there is only a tail, skip the main body
    {                                           ;   bf cog_left, .L_cog_tail                }

    //Set up K_cig_stride, which should be (15 * C_in) + 32
    //  = 16 * C_in - C_in + 32
    //  = (C_in << 4) + (32 - C_in)
    {   sub K_cig_stride, _32, C_in             ;   shl r11, C_in, 4                        }
    {   add K_cig_stride, r11, K_cig_stride     ;                                           }

    // Loop over (full) output channel groups
    .L_cog_loop:
        {   add K, K, K_cig_stride              ;   ldw pix_left, sp[STACK_PIX_COUNT]       }
        {   sub K, K, _32                       ;   sub cog_left, cog_left, 1               }
        {   shr r11, _32, 1                     ;   ldw Y, sp[STACK_Y_NEXT_COG]             }
        {   add r11, Y, r11                     ;   ldw X, sp[STACK_X]                      }
        {                                       ;   stw r11, sp[STACK_Y_NEXT_COG]           }

        // Loop over output pixels
        .L_pix_loop:    
#if CONFIG_SYMMETRIC_SATURATION_conv2d_1x1
            {                                           ;   ldw BSO, sp[STACK_BSO]                  }
#else
            {   ldaw r11, cp[VPU_VEC_0x80]              ;   ldw BSO, sp[STACK_BSO]                  }
            {   mkmsk r11, 16                           ;   vldr r11[0]                             }
            vstrpv Y[0], r11
#endif

            {   add r11, BSO, _32                       ;   vldd BSO[0]                             }
            {   shr cig_left, C_in, 5                   ;   vldr r11[0]                             }
            {   add BSO, r11, _32                       ;   bf cig_left, .L_cig_loop_end            }
            {   ldc r11, 0                              ;                                           }
            .L_cig_loop:
                {   add X, X, _32                           ;   vldc X[0]                               }

                .L_cig_tail_start:
                {   sub K, K, C_in                          ;   vlmaccr K[0]                            }
                {   sub K, K, C_in                          ;   vlmaccr K[0]                            }
                {   sub K, K, C_in                          ;   vlmaccr K[0]                            }
                {   sub K, K, C_in                          ;   vlmaccr K[0]                            }
                {   sub K, K, C_in                          ;   vlmaccr K[0]                            }
                {   sub K, K, C_in                          ;   vlmaccr K[0]                            }
                {   sub K, K, C_in                          ;   vlmaccr K[0]                            }
                {   sub K, K, C_in                          ;   vlmaccr K[0]                            }
                {   sub K, K, C_in                          ;   vlmaccr K[0]                            }
                {   sub K, K, C_in                          ;   vlmaccr K[0]                            }
                {   sub K, K, C_in                          ;   vlmaccr K[0]                            }
                {   sub K, K, C_in                          ;   vlmaccr K[0]                            }
                {   sub K, K, C_in                          ;   vlmaccr K[0]                            }
                {   sub K, K, C_in                          ;   vlmaccr K[0]                            }
                {   sub K, K, C_in                          ;   vlmaccr K[0]                            }
                {   sub cig_left, cig_left, 1               ;   vlmaccr K[0]                            }    
                {   add K, K, K_cig_stride                  ;   bt cig_left, .L_cig_loop                }
                
                //If r11 is nonzero here, then we've already dealt with the tail.
                //This prevents us from looping over the tail infinitely
                {                                           ;   bt r11, .L_get_res                      }
            .L_cig_loop_end:

            {   add K, K, cin_tail                      ;   bf cin_tail, .L_get_res                 }
            {   sub K, K, _32                           ;   sub cin_tail, _32, cin_tail             }
            {   ldaw r11, sp[STACK_VEC_TMP]             ;   vldc X[0]                               }
            {   sub r11, r11, cin_tail                  ;   vstc r11[0]                             }
            {   sub cin_tail, _32, cin_tail             ;   vldc r11[0]                             }
            {   ldc cig_left, 1                         ;   bu .L_cig_tail_start                    }

            .L_get_res:
            
#if CONFIG_SYMMETRIC_SATURATION_conv2d_1x1
            {   shl r11, _32, 3                         ;   sub pix_left, pix_left, 1               }
            {   ldaw r11, sp[STACK_VEC_TMP]             ;   vsetc r11                               }
            {   add BSO, BSO, _32                       ;   vlsat BSO[0]                            }
            {   add X, X, cin_tail                      ;   vstr r11[0]                             }   
            {   add BSO, BSO, _32                       ;   vldc BSO[0]                             }
            {                                           ;   vclrdr                                  }
            {   shl r11, _32, 4                         ;   vlmacc r11[0]                           }
            {   add BSO, BSO, _32                       ;   vldc BSO[0]                             }
            {   add BSO, BSO, _32                       ;   vlmacc BSO[0]                           }
            {   mkmsk r11, 16                           ;   vsetc r11                               }
            {   add BSO, BSO, _32                       ;   vlsat BSO[0]                            }
            vstrpv Y[0], r11
            {   sub K, K, C_in                          ;   ldw Q(cig_left), sp[STACK_C_OUT]        }
            {   add Y, Y, Q(cig_left)                   ;   bt pix_left, .L_pix_loop                }

#else
            {   shl r11, _32, 3                         ;   sub pix_left, pix_left, 1               }
            {   ldaw r11, sp[STACK_VEC_TMP]             ;   vsetc r11    /* 16-bit mode */          }
            {   add BSO, BSO, _32                       ;   vlsat BSO[0]                            }
            {   add X, X, cin_tail                      ;   vstr r11[0]                             }   
            {   add BSO, BSO, _32                       ;   vldc BSO[0]                             }
            {                                           ;   vclrdr                                  }
            {                                           ;   vlmacc r11[0]                           }
            {   add BSO, BSO, _32                       ;   vldc BSO[0]                             }
            {   add BSO, BSO, _32                       ;   vlmacc BSO[0]                           }

            {   add BSO, BSO, _32                       ;   vlsat BSO[0]                            }
            {   ldaw r11, cp[VPU_VEC_0x007F]            ;   vstr r11[0]                             }
            {   ldaw r11, sp[STACK_TMP]                 ;   vladd r11[0]                            }
            {   mkmsk Q(cig_left), 4                    ;   vdepth1                                 }
            vstrpv r11[0], Q(cig_left)
            {   ldc Q(cig_left), 0                      ;                                           }
            {   ldaw r11, sp[STACK_VEC_TMP]             ;   sub Q(cig_left), Q(cig_left), 8         }
            vlashr r11[0], Q(cig_left)
            {   mkmsk r11, 16                           ;   ldw Q(cig_left), sp[STACK_TMP]          }
            {   andnot r11, Q(cig_left)                 ;   vdepth8                                 }
            vstrpv Y[0], r11
            {   shl r11, _32, 4                         ;   ldw Q(cig_left), sp[STACK_C_OUT]        }
            {   sub K, K, C_in                          ;   vsetc r11   /* 8-bit mode */            }
            {   add Y, Y, Q(cig_left)                   ;                                           }
            bt pix_left, .L_pix_loop

#endif
        {   add K, K, C_in                          ;   stw BSO, sp[STACK_BSO]                  }
        bt cog_left, .L_cog_loop



#undef cog_left
#undef pix_left


#define macc_hop        r7
#define reset_offset    r9

    .L_cog_tail:
        // - For the tail, the K cig stride is  (C_out_tail - 1) * C_in + VPU_INT8_EPV
        //   C_out_tail =  C_out % 16
        // - K is currently pointing at the beginning of the first row of the output tail. It needs
        //   to be pointing at the beginning of the last row ->   K = K + K_cig_stride - 32
        // - macc_hop is the argument to the BRU instruction used below. It skips some number of
        //   VLMACCRs so that it only does C_out_tail VLMACCRs per input group
        //   macc_hop = 12 - C_out_tail
        // - After C_out_tail VLMACCRs, the accumulators for the tail will occupy the first indices
        //   in vD:vR. Because of the way VLMACCR rotates the accumulators, they need to be at the 
        //   end. We can reposition them by doing a store and then an offset load. That's what
        //   reset_offset is.   reset_offset = 32 - 2*C_out_tail
        {   sub K, K, _32                           ;   ldw r11, sp[STACK_JOB_OUT_TAIL]         }
        {   mkmsk Q(cig_left), r11                  ;   ldc Q(macc_hop), 12                     }
        bf r11, .Lfunc_end /* exit early if there's no output tail */
        {   sub macc_hop, Q(macc_hop), r11          ;   sub Q(reset_offset), _32, r11           }
        mul K_cig_stride, r11, C_in
        {   sub reset_offset, Q(reset_offset), r11  ;   sub K_cig_stride, K_cig_stride, C_in    }
        {   add K_cig_stride, K_cig_stride, _32     ;   stw Q(cig_left), sp[STACK_JOB_OUT_TAIL] }
        {   add K, K, K_cig_stride                  ;   ldw Q(cig_left), sp[STACK_PIX_COUNT]    }
        {                                           ;   ldw X, sp[STACK_X]                      }
        {                                           ;   ldw Y, sp[STACK_Y_NEXT_COG]             }

        //Loop expects Q(cig_left) to contain the remaining pixel count
        .L_pix_loop2:
#if CONFIG_SYMMETRIC_SATURATION_conv2d_1x1
            {                                           ;   ldw BSO, sp[STACK_BSO]                  }
#else
            {   ldaw r11, cp[VPU_VEC_0x80]              ;   ldw BSO, sp[STACK_BSO]                  }
            {                                           ;   vldr r11[0]                             }
            {                                           ;   ldw r11, sp[STACK_JOB_OUT_TAIL]         }
            vstrpv Y[0], r11
#endif
            {                                           ;   stw Y, sp[STACK_Y_NEXT_COG]             }
            {   add r11, BSO, _32                       ;   vldd BSO[0]                             }
            {   sub Q(cig_left), Q(cig_left), 1         ;   vldr r11[0]                             }
            {   shr cig_left, C_in, 5                   ;   stw Q(cig_left), sp[STACK_PIX_COUNT]    }
            {   ldc Q(Y), 0                             ;                                           }
            {   add BSO, r11, _32                       ;   bf cig_left, .L_cig_loop_end2           }
            .L_cig_loop2:
                {   ldaw r11, sp[STACK_VEC_TMP]             ;   vldc X[0]                               }
                {   add X, X, _32                           ;                                           }

                .L_cig_tail_start2:
                {   sub r11, r11, reset_offset              ;   vstr r11[0]                             }
                {   ldaw r11, sp[STACK_VEC_TMP]             ;   vldr r11[0]                             }
                {   sub r11, r11, reset_offset              ;   vstd r11[0]                             }
                {   sub cig_left, cig_left, 1               ;   vldd r11[0]                             }

                {                                           ;   bru macc_hop                            }
                {   sub K, K, C_in                          ;   vlmaccr K[0]                            }
                {   sub K, K, C_in                          ;   vlmaccr K[0]                            }
                {   sub K, K, C_in                          ;   vlmaccr K[0]                            }
                {   sub K, K, C_in                          ;   vlmaccr K[0]                            }
                {   sub K, K, C_in                          ;   vlmaccr K[0]                            }
                {   sub K, K, C_in                          ;   vlmaccr K[0]                            }
                {   sub K, K, C_in                          ;   vlmaccr K[0]                            }
                {   sub K, K, C_in                          ;   vlmaccr K[0]                            }
                {   sub K, K, C_in                          ;   vlmaccr K[0]                            }
                {   sub K, K, C_in                          ;   vlmaccr K[0]                            }
                {   sub K, K, C_in                          ;   vlmaccr K[0]                            }
                {   add K, K, K_cig_stride                  ;   vlmaccr K[0]                            }

                {                                           ;   bt cig_left, .L_cig_loop2               }

                //If Q(Y) is nonzero here, then we've already dealt with the tail.
                //This prevents us from looping over the tail infinitely
                {                                           ;   bt Q(Y), .L_get_res2                    }
            .L_cig_loop_end2:

            {   add K, K, cin_tail                      ;   bf cin_tail, .L_get_res2                }
            {   sub K, K, _32                           ;   sub cin_tail, _32, cin_tail             }
            {   ldaw r11, sp[STACK_VEC_TMP]             ;   vldc X[0]                               }
            {   sub r11, r11, cin_tail                  ;   vstc r11[0]                             }
            {   ldaw r11, sp[STACK_VEC_TMP]             ;   vldc r11[0]                             }

            {   sub cin_tail, _32, cin_tail             ;   ldc Q(Y), 1                             }
            {   ldc cig_left, 1                         ;   bu .L_cig_tail_start2                   }

            .L_get_res2:
#if CONFIG_SYMMETRIC_SATURATION_conv2d_1x1
            {   shl r11, _32, 3                         ;   ldw Y, sp[STACK_Y_NEXT_COG]             }     
            {                                           ;   vsetc r11                               }
            {   ldaw r11, sp[STACK_VEC_TMP]             ;   vlsat BSO[0]                            }
            {   add BSO, BSO, _32                       ;   vstr r11[0]                             }   
            {   add BSO, BSO, _32                       ;   vldc BSO[0]                             }
            {   add X, X, cin_tail                      ;   vclrdr                                  }
            {   shl r11, _32, 4                         ;   vlmacc r11[0]                           }
            {   add BSO, BSO, _32                       ;   vldc BSO[0]                             }
            {   add BSO, BSO, _32                       ;   vlmacc BSO[0]                           }
            {                                           ;   vsetc r11                               }
            {                                           ;   vlsat BSO[0]                            }
            {                                           ;   ldw r11, sp[STACK_JOB_OUT_TAIL]         }
            vstrpv Y[0], r11
            {                                           ;   ldw r8, sp[STACK_C_OUT]                 }
            {   add Y, Y, r8                            ;   ldw r8, sp[STACK_PIX_COUNT]             }
            {   sub K, K, C_in                          ;   bt r8, .L_pix_loop2                     }
#else
            {   shl r11, _32, 3                         ;   ldw Y, sp[STACK_Y_NEXT_COG]             }     
            {                                           ;   vsetc r11                               }
            {   ldaw r11, sp[STACK_VEC_TMP]             ;   vlsat BSO[0]                            }
            {   add BSO, BSO, _32                       ;   vstr r11[0]                             }   
            {   add BSO, BSO, _32                       ;   vldc BSO[0]                             }
            {   add X, X, cin_tail                      ;   vclrdr                                  }
            {                                           ;   vlmacc r11[0]                           }
            {   add BSO, BSO, _32                       ;   vldc BSO[0]                             }
            {   add BSO, BSO, _32                       ;   vlmacc BSO[0]                           }

            {   add BSO, BSO, _32                       ;   vlsat BSO[0]                            }
            {   ldaw r11, cp[VPU_VEC_0x007F]            ;   vstr r11[0]                             }
            {   ldaw r11, sp[STACK_TMP]                 ;   vladd r11[0]                            }
            {   mkmsk Q(cig_left), 4                    ;   vdepth1                                 }
            vstrpv r11[0], Q(cig_left)
            {   ldc Q(cig_left), 0                      ;                                           }
            {   ldaw r11, sp[STACK_VEC_TMP]             ;   sub Q(cig_left), Q(cig_left), 8         }
            vlashr r11[0], Q(cig_left)
            {                                           ;   ldw Q(cig_left), sp[STACK_TMP]          }
            {                                           ;   ldw r11, sp[STACK_JOB_OUT_TAIL]         }
            {   andnot r11, Q(cig_left)                 ;   vdepth8                                 }
            vstrpv Y[0], r11
            {   shl r11, _32, 4                         ;   ldw Q(cig_left), sp[STACK_C_OUT]        }
            {   sub K, K, C_in                          ;   vsetc r11   /* 8-bit mode */            }
            {   add Y, Y, Q(cig_left)                   ;   ldw Q(cig_left), sp[STACK_PIX_COUNT]    }
            bt Q(cig_left), .L_pix_loop2

#endif


.L_img_end:

.Lfunc_end:
    //Restore the original constant pool pointer
    {                                           ;   ldw r11, sp[STACK_CP]                   }
    {                                           ;   set cp, r11                             }

    {                                           ;   ldw r10, sp[8]                          }
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





            // Move 32-bit accumulators into memory:

            // {   shl r11, _32, 3                         ;   ldw r10, sp[STACK_0x11111111]           }
            // {   ldaw r8, sp[STACK_VEC_A]                ;   vsetc r11                               }
            // {   ldaw r9, sp[STACK_VEC_B]                ;   vstr r8[0]                              }
            // vlashr r8[0], tmp
            // {   ldap r11, .L_vec_zero                   ;   vstd r9[0]                              }
            // vstrpv r9[0], r10
            // {   ldap r11, .L_vec_0x0008                 ;   vldr r11[0]                             }
            // {   shl r10, r10, 1                         ;   vlsat r11[0]                            }
            // vstrpv r8[0], r10
            // {   shl r11, _32, 4                         ;   vclrdr                                  }
            // {   ldap r11, .L_vec_0x01                   ;   vsetc r11                               }
            // {   ldaw r11, sp[STACK_VEC_32BIT]           ;   vldc r11[0]                             }
            // {                                           ;   vlmacc r8[0]                            }
            // {   ldaw r8, sp[STACK_VEC_A + 4]            ;   vstr r11[0]                             }
            // {                                           ;   vclrdr                                  }
            // {   ldaw r11, sp[STACK_VEC_32BIT + 8]       ;   vlmacc r8[0]                            }
            // {   shl r8, r10, 2                          ;   vstr r11[0]                             }
            // {   ldap r11, .L_vec_0x7F                   ;   vclrdr                                  }
            // {   add r10, r8, r10                        ;   vldc r9[0]                              }
            // {   ldaw r9, sp[STACK_VEC_B + 4]            ;   vlmacc r11[0]                           }
            // {   ldap r11, .L_vec_0x02                   ;   vlmacc r11[0]                           }
            // {   ldaw r11, sp[STACK_VEC_32BIT]           ;   vlmacc r11[0]                           }
            // vstrpv r11[0], r10
            // {                                           ;   vclrdr                                  }
            // {   ldap r11, .L_vec_0x7F                   ;   vldc r9[0]                              }
            // {                                           ;   vlmacc r11[0]                           }
            // {   ldap r11, .L_vec_0x02                   ;   vlmacc r11[0]                           }
            // {   ldaw r11, sp[STACK_VEC_32BIT+8]         ;   vlmacc r11[0]                           }
            // vstrpv r11[0], r10

