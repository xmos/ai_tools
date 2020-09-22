
#if defined(__XS3A__)


/*  
void fully_connected_16(
    int16_t* Y,
    const int8_t* W, 
    const int8_t* X, 
    const nn_bso_block_t* BSO,
    const channel_count_t C_in,
    const channel_count_t out_chan_start,
    const channel_count_t out_chan_count);
*/

#define FUNCTION_NAME fully_connected_16

#ifndef FULLY_CONNECTED_8_SAFETY_CHECKS_ENABLED
  #define FULLY_CONNECTED_8_SAFETY_CHECKS_ENABLED (1)
#endif


#define NSTACKVECS  2
#define NSTACKWORDS  ((NSTACKVECS)*8+10)
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.type FUNCTION_NAME,@function
.align 16
.cc_top FUNCTION_NAME.function,FUNCTION_NAME


#define STACK_CIN           (NSTACKWORDS+1)
#define STACK_OUT_START     (NSTACKWORDS+2)
#define STACK_OUT_COUNT     (NSTACKWORDS+3)


#define STACK_X             1
#define STACK_ORIG_CP       8
#define STACK_ORIG_DP       9

#define STACK_X_TAIL    (NSTACKWORDS-8)
#define STACK_VEC_TMP   (NSTACKWORDS-16)

#define Y               r0
#define W               r1
#define X               r2
#define cig_left        r3
#define C_in            r4
#define _32             r5
#define cig_end         r6
#define cog_left        r7
#define maccr_jump      r8
#define tmp             r10

#define BSO             dp
#define acc_reset_ptr   cp



#define Q(R)  R


FUNCTION_NAME:
    dualentsp NSTACKWORDS
    std r4, r5, sp[1]
    std r6, r7, sp[2]
    std r8, r10, sp[3]
{                                           ;   stw X, sp[STACK_X]                      }
{   ldc _32, 32                             ;   stw cp, sp[STACK_ORIG_CP]               }
{   ldc Q(cog_left), 0                      ;   stw dp, sp[STACK_ORIG_DP]               }

// Apply our start strides to Y, W and BSO. These will be determined by C_in and out_start
//  -   Y <-- Y + 2 * out_start       (times 2 because output is 16-bits for this version)
//  -   W <-- W + out_start * C_in
//  -   BSO <-- BSO + (out_start//16) * sizeof(nn_bso_block_t)
{                                           ;   ldw tmp, sp[STACK_OUT_START]            }
{   add Y, Y, tmp                           ;   ldw C_in, sp[STACK_CIN]                 }
{   add Y, Y, tmp                           ;                                           }
    lmul Q(cog_left), W, tmp, C_in, W, Q(cog_left)  // Q(cog_left) is 0 and shouldn't change here.
    ldc r11, nn_bso_block_t_sizeof
{   shr r8, tmp, 4                          ;   vclrdr                                  }
    lmul Q(cog_left), r3, r8, r11, r3, Q(cog_left)  // Q(cog_left) is 0 and shouldn't change here.   

    // r3 originally contained BSO pointer, but we want it in the dp register.
{                                           ;   set dp, r3                              }

#if FULLY_CONNECTED_8_SAFETY_CHECKS_ENABLED
// Error checks
//  - C_in must be a multiple of 4
//  - Output start must be a multiple of 16
{   mov r11, C_in                           ;                                           }
{   zext r11, 2                             ;   zext tmp, 4                             }
{   ecallt r11                              ;   ecallt tmp                              }
#endif //FULLY_CONNECTED_8_SAFETY_CHECKS_ENABLED

// This function will process the tail of X BEFORE it processes full input groups, and so the FIRST (C_in%32) bytes
//  of X will be considered the input tail. They will be aligned to the HIGH INDICES in the tail vector, prepended with
//  zeros. (It has to be at the high indices of the vector so that W ends up pointing at the correct place afterwards.)
//  Also just adjust the place at which sp[STACK_X] points so that we don't need to do anything with X when processing 
//  the tail. The new value at sp[STACK_X] points should be X + (C_in%32)

{   mov r11, C_in                           ;   sub X, X, _32                           }
{   zext r11, 5                             ;   ldaw tmp, sp[STACK_X_TAIL]              }
{   add r11, X, r11                         ;   sub Q(cig_left), _32, r11               }
{   mkmsk Q(cig_left), Q(cig_left)          ;   vldr r11[0]                             }
{   add X, r11, _32                         ;   vstd tmp[0]                             }
{   not Q(cig_left), Q(cig_left)            ;   stw X, sp[STACK_X]                      }
    vstrpv tmp[0], Q(cig_left)


//  cig_end: moves W to start of next cig group
//  cig_end <--  C_in * ([cog output channels] - 1) + 32
// cig_end in the cog tail is (16-out_tail)*C_in less than in a full cog

//  maccr_jump: the number of VLMACCRs to skip (when processing output tail)
//  maccr_jump <-- 16 - [cog output channels]
//      Also: if we start with a 16-bit bitmask 0x0000FFFF (which is the bitmask we need to use to write out full 
//            channel out groups, maccr_jump will be how much we have to right-shift that mask by for the cout tail.)
//            (have to right-shift by twice as much for 16-bit outputs)

//  acc_reset_ptr: points at position from which to reload accumulators (when processing output tail)
//                 It will be stored in CP
//  acc_reset_ptr <-- sp[STACK_VEC_TMP] - (32 - 2*([cog output channels] | 0x01) )

//For all full output groups maccr_jump = 0 and cig_end = 15*C_in+32  and acc_reset_ptr is irrelevant
{   shl cig_end, C_in, 4                    ;   ldw cog_left, sp[STACK_OUT_COUNT]       }
{   sub cig_end, cig_end, C_in              ;   shr cog_left, cog_left, 4               }
{   add cig_end, cig_end, _32               ;   ldc maccr_jump, 0                       }
{   shl r11, _32, 4                         ;   bf cog_left, .L_cog_loop_end            }

.L_cog_loop:
    {   ldaw r11, BSO[0]                        ;   vsetc r11                               }
    {   ldaw r11, BSO[8]                        ;   vldd r11[0]                             }
    {   shr cig_left, C_in, 5                   ;   vldr r11[0]                             }
    {   add W, W, cig_end                       ;   ldw X, sp[STACK_X]                      }
    {   sub W, W, _32                           ;   mov r11, C_in                           }

    // Handle any C_in tail *first* cig iteration.
    {   zext r11, 5                             ;   ldaw tmp, sp[STACK_VEC_TMP]             }
    {   ldaw r11, sp[STACK_X_TAIL]              ;   bf r11, .L_is_not_cig_tail              }
    .L_is_cig_tail:
        // If there's a tail on X, load it into vC and increment cig_left by 1. Adjust W
        {   neg r11, C_in                           ;   vldc r11[0]                             }
        {   zext r11, 5                             ;   add cig_left, cig_left, 1               }
        {   sub W, W, r11                           ;   bu .L_cig_tail_start                    }
    .L_is_not_cig_tail:


    .L_cig_loop:
        {   add X, X, _32                           ;   vldc X[0]                               }
    .L_cig_tail_start:
        {   shl r11, _32, 3 /* 16-bit VPU mode */   ;   bf maccr_jump, .L_full_cog              }

    // Re-align the accumulators at the end of vD:vR. They're currently at the front of vD:vR
        {   ldaw r11, acc_reset_ptr[0]              ;   vstr tmp[0]                             }
        {                                           ;   vldr r11[0]                             }
        {   add tmp, maccr_jump, 1                  ;   vstd tmp[0]                             }
        {   zext tmp, 1                             ;   vldd r11[0]                             }
        {   ldaw tmp, sp[STACK_VEC_TMP]             ;   bru tmp                                 }
        {                                           ;   vlmaccr W[0]                            }
        {   shl r11, _32, 3 /* 16-bit VPU mode */   ;   bru maccr_jump                          }

        .L_full_cog:
        {   sub W, W, C_in                          ;   vlmaccr W[0]                            }
        {   sub W, W, C_in                          ;   vlmaccr W[0]                            }
        {   sub W, W, C_in                          ;   vlmaccr W[0]                            }
        {   sub W, W, C_in                          ;   vlmaccr W[0]                            }
        {   sub W, W, C_in                          ;   vlmaccr W[0]                            }
        {   sub W, W, C_in                          ;   vlmaccr W[0]                            }
        {   sub W, W, C_in                          ;   vlmaccr W[0]                            }
        {   sub W, W, C_in                          ;   vlmaccr W[0]                            }
        {   sub W, W, C_in                          ;   vlmaccr W[0]                            }
        {   sub W, W, C_in                          ;   vlmaccr W[0]                            }
        {   sub W, W, C_in                          ;   vlmaccr W[0]                            }
        {   sub W, W, C_in                          ;   vlmaccr W[0]                            }
        {   sub W, W, C_in                          ;   vlmaccr W[0]                            }
        {   sub W, W, C_in                          ;   vlmaccr W[0]                            }
        {   sub W, W, C_in                          ;   vlmaccr W[0]                            }
        {   sub cig_left, cig_left, 1               ;   vlmaccr W[0]                            }

        {   add W, W, cig_end                       ;   bt cig_left, .L_cig_loop                }
    
.L_cig_loop_end:
{   ldaw r11, BSO[16]                       ;   vsetc r11     /*  16-bit mode */        }
{   ldaw r11, BSO[24]                       ;   vlsat r11[0]                            }
{   sub cog_left, cog_left, 1               ;   vldc r11[0]                             }
{   mkmsk Q(cig_left), 32                   ;   vstr tmp[0]                             }
{   shr Q(cig_left), Q(cig_left), maccr_jump;   vclrdr                                  }
{   ldaw r11, BSO[32]                       ;   vlmacc tmp[0]                           }
{   ldaw r11, BSO[40]                       ;   vldc r11[0]                             }
{   ldaw r11, BSO[48]                       ;   vlmacc r11[0]                           }
{   ldaw BSO, BSO[56]                       ;   vlsat r11[0]                            }
{   shl r11, _32, 4                         ;   shr Q(cig_left), Q(cig_left), maccr_jump}
    vstrpv Y[0], Q(cig_left)
{   add Y, Y, _32                           ;   bt cog_left, .L_cog_loop                }
.L_cog_loop_end:

// Check if there's a cog tail. If so, do one more cog loop using the corrected maccr_jump
//  and cig_end
{   shr r11, _32, 1                         ;   ldw Q(cig_left), sp[STACK_OUT_COUNT]    }
{   zext Q(cig_left), 4                     ;   ldc cog_left, 1                         }
{   sub Q(X), r11, Q(cig_left)              ;   bf Q(cig_left), .Lfunc_end              }
{   andnot Q(X), cog_left                   ;   mov maccr_jump, Q(X)                    }
{   shl Q(X), Q(X), 1                       ;   sub cig_end, Q(cig_left), 1             }
    mul cig_end, cig_end, C_in
{   ldaw tmp, sp[STACK_VEC_TMP]             ;   ldc r11, 0                              }
{   sub Q(X), tmp, Q(X)                     ;   stw r11, sp[STACK_OUT_COUNT]            }
{   add cig_end, cig_end, _32               ;   set cp, Q(X)                            }
{   shl r11, _32, 4                         ;   bu .L_cog_loop                          }


.Lfunc_end:
    ldd r4, r5, sp[1]
    ldd r6, r7, sp[2]
    ldd r8, r10, sp[3]
{                                           ;   ldw cp, sp[STACK_ORIG_CP]               }
{                                           ;   ldw dp, sp[STACK_ORIG_DP]               }
    retsp NSTACKWORDS


    .cc_bottom FUNCTION_NAME.function
    .set   FUNCTION_NAME.nstackwords,NSTACKWORDS
    .globl FUNCTION_NAME.nstackwords
    .set   FUNCTION_NAME.maxcores,1
    .globl FUNCTION_NAME.maxcores
    .set   FUNCTION_NAME.maxtimers,0
    .globl FUNCTION_NAME.maxtimers
    .set   FUNCTION_NAME.maxchanends,0
    .globl FUNCTION_NAME.maxchanends
.Ltmp0:
    .size  FUNCTION_NAME, .Ltmp0-FUNCTION_NAME
    .issue_mode  single

#endif


