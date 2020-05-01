
#if defined(__XS3A__)

/**


void nn_conv2d_hstrip_depthwise_bias_adj_asm(
    data16_t* adj_biases,
    const int8_t* K,
    const data16_t* bias_hi_lo,
    const int8_t* zero_point_vec,
    const unsigned K_h,
    const unsigned K_w,
    const unsigned K_c,
    const unsigned pad_t,
    const unsigned pad_b);
*/

#define FUNCTION_NAME nn_conv2d_hstrip_depthwise_bias_adj_asm

#define NSTACKWORDS  48
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 4
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME


#define STACK_K_H               (NSTACKWORDS+1)
#define STACK_K_W               (NSTACKWORDS+2)
#define STACK_K_C               (NSTACKWORDS+3)
#define STACK_PAD_T             (NSTACKWORDS+4)
#define STACK_PAD_B             (NSTACKWORDS+5)

#define STACK_VEC_TMP           (NSTACKWORDS-8)
#define STACK_VEC_ZERO          (NSTACKWORDS-16)



#define arg_BSS         r2
#define arg_zpvec       r3

#define adj_biases            r0
#define K_top           r1
#define K_h             arg_BSS
#define K_w             arg_zpvec
#define K_c             r4
#define K_bot           r5
#define cols_left       r6
#define pad_t           r7
#define pad_b           r8

#define Q(RSOMETHING)    RSOMETHING

.align 16
FUNCTION_NAME:
/**/dualentsp NSTACKWORDS 
    {   ldc r11, 32                             ;   stw r10, sp[8]                          }
    std r8, r9, sp[3]   
    {   shl r11, r11, 4                         ;                                           }
/**/std r4, r5, sp[1]
    std r6, r7, sp[2]
    {   ldc r11, 32                             ;   vsetc r11                               }

    {                                           ;   vldc arg_zpvec[0]                       }
/**/{                                           ;   ldw K_c, sp[STACK_K_C]                  }
    {                                           ;   ldw K_w, sp[STACK_K_W]                  }
    mul Q(K_w), K_w, K_c
    {   add r11, arg_BSS, r11                   ;   vldd arg_BSS[0]                         }
/**/{   mov K_bot, K_top                        ;   ldw K_h, sp[STACK_K_H]                  }
    {   ldc Q(cols_left), 0                     ;   ldw pad_b, sp[STACK_PAD_B]              }
    {   sub Q(K_h), K_h, pad_b                  ;   ldw pad_t, sp[STACK_PAD_T]              }
    maccu Q(cols_left), K_bot, Q(K_h), Q(K_w)
/**/mul pad_t, pad_t, Q(K_w)
    mul pad_b, pad_b, Q(K_w)
    {                                           ;   vldr r11[0]                             }
    {   ldc r11, 32                             ;   bf pad_t, .L_bstuff                     }
    .L_top_loop:
        {   sub pad_t, pad_t, K_c                   ;   vlmacc K_top[0]                         }
        {   add K_top, K_top, K_c                   ;   bt pad_t, .L_top_loop                   }
    .L_bstuff:
/**/{                                           ;   bf pad_b, .L_finish                     }
    .L_bot_loop:
        {   sub pad_b, pad_b, K_c                   ;   vlmacc K_bot[0]                         }
        {   add K_bot, K_bot, K_c                   ;   bt pad_b, .L_bot_loop                   }
    
    .L_finish:
    {   add adj_biases, adj_biases, r11         ;   vstd adj_biases[0]                      }
/**/{                                           ;   vstr adj_biases[0]                      }

.Lfunc_end:
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


