
#if defined(__XS3A__)

/**
void nn_conv2d_hstrip_tail_shallowin(
        nn_image_t* Y,
        const nn_image_t* X,
        const nn_tensor_t* K,
        const nn_bss_block_t* BSS,
        const unsigned K_h,
        const unsigned K_h_stride,
        const channel_count_t C_in,
        const mem_stride_t x_v_stride,
        const mem_stride_t y_h_stride,
        const unsigned out_cols,
        const channel_count_t C_out_tail)
*/

#define FUNCTION_NAME nn_conv2d_hstrip_tail_shallowin


#define NSTACKVECS  (1)

#define NSTACKWORDS  ((NSTACKVECS*8)+12)
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 16
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME


#define STACK_K_H               (NSTACKWORDS+1)
#define STACK_K_h_stride        (NSTACKWORDS+2)
#define STACK_C_IN              (NSTACKWORDS+3)
#define STACK_X_V_STRIDE        (NSTACKWORDS+4)
#define STACK_Y_H_STRIDE        (NSTACKWORDS+5)
#define STACK_OUT_COLS          (NSTACKWORDS+6)
#define STACK_C_OUT_TAIL        (NSTACKWORDS+7)

#define STACK_VEC_TMP1          (NSTACKWORDS-8)

#define STACK_K                 7
#define STACK_BSS               8
#define STACK_WRITE_MASK        9
#define STACK_Y                 10
#define STACK_WIN_H_STRIDE      STACK_K_h_stride


#define Y               r0
#define X               r1
#define K               r2
#define BSS             r3
#define x_v_stride      r4
#define k_cout_str      r5
#define rows_left       r6
#define _32             r7
#define X_patch         r8
#define tail_mod        r9
#define tmp             r10

#define Q(R)      R

.align 16
FUNCTION_NAME:
    dualentsp NSTACKWORDS
    std r4, r5, sp[0]
    std r6, r7, sp[1]
    std r8, r9, sp[2]
    // We need 32 all over the place, because more instructions can't use it as an immediate.
    {   ldc _32, 32                             ;   stw r10, sp[6]                          }

    // Store Y on the stack. We'll load/store it as needed.
    {   shl r11, _32, 4                         ;   stw Y, sp[STACK_Y]                      }
    {                                           ;   stw K, sp[STACK_K]                      }

    //Set VPU mode to 8-bit
    {                                           ;   vsetc r11                               }

    // Push BSS onto the stack. We'll need to reset it for each output pixel
    {                                           ;   stw BSS, sp[STACK_BSS]                  }

    // - Compute the number of rows of the patch between the top and bottom padding. Store
    //   on the stack, as we'll need to reset that for every output pixel
    {                                           ;   ldw tmp, sp[STACK_K_H]                  }

    // - k_cout_str = 32*K_h;  The shape of K is  (C_out, K_h, K_w, C_in), with the
    //      restriction that C_in * K_w = 32, so the memory stride to increment/decrement
    //      just C_out index is K_h << 5.

    // WIN_H_STRIDE <bytes> = K_h_stride <pixels> * C_in <bytes/pixel>
    {   shl k_cout_str, tmp, 5                  ;   ldw tmp, sp[STACK_K_h_stride]           }
    {                                           ;   ldw r11, sp[STACK_C_IN]                 }
    mul tmp, tmp, r11
    {                                           ;   stw tmp, sp[STACK_WIN_H_STRIDE]         }

    // tail_mod = 2*(16-C_out_tail)
    //      tail_mod is the offset needed to reset accumulator positions to the end of vD:vR, 
    //      which is where they need to be to begin doing VLMACCRs
    // tail_mod' = (tail_mod >> 1)-4   
    //      tail_mod' is the argument to the BRU instruction that determines how many VLMACCRs
    //      are actually done.
    // Because tail_mod can be transformed to tail_mod' (and back) using only operations that
    //  can be encoded in immediates, I'll use one register for both, and use spare left-hand
    //  dual-issue slots to transform back and forth between the two.
    // OPTION:  For the cost of 4 words per VLMACCR group (2 groups in this function, so 32
    //          bytes total) I could get rid of the -4, if I don't have enough spare left-hand
    //          DI slots to perform the two-step transformations. (This would only be beneficial
    //          if there are no FNOPs where the step can be inserted)
    {   ldc r11, 16                             ;   ldw tmp, sp[STACK_C_OUT_TAIL]           }
    {   sub tmp, r11, tmp                       ;   mkmsk r11, tmp                          }
    {   shl tail_mod, tmp, 1                    ;   stw r11, sp[STACK_WRITE_MASK]           }

    // Load other needed parameters
    {                                           ;   ldw x_v_stride, sp[STACK_X_V_STRIDE]    }
    
    .L_out_col_start:
        // K, BSS and X_patch and rows_left reset each iteration.
        // Decrement out columns remaining
        {   ldc _32, 32                             ;   ldw K, sp[STACK_K]                      }
        {   sub tmp, tmp, 1                         ;   ldw BSS, sp[STACK_BSS]                  }
        {   mov X_patch, X                          ;   stw tmp, sp[STACK_OUT_COLS]             }
        {   ldaw tmp, sp[STACK_VEC_TMP1]            ;   ldw rows_left, sp[STACK_K_H]            }

        // Initialize accumulators with biases
        {   add r11, BSS, _32                       ;   vldd BSS[0]                             }
        {   add BSS, r11, _32                       ;   vldr r11[0]                             }

        .L_patch_row_start:
            // Load next patch row from X into vC
            {   sub r11, tmp, tail_mod                  ;   vldc X_patch[0]                         }

            // reset accumulator positions
            {   shr tail_mod, tail_mod, 1               ;   vstr tmp[0]                             }
            {   sub tail_mod, tail_mod, 4               ;   vldr r11[0]                             }
            {   sub rows_left, rows_left, 1             ;   vstd tmp[0]                             }
            {   mov tmp, K                              ;   vldd r11[0]                             }

            //do vlmaccrs
            {   add tail_mod, tail_mod, 4               ;   bru tail_mod                            }

            // Do VLMACCRs
            {   sub tmp, tmp, k_cout_str                ;   vlmaccr tmp[0]                          }
            {   sub tmp, tmp, k_cout_str                ;   vlmaccr tmp[0]                          }
            {   sub tmp, tmp, k_cout_str                ;   vlmaccr tmp[0]                          }
            {   sub tmp, tmp, k_cout_str                ;   vlmaccr tmp[0]                          }

            {   sub tmp, tmp, k_cout_str                ;   vlmaccr tmp[0]                          }
            {   sub tmp, tmp, k_cout_str                ;   vlmaccr tmp[0]                          }
            {   sub tmp, tmp, k_cout_str                ;   vlmaccr tmp[0]                          }
            {   sub tmp, tmp, k_cout_str                ;   vlmaccr tmp[0]                          }

            {   sub tmp, tmp, k_cout_str                ;   vlmaccr tmp[0]                          }
            {   sub tmp, tmp, k_cout_str                ;   vlmaccr tmp[0]                          }
            {   sub tmp, tmp, k_cout_str                ;   vlmaccr tmp[0]                          }
            {   shl tail_mod, tail_mod, 1               ;   vlmaccr tmp[0]                          }

            {   add X_patch, X_patch, x_v_stride        ;   ldaw tmp, sp[STACK_VEC_TMP1]            }

            // Each row of K (second dimension) is exactly 32 bytes. Iterate if rows remain.
            {   add K, K, _32                           ;   bt rows_left, .L_patch_row_start        }

        .L_patch_row_end:

        {   shl r11, _32, 3                         ;                                           }
        {   ldaw r11, sp[STACK_VEC_TMP1]            ;   vsetc r11            /*set 16-bit mode*/}
        {   add BSS, BSS, _32                       ;   vlsat BSS[0]       /*apply first shift*/}
        {                                           ;   vstr r11[0] /*save 16-bit intermediate*/}
        {   add BSS, BSS, _32                       ;   vldc BSS[0]       /*load scale into vC*/}
        {                                           ;   vclrdr   /*clear accumulate for VLMACC*/}
        {   shl r11, _32, 4                         ;   vlmacc r11[0]              /*do VLMACC*/}
        {                                           ;   vsetc r11             /*set 8-bit mode*/}
        {                                           ;   vlsat BSS[0]       /*apply final shift*/}
        {                                           ;   ldw tmp, sp[STACK_WRITE_MASK]           }
        vstrpv Y[0], tmp  /* Store output */
        {                                           ;   ldw tmp, sp[STACK_Y_H_STRIDE]           }
        {   add Y, Y, tmp                           ;   ldw tmp, sp[STACK_WIN_H_STRIDE]         }
        {   add X, X, tmp                           ;   ldw tmp, sp[STACK_OUT_COLS]             }
        {                                           ;   bt tmp, .L_out_col_start                }
    .L_out_col_end:


.Lfunc_end:
    {                                           ;   ldw r10, sp[6]                          }
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
