
#if defined(__XS3A__)

/**
void nn_conv2d_hstrip_shallowin_asm(
        nn_image_t* Y,
        const nn_image_t* X,
        const nn_tensor_t* K,
        const nn_bss_block_t* BSS,
        const unsigned K_h,
        const unsigned K_h_stride,
        const channel_count_t C_in,
        const mem_stride_t x_v_stride,
        const mem_stride_t y_h_stride,
        const unsigned out_cols)
*/

#define FUNCTION_NAME nn_conv2d_hstrip_shallowin_asm


#define NSTACKVECS  (1)

#define NSTACKWORDS  ((NSTACKVECS*8)+10)
    
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

#define STACK_VEC_TMP           (NSTACKWORDS-8)

#define STACK_K                 7
#define STACK_BSS               8
#define STACK_WIN_H_STRIDE      STACK_K_h_stride


#define Y               r0
#define X               r1
#define K               r2
#define BSS             r3
#define x_v_stride      r4
#define k_cout_str      r5
#define rows_left       r6
#define _32             r7
#define tmp             r8
#define X_patch         r9

#define Q(R)      R

.align 16
FUNCTION_NAME:
    dualentsp NSTACKWORDS
    std r4, r5, sp[0]
    std r6, r7, sp[1]
    std r8, r9, sp[2]
    {   ldc _32, 32                             ;   stw r10, sp[6]                          }
    {   shl r11, _32, 4                         ;   stw K, sp[STACK_K]                      }
    {                                           ;   vsetc r11                               }
    {                                           ;   stw BSS, sp[STACK_BSS]                  }
    {                                           ;   ldw tmp, sp[STACK_K_H]                  }
    {   shl k_cout_str, tmp, 5                  ;   ldw tmp, sp[STACK_K_h_stride]           }
    {                                           ;   ldw r11, sp[STACK_C_IN]                 }
    mul tmp, tmp, r11
    {                                           ;   stw tmp, sp[STACK_WIN_H_STRIDE]         }
    {   mov tmp, K                              ;   ldw x_v_stride, sp[STACK_X_V_STRIDE]    }
    {                                           ;   ldw tmp, sp[STACK_OUT_COLS]             }

    .L_out_col_start:
        // load number of patch rows, initialize accumulators, decrement out cols
        {   sub tmp, tmp, 1                         ;   ldw BSS, sp[STACK_BSS]                  }
        {                                           ;   stw tmp, sp[STACK_OUT_COLS]             }
        {   mov X_patch, X                          ;   ldw rows_left, sp[STACK_K_H]            }
        {   add r11, BSS, _32                       ;   vldd BSS[0]                             }
        {   add BSS, r11, _32                       ;   vldr r11[0]                             }
        
        // K resets for every output pixel
        {                                           ;   ldw K, sp[STACK_K]                      }

        .L_patch_row_start:
            {   add X_patch, X_patch, x_v_stride        ;   vldc X_patch[0]                         }

            // Do VLMACCRs
            {   sub tmp, K, k_cout_str                  ;   vlmaccr K[0]                            }
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
            {   sub tmp, tmp, k_cout_str                ;   vlmaccr tmp[0]                          }
            {   sub tmp, tmp, k_cout_str                ;   vlmaccr tmp[0]                          }
            {   sub tmp, tmp, k_cout_str                ;   vlmaccr tmp[0]                          }
            {   sub rows_left, rows_left, 1             ;   vlmaccr tmp[0]                          }

            // Each row of K (second dimension) is exactly 32 bytes. Iterate if rows remain.
            {   add K, K, _32                           ;   bt rows_left, .L_patch_row_start        }

        .L_patch_row_end:

        {   shl r11, _32, 3                         ;                                           }
        {   ldaw r11, sp[STACK_VEC_TMP]             ;   vsetc r11            /*set 16-bit mode*/}
        {   add BSS, BSS, _32                       ;   vlsat BSS[0]       /*apply first shift*/}
        {                                           ;   vstr r11[0] /*save 16-bit intermediate*/}
        {   add BSS, BSS, _32                       ;   vldc BSS[0]       /*load scale into vC*/}
        {                                           ;   vclrdr   /*clear accumulate for VLMACC*/}
        {   shl r11, _32, 4                         ;   vlmacc r11[0]              /*do VLMACC*/}
        {                                           ;   vsetc r11             /*set 8-bit mode*/}
        {   mkmsk tmp, 16        /*16 out channels*/;   vlsat BSS[0]       /*apply final shift*/}
        vstrpv Y[0], tmp         /* Store output */
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
