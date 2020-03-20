
#if defined(__XS3A__)

/**


void nn_compute_patch_depthwise_asm(
    int8_t* Y,
    const int8_t* X, 
    const int8_t* K,
    const nn_bss_block_t* BSS,
    const unsigned K_h,
    const unsigned K_w,
    const int32_t xk_col_stride,
    const int32_t x_row_stride,
    const unsigned chans_to_write);
*/

#define FUNCTION_NAME nn_compute_patch_depthwise_asm

#define NSTACKWORDS  20
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 4
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME


#define STACK_K_H               (NSTACKWORDS+1)
#define STACK_K_W               (NSTACKWORDS+2)
#define STACK_XK_COL_STRIDE     (NSTACKWORDS+3)
#define STACK_X_ROW_STRIDE      (NSTACKWORDS+4)
#define STACK_CHAN2WRITE        (NSTACKWORDS+5)

#define STACK_VEC_TMP           (NSTACKWORDS-8)

// Use of this macro is a hint to you that the register isn't being used for it's nominal purpose
#define Q(RSOMETHING)    RSOMETHING

#define Y               r0
#define X               r1
#define K               r2
#define BSS             r3
#define xk_col_stride   r4
#define x_row_stride    r5
#define _32             r6
#define rows_left       r7
#define cols_left       r8


.align 16
FUNCTION_NAME:
/**/dualentsp NSTACKWORDS 
    {   ldc r11, 32                             ;   stw r10, sp[8]                          }
    std r8, r9, sp[3]   
    {   shl r11, r11, 4                         ;                                           }
/**/std r4, r5, sp[1]
    std r6, r7, sp[2]
    {   ldc r10, 32                             ;   vsetc r11                               }

    {   add r11, BSS, r10                       ;   vldd BSS[0]                             }
/**/{   add BSS, r11, r10                       ;   vldr r11[0]                             }
    {   ldc _32, 32                             ;   ldw xk_col_stride, sp[STACK_XK_COL_STRIDE] }
    {                                           ;   ldw x_row_stride, sp[STACK_X_ROW_STRIDE]}
    {                                           ;   ldw rows_left, sp[STACK_K_H]            }
    
    .align 16
    .L_pat_row:
/**/    {   sub rows_left, rows_left, 1             ;   ldw cols_left, sp[STACK_K_W]            }
        .L_pat_col:     
            {   sub cols_left, cols_left, 1             ;   vldc X[0]                               }
            {   add X, X, xk_col_stride                 ;   vlmacc K[0]                             }
            {   add K, K, xk_col_stride                 ;   bt cols_left, .L_pat_col                }
/**/    {   add X, X, x_row_stride                  ;   bt rows_left, .L_pat_row                }
        {   shl r11, _32, 3                         ;   ldw Q(cols_left), sp[STACK_CHAN2WRITE]  }
        {   ldaw r11, sp[STACK_VEC_TMP]             ;   vsetc r11                               }
        {   add BSS, BSS, _32                       ;   vlsat BSS[0]                            }
/**/    {                                           ;   vstr r11[0]                             }   
        {   add BSS, BSS, _32                       ;   vldc BSS[0]                             }
        {                                           ;   vclrdr                                  }
        {   shl r11, _32, 4                         ;   vlmacc r11[0]                           }
/**/    {   mkmsk Q(cols_left), Q(cols_left)        ;   vsetc r11                               }
        {                                           ;   vlsat BSS[0]                            }
        vstrpv Y[0], Q(cols_left)

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



