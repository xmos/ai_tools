
#if defined(__XS3A__)

#include "nn_config.h"
#include "../../asm_constants.h"

/**


void nn_conv2d_hstrip_depthwise(
    int8_t* Y,
    const int8_t* X, 
    const int8_t* K,
    const nn_bso_block_t* BSO,
    const unsigned K_h,
    const unsigned K_w,
    const int32_t xk_col_stride,
    const int32_t x_row_stride,
    const int32_t window_hstride,
    const int32_t y_col_stride,
    const unsigned out_cols,
    const unsigned chans_to_write);
*/

#define FUNCTION_NAME nn_conv2d_hstrip_depthwise

#define NSTACKVECS   1
#define NSTACKWORDS  (8*NSTACKVECS+12)
    
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
#define STACK_WINDOW_HSTRIDE    (NSTACKWORDS+5)
#define STACK_Y_COL_STRIDE      (NSTACKWORDS+6)
#define STACK_OUT_COLS          (NSTACKWORDS+7)
#define STACK_CHAN2WRITE        (NSTACKWORDS+8)

#define STACK_VEC_TMP           (NSTACKWORDS-8)

#define STACK_CP            (1)
#define STACK_K_PSTART      (9)
#define STACK_BSO_PSTART    (10)
#define STACK_TMP           (11)


#define Y               r0
#define X               r1
#define K               r2
#define xk_col_stride   r3
#define x_row_stride    r4
#define _32             r5
#define rows_left       r6
#define cols_left       r7
#define window_hstride  r8
#define pix_left        r9
#define X_start         r10

#define Q(RSOMETHING)    RSOMETHING

.align 16


FUNCTION_NAME:
    dualentsp NSTACKWORDS 
    {   ldc r11, 32                             ;   stw r10, sp[8]                          }
    std r8, r9, sp[3]   
    {   shl r11, r11, 4                         ;                                           }
    std r4, r5, sp[1]
    std r6, r7, sp[2]
    {   ldc r10, 32                             ;   vsetc r11                               }

    //Change constant pool pointer to refer to the constant VPU vects needed here
    ldaw r11, cp[vpu_vects]
    {   ldaw r11, cp[0]                         ;   set cp, r11                             }
    {                                           ;   stw r11, sp[STACK_CP]                   }

    {                                           ;   stw r3, sp[STACK_BSO_PSTART]            }
    {                                           ;   ldw r5, sp[STACK_XK_COL_STRIDE]         }
    {                                           ;   stw K, sp[STACK_K_PSTART]               }
    {                                           ;   ldw x_row_stride, sp[STACK_X_ROW_STRIDE]}
    

// #define Y               r0
// #define X               r1
// #define K               r2

    {   mov xk_col_stride, r5                      ;   ldw window_hstride, sp[STACK_WINDOW_HSTRIDE]}
    {   mov X_start, X                             ;   ldw pix_left, sp[STACK_OUT_COLS]        }
    {   ldc _32, 32                                ;   bu .L_pix_start                         }

    .align 16
    .L_pix_start:
        {   sub pix_left, pix_left, 1               ;   mov X, X_start                          }
        {    add X_start, X_start, window_hstride   ;   ldw r11, sp[STACK_BSO_PSTART]           }
        {                                           ;   ldw K, sp[STACK_K_PSTART]               }
#if CONFIG_SYMMETRIC_SATURATION_conv2d_depthwise
        
#else
        {   ldaw r11, cp[VPU_VEC_0x80]              ;   ldw Q(rows_left), sp[STACK_CHAN2WRITE]  }
        {   mkmsk Q(rows_left), Q(rows_left)        ;   vldr r11[0]                             }
        vstrpv Y[0], Q(rows_left)
        {                                           ;   ldw r11, sp[STACK_BSO_PSTART]           }
#endif
        {   add r11, r11, _32                       ;   vldd r11[0]                             }
        {                                           ;   vldr r11[0]                             }
        {                                           ;   ldw rows_left, sp[STACK_K_H]            }
        .L_patch_row:
            {   sub rows_left, rows_left, 1             ;   ldw cols_left, sp[STACK_K_W]            }
            .L_patch_col:
                {   sub cols_left, cols_left, 1             ;   vldc X[0]                               }
                {   add X, X, xk_col_stride                 ;   vlmacc K[0]                             }
                {   add K, K, xk_col_stride                 ;   bt cols_left, .L_patch_col              }
            {   add X, X, x_row_stride                  ;   bt rows_left, .L_patch_row              }

#define BSO     rows_left
    .L_pix_finish:
#if CONFIG_SYMMETRIC_SATURATION_conv2d_depthwise
        {   shl r11, _32, 3                         ;   ldw BSO, sp[STACK_BSO_PSTART]           }
        {   shl Q(cols_left), _32, 1                ;   vsetc r11                               }
        {   add BSO, BSO, Q(cols_left)              ;   ldw Q(cols_left), sp[STACK_CHAN2WRITE]  }
        {   ldaw r11, sp[STACK_VEC_TMP]             ;   vlsat BSO[0]                            }
        {   add BSO, BSO, _32                       ;   vstr r11[0]                             }   
        {   add BSO, BSO, _32                       ;   vldc BSO[0]                             }
        {                                           ;   vclrdr                                  }
        {   shl r11, _32, 4                         ;   vlmacc r11[0]                           }
        {   add BSO, BSO, _32                       ;   vldc BSO[0]                             }
        {   add BSO, BSO, _32                       ;   vlmacc BSO[0]                           }

        {   mkmsk Q(cols_left), Q(cols_left)        ;   vsetc r11                               }
        {                                           ;   vlsat BSO[0]                            }
        vstrpv Y[0], Q(cols_left)
        {                                           ;   ldw Q(cols_left), sp[STACK_Y_COL_STRIDE]}
        {   add Y, Y, Q(cols_left)                  ;   bt pix_left, .L_pix_start               }
#else
        {   shl r11, _32, 3                         ;   ldw BSO, sp[STACK_BSO_PSTART]           }
        {   shl Q(cols_left), _32, 1                ;   vsetc r11                               }
        {   add BSO, BSO, Q(cols_left)              ;   ldw Q(cols_left), sp[STACK_CHAN2WRITE]  }
        {   ldaw r11, sp[STACK_VEC_TMP]             ;   vlsat BSO[0]                            }
        {   add BSO, BSO, _32                       ;   vstr r11[0]                             }   
        {   add BSO, BSO, _32                       ;   vldc BSO[0]                             }
        {   mkmsk Q(cols_left), Q(cols_left)        ;   vclrdr                                  }
        {                                           ;   vlmacc r11[0]                           }
        {   add BSO, BSO, _32                       ;   vldc BSO[0]                             }
        {   add BSO, BSO, _32                       ;   vlmacc BSO[0]                           }
        {                                           ;   vlsat BSO[0]                            }

        {   ldaw r11, cp[VPU_VEC_0x007F]            ;   vstr r11[0]                             }
        {   ldaw r11, sp[STACK_TMP]                 ;   vladd r11[0]                            }
        {   mkmsk Q(rows_left), 4                   ;   vdepth1                                 }
        vstrpv r11[0], Q(rows_left)
        {   ldc Q(rows_left), 0                     ;                                           }
        {   ldaw r11, sp[STACK_VEC_TMP]             ;   sub Q(rows_left), Q(rows_left), 8       }
        vlashr r11[0], Q(rows_left)
        {   ldc r11, 32                             ;   ldw Q(rows_left), sp[STACK_TMP]         }
        {   andnot Q(cols_left), Q(rows_left)       ;   vdepth8                                 }
        vstrpv Y[0], Q(cols_left)
        {   shl r11, r11, 4                         ;   ldw Q(cols_left), sp[STACK_Y_COL_STRIDE]}
        {   add Y, Y, Q(cols_left)                  ;   vsetc r11                               }
        {                                           ;   bt pix_left, .L_pix_start               }

#endif


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



