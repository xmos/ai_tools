
#if defined(__XS3A__)


/*  
void maxpool2d_deep_asm(
    const int8_t* X, 
    int8_t* Y,
    const int32_t height, 
    const int32_t width,
    const int32_t C_in);
*/

#define FUNCTION_NAME maxpool2d_deep_asm

#define NSTACKWORDS  20
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 4
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

#define NOOOOP          nop
    
#define X    r0
#define Y    r1
#define height          r2
#define ROW       height
#define width           r3
#define COL       width
#define C_in            r4      //After loading C_in, this holds C_in/32
#define PXL_JMP         r5      //Bytes between each pixel (C_in)
#define TMP_BUFF        r6      //Temporary storage space on the stack
#define ROW_BYTES       r7
#define X2              r8      //

#define STACK_X_BCK_TRK     7
#define STACK_Y_BCK_TRK     8
#define STACK_HEIGHT        9
#define STACK_WIDTH         10
#define STACK_CHANS         (NSTACKWORDS+1)
#define STACK_ROW_BYTES     11
#define STACK_TMP           12

.align 8


FUNCTION_NAME:
    dualentsp NSTACKWORDS
    std r4, r5, sp[0]
    std r6, r7, sp[1]
    std r8, r9, sp[2]
    {   ldc r8, 32                              ;   stw r10, sp[6]                      }
    {   shl r11, r8, 4                          ;   ldw C_in, sp[STACK_CHANS]           }
    {   mov PXL_JMP, C_in                       ;   vsetc r11                           }
    mul r7, width, C_in
    {   NOOOOP                                  ;   stw r7, sp[STACK_ROW_BYTES]         }
    mul r7, r7, height
    {   shr r9, r7, 2                           ;   stw height, sp[STACK_HEIGHT]        }
    {   sub r7, r7, r8                          ;   stw width, sp[STACK_WIDTH]          }
    {   sub r9, r9, r8                          ;   stw r7, sp[STACK_X_BCK_TRK]         }
    {   shr C_in, C_in, 5                       ;   stw r9, sp[STACK_Y_BCK_TRK]         }
    {   ldaw TMP_BUFF, sp[STACK_TMP]            ;   ldw ROW_BYTES, sp[STACK_ROW_BYTES]  }
    {   add X2, X, ROW_BYTES                    ;   sub Y, Y, PXL_JMP                   }

.L_chan_start:
    {   NOOOOP                                  ;   ldw ROW, sp[STACK_HEIGHT]           }

.L_row_start:
    {   mov r11, X                              ;   ldw COL, sp[STACK_WIDTH]            }

.L_pix_start:
    {   add Y, Y, PXL_JMP                       ;   vldr r11[0]                         } //vR = A
    {   add X, X, PXL_JMP                       ;   vstr Y[0]                           } //Y = A
    {   NOOOOP                                  ;   vlsub X[0]                          } //vR = B - A
    {   NOOOOP                                  ;   vdepth1                             } //vR = [bitmask of A>B]
    {   mov r11, X                              ;   vstr TMP_BUFF[0]                    } //TMP = [bitmask of A>B]
    {   add X, X, PXL_JMP                       ;   ldw r9, TMP_BUFF[0]                 } //r9 = [bitmask of A>B]
    {   not r9, r9                              ;   vldr r11[0]                         } //vR = B; r9 = [bitmask of B>=A]
    vstrpv Y[0], r9                                                                       //Y = MAX(A,B)

    {   mov r11, Y                              ;   NOOOOP                              } 
    {   mov r11, X2                             ;   vldr r11[0]                         } //vR  = MAX(A,B)
    {   add X2, X2, PXL_JMP                     ;   vlsub r11[0]                        } //vR  = C - MAX(A,B)
    {   NOOOOP                                  ;   vdepth1                             } //vR  = [bitmask of MAX(A,B)>C]
    {   NOOOOP                                  ;   vstr TMP_BUFF[0]                    } //TMP = [bitmask of MAX(A,B)>C]
    {   NOOOOP                                  ;   ldw r9, TMP_BUFF[0]                 } //r9  = [BITMASK of MAX(A,B)>C]
    {   not r9, r9                              ;   vldr r11[0]                         } //vR  = C; r9 = [bitmask of C>=MAX(A,B)]
    vstrpv Y[0], r9                                                                       //Y = MAX(A,B,C)
    
    {   mov r11, Y                              ;   NOOOOP                              }
    {   mov r11, X2                             ;   vldr r11[0]                         }
    {   add X2, X2, PXL_JMP                     ;   vlsub r11[0]                        }
    {   NOOOOP                                  ;   vdepth1                             }
    {   NOOOOP                                  ;   vstr TMP_BUFF[0]                    }
    {   sub COL, COL, 2                         ;   ldw r9, TMP_BUFF[0]                 }
    {   not r9, r9                              ;   vldr r11[0]                         } 
    vstrpv Y[0], r9                                                                       //Y = MAX(A,B,C,D)
.L_pix_end:
    {   mov r11, X                              ;   bt COL, .L_pix_start                }
.L_row_end:

    {   sub ROW, ROW, 2                         ;   ldw r9, sp[STACK_X_BCK_TRK]         }
    {   add X, X, ROW_BYTES                     ;   NOOOOP                              }
    {   add X2, X2, ROW_BYTES                   ;   bt ROW, .L_row_start                }

.L_img_end:
    {   sub X, X, r9                            ;   sub X2, X2, r9                      }
    {   sub C_in, C_in, 1                       ;   ldw r9, sp[STACK_Y_BCK_TRK]         }
    {   sub Y, Y, r9                            ;   bt C_in, .L_chan_start              }

.Lfunc_end:
    {   NOOOOP                                  ;   ldw r10, sp[6]                      }
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



