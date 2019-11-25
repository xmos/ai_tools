
#if defined(__XS3A__)


/*  
void averagepool2d_deep_asm(
    const int8_t* X, 
    int8_t* Y,
    const int32_t height, 
    const int32_t width,
    const int32_t C_in);
*/

#define FUNCTION_NAME averagepool2d_deep_asm

#define NSTACKWORDS  20
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 4
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

#define NOOOOP          nop

#define arg_X           r0
#define arg_Y           r1
#define arg_height      r2
#define arg_width       r3
    


#define XA                  r0
#define XB                  r5
#define XC                  r8
#define XD                  r6
#define Y                   r1
#define cols_iter           r3
#define rows_iter           r2
#define _16                 r7
#define mask                _16
#define c_in                r9
#define chgrps_iter         r10
#define row_bytes           r4
#define sat_vec             r11

#define STACK_C_IN          (NSTACKWORDS+1)
#define STACK_WIDTH         1

.align 4
.L_macc_const: //multiplies input in VLMACC instruction
// .byte 0x80,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40
.byte 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1

.align 4
.L_sat_vec:  //used in VLSAT instruction
.short 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2



.align 4
FUNCTION_NAME:
    dualentsp NSTACKWORDS
    std r4, r5, sp[1]
    std r6, r7, sp[2]
    std r8, r9, sp[3]
    {   ldc r7, 32                              ;   stw r10, sp[8]                      }
    {   shl r11, r7, 4                          ;   ldw c_in, sp[STACK_C_IN]            }
    {   shr rows_iter, rows_iter, 1             ;   vsetc r11                           }
    mul row_bytes, arg_width, c_in
    {   shr cols_iter, arg_width, 1             ;   ldc _16, 16                         }
    {   ldap r11, .L_macc_const                 ;   stw cols_iter, sp[STACK_WIDTH]      }
    {   sub rows_iter, rows_iter, 1             ;   vldc r11[0]                         }
    
    {   add XC, XA, row_bytes                   ;   NOOOOP                              }
    {   add XB, XA, c_in                        ;   add XD, XC, c_in                    }
    
    // {   NOOOOP                                  ;   NOOOOP                              }
    // {   NOOOOP                                  ;   NOOOOP                              }
    // {   NOOOOP                                  ;   NOOOOP                              }



.align 4
.L_row_start:
    {   NOOOOP                                  ;   ldw cols_iter, sp[STACK_WIDTH]      }

.align 4
.L_pix_start:
    {   shr chgrps_iter, c_in, 4                ;   NOOOOP                              }

.align 4
.L_chgrp_start:
    {   ldap r11, .L_sat_vec                    ;   vclrdr                              }
    {   add XA, XA, _16                         ;   vlmacc XA[0]                        }
    {   add XB, XB, _16                         ;   vlmacc XB[0]                        }
    {   add XC, XC, _16                         ;   vlmacc XC[0]                        }
    {   add XD, XD, _16                         ;   vlmacc XD[0]                        }
    {   mkmsk mask, 16                          ;   vlsat r11[0]                        }
    vstrpv Y[0], mask
    {   sub chgrps_iter, chgrps_iter, 1         ;   ldc _16, 16                         }
.L_chgrp_end:
    {   add Y, Y, _16                           ;   bt chgrps_iter, .L_chgrp_start      }

.L_pix_end:
    {   add XA, XA, c_in                        ;   add XB, XB, c_in                    }
    {   sub cols_iter, cols_iter, 1             ;   add XD, XD, c_in                    }
    {   add XC, XC, c_in                        ;   bt cols_iter, .L_pix_start          }

.L_row_end:
    {   add XA, XA, row_bytes                   ;   add XB, XB, row_bytes               }
    {   add XC, XC, row_bytes                   ;   add XD, XD, row_bytes               }
    {   sub rows_iter, rows_iter, 1             ;   bt rows_iter, .L_row_start          }

.L_img_end:

.Lfunc_end:
    {   NOOOOP                                  ;   ldw r10, sp[8]                      }
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



