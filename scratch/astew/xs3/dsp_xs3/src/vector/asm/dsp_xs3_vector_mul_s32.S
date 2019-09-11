
#if defined(__XS3A__)


/*  
void dsp_xs3_vector_mul_s32_asm(
    int32_t* A,
    unsigned* A_hr,
    const int32_t* B,
    const int32_t* C, 
    const int B_shr,
    const int C_shr,
    const unsigned length);
*/

#define FUNCTION_NAME	dsp_xs3_vector_mul_s32_asm

#define NSTACKWORDS  16
    
.text
.issue_mode  dual
.globl	FUNCTION_NAME
.align	4
.type	FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

/**
 * Timing analysis (instruction fetches not included):
 * 
 *  J = (length / 16)            ---   J  >=    0
 *  K = (length / 8) & 0x01      ---   K  in   {0,1}
 *  L = (length % 8) > 0         ---   L  in   {0,1}
 * 
 *  [Total instructions] = 13*J + 6*K + 10*L + 26
 * 
 * 
 * def dsp_xs3_calc_headroom_s32_asm_timing(N):
 *     J = (length // 16)
 *     K = (length // 8) & 0x01
 *     L = (length % 8) > 0
 *     return 13*J + 6*K + 10*L + 26
 */
    
#define A			r0
#define A_hr		r1
#define B			r2
#define C			r3
#define B_shr		r4
#define C_shr		r5
#define length		r6

#define vr_tmp      sp[8]
#define sz_tmp		sp[7]

#define NOOOOOOOOP  nop

FUNCTION_NAME:
    {	dualentsp NSTACKWORDS	        ;	ldc r11, 0              }
    std r4, r5, sp[0]
    std r6, r7, sp[1]
    std r8, r9, sp[2]
    {	stw r10, sp[6]			        ;	mkmsk r10, 3            }
    {   ldw length, sp[NSTACKWORDS+3]   ;   NOOOOOOOOP              }
    {   ldw B_shr,  sp[NSTACKWORDS+1]   ;   and r10, r10, length    }
    {   ldw C_shr,  sp[NSTACKWORDS+2]   ;   shr length, length, 3   }



.L_part1:
    {   vsetc r11                       ;   NOOOOOOOOP              }
    {   stw r11, sz_tmp                 ;   NOOOOOOOOP              }
    {   bf r10, .L_part2                ;   ldaw r11, vr_tmp        }
    {   vclrdr                          ;   shl r10, r10, 2         }
    {   vstr r11[0]                     ;   mkmsk r9, r10           }
    vlashr B[0], B_shr
    vstrpv r11[0], r9
    vlashr C[0], C_shr
    {   vlmul r11[0]                    ;   add B, B, r10           }
    vstrpv A[0], r9                                                             //Doesn't update vHR
    {   vstr r11[0]		                ;   add A, A, r10           }           //Does
    {   vgetc r11                       ;   add C, C, r10           }
    {   stw r11, sz_tmp                 ;   NOOOOOOOOP              }

.L_part2:   
.L_part2_A:

    {   ldc r9, 1                       ;   mov r7, A               }
    {   and r9, r9, length              ;   shr r8, length, 1       }
    {   bf r9, .L_part2_B               ;   ldc r10, 32             }
    vlashr B[0], B_shr
    {   vstr r7[0]                      ;   add r7, r7, r10         }
    {   NOOOOOOOOP                      ;   add B, B, r10           }

.L_part2_B:
    {   bf r8, .L_part3                 ;   NOOOOOOOOP              }
.L_part2_loop_top:
    vlashr B[0], B_shr
    {   vstr r7[0]                      ;   add B, B, r10           }
    vlashr B[0], B_shr
    {   add r7, r7, r10                 ;   add B, B, r10           }
    {   vstr r7[0]                      ;   sub r8, r8, 1           }
    {   bt r8, .L_part2_loop_top        ;   add r7, r7, r10         }
    
    

.L_part3:
.L_part3_A:
    {   ldw r11, sz_tmp                 ;   shr length, length, 1   }
    {   vsetc r11                       ;   NOOOOOOOOP              }
    {   bf r9, .L_part3_B               ;   NOOOOOOOOP              }

    vlashr C[0], C_shr
    {   vlmul A[0]                      ;   add C, C, r10           }
    {   vstr A[0]                       ;   add A, A, r10           }


.L_part3_B:
    {   bf length, .Lfunc_end           ;   sub length, length, 1   }
.L_part3_loop_top:
    vlashr C[0], C_shr
    {   vlmul  A[0]                     ;   add C, C, r10           }
    {   vstr   A[0]                     ;   add A, A, r10           }
    vlashr C[0], C_shr
    {   vlmul  A[0]                     ;   add C, C, r10           }
    {   vstr   A[0]                     ;   add A, A, r10           }
    {   bt length, .L_part3_loop_top    ;   sub length, length, 1   }
    
    


.Lfunc_end:
    {   vgetc r11                       ;   ldc r0, 31              }
    {   ldw r10, sp[6]                  ;   sub r0, r0, r11         }
    {   stw r0, A_hr[0]                 ;   NOOOOOOOOP              }
    ldd r8, r9, sp[2]
    ldd r6, r7, sp[1]
    ldd r4, r5, sp[0]
    {   retsp NSTACKWORDS              ;   NOOOOOOOOP               }
    

    .cc_bottom FUNCTION_NAME.function
    .set	FUNCTION_NAME.nstackwords,NSTACKWORDS
    .globl	FUNCTION_NAME.nstackwords
    .set	FUNCTION_NAME.maxcores,1
    .globl	FUNCTION_NAME.maxcores
    .set	FUNCTION_NAME.maxtimers,0
    .globl	FUNCTION_NAME.maxtimers
    .set	FUNCTION_NAME.maxchanends,0
    .globl	FUNCTION_NAME.maxchanends
.Ltmp0:
    .size	FUNCTION_NAME, .Ltmp0-FUNCTION_NAME
    .issue_mode  single

#endif



